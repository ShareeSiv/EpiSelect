# EpiSelect

[![arXiv](https://img.shields.io/badge/arXiv-2608.11746-b31b1b.svg)](https://arxiv.org/abs/2608.11746)

Code for [**Epiplexity Guided Data Selection and Generation for Out-of-Distribution
Generalization**](https://arxiv.org/abs/2608.11746) — epiplexity-maximizing data selection for
curriculum learning over mixed-domain pretraining corpora. EpiSelect fits a cross-domain scaling law
`L_m = beta_m (sum_k gamma_mk n_k)^-alpha_m + eps_m` and reweights batches toward the domains with the
largest marginal contribution to epiplexity, refitting every 1,000 steps. No proxy model required.
Adapted from the [ADO reference implementation](https://github.com/yidingjiang/ado).

## Setup

```bash
bash requirements.sh          # or: bash requirements.sh tpu
source .venv/bin/activate
export WANDB_API_KEY=...      # or write the key to .wandbkey
```

## Data

```bash
python scripts/build_commonpile_tfds.py --all --data_dir gs://your-bucket/commonpile_tfds
```

Streams the 31 `common-pile/*_filtered` domains and tokenizes with GPT-NeoX-20B — 43.3B tokens,
~95 GB, capped at 2B tokens per domain (`src/tfds/common_pile_grouped.py`).

## Train

Six experiments: `{124M, 1.3B} x {Natural, ADO, EpiSelect}` under `configs/experiment/`.

```bash
# single host
python launch.py +experiment=124M/EpiSelect \
  loader.data_dir=/path/to/commonpile_tfds \
  rundir=/path/to/checkpoints/124M/EpiSelect \
  multihost=false

# TPU pod
./scripts/tpu_setup.sh node1 us-central2-b
./scripts/tpu_launch.sh 124M/EpiSelect gs://your-bucket/commonpile_tfds gs://your-bucket/checkpoints
```

## Small models on a Mac (MLX)

`scripts/train_mlx.py` trains 5-20M-parameter models on the Apple-silicon GPU with MLX (`src/mlx_model.py`, a port of `src/model.py`), using the same loaders, data selectors, WSD schedule and exploration options as the JAX path. The default config (`configs/mlx.yaml`) trains a 10M model on 8 domains for 12k steps x 32 x 256 = 98M tokens, about an hour on an M5 Pro.

```bash
uv pip install -e ".[dev,mlx]"
python scripts/train_mlx.py data_dir=/path/to/commonpile_tfds rundir=outputs/mlx/episelect_s0 seed=0
python scripts/train_mlx.py sampler=synthetic  # no data on disk: bigram stand-in domains, for smoke runs
```

Each run writes `metrics.jsonl` (per-domain `val_loss/task{i}`, `val_balanced_loss`, `val_worst_loss`, the selector's metrics), `run_info.json` (domain names), `fit_data_{step}.npz` from each refit, and the final weights. Matmuls run in TF32 by default (`tf32: false` for full fp32, ~1.9x slower).

To compare the paper's joint fit (`src/scaling_law_paper.py`, from the initial commit) with the own-weight-1 fit on a run's refits, fitting on the first 50% / 70% of each curve and predicting the rest:

```bash
python scripts/compare_fits.py outputs/mlx/episelect_s0 --every 2  # --bayes adds posterior 90% intervals on gamma
```

End to end, `scripts/sweep_mlx.py` runs five selectors x seeds, one run at a time, and summarises final and stable-phase balanced and worst-domain validation loss as mean ± std over seeds:

| Arm | Loader config |
|---|---|
| `natural` | `mlx_natural`: token-proportional mixture throughout |
| `ado` | `mlx_ado` |
| `episelect_paper` | `mlx_episelect_paper`: the paper's joint fit (`fit_method: paper`) |
| `episelect_own` | `mlx_episelect`: the own-weight-1 fit |
| `gamma_identity` | `mlx_episelect_identity`: `gamma_identity: true` ablation |

The two EpiSelect fits and the ablation share exploration, refit schedule and fit window, so they differ only in the fit.

```bash
python scripts/sweep_mlx.py plan data_dir=/path/to/commonpile_tfds   # print the 15 commands
python scripts/sweep_mlx.py run data_dir=/path/to/commonpile_tfds    # run them; finished runs are skipped
python scripts/sweep_mlx.py summarize                                # table + outputs/mlx_sweep/summary.json
```

## Evaluate

Point `configs/eval.yaml` at the checkpoints, then:

```bash
python -m src.eval output=eval_results.json
```

## Citation

```bibtex
@article{su2026epiplexity,
  title   = {Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization},
  author  = {Su, Ellen and Potapczynski, Andres and Qiu, Shikai and Hughes, Edward and Wilson, Andrew Gordon},
  journal = {arXiv preprint arXiv:2608.11746},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.11746},
}
```

## Scaling-law fit (own-weight-1 variant)

The law is `L_m = eps_m + beta_m * z_m^(-alpha_m)` with `z_m = n_m + sum_{k!=m} gamma_mk n_k`: `gamma_mm = 1` and the off-diagonal weights are free and non-negative. `src/scaling_law.py` splits the fit by how each parameter enters the law:

| Parameter | How it is fit |
|---|---|
| `gamma_mk` | weighted NNLS of `(L_m - eps_m)^(-1/alpha_m)` on the token counts: convex, one answer |
| `eps_m`, `beta_m` | OLS of `L_m` on `z_m^(-alpha_m)`: exact, with `eps_m < min L_m` |
| `alpha_m` | 1-d grid search, per domain or one shared alpha (`shared_alpha`) |
| all of them | projected Levenberg-Marquardt polish in log-loss space, kept only if it does not worsen the Huber score |

For each alpha, the (eps, beta) and gamma steps alternate from `gamma = I`. Every refit narrows each domain's alpha range around its previous alpha (`alpha_range`, `alpha_shrink`, `alpha_min_halfwidth`) and can smooth gamma across refits (`gamma_smooth`). With own weight 1 nothing bounds gamma, so early refits on few points can blow up: below `min_fit_points` points, or with any off-diagonal gamma above `gamma_max`, a refit falls back to the gamma = I law (logged as `gamma_guarded`). `gamma_spread` and `fit_cond` are logged to check that gamma is identified; `gamma_l1`, `gamma_identity` (ablation), `fit_max_frac`/`fit_max_step`, the `lr_schedule: wsd` option and the early `explore_steps` phase are in the configs.

For uncertainty on gamma, set `fit_data_path` to dump each refit's `(D, L)` and run the same model with full Bayes (NUTS, needs `pip install -e .[bayes]`):

```bash
python scripts/fit_scaling_law_bayes.py fit_data_60000.npz --out posterior.npz
```

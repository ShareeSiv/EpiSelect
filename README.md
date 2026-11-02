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

## 124M on NVIDIA GPUs (Vast.ai, RunPod)

`configs/experiment/124M/EpiSelect_own_nvidia.yaml` is the paper's 124M EpiSelect run (same model, the 31 Common Pile domains, seed 21, 60k steps x 256 x 1024 = 15.7B tokens, cosine schedule, refit every 1,000 steps) with the own-weight-1 fit, on one GPU host. `loader.fit_method=paper` runs the paper's fit on the same hardware.

1. **Tokenize once, off the GPU box.** The tfds build stores GPT-NeoX-20B token ids as int32, so training never tokenizes. Build on a cheap many-core CPU instance and push it to storage:
   ```bash
   bash requirements.sh && bash scripts/gpu/build_data.sh /data/commonpile_tfds 16
   bash scripts/gpu/sync.sh /data/commonpile_tfds r2:bucket/commonpile_tfds   # local, gs://, s3:// or any rclone remote
   ```
2. **On the GPU box** (needs a CUDA 12 driver and ~250 GB disk: 95 GB of data plus checkpoints):
   ```bash
   bash scripts/gpu/setup.sh
   bash scripts/gpu/sync.sh r2:bucket/commonpile_tfds /data/commonpile_tfds
   export WANDB_API_KEY=...
   MICRO_BS=32 bash scripts/gpu/launch_124M.sh /data/commonpile_tfds /data/runs/124M_EpiSelect_own
   ```
   `MICRO_BS` is the sequences each GPU processes per forward and backward pass: 32 for 80 GB, 16 for 40-48 GB, 8 for 24 GB. The script uses gradient accumulation to make up the rest, so every step is still 256 sequences, and the domain draws are the same for any GPU count. The val set stays at the paper's 150 x 256 sequences (`eval_batch_size`), evaluated in chunks of `eval_microbatch`. Only the last 3 checkpoints are kept (`ckpt_max_to_keep`). Each one is about 1.5 GB.
3. Copy the run back: `bash scripts/gpu/sync.sh /data/runs/124M_EpiSelect_own <remote>`. It holds the checkpoints, `fit_data_*.npz` for `scripts/compare_fits.py` and `token_stats.json`.

Things to know before renting:
- **Use an on-demand (non-interruptible) instance.** Resuming restores the model and optimizer, but not the selector's loss history or the data iterators. After a restart, EpiSelect refits on a fresh history, so a resumed run no longer matches an uninterrupted one.
- Seed 21 makes the initialization identical to the paper's, because JAX's threefry RNG is platform-independent. The run is not bit-identical, because GPU kernels and bf16 rounding differ from TPU, and the paper's run sharded the tf.data streams across TPU hosts. Expect a close match, not an exact one, and compare against `fit_method=paper` on the same GPUs for a like-for-like difference.
- bf16 compute needs an Ampere or newer GPU (A100, A6000, L40S, RTX 3090/4090, H100).

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

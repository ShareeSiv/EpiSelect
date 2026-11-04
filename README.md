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

## 124M on an NVIDIA GPU in PyTorch (Vast.ai, RunPod)

`scripts/train_torch.py` trains the paper's Table 1 124M EpiSelect run in PyTorch on one GPU, with the own-weight-1 fit
(`configs/experiment/124M/EpiSelect_own_nvidia.yaml`). It uses the JAX path's configs, loaders, data selectors and
validation set. Seed 21, 60k steps x 256 x 1024 = 15.7B tokens, AdamW (lr 1e-3, beta2 0.95, weight decay 1e-4),
500 warm-up steps and cosine decay to 1e-5, refit every 1,000 steps (Appendix E.1). `loader.fit_method=paper`
runs the paper's fit on the same hardware.

- **Same model:** `src/torch_model.py` ports `src/model.py`. The initial weights come from the JAX model for the
  same seed, so the init is the paper's. As in the Equinox model, the embedding and output head start from the same
  matrix but are separate parameters that train independently. Tests check the forward pass, the optimizer update,
  harness scoring and the full loop against the JAX code (`tests/test_torch_port.py`).
- **Faster:** flash attention through `scaled_dot_product_attention`, `torch.compile`, bf16 autocast, fused AdamW, and
  no activation recompute. JAX stays on the CPU, where it builds the init and runs the selector's scaling-law fits.

1. **Tokenize once, off the GPU box.** The tfds build stores GPT-NeoX-20B token ids as int32, so training never
   tokenizes. Build on a cheap many-core CPU instance and push the result to storage:
   ```bash
   bash requirements.sh && bash scripts/gpu/build_data.sh /data/commonpile_tfds 16
   bash scripts/gpu/sync.sh /data/commonpile_tfds r2:bucket/commonpile_tfds   # local, gs://, s3:// or any rclone remote
   ```
2. **On the GPU box** (CUDA >= 12.6 driver, Ampere or newer, ~150 GB disk):
   ```bash
   bash scripts/gpu/setup.sh          # picks the PyTorch build for the driver's CUDA version
   bash scripts/gpu/sync.sh r2:bucket/commonpile_tfds /data/commonpile_tfds
   export WANDB_API_KEY=...
   MICRO_BS=32 bash scripts/gpu/launch_124M.sh /data/commonpile_tfds /data/runs/124M_EpiSelect_own
   ```
   `MICRO_BS` is the sequences per forward and backward pass. By default it's set from the GPU's memory: 32 for
   80 GB, 16 for 40-48 GB, 8 for 24-32 GB (halve it on an out-of-memory error). Gradient accumulation makes up the rest, so every step is still 256 sequences with the same
   domain draws. The val set stays at the paper's 150 x 256 sequences (`eval_batch_size`). Only the last 3
   checkpoints are kept, and the final weights go to `model.pt`. Throughput is logged as `perf/tps`.
3. **Score it on Table 1's ten tasks** (`configs/eval.yaml`; the ARC column is ARC-Challenge), then copy the run
   back:
   ```bash
   .venv/bin/python scripts/eval_torch.py 'runs=[{label: EpiSelect_own, path: /data/runs/124M_EpiSelect_own}]' \
     data_dir=/data/commonpile_tfds evals=[val,harness] output=/data/runs/eval_results.json
   bash scripts/gpu/sync.sh /data/runs <remote>
   ```
   Table 1's 124M averages: EpiSelect 0.394, ADO 0.379, Natural 0.377.

### Renting on Vast.ai from your own machine

`scripts/vast/vast.py` uses the `vastai` Python SDK to rent the GPU and run the whole job on it. On the
instance, `scripts/vast/pipeline.sh` sets up, gets the data, trains, scores Table 1's tasks and uploads the run,
writing each stage to `/workspace/STATUS`.

```bash
pip install vastai                                  # or: uv pip install -e ".[vast]"
export VAST_API_KEY=...                             # https://console.vast.ai/manage-keys/
vastai create ssh-key ~/.ssh/id_ed25519.pub         # once; status, fetch and watch use ssh
export WANDB_API_KEY=...                            # forwarded to the instance
python scripts/vast/vast.py offers                  # suitable on-demand GPUs, cheapest estimated total first
python scripts/vast/vast.py launch OFFER_ID --data-src r2:bucket/commonpile_tfds
python scripts/vast/vast.py status                  # stage, training progress, $ so far
python scripts/vast/vast.py watch --destroy         # when DONE: copy the run to ./vast_runs, then destroy
```

- `offers` filters for 1 GPU, Ampere or newer, a CUDA >= 12.6 driver, reliability >= 0.98, >= 200 GB disk, fast
  download, and bandwidth at most $2/TB each way (`--max-bw-cost`; `0` for hosts that don't charge for it). The
  `bw $` column is the bandwidth charge for this job: ~110 GB in when pulling the tokenized data (~250 GB when
  tokenizing on the instance) and ~5 GB out. `est $` adds it to GPU time, which comes from guessed tokens/s per
  GPU (`TPS_GUESS` in `vast.py`). Replace the guesses with your measured `perf/tps`.
- `launch` always rents on-demand, never interruptible. The instance clones this branch at your pushed `HEAD` and
  runs the pipeline. A restarted container picks it up again, and a finished run is left alone.
- **Data:** with `--data-src`, the instance pulls the tokenized data from an rclone remote, `s3://` or `gs://`.
  The `RCLONE_CONFIG_*` and `AWS_*` variables in your shell are forwarded. Without it, the instance tokenizes the
  data itself, which takes about 1-2 extra GPU-hours (`offers --build-data` adds that to the estimate).
- `--results-dst` uploads the run directory when it's done. Otherwise use `fetch` or `watch`. `--overrides` passes
  hydra overrides, e.g. `"loader.fit_method=paper"` for the paper's fit with `--run-name 124M_EpiSelect_paper`.
- **Billing doesn't stop by itself.** The instance keeps billing after `DONE` until it's destroyed. Run
  `watch --destroy` (your machine must stay on), or `destroy` by hand.
- The forwarded keys (W&B, HF, storage) are stored in the instance's environment, which the host can read. Use
  scoped or temporary keys.

Keys and accounts:

| What | Needed? | For |
|---|---|---|
| Vast.ai API key (`VAST_API_KEY`) | yes | renting and managing the instance |
| SSH key registered on Vast (`vastai create ssh-key`) | yes | `status`, `fetch`, `watch`, `ssh` |
| W&B API key (`WANDB_API_KEY`) | recommended | live curves; without it metrics only go to `metrics.jsonl` on the instance |
| Hugging Face token (`HF_TOKEN`, read-only) | recommended | avoids rate limits when streaming Common Pile, the tokenizer and the harness datasets |
| Storage keys (`RCLONE_CONFIG_*` or `AWS_*`) | only with `--data-src` / `--results-dst` | pulling the tokenized data, uploading results |

No GitHub token is needed: the instance clones this public repository.

Before renting:
- **Use an on-demand (non-interruptible) instance.** A restart resumes the model and optimizer, but not the selector's
  loss history or the data streams. EpiSelect then refits on a fresh history, so a resumed run no longer matches an
  uninterrupted one.
- The run will be close to the paper's but not bit-identical. The init is the same, but GPU kernels and bf16
  rounding differ from TPU, the paper's tf.data streams were split across 4 TPU hosts, and the shuffle buffers are
  unseeded. Compare against `fit_method=paper` on the same GPU for a like-for-like difference.
- `launch.py` with `requirements.sh cuda` still runs the JAX trainer on NVIDIA GPUs (any count), but it is slower.

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

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

The law is `L_m = eps_m + beta_m (n_m + sum_{k!=m} gamma_mk n_k)^(-alpha_m)`: `gamma_mm = 1` and the off-diagonal weights are free and non-negative. For a fixed `(alpha_m, eps_m)` the transformed loss `(L_m - eps_m)^(-1/alpha_m)` is linear in the token counts, so `src/scaling_law.py` solves a weighted non-negative least squares exactly and grid-searches `(alpha, eps)` outside it (with `eps < min L_m`). `gamma_spread` and `fit_cond` are logged to check that gamma is identified; `gamma_l1`, `gamma_smooth`, `gamma_identity` (ablation) and `fit_max_frac` are in `configs/loader/commonpile_episelect.yaml`.

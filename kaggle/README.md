# 124M EpiSelect on Kaggle (TPU v5e-8 or v6e-8)

This pretrains the paper's 124M model with the **own-weight-1 scaling-law fit** on Kaggle's free TPU (v5e-8, or v6e-8 if that is what Kaggle offers), then compares it with the paper on two things:

- **Downstream accuracy.** Zero-shot LM-Eval scores against the paper's Table 1. For 124M the paper reports Natural 0.377, ADO 0.379 and EpiSelect 0.394.
- **The fitted γ_{m,k} matrix.** Compared against the paper's joint fit, either from a second run trained with that fit, or by replaying our run's data through it.

The setup matches the paper (App. E.1): 12 layers, 768 wide, 12 heads, 60k steps × 256 × 1024 tokens = 15.7B tokens. The optimizer is AdamW with lr 1e-3, β2 0.95, weight decay 1e-4, 500 warmup steps and a cosine schedule down to 1e-5. The mixture is refit every 1k steps from step 1k. Each step uses 256 sequences, as 128 × 2 gradient-accumulation micro-steps.

| Config | Fit |
|---|---|
| `configs/experiment/124M/EpiSelect_kaggle.yaml` | own-weight-1 fit (`src/scaling_law.py`) |
| `configs/experiment/124M/EpiSelect_paper_kaggle.yaml` | the paper's joint fit (`src/scaling_law_paper.py`), everything else identical |

## 1. Build the corpus and upload it as Kaggle Datasets

Run this once, on any machine with good network access and a few CPU cores, not on Kaggle. It streams each Common Pile domain from Hugging Face, tokenizes it with GPT-NeoX, and writes one flat uint16 stream per domain and split.

```bash
pip install -e .
python scripts/build_compact_commonpile.py --out commonpile_compact --cap 1e9 --workers 16
```

What the build produces:

- **Size.** `--cap 1e9` gives about 23.3B tokens and about 47 GB. `--cap 2e9` gives the repo's 43.3B tokens and about 87 GB.
- **Domains.** All 31 Common Pile domains, with the same document split as `src/tfds/common_pile_grouped.py`.
- **Resumability.** The build is per domain and resumable. You can build part of the corpus with `--domains ...`, upload it, delete it, and continue.

The loader finds domains anywhere under `/kaggle/input`, so you can upload the corpus as several datasets. For example, 2–3 parts of about 20 GB each, if your Kaggle storage quota needs that. For each part:

```bash
kaggle datasets init -p commonpile_compact      # set "id": "<you>/commonpile-compact-part1" in dataset-metadata.json
kaggle datasets create -p commonpile_compact --dir-mode zip
```

Make the datasets private. After uploading, start one `bench` version to check the layout. It prints the number of windows it found for each domain.

## 2. Notebook

Kaggle runs notebooks, and this one is `kaggle/train_124m.ipynb`. Kaggle does not pick it up from GitHub on its own, so get it there one of two ways.

**In the browser.** On kaggle.com choose Create → New Notebook, then File → Import Notebook, and paste the GitHub URL of `kaggle/train_124m.ipynb` (or upload the file).

**From the command line.** In `kaggle/kernel-metadata.json`:

- replace `YOUR_KAGGLE_USERNAME` in `id`;
- list your corpus datasets in `dataset_sources`, for example `"you/commonpile-compact-part1"`;
- to resume, put the notebook itself (`"you/episelect-124m"`) in `kernel_sources`, so the previous version's output is attached.

Then push it and follow it:

```bash
kaggle kernels push -p kaggle --accelerator TpuV6E8
```

```bash
kaggle kernels status YOUR_KAGGLE_USERNAME/episelect-124m
```

```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/episelect-124m -p out/
```

Use the accelerator value Kaggle lists for the TPU it offers. Kaggle's CLI docs list `TpuV6E8`; I found no listed value for v5e-8. A v6e-8 is faster and has 32 GB of memory per chip, so the configs still work, and `bench` shows the real speed. Each push runs the notebook as a new version.

For the paper-fit arm, use a second notebook id (for example `episelect-124m-paper`) so the two arms' outputs never mix.

Either way, set the notebook up like this:

- Accelerator: **TPU v5e-8 or v6e-8**.
- Internet: on.
- Add your corpus dataset(s) as input.
- Optional: add `WANDB_API_KEY` under Add-ons → Secrets. Without it, wandb logs offline into the output.

Start every version with **Save Version → Save & Run All** so it runs in the background for up to 9 h, then set `MODE` in the first cell:

| MODE | What it does | Quota |
|---|---|---|
| `bench` | Runs 300 steps and prints s/step, the projected hours for 60k steps, and peak HBM. With no corpus attached, it uses synthetic data. **Run this first.** | ~15 min |
| `train` | Trains, or resumes from an attached previous output. It stops cleanly and checkpointed after 8.4 h and writes `run_status.json`. | ≤ 9 h per version |
| `eval` | Runs the Table 1 zero-shot tasks plus Common Pile val on the final checkpoint, and writes `eval_results.json`. | ~1 h |
| `report` | Runs `scripts/gamma_report.py`: γ heatmaps (paper Fig. 5 bins), γ statistics, agreement with the paper fit, held-out fit error, and the Table 1 comparison. | ~15 min |

Throughput sets everything else. The estimate is about 1.35e19 FLOPs per run, which at ~35% MFU on 8 × v5e chips is about 7 h. If `bench` projects more than ~16 h for 60k steps, stop and rethink before spending quota (a larger `batch_size` with `g_accum_iters=1`, or fewer steps).

### Resuming across sessions

A `train` version that runs out of time leaves `run/` in its output: the last 2 checkpoints, the selector state, `fit_data/` and `run_status.json`. To continue, start a new version with that output attached as input (Add Input → Notebook Output) and `MODE = "train"`. The notebook copies the newest attached run into `/kaggle/working/run`, and training resumes exactly where it stopped. That includes the data-selection history, the fitted scaling law, the sampling distribution and the position in every domain's data. A smoke test checks that a resumed run is bit-identical to an uninterrupted one.

**Attach only one arm's output at a time.** The notebook resumes the attached run with the highest step.

## 3. Arms and quota

Kaggle gives about 20 TPU hours per week.

1. Run the own fit (`EXPERIMENT = "124M/EpiSelect_kaggle"`): `bench`, then 1–2 `train` versions, then `eval`, then `report`. The report runs `--replay_paper`, which refits our run's final (tokens, loss) data with the paper's joint fit. That gives a γ comparison on identical data at no training cost.
2. If quota is left, run the paper fit (`EXPERIMENT = "124M/EpiSelect_paper_kaggle"`) as a separate notebook, so the outputs don't mix. Then run `report` with `paper_run` pointing at its run dir, to compare the two trained γ matrices and the two curricula end to end.

The run also writes `fit_data/fit_data_{step}.npz` at every refit. Each file holds the (D, L) data and the fitted law, in its own form and in the γ_mm = 1 form. To redo or extend the analysis offline:

```bash
python scripts/gamma_report.py run --replay_paper --heldout --eval eval_results.json
python scripts/compare_fits.py run/fit_data --every 10
```

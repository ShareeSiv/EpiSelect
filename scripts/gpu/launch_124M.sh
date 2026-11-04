#!/bin/bash
# The paper's 124M EpiSelect run (seed 21) with the own-weight fit, in PyTorch on one NVIDIA GPU, in tmux.
#   bash scripts/gpu/launch_124M.sh DATA_DIR RUNDIR [extra hydra overrides...]
#   MICRO_BS=16 bash scripts/gpu/launch_124M.sh /data/commonpile_tfds /data/runs/124M_EpiSelect_own
# Every step is the paper's 256 sequences x 1024 tokens. MICRO_BS is the sequences per forward/backward
# (default: 32 for 80 GB cards, 16 for 40-48 GB, 8 for 24-32 GB; halve it on an out-of-memory error). The rest
# is gradient accumulation, which gives the same update and the same domain draws.
# FOREGROUND=1 runs in the current shell instead of a tmux session (scripts/vast/pipeline.sh uses this).
# Rerunning with the same RUNDIR resumes from the last checkpoint -- but see the README on resuming.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_DIR=${1:?usage: launch_124M.sh DATA_DIR RUNDIR [overrides...]}
RUNDIR=${2:?usage: launch_124M.sh DATA_DIR RUNDIR [overrides...]}
shift 2
EXPERIMENT=${EXPERIMENT:-124M/EpiSelect_own_nvidia}
GLOBAL_BS=256
if [[ -z "${MICRO_BS:-}" ]]; then  # ~1.1 GB per sequence without activation recompute
  MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if (( MEM_MB >= 70000 )); then MICRO_BS=32; elif (( MEM_MB >= 38000 )); then MICRO_BS=16; else MICRO_BS=8; fi
fi

(( GLOBAL_BS % MICRO_BS == 0 )) || { echo "MICRO_BS ($MICRO_BS) must divide $GLOBAL_BS"; exit 1; }
G=$((GLOBAL_BS / MICRO_BS))
(( $(nvidia-smi -L | wc -l) > 1 )) && echo "note: the torch trainer uses one GPU (CUDA_VISIBLE_DEVICES picks which)"
echo "batch_size=$MICRO_BS x g_accum_iters=$G = $GLOBAL_BS sequences/step"

for t in arxiv_abstracts youtube; do  # quick check that the data is where it should be
  [ -d "$DATA_DIR/common_pile_grouped/$t" ] || { echo "no $DATA_DIR/common_pile_grouped/$t -- build or sync the data first"; exit 1; }
done
mkdir -p "$RUNDIR"

ENV="export TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false WANDB_PROJECT=\${WANDB_PROJECT:-episelect}"
CMD=".venv/bin/python -u scripts/train_torch.py +experiment=$EXPERIMENT loader.data_dir=$DATA_DIR rundir=$RUNDIR"
CMD="$CMD batch_size=$MICRO_BS g_accum_iters=$G eval_microbatch=$MICRO_BS $*"
echo "$CMD"

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  eval "$ENV" && eval "$CMD" 2>&1 | tee -a "$RUNDIR/train.log"
  exit "${PIPESTATUS[0]}"
fi
tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train "$ENV && $CMD 2>&1 | tee -a $RUNDIR/train.log"
echo "Started in tmux session 'train'. Attach: tmux attach -t train   Log: $RUNDIR/train.log"

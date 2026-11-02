#!/bin/bash
# The paper's 124M EpiSelect run (seed 21) with the own-weight fit, on one NVIDIA host, in tmux.
#   bash scripts/gpu/launch_124M.sh DATA_DIR RUNDIR [extra hydra overrides...]
#   MICRO_BS=16 bash scripts/gpu/launch_124M.sh /data/commonpile_tfds /data/runs/124M_own
# Every step is the paper's 256 sequences x 1024 tokens. MICRO_BS is the sequences per GPU per
# forward/backward (32 fits an 80 GB card, 16 a 40-48 GB one, 8 a 24 GB one); when
# GPUs x MICRO_BS < 256 the rest is made up with gradient accumulation, which gives the same update.
# Rerunning with the same RUNDIR resumes from the last checkpoint -- but see the README on resuming.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_DIR=${1:?usage: launch_124M.sh DATA_DIR RUNDIR [overrides...]}
RUNDIR=${2:?usage: launch_124M.sh DATA_DIR RUNDIR [overrides...]}
shift 2
EXPERIMENT=${EXPERIMENT:-124M/EpiSelect_own_nvidia}
GLOBAL_BS=256
MICRO_BS=${MICRO_BS:-32}

N_GPU=$(nvidia-smi -L | wc -l)
PER_STEP=$((N_GPU * MICRO_BS))
if (( PER_STEP >= GLOBAL_BS )); then
  G=1
else
  (( GLOBAL_BS % PER_STEP == 0 )) || { echo "GPUs ($N_GPU) x MICRO_BS ($MICRO_BS) must divide $GLOBAL_BS"; exit 1; }
  G=$((GLOBAL_BS / PER_STEP))
fi
BS=$((GLOBAL_BS / G))
EVAL_MB=$(( PER_STEP < GLOBAL_BS ? PER_STEP : GLOBAL_BS ))
(( BS % N_GPU == 0 )) || { echo "batch_size $BS not divisible by $N_GPU GPUs"; exit 1; }
echo "$N_GPU GPU(s): batch_size=$BS x g_accum_iters=$G = $GLOBAL_BS sequences/step"

for t in arxiv_abstracts youtube; do  # quick check that the data is where it should be
  [ -d "$DATA_DIR/common_pile_grouped/$t" ] || { echo "no $DATA_DIR/common_pile_grouped/$t -- build or sync the data first"; exit 1; }
done
mkdir -p "$RUNDIR"

ENV="export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false"
ENV="$ENV WANDB_PROJECT=\${WANDB_PROJECT:-episelect}"
CMD=".venv/bin/python -u launch.py +experiment=$EXPERIMENT loader.data_dir=$DATA_DIR rundir=$RUNDIR"
CMD="$CMD batch_size=$BS g_accum_iters=$G eval_microbatch=$EVAL_MB $*"
echo "$CMD"

tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train "$ENV && $CMD 2>&1 | tee -a $RUNDIR/train.log"
echo "Started in tmux session 'train'. Attach: tmux attach -t train   Log: $RUNDIR/train.log"

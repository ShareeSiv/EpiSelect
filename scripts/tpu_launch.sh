#!/bin/bash
set -euo pipefail

EXPERIMENT=${1:?usage: tpu_launch.sh EXPERIMENT DATA_DIR CKPT_BASE [NODE] [ZONE]}
DATA_DIR=${2:?}
CKPT_BASE=${3:?}
NODE=${4:-node1}
ZONE=${5:-us-central2-b}

RUN="cd ~/EpiSelect && source ~/venv/bin/activate && export WANDB_PROJECT=\${WANDB_PROJECT:-episelect} && python -u launch.py +experiment=$EXPERIMENT loader.data_dir=$DATA_DIR rundir=$CKPT_BASE/$EXPERIMENT multihost=true 2>&1 | tee -a ~/EpiSelect/launch.log"

gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker=all --command="
  set -euo pipefail
  cd ~/EpiSelect
  [ '${PULL:-0}' = '1' ] && git pull --ff-only
  tmux kill-session -t launch 2>/dev/null || true
  tmux new -d -s launch '$RUN'
  echo \"launched on \$(hostname)\"
"

echo
echo "Started detached. Follow with: ./scripts/tpu_check.sh $NODE $ZONE"

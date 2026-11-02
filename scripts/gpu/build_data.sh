#!/bin/bash
# Tokenize Common Pile into the tfds format the trainer reads (int32 GPT-NeoX-20B ids), several
# domains at a time. CPU and network bound: run it on a cheap CPU box, push the result to storage
# with scripts/gpu/sync.sh, and pull it onto the GPU box, so no GPU hours are spent tokenizing.
#   bash scripts/gpu/build_data.sh /data/commonpile_tfds [JOBS]
# ~95 GB out, ~43B tokens. Domains already built are skipped by tfds, so it is safe to rerun.
# Set HF_TOKEN to avoid Hugging Face rate limits on the streaming download.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_DIR=${1:?usage: build_data.sh DATA_DIR [JOBS]}
JOBS=${2:-8}
mkdir -p "$DATA_DIR/logs"
PY=${PY:-.venv/bin/python}

DOMAINS=$($PY -c "from src.tfds.common_pile_grouped import TASKS; print(' '.join(TASKS))")
# a domain that fails is reported at the end; the others keep going
printf '%s\n' $DOMAINS | xargs -P "$JOBS" -I{} sh -c \
  "$PY -u scripts/build_commonpile_tfds.py {} --data_dir '$DATA_DIR' > '$DATA_DIR/logs/{}.log' 2>&1 \
   && echo 'done   {}' || echo 'FAILED {} (see $DATA_DIR/logs/{}.log)'"

du -sh "$DATA_DIR"

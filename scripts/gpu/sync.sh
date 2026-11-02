#!/bin/bash
# Copy a directory (the tokenized data, or a run's checkpoints) to or from storage.
#   bash scripts/gpu/sync.sh SRC DST
# Either side may be local, gs://..., s3://..., or any rclone remote (remote:path).
#   bash scripts/gpu/sync.sh /data/commonpile_tfds r2:bucket/commonpile_tfds   # after building
#   bash scripts/gpu/sync.sh r2:bucket/commonpile_tfds /data/commonpile_tfds   # on the gpu box
set -euo pipefail
SRC=${1:?usage: sync.sh SRC DST}
DST=${2:?usage: sync.sh SRC DST}

if [[ "$SRC" == gs://* || "$DST" == gs://* ]]; then
  gcloud storage rsync --recursive "$SRC" "$DST"
elif [[ "$SRC" == s3://* || "$DST" == s3://* ]]; then
  aws s3 sync "$SRC" "$DST"
elif [[ "$SRC" == *:* || "$DST" == *:* ]]; then
  rclone copy --transfers 32 --checkers 32 --progress "$SRC" "$DST"
else
  mkdir -p "$DST"
  rsync -a --info=progress2 "$SRC/" "$DST/"
fi

#!/bin/bash
# The whole job on a Vast.ai instance: setup -> data -> train -> Table 1 eval -> upload. onstart.sh runs it in
# tmux. Each stage is written to /workspace/STATUS, which `scripts/vast/vast.py status` reads.
# Settings come from the instance's environment (vast.py launch passes them):
#   RUN_NAME      run directory name under /workspace/runs (default 124M_EpiSelect_own)
#   DATA_SRC      where the tokenized data is (rclone remote, s3://, gs://). Unset: tokenize here (~1-2 h)
#   RESULTS_DST   where to upload the run directory when done (optional; otherwise `vast.py fetch`)
#   TRAIN_OVERRIDES  extra hydra overrides, e.g. "loader.fit_method=paper"
#   WANDB_API_KEY, HF_TOKEN, RCLONE_CONFIG_* / AWS_*  credentials, if used
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
DATA_DIR=/workspace/commonpile_tfds
RUNDIR=/workspace/runs/${RUN_NAME:-124M_EpiSelect_own}
mkdir -p "$RUNDIR"

stage() { echo "$1 $(date -u +%FT%TZ)" > /workspace/STATUS; echo "=== $1"; }
trap 'stage "FAILED(${CURRENT:-?}) see /workspace/pipeline.log"' ERR

CURRENT=setup; stage setup
if [[ ! -f .venv/.setup_done ]]; then
  bash scripts/gpu/setup.sh
  uv pip install --python .venv/bin/python -q "lm-eval==0.4.11"  # for the Table 1 eval at the end
  touch .venv/.setup_done
fi
export PATH="$HOME/.local/bin:$PATH"

CURRENT=data; stage data
if [[ ! -f "$DATA_DIR/.complete" ]]; then
  if [[ -n "${DATA_SRC:-}" ]]; then
    if [[ "$DATA_SRC" == *:* && "$DATA_SRC" != gs://* && "$DATA_SRC" != s3://* ]]; then
      command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | bash
    elif [[ "$DATA_SRC" == s3://* ]]; then
      command -v aws >/dev/null || uv pip install --python .venv/bin/python -q awscli
      export PATH="$REPO/.venv/bin:$PATH"
    fi
    bash scripts/gpu/sync.sh "$DATA_SRC" "$DATA_DIR"
  else
    bash scripts/gpu/build_data.sh "$DATA_DIR" "$(nproc)"
  fi
  # every domain must be there before training starts
  missing=$(.venv/bin/python - "$DATA_DIR" <<'PY'
import os, sys
from src.tfds.common_pile_grouped import TASKS, CommonPileGrouped
v = str(CommonPileGrouped.VERSION)
print(" ".join(t for t in TASKS if not os.path.isdir(os.path.join(sys.argv[1], "common_pile_grouped", t, v))))
PY
)
  [[ -z "$missing" ]] || { echo "missing domains: $missing"; false; }
  touch "$DATA_DIR/.complete"
fi

CURRENT=train; stage train
if [[ ! -f "$RUNDIR/model.pt" ]]; then
  # shellcheck disable=SC2086  # TRAIN_OVERRIDES is a list of hydra overrides
  FOREGROUND=1 bash scripts/gpu/launch_124M.sh "$DATA_DIR" "$RUNDIR" ${TRAIN_OVERRIDES:-}
fi

CURRENT=eval; stage eval
if [[ ! -f "$RUNDIR/eval_results.json" ]]; then
  TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python scripts/eval_torch.py \
    "runs=[{label: ${RUN_NAME:-124M_EpiSelect_own}, path: $RUNDIR}]" \
    data_dir="$DATA_DIR" "evals=[val,harness]" output="$RUNDIR/eval_results.json" 2>&1 | tee -a "$RUNDIR/eval.log"
fi

CURRENT=upload; stage upload
if [[ -n "${RESULTS_DST:-}" ]]; then
  bash scripts/gpu/sync.sh "$RUNDIR" "$RESULTS_DST/${RUN_NAME:-124M_EpiSelect_own}"
fi
touch "$RUNDIR/DONE"
stage DONE

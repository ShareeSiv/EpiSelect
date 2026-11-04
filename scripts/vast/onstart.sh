#!/bin/bash
# Runs on the Vast.ai instance every time its container starts (scripts/vast/vast.py launch sets this up).
# Idempotent: a finished run is left alone, a running pipeline is not started twice, and after a container
# restart the pipeline picks up where it stopped (setup and data are skipped once done; training resumes from
# the last checkpoint, with the caveat in the README).
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p /workspace
exec >> /workspace/onstart.log 2>&1
echo "=== onstart $(date -u +%FT%TZ) at $(git rev-parse --short HEAD)"

RUNDIR=/workspace/runs/${RUN_NAME:-124M_EpiSelect_own}
if [[ -f "$RUNDIR/DONE" ]]; then
  echo "run already finished: $RUNDIR"
  exit 0
fi
if tmux has-session -t pipeline 2>/dev/null; then
  echo "pipeline already running"
  exit 0
fi
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux >/dev/null; }
tmux new -d -s pipeline "bash scripts/vast/pipeline.sh 2>&1 | tee -a /workspace/pipeline.log"
echo "pipeline started in tmux session 'pipeline'"

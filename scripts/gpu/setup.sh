#!/bin/bash
# Provision a rented NVIDIA box (Vast.ai, RunPod): system packages, uv, the venv with jax[cuda12].
#   bash scripts/gpu/setup.sh
# Run from the repo root. Needs an NVIDIA driver new enough for CUDA 12 (nvidia-smi shows >= 12.x).
set -euo pipefail
cd "$(dirname "$0")/../.."

if command -v apt-get >/dev/null 2>&1; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq tmux git curl rsync >/dev/null
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash requirements.sh cuda

.venv/bin/python - <<'PY'
import jax, jax.numpy as jnp
devs = jax.devices()
assert devs[0].platform == "gpu", f"jax sees no gpu: {devs} -- check the driver / CUDA version"
x = jnp.ones((1024, 1024), jnp.bfloat16)
print(f"jax {jax.__version__}: {len(devs)} x {devs[0].device_kind}, bf16 matmul ok: {float((x @ x)[0, 0])}")
PY
echo
echo "Done. Next: put the tokenized data on local disk (scripts/gpu/build_data.sh or scripts/gpu/sync.sh),"
echo "export WANDB_API_KEY=..., then scripts/gpu/launch_124M.sh DATA_DIR RUNDIR"

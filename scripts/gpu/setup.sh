#!/bin/bash
# Provision a rented NVIDIA box (Vast.ai, RunPod): system packages, uv, the venv with PyTorch (and cpu jax,
# which builds the init and runs the data selectors' fits).
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
# pytorch's default wheel targets the newest CUDA; pick the build the host driver supports (TORCH_INDEX overrides)
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
if [[ -z "${TORCH_INDEX:-}" ]]; then
  case "$DRIVER_CUDA" in
    1[3-9].*) TORCH_INDEX=https://pypi.org/simple ;;
    12.[89]) TORCH_INDEX=https://download.pytorch.org/whl/cu128 ;;
    12.[67]) TORCH_INDEX=https://download.pytorch.org/whl/cu126 ;;
    *) echo "driver supports CUDA $DRIVER_CUDA; pick a host with CUDA >= 12.6 (or set TORCH_INDEX)"; exit 1 ;;
  esac
fi
echo "driver CUDA $DRIVER_CUDA -> torch from $TORCH_INDEX"
[ -d .venv ] || uv venv .venv
uv pip install --python .venv/bin/python torch --index-url "$TORCH_INDEX"
bash requirements.sh torch  # keeps the torch above, which already satisfies the extra

.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch sees no gpu -- check the driver / CUDA version"
assert torch.cuda.is_bf16_supported(), "bf16 needs an Ampere or newer gpu"
x = torch.ones(1024, 1024, device="cuda", dtype=torch.bfloat16)
print(f"torch {torch.__version__}: {torch.cuda.device_count()} x {torch.cuda.get_device_name()}, bf16 ok: {float((x @ x)[0, 0])}")
PY
echo
echo "Done. Next: put the tokenized data on local disk (scripts/gpu/build_data.sh or scripts/gpu/sync.sh),"
echo "export WANDB_API_KEY=..., then scripts/gpu/launch_124M.sh DATA_DIR RUNDIR"

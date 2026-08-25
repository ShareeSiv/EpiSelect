#!/bin/bash
set -euo pipefail

NODE=${1:-node1}
ZONE=${2:-us-central2-b}
REPO=${3:-https://github.com/eysu35/EpiSelect.git}

# --worker=all runs this on every host of the pod; each needs its own venv and copy of the code.
gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker=all --command="
  set -euo pipefail
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv tmux git

  if [ -d ~/EpiSelect/.git ]; then
    git -C ~/EpiSelect pull --ff-only
  else
    git clone $REPO ~/EpiSelect
  fi

  python3 -m venv ~/venv
  source ~/venv/bin/activate
  pip install -q --upgrade pip
  pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install -q -e ~/EpiSelect

  # frees the TPU after a crashed run
  grep -q tpu-reset ~/.bashrc || echo 'alias tpu-reset=\"pkill -9 python; pkill -9 wandb; sudo rm -rf /tmp/tpu_logs; sudo rm -f /tmp/libtpu_lockfile\"' >> ~/.bashrc

  python -c 'import jax; print(f\"host {jax.process_index()}: {jax.local_device_count()} local / {jax.device_count()} global devices\")'
"

echo
echo "All hosts provisioned. Set your W&B key on every host before training:"
echo "  gcloud compute tpus tpu-vm ssh $NODE --zone=$ZONE --worker=all --command=\"echo YOUR_KEY > ~/EpiSelect/.wandbkey\""

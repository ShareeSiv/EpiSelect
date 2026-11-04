# PyTorch entry point for one NVIDIA GPU, with the same hydra configs as launch.py:
#   python scripts/train_torch.py +experiment=124M/EpiSelect_own_nvidia loader.data_dir=... rundir=...
# scripts/gpu/launch_124M.sh wraps this with the batch / accumulation split for the GPU's memory
import os
import sys

# jax only builds the init and runs the selectors' fits; keep it (and tf.data) off the gpu torch uses
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# isort: split
from src.torch_train import train  # noqa: E402  # first: loads triton before tensorflow imports

# isort: split

import json  # noqa: E402
from datetime import datetime  # noqa: E402

import hydra  # noqa: E402
import wandb  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


@hydra.main(config_name="main.yaml", config_path="../configs", version_base=None)
def main(cfg=None):
    if cfg.rundir is None:
        cfg.rundir = os.path.join("outputs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    assert not str(cfg.rundir).startswith("gs://"), "the torch path writes to local disk; sync it afterwards"
    cfg.rundir = os.path.abspath(cfg.rundir)
    OmegaConf.resolve(cfg)  # fills ${rundir} in loader.fit_data_path
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.rundir, exist_ok=True)
    cfg_dict = OmegaConf.to_object(cfg)
    with open(os.path.join(cfg.rundir, "cfg.json"), "w") as f:
        json.dump({**cfg_dict, "framework": "torch"}, f)

    key = os.environ.get("WANDB_API_KEY")
    if key is None and os.path.exists(".wandbkey"):
        key = open(".wandbkey").read().strip()
    if key and not cfg.debug:
        wandb.login(key=key)
        id_path = os.path.join(cfg.rundir, "wandb_id.txt")  # a restart resumes the same wandb run
        if not os.path.exists(id_path):
            with open(id_path, "w") as f:
                f.write(wandb.util.generate_id())
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "episelect"),
            id=open(id_path).read().strip(),
            resume="allow",
            config=cfg_dict,
            tags=[cfg.tag] if cfg.get("tag") else None,
        )
    else:
        print("No W&B key (WANDB_API_KEY or .wandbkey): logging to metrics.jsonl only.")
    train(cfg)


if __name__ == "__main__":
    main()

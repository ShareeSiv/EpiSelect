import os
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from src.mlx_train import train


@hydra.main(config_name="mlx.yaml", config_path="../configs", version_base=None)
def main(cfg=None):
    if cfg.rundir is None:
        cfg.rundir = os.path.join("outputs", "mlx", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    cfg.rundir = os.path.abspath(cfg.rundir)
    OmegaConf.resolve(cfg)  # fills ${rundir} in fit_data_path and the sampler/tasks references
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.rundir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.rundir, "config.yaml"))
    train(cfg)


if __name__ == "__main__":
    main()

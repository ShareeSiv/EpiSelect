import optax


def make_lr_schedule(cfg):
    # cosine: warmup + cosine decay over the whole run (the original schedule).
    # wsd: warmup, constant, then a linear decay over the last wsd_decay_frac of the run
    if cfg.get("lr_schedule", "cosine") == "cosine":
        return optax.warmup_cosine_decay_schedule(
            0, cfg.learning_rate, cfg.warmup_steps, cfg.lr_decay_steps, end_value=cfg.min_lr
        )
    decay_start = stable_end_step(cfg)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, cfg.learning_rate, cfg.warmup_steps),
            optax.constant_schedule(cfg.learning_rate),
            optax.linear_schedule(cfg.learning_rate, cfg.min_lr, cfg.max_steps - decay_start),
        ],
        [cfg.warmup_steps, decay_start],
    )


def stable_end_step(cfg):
    # first step at which the lr starts to fall. None when there is no stable phase to restrict the fit to
    if cfg.get("lr_schedule", "cosine") == "cosine":
        return None
    return cfg.max_steps - round(cfg.wsd_decay_frac * cfg.max_steps)

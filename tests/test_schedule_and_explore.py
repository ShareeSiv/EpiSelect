import numpy as np
from omegaconf import OmegaConf

from src.dataloader.explore import exploration_dist
from src.lr_schedule import make_lr_schedule, stable_end_step

CFG = dict(learning_rate=1e-3, min_lr=1e-5, warmup_steps=10, lr_decay_steps=100, max_steps=100, wsd_decay_frac=0.2)


def test_wsd_is_flat_then_decays():
    cfg = OmegaConf.create({**CFG, "lr_schedule": "wsd"})
    sched = make_lr_schedule(cfg)
    assert stable_end_step(cfg) == 80
    assert float(sched(0)) == 0.0
    assert np.allclose([float(sched(s)) for s in (10, 40, 79)], 1e-3)
    assert float(sched(99)) < 1e-3 * 0.2
    assert np.isclose(float(sched(100)), 1e-5)


def test_cosine_has_no_stable_phase():
    cfg = OmegaConf.create({**CFG, "lr_schedule": "cosine"})
    assert stable_end_step(cfg) is None
    assert float(make_lr_schedule(cfg)(10)) > float(make_lr_schedule(cfg)(90))


def test_burst_dist_cycles_domains_and_is_deterministic():
    base = np.full(5, 0.2)
    kw = dict(mode="burst", seed=3, burst_len=10, burst_mass=0.8)
    p0 = exploration_dist(0, base, **kw)
    assert np.isclose(p0.sum(), 1.0) and p0.max() >= 0.8
    assert np.array_equal(p0, exploration_dist(9, base, **kw))  # same burst
    peaks = [int(np.argmax(exploration_dist(10 * i, base, **kw))) for i in range(5)]
    assert sorted(peaks) == list(range(5))  # every domain gets one burst per cycle


def test_dirichlet_dist_varies_per_block():
    base = np.full(4, 0.25)
    kw = dict(mode="dirichlet", seed=0, burst_len=10, concentration=0.3)
    assert np.isclose(exploration_dist(0, base, **kw).sum(), 1.0)
    assert not np.allclose(exploration_dist(0, base, **kw), exploration_dist(10, base, **kw))

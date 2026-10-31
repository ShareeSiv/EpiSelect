import json
import os
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

mx = pytest.importorskip("mlx.core")
# full fp32 matmuls (mlx reads this at the first matmul), so the jax comparison is exact up to fp32 rounding
os.environ["MLX_ENABLE_TF32"] = "0"

from src.dataloader.synthetic_loader import SyntheticTasksSampler  # noqa: E402
from src.mlx_model import GPT, count_params, loss_fn, make_optimizer, make_train_step  # noqa: E402
from src.mlx_train import domain_metrics, train  # noqa: E402
from src.model import GPT as JaxGPT  # noqa: E402
from src.model import count_params as jax_count_params  # noqa: E402

CFG = SimpleNamespace(vocab_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0, mup=False)


def _copy_jax_weights(mlx_model, jax_model):
    # blocks are stacked along a leading layer axis in the jax model
    mlx_model.wte.weight = mx.array(np.asarray(jax_model.wte.weight_VxD))
    b = jax_model.blocks
    for i, block in enumerate(mlx_model.blocks):
        block.attn.c_attn.weight = mx.array(np.asarray(b.attn.c_attn.weight_MxN[i]))
        block.attn.c_proj.weight = mx.array(np.asarray(b.attn.c_proj.weight_MxN[i]))
        block.attn.q_ln.weight = mx.array(np.asarray(b.attn.q_ln.weight[i]))
        block.attn.k_ln.weight = mx.array(np.asarray(b.attn.k_ln.weight[i]))
        for name in ("w1", "w2", "w3"):
            getattr(block.mlp, name).weight = mx.array(np.asarray(getattr(b.mlp, name).weight_MxN[i]))


def test_mlx_gpt_matches_jax_gpt():
    jax_model = JaxGPT(CFG, jax.random.PRNGKey(0))
    model = GPT(CFG)
    assert count_params(model) == jax_count_params(jax_model)
    _copy_jax_weights(model, jax_model)
    x_BxT = np.random.default_rng(0).integers(0, CFG.vocab_size, (3, 16)).astype(np.int32)
    expected = np.asarray(jax.vmap(jax_model)(x_BxT))
    got = np.array(model(mx.array(x_BxT)))
    assert np.abs(got - expected).max() < 1e-4


def test_train_step_lowers_loss_and_returns_per_sequence_loss():
    mx.random.seed(0)
    model = GPT(CFG)
    opt_cfg = SimpleNamespace(learning_rate=3e-3, beta2=0.95, weight_decay=1e-4)
    optimizer = make_optimizer(opt_cfg)
    step, state = make_train_step(model, optimizer)
    sampler = SyntheticTasksSampler("train", 32, k=2, vocab=CFG.vocab_size)
    x, y, _ = sampler.get_batch(np.array([4, 4]), 8)
    x, y = mx.array(x), mx.array(y)
    first, _ = loss_fn(model, x, y)
    for _ in range(30):
        loss, loss_B = step(x, y)
        mx.eval(loss, loss_B, state)
    assert loss_B.shape == (8,)
    assert loss.item() < first.item() - 0.5


def test_synthetic_sampler_batches():
    train_s = SyntheticTasksSampler("train", 16, tasks=["a", "b", "c"], vocab=32)
    val_s = SyntheticTasksSampler("val", 16, tasks=["a", "b", "c"], vocab=32)
    x, y, ids = train_s.get_batch(np.array([2, 0, 3]), 5)
    assert x.shape == y.shape == (5, 16) and (x[:, 1:] == y[:, :-1]).all()
    assert ids.tolist() == [0, 0, 2, 2, 2] and x.max() < 32
    assert train_s.k == 3 and np.isclose(train_s.init_dist.sum(), 1.0)
    # same sources, different draws for the val split
    assert np.allclose(train_s.cdf_KxVxV, val_s.cdf_KxVxV)
    assert not (val_s.get_batch(np.array([2, 0, 3]), 5)[0] == x).all()


def test_domain_metrics_balanced_and_worst():
    loss_N = np.array([1.0, 3.0, 2.0, 2.0, 2.0])
    m = domain_metrics(loss_N, np.array([0, 0, 1, 1, 1]))
    assert m["loss/task0"] == 2.0 and m["loss/task1"] == 2.0
    m = domain_metrics(np.array([1.0, 5.0]), np.array([0, 1]))
    assert m["balanced_loss"] == 3.0 and m["worst_loss"] == 5.0 and m["empirical_loss"] == 3.0


def test_train_mlx_end_to_end(tmp_path):
    # tiny model on synthetic domains: the selector refits, dumps (D, L) and logs per-domain val loss
    overrides = [
        "sampler=synthetic",
        f"rundir={tmp_path}",
        "tf32=false",
        "tasks=[a,b,c,d]",
        "model.vocab_size=64",
        "model.n_layer=1",
        "model.n_embd=32",
        "model.n_head=2",
        "batch_size=8",
        "block_size=32",
        "max_steps=60",
        "lr_decay_steps=60",
        "warmup_steps=5",
        "eval_interval=20",
        "val_seqs_per_domain=4",
        "loader.ignore_steps=5",
        "loader.start_step=30",
        "loader.update_interval=15",
        "loader.explore_steps=30",
        "loader.burst_len=2",
        "loader.window_length=5",
        "loader.polyorder=2",
        "loader.fit_freq=1",
        "sampler.vocab=64",
    ]
    with initialize(config_path="../configs", version_base=None):
        cfg = compose("mlx", overrides=overrides)
    OmegaConf.resolve(cfg)
    train(cfg)

    rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
    evals = [r for r in rows if "val_balanced_loss" in r]
    assert [r["step"] for r in evals] == [0, 20, 40, 60]
    assert all(f"val_loss/task{i}" in evals[-1] for i in range(4))
    assert evals[-1]["val_balanced_loss"] < evals[0]["val_balanced_loss"]
    assert any("opt_param_gamma/task0_from1" in r for r in rows)
    dumps = sorted(p for p in os.listdir(tmp_path) if p.startswith("fit_data_"))
    assert dumps, "refits should dump (D, L)"
    data = np.load(tmp_path / dumps[0])
    assert data["D"].shape[0] == data["L"].shape[0] == 4
    assert json.load(open(tmp_path / "run_info.json"))["domains"] == ["a", "b", "c", "d"]

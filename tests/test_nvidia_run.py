import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import tensorflow as tf  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.model import GPT  # noqa: E402
from src.train import make_training_fns  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_experiment_matches_paper_124M_run():
    with initialize_config_dir(config_dir=os.path.abspath(CONFIGS), version_base=None):
        paper = compose("main", overrides=["+experiment=124M/EpiSelect", "rundir=/tmp/r"])
        ours = compose("main", overrides=["+experiment=124M/EpiSelect_own_nvidia", "rundir=/tmp/r"])
    assert ours.seed == paper.seed == 21
    assert ours.model == paper.model
    assert ours.loader.fit_method == "own" and not ours.multihost
    assert ours.eval_batch_size == 256 and ours.loader.fit_data_path == "/tmp/r/fit_data_{step}.npz"
    # nothing else about the training run moves
    skip = {"tag", "multihost", "eval_batch_size", "ckpt_max_to_keep", "loader"}
    for k in paper:
        if k not in skip:
            assert ours[k] == paper[k], k
    for k in paper.loader:
        if k not in {"fit_method", "fit_data_path"}:
            assert ours.loader[k] == paper.loader[k], k


def test_chunked_eval_matches_full_batch():
    devices = jax.devices()
    assert len(devices) == 2
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 2)), axis_names=("replica", "data"))
    model_cfg = OmegaConf.create(dict(vocab_size=64, n_layer=2, n_head=2, n_embd=16, dropout=0.0, mup=False))
    model = GPT(model_cfg, jax.random.PRNGKey(0))

    rng = np.random.default_rng(0)
    B, T = 8, 9
    seqs = rng.integers(0, 64, size=(3 * B, T + 1)).astype(np.int32)
    tasks = rng.integers(0, 4, size=(3 * B,)).astype(np.int32)
    dset = tf.data.Dataset.from_tensor_slices((tasks, seqs)).batch(B)

    def run(chunk):
        cfg = OmegaConf.create(dict(compute_dtype="float32", g_accum_iters=1, shard_model=False, eval_microbatch=chunk))
        _, evaluate = make_training_fns(cfg, optax.sgd(0.1), mesh)
        return evaluate(model, dset)

    loss_full, task_full, m_full = run(None)
    loss_chunk, task_chunk, m_chunk = run(2)
    assert loss_full.shape == (3 * B,)
    np.testing.assert_array_equal(task_full, task_chunk)
    np.testing.assert_allclose(loss_full, loss_chunk, rtol=1e-5)
    np.testing.assert_allclose(m_full["balanced_loss"], m_chunk["balanced_loss"], rtol=1e-5)

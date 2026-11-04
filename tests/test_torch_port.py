# the pytorch trainer (src/torch_model.py, src/torch_train.py, scripts/eval_torch.py) against the jax one
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

torch = pytest.importorskip("torch")  # triton and the tokenizer stub are set up in tests/conftest.py

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import tensorflow as tf  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.model import GPT as EqxGPT  # noqa: E402
from src.torch_model import GPT, init_from_jax, seq_losses  # noqa: E402
from src.torch_train import evaluate, make_optimizer, optimizer_step  # noqa: E402

MODEL = OmegaConf.create(dict(vocab_size=64, n_layer=2, n_head=2, n_embd=16, dropout=0.0, mup=False))


def _models(seed=21):
    _, key1 = jax.random.split(jax.random.PRNGKey(seed))
    tm = GPT(MODEL)
    tm.load_state_dict(init_from_jax(MODEL, seed))
    return EqxGPT(MODEL, key1), tm


def test_forward_and_init_match_jax():
    em, tm = _models()
    x = np.random.default_rng(0).integers(0, 64, (3, 11))
    le = np.asarray(jax.vmap(lambda s: em(s, inference=True))(x))
    lt = tm(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(lt, le, atol=2e-5)
    # untied: two parameters with the same initial values, as the equinox model's two leaves
    assert tm.wte.weight.data_ptr() != tm.lm_head.weight.data_ptr()
    assert torch.equal(tm.wte.weight, tm.lm_head.weight)


def test_optimizer_matches_optax_chain():
    cfg = OmegaConf.create(dict(learning_rate=1e-2, weight_decay=1e-4, beta2=0.95))
    em, tm = _models()
    rng = np.random.default_rng(1)
    lrs = [3e-3, 1e-2, 5e-3, 1e-2]

    # src/train.py's chain, with the schedule replaced by the lrs above
    chain = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=cfg.beta2),
        optax.add_decayed_weights(cfg.weight_decay / cfg.learning_rate),
        optax.scale_by_schedule(lambda c: jax.numpy.asarray(lrs)[c]),
        optax.scale(-1),
    )
    params, static = eqx.partition(em, eqx.is_array)
    opt_state = chain.init(params)

    def jax_loss(p, x, y):
        logits = jax.vmap(lambda s: eqx.combine(p, static)(s, inference=True))(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    opt = make_optimizer(tm, cfg)
    for lr in lrs:
        seq = rng.integers(0, 64, (4, 10))
        x, y = seq[:, :-1], seq[:, 1:]
        grads = jax.grad(jax_loss)(params, x, y)
        updates, opt_state = chain.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        seq_losses(tm, torch.tensor(x), torch.tensor(y)).mean().backward()
        optimizer_step(tm, opt, lr)

    em = eqx.combine(params, static)
    sd = tm.state_dict()
    np.testing.assert_allclose(sd["wte.weight"].numpy(), np.asarray(em.wte.weight_VxD), atol=1e-5)
    np.testing.assert_allclose(sd["lm_head.weight"].numpy(), np.asarray(em.lm_head.weight_MxN), atol=1e-5)
    np.testing.assert_allclose(
        sd["blocks.1.mlp.w2.weight"].numpy(), np.asarray(em.blocks.mlp.w2.weight_MxN[1]), atol=1e-5
    )
    # the two vocab matrices got different gradients, so they no longer agree
    assert not torch.allclose(sd["wte.weight"], sd["lm_head.weight"])


def test_chunked_eval_matches_full_batch():
    _, tm = _models()
    rng = np.random.default_rng(2)
    seqs = rng.integers(0, 64, (24, 10)).astype(np.int32)
    tasks = rng.integers(0, 3, (24,)).astype(np.int32)
    dset = tf.data.Dataset.from_tensor_slices((tasks, seqs)).batch(8)
    fn = lambda x, y: seq_losses(tm, x, y)  # noqa: E731
    full, tid_full, m_full = evaluate(fn, dset, 8, torch.device("cpu"))
    chunk, tid_chunk, m_chunk = evaluate(fn, dset, 2, torch.device("cpu"))
    np.testing.assert_array_equal(tid_full, tid_chunk)
    np.testing.assert_allclose(full, chunk, rtol=1e-5)
    np.testing.assert_allclose(m_full["balanced_loss"], m_chunk["balanced_loss"], rtol=1e-5)


def test_harness_scores_match_jax_evaluator():
    pytest.importorskip("lm_eval")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eval_torch import TorchLM

    from src.harness.lm_harness_eval import LMEvaluator

    em, tm = _models()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("replica", "data"))
    pairs = [("The cat sat on", " the mat"), ("Q: two plus two? A:", " four"), ("x", " yz yz"), ("Hello ", "world")]
    reqs = [SimpleNamespace(args=p) for p in pairs]
    ours = TorchLM(tm, torch.device("cpu"), batch_size=3, dtype=torch.float32).loglikelihood(reqs)
    theirs = LMEvaluator(em, mesh, batch_size=3).loglikelihood(reqs)
    for (lp_t, g_t), (lp_j, g_j) in zip(ours, theirs):
        assert abs(lp_t - lp_j) < 1e-3 and g_t == g_j


def test_train_loop_runs_and_refits(tmp_path, monkeypatch):
    # a few dozen cpu steps on synthetic domains through the real loop, the episelect loader and its own fit
    from src.dataloader import commonpile_loader
    from src.torch_train import train

    def fake_eval_dset(batch_size, block_size, data_dir):
        rng = np.random.default_rng(0)
        seqs = rng.integers(0, 64, (2 * batch_size, block_size + 1)).astype(np.int32)
        tasks = rng.integers(0, 4, (2 * batch_size,)).astype(np.int32)
        return tf.data.Dataset.from_tensor_slices((tasks, seqs)).batch(batch_size)

    monkeypatch.setattr(commonpile_loader, "make_eval_dset", fake_eval_dset)
    here = os.path.dirname(__file__)
    loader = OmegaConf.load(os.path.join(here, "..", "configs", "loader", "commonpile_episelect.yaml"))
    loader.pop("defaults")
    loader.tasks_sampler_cfg = OmegaConf.create(
        dict(_target_="src.dataloader.synthetic_loader.SyntheticTasksSampler", _recursive_=False, k=4, vocab=64)
    )
    loader.update(
        dict(
            ignore_steps=0,
            start_step=20,
            update_interval=20,
            fit_freq=1,
            window_length=5,
            polyorder=2,
            min_fit_points=5,
            explore_steps=20,
            burst_len=5,
            fit_data_path=str(tmp_path / "fit_{step}.npz"),
        )
    )
    cfg = OmegaConf.create(
        dict(
            model=MODEL,
            loader=loader,
            seed=21,
            rundir=str(tmp_path),
            learning_rate=1e-3,
            min_lr=1e-5,
            beta2=0.95,
            weight_decay=1e-4,
            warmup_steps=5,
            lr_decay_steps=45,
            max_steps=45,
            lr_schedule="cosine",
            wsd_decay_frac=0.2,
            batch_size=4,
            g_accum_iters=2,
            block_size=16,
            compute_dtype="float32",
            param_dtype="float32",
            eval_interval=20,
            eval_batch_size=8,
            eval_microbatch=4,
            ckpt_max_to_keep=1,
            allow_cpu=True,
            compile=False,
        )
    )
    train(cfg)
    names = sorted(os.listdir(tmp_path))
    assert "model.pt" in names and "metrics.jsonl" in names and "token_stats.json" in names
    assert [n for n in names if n.startswith("ckpt_")] == ["ckpt_44.pt"]  # older ones pruned
    assert any(n.startswith("fit_") for n in names)  # the selector refit at step 20 / 40

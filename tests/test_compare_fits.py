import jax.numpy as jnp
import numpy as np
import pytest

from scripts.compare_fits import compare, fit_own, gamma_stability, held_out_error, load_dumps
from src import scaling_law, scaling_law_paper
from tests.test_scaling_law import _synthetic


def test_paper_theta_converts_to_own_weight_exactly():
    K, T = 3, 20
    rng = np.random.default_rng(0)
    theta = np.concatenate([rng.normal(size=K * K), [0.3, 0.4, 0.5], rng.normal(size=K), [0.6, 0.7, 0.8]])
    D_T = jnp.asarray(np.cumsum(rng.uniform(1, 10, (T, K)), 0))
    paper = scaling_law_paper._log_loss(jnp.asarray(theta), D_T)
    own_theta = scaling_law_paper.to_own_weight(theta, K)
    assert np.allclose(np.diag(scaling_law.readout_params(own_theta, K)[0]), 1.0)
    assert np.allclose(scaling_law._log_loss(own_theta, D_T), paper, atol=1e-5)


def test_gamma_stability():
    g = np.stack([np.eye(2), np.eye(2) + np.array([[0, 0.2], [0, 0]]), np.eye(2) + np.array([[0, 0.4], [0, 0]])])
    s = gamma_stability(g)
    assert np.isclose(s["mean_abs_change"], 0.1) and np.isclose(s["median_abs_change"], 0.1)
    assert np.isclose(s["max_offdiag_last"], 0.4) and np.allclose(s["max_offdiag_per_refit"], [0, 0.2, 0.4])
    assert np.isnan(gamma_stability(g[:1])["mean_abs_change"])


def test_held_out_error_small_for_the_true_law():
    D, L, *_ = _synthetic()
    h = held_out_error(fit_own, D, L, 0.5)
    assert h["n_fit"] == 100 and h["heldout_rmse"] < 1e-3


def test_compare_on_dumps(tmp_path):
    D, L, *_ = _synthetic(T=120)
    for n in (60, 90, 120):
        np.savez(tmp_path / f"fit_data_{n}.npz", D=D[:, :n], L=L[:, :n])
    dumps = load_dumps([str(tmp_path)])
    assert [s for s, _, _ in dumps] == [60, 90, 120]
    report = compare(dumps, fracs=(0.6,), methods=("own", "own_guarded", "identity"))
    # 60 points at the first refit: the guard (min 50 points) keeps the own-weight fit throughout
    assert report["methods"]["own_guarded"]["guarded"] == [False, False, False]
    own, eye = report["methods"]["own"], report["methods"]["identity"]
    assert np.allclose(eye["gamma_last"], np.eye(4))
    # the data have transfer (gamma_01 = 0.4), so the own-weight fit predicts better than gamma = I
    assert own["heldout"]["0.6"]["heldout_rmse"] < eye["heldout"]["0.6"]["heldout_rmse"]
    assert len(own["gamma_spread"]) == 3 and len(report["fit_cond"]) == 3
    assert not any("tie" in f for f in report["flags"])


def test_paper_fit_runs():
    # the original lbfgs fit from 729 inits: slow on the cpu, so a small problem
    D, L, *_ = _synthetic(K=4, T=40)
    report = compare([(40, D, L)], fracs=(0.7,), methods=("paper",))
    m = report["methods"]["paper"]
    assert np.isfinite(m["heldout"]["0.7"]["heldout_rmse"])
    assert np.allclose(np.diag(m["gamma_last"]), 1.0)


def test_bayes_intervals_attach_to_the_report():
    pytest.importorskip("numpyro")
    from scripts.compare_fits import add_bayes

    D, L, *_ = _synthetic(T=60, noise=5e-4)
    report = compare([(60, D, L)], fracs=(0.7,), methods=("own",))
    add_bayes(report, D, L, num_warmup=50, num_samples=50, num_chains=1)
    assert np.asarray(report["bayes"]["gamma_q05"]).shape == (4, 4)
    assert 0.0 <= report["methods"]["own"]["frac_offdiag_in_bayes_90"] <= 1.0

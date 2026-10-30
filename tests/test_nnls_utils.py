import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import nnls as scipy_nnls

from src.nnls_utils import nnls, nnls_np


def _cumulative_counts(rng, T, K):
    # near-collinear columns, like cumulative token counts under a fixed mixture
    steps = rng.multinomial(256, rng.dirichlet(np.ones(K)), size=T).astype(float) * 40
    D = np.cumsum(steps, 0) + 1e3
    return D / D.max(0)


def test_matches_scipy_without_linear_term():
    rng = np.random.default_rng(0)
    A = _cumulative_counts(rng, 300, 31)
    b = A @ np.where(rng.random(31) < 0.4, rng.random(31), 0.0) + 1e-3 * rng.standard_normal(300)
    expected, _ = scipy_nnls(A, b, maxiter=10000)
    assert np.abs(nnls_np(A, b) - expected).max() <= 1e-6 * np.abs(expected).max()


def test_linear_term_satisfies_kkt():
    rng = np.random.default_rng(1)
    A = _cumulative_counts(rng, 300, 31)
    b = A @ np.abs(rng.standard_normal(31))
    c = np.full(31, 0.5)
    w = nnls_np(A, b, c)
    g = A.T @ (A @ w - b) + c
    scale = np.abs(A.T @ b).max()
    assert (w >= 0).all()
    assert np.abs(g[w > 0]).max() < 1e-8 * scale  # stationary on the support
    assert g[w == 0].min() > -1e-8 * scale  # no descent direction off it


def test_runs_under_jit_and_nested_vmap():
    A = jnp.asarray(_cumulative_counts(np.random.default_rng(2), 50, 5), jnp.float32)
    w_true = jnp.array([[[1.0, 0, 0.5, 0, 0], [0, 2.0, 0, 0, 1.0]]] * 3)  # [3,2,5]
    B = jnp.einsum("tk,ijk->ijt", A, w_true)
    W = jax.jit(jax.vmap(jax.vmap(lambda b: nnls(A, b))))(B)
    assert W.shape == (3, 2, 5) and W.dtype == jnp.float32
    assert np.allclose(W, w_true, atol=1e-4)


def test_nonfinite_input_gives_nan_instead_of_raising():
    A = np.ones((4, 2))
    assert np.isnan(nnls_np(A, np.array([1.0, np.nan, 1.0, 1.0]))).all()


def test_inactive_problems_are_skipped():
    A = jnp.asarray(_cumulative_counts(np.random.default_rng(3), 50, 5), jnp.float32)
    B = jnp.stack([A @ jnp.ones(5), A @ jnp.ones(5)])
    W = jax.vmap(lambda b, on: nnls(A, b, active=on))(B, jnp.array([True, False]))
    assert np.allclose(W[0], 1.0, atol=1e-4) and np.isnan(W[1]).all()

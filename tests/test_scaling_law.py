import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.sharding import Mesh

from src.scaling_law import _score, epiplexity_grads, fit_scaling_law, readout_params


def np_loss(gamma, alpha, beta, eps, x):
    # reference implementation of the cross-domain law, in plain loops
    # gamma: [K,K], alpha/beta/eps: [K,], x: [N,K]
    preds = np.zeros(shape=x.shape, dtype=x.dtype)
    for i in range(x.shape[0]):
        for m in range(x.shape[1]):
            for k in range(x.shape[1]):
                preds[i, m] += gamma[m, k] * x[i, k]
            preds[i, m] = beta[m] / preds[i, m] ** alpha[m] + eps[m]
    return preds


def np_grads(gamma, alpha, beta, eps, x):
    # reference dS/dn_k. x: [1,K]
    assert x.shape[0] == 1
    loss = np_loss(gamma=gamma, alpha=alpha, beta=beta, eps=eps, x=x)

    denom = np.zeros(shape=x.shape[-1], dtype=x.dtype)
    for m in range(x.shape[1]):
        denom[m] = np.sum(gamma[m, :] * x[0, :])

    grads = np.zeros(shape=x.shape[-1], dtype=x.dtype)
    for j in range(x.shape[1]):
        for m in range(x.shape[1]):
            grads[j] += x[0, m] * alpha[m] * (gamma[m, j] / denom[m]) * (loss[0, m] - eps[m])
    return grads


def test_fit_scaling_law():
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(1, len(devices)), ("replica", "data"))
    T, K, dtype, key = 111, 17, jnp.float32, jr.PRNGKey(18)
    D = jr.randint(key, minval=1, maxval=100, shape=(K, T)).astype(dtype)
    L = jr.normal(key, shape=(K, T), dtype=dtype) + 1.8
    theta_hat, _ = fit_scaling_law(D, L, mesh, K=K)
    assert theta_hat.shape == (K * (K + 3),)


def test_epiplexity_grads_match_reference():
    dtype = jnp.float32
    N = 10
    x1, x2 = jnp.linspace(10, 50, N), jnp.linspace(30, 70, N)
    X, Y = jnp.meshgrid(x1, x2, indexing="ij")
    x_grid = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    K = x_grid.shape[-1]

    log_gamma = jnp.array([[-0.1, -2.3], [-1.6, -0.2]], dtype=dtype)
    alpha = jnp.array([0.6, 0.3], dtype=dtype)
    log_beta = jnp.array([0.0, -0.5], dtype=dtype)
    log_eps = jnp.array([0.6, 0.7], dtype=dtype)
    theta = jnp.concat([log_gamma.flatten(), alpha.flatten(), log_beta.flatten(), log_eps])
    tokens = jnp.sum(x_grid, axis=0, keepdims=True)

    approx = epiplexity_grads(tokens.T, params=theta)
    assert approx.shape == (K,)

    # own weight is fixed to 1, so the reference must pin the diagonal too
    gamma = np.exp(np.where(np.eye(K, dtype=bool), 0.0, np.asarray(log_gamma)))
    soln = np_grads(gamma, alpha, np.exp(log_beta), np.exp(log_eps), x=tokens)
    assert np.sum(np.abs(soln - approx)) < 1e-7


def test_unpack_fixes_own_weight_and_keeps_gamma_nonnegative():
    K = 3
    theta = np.array(jr.normal(jr.PRNGKey(0), (K * (K + 3),)))
    gamma, _, _, _ = readout_params(jnp.asarray(theta), K)
    assert np.allclose(np.diag(gamma), 1.0)
    assert (gamma >= 0).all()


def _synthetic(K=4, T=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    gamma = np.eye(K)
    gamma[0, 1], gamma[2, 3] = 0.4, 0.2
    alpha, beta, eps = np.array([0.3, 0.4, 0.5, 0.35]), np.array([5.0, 4.0, 6.0, 3.0]), np.array([1.7, 1.5, 1.9, 1.6])
    # single-domain bursts so the columns of N move independently (random mixtures leave them near-collinear)
    steps = np.zeros((T, K))
    for start in range(0, T, 10):
        steps[start : start + 10, rng.integers(K)] = 1e4
    steps += 1e2
    D = np.cumsum(steps, axis=0).T + 1e3  # [K,T]
    L = eps[:, None] + beta[:, None] * (gamma @ D) ** (-alpha[:, None])
    L = L * np.exp(noise * rng.standard_normal(L.shape))
    return D.astype(np.float32), L.astype(np.float32), gamma, alpha, eps


def test_fit_recovers_synthetic_law():
    mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ("replica", "data"))
    D, L, gamma, alpha, eps = _synthetic()
    K = D.shape[0]
    theta, (value, spread) = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), mesh, K=K)
    g_hat, a_hat, _, e_hat = readout_params(theta, K)
    assert np.allclose(np.diag(g_hat), 1.0)
    assert np.abs(g_hat - gamma).max() < 0.15
    assert np.abs(e_hat - eps).max() < 0.3
    assert np.isfinite(value) and np.isfinite(spread)


def test_fit_never_nan_with_eps_grid_near_loss():
    mesh = Mesh(np.array(jax.devices()).reshape(1, -1), ("replica", "data"))
    D, L, *_ = _synthetic(noise=0.01)
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), mesh, K=D.shape[0])
    assert np.isfinite(np.asarray(theta)).all()


def test_fit_matches_true_params_under_noise():
    # with noise gamma need not be recovered, but the optimiser should fit at least as well as the truth
    D, L, gamma, alpha, eps = _synthetic(noise=0.01)
    beta = np.array([5.0, 4.0, 6.0, 3.0])
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=D.shape[0])
    g, a, b, e = readout_params(theta, D.shape[0])
    for m in range(D.shape[0]):
        true = float(
            _score(jnp.asarray(D.T), jnp.asarray(L[m]), jnp.asarray(gamma[m], jnp.float32), alpha[m], eps[m], beta[m])
        )
        fit = float(_score(jnp.asarray(D.T), jnp.asarray(L[m]), jnp.asarray(g[m]), a[m], e[m], b[m]))
        assert fit <= true * 1.05


def test_shared_alpha_and_narrowed_range():
    D, L, *_ = _synthetic()
    K = D.shape[0]
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=K, shared_alpha=True)
    _, a, _, _ = readout_params(theta, K)
    assert np.allclose(a, a[0])
    lo, hi = jnp.full((K,), 0.2), jnp.full((K,), 0.3)
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=K, alpha_lo=lo, alpha_hi=hi)
    _, a, _, _ = readout_params(theta, K)
    assert (a >= 0.2 - 1e-6).all() and (a <= 0.3 + 1e-6).all()


def test_fit_recovers_31_domains():
    # 31 domains with single-domain bursts: the alternation must run to convergence to recover the transfer weights
    rng = np.random.default_rng(0)
    K, T = 31, 400
    steps = np.full((T, K), 1e2)
    for start in range(0, T, 5):
        steps[start : start + 5, rng.integers(K)] += 1e4
    D = (np.cumsum(steps, 0).T + 1e3).astype(np.float32)
    gamma = np.eye(K)
    gamma[0, 1], gamma[2, 3], gamma[5, 9] = 0.4, 0.2, 0.3
    L = (1.7 + 5 * (gamma @ D) ** -0.3).astype(np.float32)
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=K)
    g, a, _, e = readout_params(theta, K)
    assert np.abs(g - gamma).max() < 1e-2
    assert np.abs(a - 0.3).max() < 1e-2 and np.abs(e - 1.7).max() < 1e-2


def test_l1_zeros_survive_the_polish():
    # the lm polish ignores the l1 penalty, so it must not move weights the penalised nnls set to 0
    D, L, *_ = _synthetic(noise=0.003)
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=D.shape[0], l1=1e6)
    g, *_ = readout_params(theta, D.shape[0])
    assert np.allclose(g, np.eye(D.shape[0]))

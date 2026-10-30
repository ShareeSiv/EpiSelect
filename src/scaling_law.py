from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.nn import logsumexp


def huber_loss(y_hat, y, delta=1e-3):
    abs_diff = jnp.abs(y - y_hat)
    part1 = delta * (abs_diff - 0.5 * delta)
    part2 = 0.5 * abs_diff**2
    return jnp.where(abs_diff > delta, part1, part2)


def _unpack(theta, K):
    # own weight is 1: gamma_mm = 1, other rows are free non-negative weights (no simplex constraint)
    K2 = K * K
    log_gamma = theta[:K2].reshape(K, K)
    log_gamma = jnp.where(jnp.eye(K, dtype=bool), 0.0, log_gamma)
    return log_gamma, theta[K2 : K2 + K], theta[K2 + K : K2 + 2 * K], theta[-K:]


def _log_loss(theta, x):
    # theta: [K(K+3),], x: [T,K] -> [T,K]
    log_gamma, alpha, log_beta, log_eps = _unpack(theta, K=x.shape[-1])
    term1 = logsumexp(log_gamma[None] + jnp.log(x)[:, None], axis=-1)  # [T,K]
    term1 = log_beta[None] - alpha[None] * term1
    term2 = log_eps[None] * jnp.ones_like(x)
    return logsumexp(jnp.stack([term1, term2], axis=0), axis=0)


def readout_params(theta, K):
    log_gamma, alpha, log_beta, log_eps = _unpack(theta, K=K)
    return np.exp(log_gamma), np.array(alpha), np.exp(log_beta), np.exp(log_eps)


def epiplexity_grads(tokens, params):
    log_gamma, alpha, _, log_eps = _unpack(params, K=tokens.shape[0])
    gamma = np.exp(log_gamma)  # [K,K]
    alpha = np.array(alpha)  # [K,]
    eps = np.exp(log_eps)  # [K,]
    loss = np.exp(_log_loss(params, tokens.T).T)[:, 0]  # [K,]

    dS = (alpha * (loss - eps))[:, None]  # [K,1]
    num = gamma * tokens  # [K,K]
    denom = np.sum(gamma * tokens.T, axis=-1, keepdims=True)  # [K,1]
    return np.sum((num / denom) * dS, axis=0)  # [K,]


def _nnls(A, b, l1_K, iters):
    # min 0.5 |A w - b|^2 + l1_K . w  s.t. w >= 0, by FISTA on the normal equations. convex, init-independent
    G, h = A.T @ A, A.T @ b
    lip = jnp.linalg.eigvalsh(G)[-1] + 1e-12

    def body(_, carry):
        w, v, t = carry
        w_new = jnp.maximum(v - (G @ v - h + l1_K) / lip, 0.0)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t**2))
        return w_new, w_new + (t - 1.0) / t_new * (w_new - w), t_new

    w0 = jnp.zeros_like(h)
    return jax.lax.fori_loop(0, iters, body, (w0, w0, 1.0))[0]


def _fit_pair(Dn, L_m, m, alpha, eps, l1, iters):
    # Dn: [T,K] column-normalised tokens, L_m: [T,]. fixed (alpha, eps) => linear regression of
    # y = (L-eps)^(-1/alpha) on the token counts, with coefficients w_mk = beta^(-1/alpha) gamma_mk.
    K = Dn.shape[-1]
    d = L_m - eps
    # delta-method weights: noise on L becomes |dy/dL| times larger on y, so rows get s = 1/|dy/dL| / L.
    # s*y = alpha*d/L, so the weighted problem never needs the (possibly huge) y itself
    log_s = jnp.log(alpha) + (1.0 / alpha + 1.0) * jnp.log(d) - jnp.log(L_m)
    log_sy = jnp.log(alpha) + jnp.log(d) - jnp.log(L_m)
    shift = jnp.maximum(log_s.max(), log_sy.max())
    A = jnp.exp(log_s - shift)[:, None] * Dn
    b = jnp.exp(log_sy - shift)

    l1_K = l1 * (jnp.arange(K) != m)
    w = _nnls(A, b, l1_K, iters)  # normalised coefficients [K,]

    z = jnp.maximum(Dn @ w, 1e-30)
    log_L_hat = jnp.log(eps + z ** (-alpha))
    score = jnp.sum(huber_loss(log_L_hat, jnp.log(L_m), delta=1e-3))
    valid = (w[m] > 0) & jnp.isfinite(score)
    return jnp.where(valid, score, jnp.inf), w


@partial(jit, static_argnames=("mesh", "K", "n_alpha", "n_eps", "iters"))
def fit_scaling_law(D, L, mesh, K, l1=0.0, n_alpha=31, n_eps=60, iters=500):
    # D, L: [K,T]. per domain: grid over (alpha, eps) outside, exact weighted NNLS for gamma inside.
    # eps = f * min(L) with f < 1, so L - eps > 0 at every point (f = 0 is the eps = 0 end of the grid)
    del mesh
    colscale = jnp.maximum(D.max(axis=-1), 1.0)  # [K,]
    Dn = D.T / colscale  # [T,K]
    alphas = jnp.linspace(0.05, 0.8, n_alpha)
    # losses plateau near eps, so resolve the gap min(L) - eps log-spaced: eps = min(L) * (1 - g), g in [1e-3, 1]
    fracs = 1.0 - jnp.logspace(-3, 0, n_eps)[::-1]
    A_p = jnp.repeat(alphas, n_eps)  # [P,]
    f_p = jnp.tile(fracs, n_alpha)

    def fit_domain(L_m, m):
        eps_p = f_p * L_m.min()
        scores, w = vmap(_fit_pair, in_axes=(None, None, None, 0, 0, None, None))(Dn, L_m, m, A_p, eps_p, l1, iters)
        best = jnp.argmin(scores)
        w_orig = w / colscale[None]  # [P,K]
        w_mm = jnp.maximum(w_orig[:, m], 1e-30)
        gamma_p = w_orig / w_mm[:, None]  # [P,K], gamma_mm = 1

        # identifiability check: how much does gamma move across pairs that fit about as well as the best?
        near = (scores <= scores[best] * 1.1 + 1e-4)[:, None]
        spread = jnp.where(near, gamma_p, -jnp.inf).max(0) - jnp.where(near, gamma_p, jnp.inf).min(0)
        spread = jnp.where(jnp.arange(K) == m, 0.0, spread).max()

        log_beta = -A_p[best] * jnp.log(w_mm[best])  # beta = w_mm^(-alpha)
        log_gamma = jnp.log(jnp.maximum(gamma_p[best], 1e-12))
        return log_gamma, A_p[best], log_beta, jnp.log(jnp.maximum(eps_p[best], 1e-12)), scores[best], spread

    log_gamma, alpha, log_beta, log_eps, value, spread = vmap(fit_domain)(L, jnp.arange(K))
    theta = jnp.concatenate([log_gamma.reshape(-1), alpha, log_beta, log_eps])
    return theta, (jnp.sum(value), jnp.mean(spread))

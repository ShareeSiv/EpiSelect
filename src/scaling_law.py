from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.nn import logsumexp

from src.nnls_utils import nnls


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


def _nnls_gamma(Dn, colscale, L_m, m, alpha, eps, l1, active=True):
    # gamma step: with (alpha, eps) fixed, y = (L-eps)^(-1/alpha) is linear in the token counts with
    # coefficients w_mk = beta^(-1/alpha) gamma_mk, so gamma = w / w_mm comes from one convex weighted NNLS.
    # Dn: [T,K] column-normalised tokens, L_m: [T,]
    K = Dn.shape[-1]
    d = L_m - eps
    # delta-method weights: noise on L becomes |dy/dL| times larger on y, so rows get s = 1/|dy/dL| / L.
    # s*y = alpha*d/L, so the weighted problem never needs the (possibly huge) y itself
    log_s = jnp.log(alpha) + (1.0 / alpha + 1.0) * jnp.log(d) - jnp.log(L_m)
    log_sy = jnp.log(alpha) + jnp.log(d) - jnp.log(L_m)
    shift = jnp.maximum(log_s.max(), log_sy.max())
    A = jnp.exp(log_s - shift)[:, None] * Dn
    b = jnp.exp(log_sy - shift)

    w = (
        nnls(A, b, c=l1 * (jnp.arange(K) != m), active=active) / colscale
    )  # [K,] in token units, exact solve on the host
    return w / jnp.maximum(w[m], 1e-30), w[m] > 0


def _ols_eps_beta(z, L_m, alpha, eps_max):
    # (eps, beta) step: with gamma and alpha fixed, L = eps + beta * z^(-alpha) is linear in (eps, beta).
    # exact 2x2 least squares, then clip eps into [0, eps_max] and refit beta if the clip was active
    x = z ** (-alpha)
    xc, Lc = x - x.mean(), L_m - L_m.mean()
    beta = jnp.sum(xc * Lc) / jnp.maximum(jnp.sum(xc**2), 1e-30)
    eps = jnp.clip(L_m.mean() - beta * x.mean(), 0.0, eps_max)
    beta = jnp.maximum(jnp.sum(x * (L_m - eps)) / jnp.maximum(jnp.sum(x**2), 1e-30), 1e-12)
    return eps, beta


def _log_pred(D_T, gamma, alpha, eps, beta):
    z = jnp.maximum(D_T @ gamma, 1e-30)
    return jnp.log(eps + beta * z ** (-alpha))


def _score(D_T, L_m, gamma, alpha, eps, beta):
    s = jnp.sum(huber_loss(_log_pred(D_T, gamma, alpha, eps, beta), jnp.log(L_m), delta=1e-3))
    return jnp.where(jnp.isfinite(s), s, jnp.inf)


def _fit_alpha(D_T, Dn, colscale, L_m, m, alpha, l1, max_alt, alt_tol):
    # for one alpha: alternate exact OLS for (eps, beta) and NNLS for gamma, starting from gamma = I.
    # the gamma step does not depend on beta, so one round is a scalar map eps -> F(eps) and the alternation is
    # a fixed-point iteration on eps. plain iteration converges slowly, so it is accelerated with steffensen
    # (aitken) steps, falling back to the plain iterate when the extrapolation is unusable.
    # stops when eps moves by <= alt_tol * eps_max, or after max_alt steps (2 rounds each)
    K = D_T.shape[-1]
    eps_max = L_m.min() * (1.0 - 1e-3)  # keep eps safely below every observed loss

    def F(eps, gamma, active=True):
        new_gamma, ok = _nnls_gamma(Dn, colscale, L_m, m, alpha, eps, l1, active)
        new_gamma = jnp.where(ok, new_gamma, gamma)
        return _ols_eps_beta(D_T @ new_gamma, L_m, alpha, eps_max)[0], new_gamma

    def cond(carry):
        _, _, it, change = carry
        return (it < max_alt) & (change > alt_tol * eps_max)

    def body(carry):
        # under vmap every problem runs this body until the slowest converges; only active ones are solved
        e0, gamma, it, _ = carry
        active = cond(carry)
        e1, gamma = F(e0, gamma, active)
        e2, gamma = F(e1, gamma, active)
        denom = e2 - 2.0 * e1 + e0
        e_acc = e0 - (e1 - e0) ** 2 / jnp.where(denom == 0, 1.0, denom)
        usable = (jnp.abs(denom) > 1e-12 * eps_max) & jnp.isfinite(e_acc)
        e_new = jnp.clip(jnp.where(usable, e_acc, e2), 0.0, eps_max)
        return e_new, gamma, it + 1, jnp.abs(e_new - e0)

    gamma0 = (jnp.arange(K) == m).astype(L_m.dtype)
    eps0 = _ols_eps_beta(D_T @ gamma0, L_m, alpha, eps_max)[0]
    eps, gamma, _, _ = jax.lax.while_loop(cond, body, (eps0, gamma0, 0, jnp.array(jnp.inf, L_m.dtype)))
    gamma = F(eps, gamma)[1]  # gamma for the converged eps
    eps, beta = _ols_eps_beta(D_T @ gamma, L_m, alpha, eps_max)  # final (eps, beta) for the final gamma
    return _score(D_T, L_m, gamma, alpha, eps, beta), gamma, eps, beta


def _lm_polish(D_T, L_m, m, gamma, alpha, eps, beta, fix_alpha, alpha_bounds, iters):
    # joint levenberg-marquardt on all parameters, in log-loss space, projected onto the feasible set.
    # params: [gamma (K), log_beta, alpha, eps]; gamma_mm (and alpha, if shared) are held fixed
    K = D_T.shape[-1]
    log_L = jnp.log(L_m)
    eps_max = L_m.min() * (1.0 - 1e-3)
    free = jnp.concatenate([jnp.arange(K) != m, jnp.array([True, not fix_alpha, True])]).astype(L_m.dtype)

    def resid(p):
        return _log_pred(D_T, p[:K], p[K + 1], p[K + 2], jnp.exp(p[K])) - log_L

    def project(p):
        p = p.at[:K].set(jnp.maximum(p[:K], 0.0))
        p = p.at[K + 1].set(jnp.clip(p[K + 1], *alpha_bounds))
        return p.at[K + 2].set(jnp.clip(p[K + 2], 0.0, eps_max))

    def cost(p):
        c = jnp.sum(resid(p) ** 2)
        return jnp.where(jnp.isfinite(c), c, jnp.inf)

    def body(_, carry):
        p, lam, c = carry
        r, J = resid(p), jax.jacfwd(resid)(p) * free[None]
        H = J.T @ J
        H = H + lam * jnp.diag(jnp.diag(H) + 1e-12) + (1.0 - free) * jnp.eye(p.shape[0])  # pin fixed params
        step = jnp.linalg.solve(H, -J.T @ r) * free
        p_new = project(p + step)
        c_new = cost(p_new)
        better = c_new < c
        return jnp.where(better, p_new, p), jnp.where(better, lam / 3.0, lam * 3.0), jnp.minimum(c, c_new)

    p0 = jnp.concatenate([gamma, jnp.array([jnp.log(beta), alpha, eps])])
    p, _, _ = jax.lax.fori_loop(0, iters, body, (p0, 1e-2, cost(p0)))
    return p[:K], p[K + 1], p[K + 2], jnp.exp(p[K])


@partial(jit, static_argnames=("mesh", "K", "n_alpha", "shared_alpha", "max_alt", "lm_iters"))
def fit_scaling_law(
    D,
    L,
    mesh,
    K,
    l1=0.0,
    alpha_lo=None,
    alpha_hi=None,
    n_alpha=31,
    shared_alpha=False,
    max_alt=50,
    alt_tol=1e-6,
    lm_iters=50,
):
    # D, L: [K,T]. split the problem by how each parameter enters the law:
    #   alpha: 1-d grid search (per domain, or one alpha shared by all domains)
    #   inside, per alpha: alternate exact OLS for (eps, beta) and weighted NNLS for gamma
    #   then a joint levenberg-marquardt polish in log-loss space from the best grid point.
    # alpha_lo/alpha_hi: [K,] per-domain search range, so refits can narrow it around the previous alpha
    del mesh
    alpha_lo = jnp.full((K,), 0.05) if alpha_lo is None else alpha_lo
    alpha_hi = jnp.full((K,), 0.8) if alpha_hi is None else alpha_hi
    if shared_alpha:  # one grid for everyone, so scores are comparable across domains
        alpha_lo, alpha_hi = jnp.full((K,), alpha_lo.min()), jnp.full((K,), alpha_hi.max())
    alphas_KxA = alpha_lo[:, None] + (alpha_hi - alpha_lo)[:, None] * jnp.linspace(0.0, 1.0, n_alpha)[None]

    D_T = D.T  # [T,K]
    colscale = jnp.maximum(D.max(axis=-1), 1.0)  # [K,]
    Dn = D_T / colscale

    fit_grid = vmap(_fit_alpha, in_axes=(None, None, None, None, None, 0, None, None, None))
    scores_KxA, gamma_KxAxK, eps_KxA, beta_KxA = vmap(
        lambda L_m, m, alphas: fit_grid(D_T, Dn, colscale, L_m, m, alphas, l1, max_alt, alt_tol)
    )(L, jnp.arange(K), alphas_KxA)

    if shared_alpha:
        best_K = jnp.full((K,), jnp.argmin(scores_KxA.sum(0)))
    else:
        best_K = jnp.argmin(scores_KxA, axis=-1)
    take = lambda x: jnp.take_along_axis(x, best_K.reshape((K,) + (1,) * (x.ndim - 1)), axis=1)[:, 0]  # noqa: E731
    gamma, eps, beta, alpha = take(gamma_KxAxK), take(eps_KxA), take(beta_KxA), take(alphas_KxA)
    grid_score = take(scores_KxA)

    # identifiability check: how much does gamma move across alphas that fit about as well as the best?
    near = (scores_KxA <= grid_score[:, None] * 1.1 + 1e-4)[..., None]
    spread = jnp.where(near, gamma_KxAxK, -jnp.inf).max(1) - jnp.where(near, gamma_KxAxK, jnp.inf).min(1)
    spread = jnp.where(jnp.eye(K, dtype=bool), 0.0, spread).max(-1)  # [K,]

    def polish(L_m, m, g, a, e, b, lo, hi):
        g2, a2, e2, b2 = _lm_polish(D_T, L_m, m, g, a, e, b, shared_alpha, (lo, hi), lm_iters)
        s2 = _score(D_T, L_m, g2, a2, e2, b2)
        return s2, g2, a2, e2, b2

    s2, g2, a2, e2, b2 = vmap(polish)(L, jnp.arange(K), gamma, alpha, eps, beta, alpha_lo, alpha_hi)
    # lm minimises squared error, the grid minimises huber; keep the polish only where it did not hurt
    keep = s2 <= grid_score
    pick = lambda new, old: jnp.where(keep.reshape((K,) + (1,) * (old.ndim - 1)), new, old)  # noqa: E731
    gamma, alpha, eps, beta = pick(g2, gamma), pick(a2, alpha), pick(e2, eps), pick(b2, beta)
    value = jnp.where(keep, s2, grid_score)

    log_gamma = jnp.log(jnp.maximum(gamma, 1e-12))
    theta = jnp.concatenate([log_gamma.reshape(-1), alpha, jnp.log(beta), jnp.log(jnp.maximum(eps, 1e-12))])
    return theta, (jnp.sum(value), jnp.mean(spread))

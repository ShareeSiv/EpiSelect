# offline full-bayes fit of the own-weight-1 law, for uncertainty on gamma. needs the `bayes` extra (numpyro)
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from src.scaling_law import fit_scaling_law, readout_params


def domain_model(D_T, log_L, m, eps_max, gamma_scale=1.0):
    # log L_m ~ studentT(log(eps + beta * (n_m + sum_k gamma_mk n_k)^-alpha), sigma). gamma_mm = 1
    K = D_T.shape[-1]
    gamma_off = numpyro.sample("gamma_off", dist.HalfNormal(gamma_scale).expand([K - 1]))
    gamma = jnp.insert(gamma_off, m, 1.0)
    alpha = numpyro.sample("alpha", dist.Uniform(0.01, 1.0))
    eps = numpyro.sample("eps", dist.Uniform(0.0, eps_max))
    log_beta = numpyro.sample("log_beta", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.05))
    pred = jnp.log(eps + jnp.exp(log_beta) * jnp.maximum(D_T @ gamma, 1e-30) ** (-alpha))
    numpyro.sample("obs", dist.StudentT(4.0, pred, sigma), obs=log_L)


def fit_bayes(D, L, num_warmup=500, num_samples=1000, num_chains=2, seed=0, gamma_scale=1.0):
    # D, L: [K,T]. domains are independent given the law, so each gets its own NUTS run,
    # initialised at the point fit. returns posterior samples per domain
    K = D.shape[0]
    theta, _ = fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=K)
    gamma_hat, alpha_hat, beta_hat, eps_hat = readout_params(theta, K)
    D_T = jnp.asarray(D.T)
    out = {"gamma": np.zeros((K, num_chains * num_samples, K)), "point_gamma": gamma_hat}
    for name in ("alpha", "eps", "log_beta", "sigma"):
        out[name] = np.zeros((K, num_chains * num_samples))

    for m in range(K):
        eps_max = float(L[m].min()) * (1.0 - 1e-3)
        init = {
            "gamma_off": jnp.asarray(np.delete(np.maximum(gamma_hat[m], 1e-6), m)),
            "alpha": float(np.clip(alpha_hat[m], 0.02, 0.99)),
            "eps": float(np.clip(eps_hat[m], 1e-6, eps_max * 0.999)),
            "log_beta": float(np.log(beta_hat[m])),
            "sigma": 0.01,
        }
        kernel = NUTS(domain_model, init_strategy=init_to_value(values=init))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed + m), D_T, jnp.log(jnp.asarray(L[m])), m, eps_max, gamma_scale)
        s = mcmc.get_samples()
        out["gamma"][m] = np.insert(np.asarray(s["gamma_off"]), m, 1.0, axis=-1)
        for name in ("alpha", "eps", "log_beta", "sigma"):
            out[name][m] = np.asarray(s[name])
    return out


def summarize(samples, names=None, threshold=0.05):
    # print gamma entries whose posterior puts most of its mass above `threshold`, with 90% intervals
    gamma = samples["gamma"]  # [K,S,K]
    K = gamma.shape[0]
    names = names or [str(i) for i in range(K)]
    lo, med, hi = np.quantile(gamma, [0.05, 0.5, 0.95], axis=1)
    p_on = (gamma > threshold).mean(1)
    for m in range(K):
        for k in range(K):
            if k != m and p_on[m, k] > 0.5:
                print(
                    f"gamma[{names[m]} <- {names[k]}] median {med[m, k]:.3f} 90% [{lo[m, k]:.3f}, {hi[m, k]:.3f}]"
                    f" P(>{threshold}) {p_on[m, k]:.2f}"
                )

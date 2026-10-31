import numpy as np


def exploration_dist(itr, base_dist_K, mode, seed, burst_len=10, burst_mass=0.8, concentration=0.3):
    # sampling distribution for the opening phase. the columns of the token-count matrix N only move
    # independently if the mixture varies, so gamma is identifiable. depends on (itr, seed) only, so every host agrees
    K = len(base_dist_K)
    if mode == "burst":
        # one domain gets burst_mass for burst_len steps, visiting domains in a fixed random order
        order = np.random.default_rng(seed).permutation(K)
        p = (1.0 - burst_mass) * np.asarray(base_dist_K, dtype=np.float64)
        p[order[(itr // burst_len) % K]] += burst_mass
    elif mode == "dirichlet":
        p = np.random.default_rng((seed, itr // burst_len)).dirichlet(np.full(K, concentration))
    else:
        raise ValueError(f"Unrecognized explore_mode: {mode}. Options: ['burst', 'dirichlet']")
    return p / p.sum()

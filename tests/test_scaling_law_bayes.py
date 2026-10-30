import numpy as np
import pytest

pytest.importorskip("numpyro")

from src.scaling_law_bayes import fit_bayes  # noqa: E402
from tests.test_scaling_law import _synthetic  # noqa: E402


def test_bayes_posterior_covers_true_gamma():
    # low noise: at higher noise one draw can legitimately shift the posterior along the alpha-gamma ridge
    D, L, gamma, *_ = _synthetic(noise=5e-4)
    s = fit_bayes(D, L, num_warmup=200, num_samples=200, num_chains=1)
    K = D.shape[0]
    assert s["gamma"].shape == (K, 200, K)
    assert all(np.allclose(s["gamma"][m, :, m], 1.0) for m in range(K))
    lo, hi = np.quantile(s["gamma"], [0.01, 0.99], axis=1)
    assert lo[0, 1] <= gamma[0, 1] <= hi[0, 1]

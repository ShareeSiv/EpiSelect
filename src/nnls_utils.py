import jax
import numpy as np
from scipy.optimize import nnls as scipy_nnls


def nnls_np(A, b, c=None):
    # exact min 0.5 |A w - b|^2 + c . w  s.t. w >= 0, in float64 with scipy's active-set solver.
    # A: [T,K], b: [T,], c: [K,] optional linear term (e.g. an l1 penalty on non-negative weights).
    # the linear term is folded into the right-hand side via a thin qr, A = Q R:
    #   0.5 |A w - b|^2 + c . w = 0.5 |R w - (Q^T b - R^-T c)|^2 + const
    # which is also a K x K problem instead of T x K
    A, b = np.asarray(A, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        return np.full(A.shape[1], np.nan)  # callers treat a nan solve as failed
    Q, R = np.linalg.qr(A)
    rhs = Q.T @ b
    if c is not None and np.any(c):
        rhs = rhs - np.linalg.lstsq(R.T, np.asarray(c, np.float64), rcond=None)[0]
    return scipy_nnls(R, rhs, maxiter=50 * A.shape[1])[0]


def _batched_nnls_np(A, b, c, active):
    # A: [...,T,K], b: [...,T], c: [...,K], active: [...] with broadcastable batch dims (from vmap's expand_dims).
    # inactive problems are skipped and come back as nan
    batch = np.broadcast_shapes(A.shape[:-2], b.shape[:-1], c.shape[:-1], active.shape)
    A = np.broadcast_to(A, batch + A.shape[-2:])
    b = np.broadcast_to(b, batch + b.shape[-1:])
    c = np.broadcast_to(c, batch + c.shape[-1:])
    active = np.broadcast_to(active, batch)
    out = np.full(batch + A.shape[-1:], np.nan)
    for idx in np.ndindex(*batch):
        if active[idx]:
            out[idx] = nnls_np(A[idx], b[idx], c[idx])
    return out


def nnls(A, b, c=None, active=True):
    # jax entry point: runs nnls_np on the host through a callback, so it can sit inside jit and vmap.
    # under vmap the whole batch goes to the host in one call. active=False skips the solve (returns nan),
    # e.g. for problems in a batched while_loop that have already converged
    c = jax.numpy.zeros(A.shape[-1], A.dtype) if c is None else c
    out = jax.ShapeDtypeStruct(A.shape[-1:], A.dtype)

    def host(A, b, c, active):
        return _batched_nnls_np(A, b, c, np.asarray(active, bool)).astype(A.dtype)

    return jax.pure_callback(host, out, A, b, c, jax.numpy.asarray(active), vmap_method="expand_dims")

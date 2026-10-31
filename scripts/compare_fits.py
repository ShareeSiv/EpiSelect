import argparse
import glob
import json
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from src import scaling_law, scaling_law_paper

IDENTITY_L1 = 1e6  # an l1 this large zeroes every off-diagonal weight: the gamma = I law, refit


def fit_paper(D, L, near=1.1):
    # the original joint fit. gamma_spread here is across near-best LBFGS inits, not near-best alphas
    K = D.shape[0]
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("replica", "data"))
    thetas, values, _ = scaling_law_paper.fit_scaling_law_all(jnp.asarray(D), jnp.asarray(L), mesh, K)
    thetas, values = np.asarray(thetas), np.asarray(values)
    values = np.where(np.isfinite(values), values, np.inf)
    best = int(np.argmin(values))
    gammas = np.stack([np.asarray(scaling_law_paper.to_own_weight(t, K))[: K * K].reshape(K, K) for t in thetas])
    # the objective's alpha barrier adds a constant -0.001 per domain, so it can be negative: compare against |best|
    gammas = np.exp(gammas[values <= values[best] + (near - 1) * abs(values[best]) + 1e-4])
    spread = np.where(np.eye(K, dtype=bool), 0.0, gammas.max(0) - gammas.min(0)).max(-1).mean()
    theta = thetas[best]
    return {
        "theta": np.asarray(scaling_law_paper.to_own_weight(theta, K)),
        "log_pred": lambda D_: np.asarray(scaling_law_paper._log_loss(theta, jnp.asarray(D_).T)).T,
        "spread": float(spread),
    }


def fit_own(D, L, l1=0.0):
    K = D.shape[0]
    theta, (_, spread) = scaling_law.fit_scaling_law(jnp.asarray(D), jnp.asarray(L), None, K=K, l1=l1)
    theta = np.asarray(theta)
    return {
        "theta": theta,
        "log_pred": lambda D_: np.asarray(scaling_law._log_loss(theta, jnp.asarray(D_).T)).T,
        "spread": float(spread),
    }


METHODS = {
    "paper": fit_paper,
    "own": fit_own,
    "identity": lambda D, L: fit_own(D, L, l1=IDENTITY_L1),
}


def fit_cond(D):
    # same check the selector logs: the columns of N must move independently for gamma to be identified
    cols = np.asarray(D, dtype=np.float64).T
    return float(np.linalg.cond(cols / np.maximum(cols.max(axis=0), 1.0)))


def gamma_of(theta, K):
    return scaling_law.readout_params(jnp.asarray(theta), K)[0]


def held_out_error(fit_fn, D, L, frac):
    # fit on the first frac of each curve, predict the rest. errors in log loss, per domain
    T = D.shape[1]
    n = min(max(3, int(round(frac * T))), T - 1)
    start = time.time()
    fit = fit_fn(D[:, :n], L[:, :n])
    seconds = time.time() - start
    log_L = np.log(L)
    err = fit["log_pred"](D) - log_L  # [K,T]
    rmse_K = np.sqrt(np.mean(err[:, n:] ** 2, axis=-1))
    return {
        "n_fit": n,
        "seconds": seconds,
        "train_rmse": float(np.sqrt(np.mean(err[:, :n] ** 2))),
        "heldout_rmse": float(rmse_K.mean()),
        "heldout_rmse_worst": float(rmse_K.max()),
        "final_rel_err": float(np.mean(np.abs(np.expm1(err[:, -1])))),
        "heldout_rmse_per_domain": rmse_K.tolist(),
    }


def gamma_stability(gammas_RxKxK):
    # how much the off-diagonal weights move between consecutive refits, and their spread over all refits
    g = np.asarray(gammas_RxKxK)
    off = ~np.eye(g.shape[-1], dtype=bool)
    # gamma is unbounded with own weight 1, so one badly identified refit can dominate the mean: also the median
    out = {"max_offdiag_last": float(g[-1][off].max()), "max_offdiag_per_refit": g[:, off].max(-1).tolist()}
    if len(g) < 2:
        return {**out, "mean_abs_change": float("nan"), "median_abs_change": float("nan"), "mean_std": float("nan")}
    change = np.abs(np.diff(g, axis=0))[:, off]
    out["mean_abs_change"], out["median_abs_change"] = float(change.mean()), float(np.median(change))
    out["mean_std"] = float(g.std(axis=0)[off].mean())
    return out


def load_dumps(paths):
    # a run dir (fit_data_{step}.npz inside) or explicit npz files, ordered by step
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = glob.glob(os.path.join(paths[0], "fit_data_*.npz"))
    step = lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1])  # noqa: E731
    dumps = []
    for p in sorted(paths, key=step):
        data = np.load(p)
        dumps.append((step(p), data["D"].astype(np.float32), data["L"].astype(np.float32)))
    return dumps


def compare(dumps, fracs=(0.5, 0.7), methods=tuple(METHODS), tie_tol=0.05, weak_gamma=0.05):
    K = dumps[-1][1].shape[0]
    _, D_last, L_last = dumps[-1]
    report = {"steps": [s for s, _, _ in dumps], "fit_cond": [fit_cond(D) for _, D, _ in dumps], "methods": {}}
    for name in methods:
        fit_fn = METHODS[name]
        refits = [fit_fn(D, L) for _, D, L in dumps]
        gammas = [gamma_of(f["theta"], K) for f in refits]
        report["methods"][name] = {
            "heldout": {str(frac): held_out_error(fit_fn, D_last, L_last, frac) for frac in fracs},
            "gamma_spread": [f["spread"] for f in refits],
            "stability": gamma_stability(gammas),
            "gamma_last": gammas[-1].tolist(),
        }

    flags = []
    m = report["methods"]
    if "own" in m:
        if m["own"]["stability"]["max_offdiag_last"] < weak_gamma:
            flags.append(f"weak transfer: every off-diagonal gamma of the own-weight fit is < {weak_gamma}")
        if "identity" in m:
            for frac in map(str, fracs):
                own, eye = m["own"]["heldout"][frac]["heldout_rmse"], m["identity"]["heldout"][frac]["heldout_rmse"]
                if abs(own - eye) <= tie_tol * max(own, eye):
                    flags.append(
                        f"own-weight fit and gamma = I tie on held-out error at frac={frac} "
                        f"({own:.4g} vs {eye:.4g}); expect the gamma = I ablation to tie end to end"
                    )
    report["flags"] = flags
    return report


def add_bayes(report, D, L, **kwargs):
    # posterior 90% intervals on gamma, and how many point estimates of each method fall inside them
    from src.scaling_law_bayes import fit_bayes

    samples = fit_bayes(D, L, **kwargs)
    lo, hi = np.percentile(samples["gamma"], [5, 95], axis=1)  # [K,K]
    off = ~np.eye(D.shape[0], dtype=bool)
    report["bayes"] = {
        "gamma_mean": samples["gamma"].mean(1).tolist(),
        "gamma_q05": lo.tolist(),
        "gamma_q95": hi.tolist(),
    }
    for name, m in report["methods"].items():
        g = np.asarray(m["gamma_last"])
        m["frac_offdiag_in_bayes_90"] = float(((g >= lo - 1e-6) & (g <= hi + 1e-6))[off].mean())


def print_report(report):
    print(f"refits at steps {report['steps']}")
    print(f"fit_cond: first {report['fit_cond'][0]:.3g}, last {report['fit_cond'][-1]:.3g}")
    fracs = list(next(iter(report["methods"].values()))["heldout"])
    for frac in fracs:
        print(f"\nfit on the first {float(frac):.0%} of the last refit's curve, predict the rest (log-loss errors)")
        print(f"{'method':9s} {'train rmse':>11s} {'held-out':>10s} {'worst dom':>10s} {'final rel':>10s} {'secs':>7s}")
        for name, m in report["methods"].items():
            h = m["heldout"][frac]
            print(
                f"{name:9s} {h['train_rmse']:11.3e} {h['heldout_rmse']:10.3e} {h['heldout_rmse_worst']:10.3e} "
                f"{h['final_rel_err']:10.3e} {h['seconds']:7.1f}"
            )
    print("\ngamma across refits (own-weight-1 normalisation, off-diagonal)")
    print(
        f"{'method':9s} {'mean |chg|':>11s} {'median |chg|':>13s} {'mean std':>9s} {'max last':>9s} {'spread last':>12s}",
        end="",
    )
    print(f" {'in bayes 90%':>13s}" if "bayes" in report else "")
    for name, m in report["methods"].items():
        s = m["stability"]
        print(
            f"{name:9s} {s['mean_abs_change']:11.3e} {s['median_abs_change']:13.3e} {s['mean_std']:9.3e} "
            f"{s['max_offdiag_last']:9.3f} "
            f"{m['gamma_spread'][-1]:12.3e}",
            end="",
        )
        print(f" {m['frac_offdiag_in_bayes_90']:13.2f}" if "bayes" in report else "")
    print("\nlargest off-diagonal gamma at each refit")
    for name, m in report["methods"].items():
        print(f"{name:9s} " + " ".join(f"{v:9.3g}" for v in m["stability"]["max_offdiag_per_refit"]))
    for flag in report["flags"]:
        print(f"\nNOTE: {flag}")


def main():
    parser = argparse.ArgumentParser(description="replay a run's dumped (D, L) refits through both scaling-law fits")
    parser.add_argument("paths", nargs="+", help="a run dir with fit_data_{step}.npz files, or the npz files")
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.7], help="leading fractions to fit on")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--last", type=int, default=None, help="only use the last N refits")
    parser.add_argument("--every", type=int, default=1, help="only use every Nth refit (the last is always kept)")
    parser.add_argument("--bayes", action="store_true", help="also run the full-Bayes fit on the last refit")
    parser.add_argument("--out", default=None, help="json report (default: <run dir>/compare_fits.json)")
    args = parser.parse_args()

    dumps = load_dumps(args.paths)
    if args.last:
        dumps = dumps[-args.last :]
    dumps = dumps[::-1][:: args.every][::-1]  # the paper fit takes ~0.5-3 min per refit on the cpu
    report = compare(dumps, fracs=args.fracs, methods=args.methods)
    if args.bayes:
        add_bayes(report, dumps[-1][1], dumps[-1][2])
    print_report(report)
    run_dir = args.paths[0] if os.path.isdir(args.paths[0]) else os.path.dirname(os.path.abspath(args.paths[0]))
    out = args.out or os.path.join(run_dir, "compare_fits.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

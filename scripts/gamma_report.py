"""Compare the fitted cross-domain matrix gamma_{m,k} (and downstream accuracy) between the own-weight-1 fit
and the paper's joint fit, for the 124M runs.

    python scripts/gamma_report.py /kaggle/working/run                       # own run only
    python scripts/gamma_report.py runs/own --replay_paper                   # + paper fit on the own run's (D, L)
    python scripts/gamma_report.py runs/own --paper runs/paper --eval eval_results.json --heldout

Reads <run>/fit_data/fit_data_{step}.npz (written at every refit) and <run>/cfg.json. gamma is compared in two
forms: own weight (gamma_mm = 1, src/scaling_law.py) and row-simplex (rows sum to 1, as the paper's Fig. 5).
Writes gamma_report.{md,json} and gamma_heatmaps.png (if matplotlib is installed) to --out.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # `scripts.` imports as a script

from src.common_pile_domains import TASKS

# Table 1 of the paper, 124M rows (zero-shot accuracy). ARC is arc_challenge
TABLE1_TASKS = {
    "ARC": "arc_challenge",
    "BBQ": "bbq",
    "BoolQ": "boolq",
    "CSQA": "commonsense_qa",
    "HSwag": "hellaswag",
    "LAM": "lambada_openai",
    "OBQA": "openbookqa",
    "PIQA": "piqa",
    "SciQ": "sciq",
    "WinoG": "winogrande",
}
TABLE1_124M = {
    "Natural (paper)": [0.199, 0.352, 0.529, 0.194, 0.279, 0.263, 0.134, 0.597, 0.708, 0.519],
    "ADO (paper)": [0.206, 0.365, 0.493, 0.197, 0.277, 0.252, 0.146, 0.596, 0.734, 0.524],
    "EpiSelect (paper)": [0.212, 0.409, 0.587, 0.210, 0.273, 0.260, 0.156, 0.595, 0.707, 0.529],
}
FIG5_BINS = [0.0, 0.1, 0.2, 0.5, 1.0]  # the paper's Fig. 5 colour bins


def _step(path):
    return int(re.findall(r"\d+", os.path.basename(path))[-1])


def load_run(rundir):
    cfg_path = os.path.join(rundir, "cfg.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    tasks = (cfg.get("loader", {}).get("tasks_sampler_cfg") or {}).get("tasks") or TASKS
    paths = sorted(glob.glob(os.path.join(rundir, "fit_data", "fit_data_*.npz")), key=_step)
    if not paths:
        paths = sorted(glob.glob(os.path.join(rundir, "fit_data_*.npz")), key=_step)
    assert paths, f"no fit_data_*.npz in {rundir}"
    K = len(tasks)
    refits = []
    for p in paths:
        z = np.load(p)
        assert "params_own" in z, f"{p} has no fitted params (dumped before this script existed)"
        assert z["D"].shape[0] == K, f"{p}: {z['D'].shape[0]} domains, cfg has {K}"
        refits.append(
            {
                "step": _step(p),
                "D": z["D"].astype(np.float32),
                "L": z["L"].astype(np.float32),
                "gamma": gamma_from_params(z["params_own"], K),
                "guarded": bool(z["gamma_guarded"]) if "gamma_guarded" in z else False,
                "fit_method": str(z["fit_method"]) if "fit_method" in z else cfg.get("loader", {}).get("fit_method"),
            }
        )
    return {"rundir": rundir, "tasks": list(tasks), "refits": refits, "seed": cfg.get("seed")}


def gamma_from_params(params_own, K):
    # own-weight parametrisation: [log_gamma (K*K, diagonal forced to 0), alpha, log_beta, log_eps]
    log_gamma = np.asarray(params_own, dtype=np.float64)[: K * K].reshape(K, K)
    return np.exp(np.where(np.eye(K, dtype=bool), 0.0, log_gamma))


def to_simplex(G):
    return G / G.sum(axis=1, keepdims=True)


def gamma_stats(G, tasks, top=10):
    K = G.shape[0]
    off = ~np.eye(K, dtype=bool)
    S = to_simplex(G)
    S_off = np.where(off, S, 0.0)
    edges = sorted(((S[m, k], m, k) for m in range(K) for k in range(K) if m != k and S[m, k] >= 1e-3), reverse=True)[:top]
    return {
        "diag_share_mean": float(np.diag(S).mean()),
        "diag_share_min": float(np.diag(S).min()),
        "diag_dominant_rows": int((np.diag(S) > S_off.max(axis=1)).sum()),
        "offdiag_frac_gt_0.1": float((S[off] > 0.1).mean()),
        "offdiag_frac_gt_0.2": float((S[off] > 0.2).mean()),
        "offdiag_own_weight_max": float(G[off].max()),
        "offdiag_own_weight_median": float(np.median(G[off])),
        "asymmetry": float(np.linalg.norm(S_off - S_off.T) / max(np.linalg.norm(S_off), 1e-12)),
        "top_edges": [
            {"to": tasks[m], "from": tasks[k], "simplex": float(s), "own_weight": float(G[m, k])} for s, m, k in edges
        ],
    }


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")


def compare_gammas(G1, G2, thresh=0.1):
    off = ~np.eye(G1.shape[0], dtype=bool)
    S1, S2 = to_simplex(G1)[off], to_simplex(G2)[off]
    e1, e2 = S1 > thresh, S2 > thresh
    union = (e1 | e2).sum()
    return {
        "spearman_offdiag_simplex": _spearman(S1, S2),
        "pearson_offdiag_log_own_weight": float(
            np.corrcoef(np.log(np.maximum(G1[off], 1e-12)), np.log(np.maximum(G2[off], 1e-12)))[0, 1]
        ),
        f"jaccard_edges_gt_{thresh}": float((e1 & e2).sum() / union) if union else float("nan"),
        "mean_abs_diff_simplex": float(np.abs(S1 - S2).mean()),
        "diag_share_diff": float(np.abs(np.diag(to_simplex(G1)) - np.diag(to_simplex(G2))).mean()),
    }


def run_summary(run):
    from scripts.compare_fits import fit_cond, gamma_stability

    refits = run["refits"]
    trusted = [r for r in refits if not r["guarded"]]
    return {
        "rundir": run["rundir"],
        "fit_method": refits[-1]["fit_method"],
        "n_refits": len(refits),
        "last_step": refits[-1]["step"],
        "guarded_refits": int(sum(r["guarded"] for r in refits)),
        "fit_cond_last": fit_cond(refits[-1]["D"]),
        "stability_all": gamma_stability([r["gamma"] for r in refits]),
        "stability_trusted": gamma_stability([r["gamma"] for r in trusted]) if trusted else None,
    }


def eval_table(eval_json):
    results = json.load(open(eval_json))["runs"]
    rows = {k: list(v) for k, v in TABLE1_124M.items()}
    for key, r in results.items():
        h = r.get("harness", {})
        rows[f"{r.get('label', key)} (ours)"] = [h.get(t, {}).get("acc", float("nan")) for t in TABLE1_TASKS.values()]
    return {name: {"per_task": dict(zip(TABLE1_TASKS, accs)), "avg": float(np.nanmean(accs))} for name, accs in rows.items()}


def plot(gammas, tasks, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
    except ImportError:
        print("matplotlib not installed: skipping heatmaps")
        return None
    n = len(gammas)
    K = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.2), squeeze=False)
    fig.subplots_adjust(wspace=0.45)
    norm = BoundaryNorm(FIG5_BINS, ncolors=256)
    for ax, (name, G) in zip(axes[0], gammas.items()):
        im = ax.imshow(to_simplex(G), cmap="Blues", norm=norm)
        ax.set_title(f"{name}\n(row-simplex gamma, Fig. 5 bins)", fontsize=10)
        ax.set_xticks(range(K), tasks, rotation=90, fontsize=6)
        ax.set_yticks(range(K), tasks, fontsize=6)
        ax.set_xlabel("from domain k")
        ax.set_ylabel("to domain m")
    fig.colorbar(im, ax=axes[0].tolist(), ticks=FIG5_BINS, shrink=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def to_markdown(report):
    lines = ["# gamma_{m,k}: own-weight-1 fit vs the paper's joint fit", ""]
    for name, s in report["runs"].items():
        lines.append(
            f"- **{name}**: `{s['rundir']}`, fit `{s['fit_method']}`, {s['n_refits']} refits to step {s['last_step']}, "
            f"{s['guarded_refits']} guarded (gamma = I), fit_cond {s['fit_cond_last']:.3g}"
        )
    lines += ["", "## Final gamma", "", "| fit | diag share (mean / min) | diag-dominant rows | off-diag > 0.1 | > 0.2 "
              "| own-weight max / median | asymmetry |", "|---|---|---|---|---|---|---|"]
    K = report["K"]
    for name, s in report["gamma_stats"].items():
        lines.append(
            f"| {name} | {s['diag_share_mean']:.3f} / {s['diag_share_min']:.3f} | {s['diag_dominant_rows']}/{K} "
            f"| {s['offdiag_frac_gt_0.1']:.3f} | {s['offdiag_frac_gt_0.2']:.3f} "
            f"| {s['offdiag_own_weight_max']:.3g} / {s['offdiag_own_weight_median']:.3g} | {s['asymmetry']:.3f} |"
        )
    if report["pairwise"]:
        lines += ["", "## Agreement between fits (off-diagonals)", ""]
        for pair, c in report["pairwise"].items():
            lines.append(f"- **{pair}**: " + ", ".join(f"{k} {v:.3f}" for k, v in c.items()))
    for name, s in report["gamma_stats"].items():
        lines += ["", f"### Strongest cross-domain edges, {name}", "", "| to m | from k | simplex | own weight |",
                  "|---|---|---|---|"]
        lines += [f"| {e['to']} | {e['from']} | {e['simplex']:.3f} | {e['own_weight']:.3g} |" for e in s["top_edges"]]
    if report.get("heldout"):
        lines += ["", "## Held-out fit error on the own run's last refit (log-loss RMSE)", "",
                  "| fit | frac | n fit | heldout RMSE | worst domain |", "|---|---|---|---|---|"]
        for method, by_frac in report["heldout"].items():
            for frac, h in by_frac.items():
                lines.append(f"| {method} | {frac} | {h['n_fit']} | {h['heldout_rmse']:.4f} | {h['heldout_rmse_worst']:.4f} |")
    if report.get("eval"):
        lines += ["", "## Zero-shot accuracy vs Table 1 (124M)", "", "| run | " + " | ".join(TABLE1_TASKS) + " | Avg |",
                  "|---|" + "---|" * (len(TABLE1_TASKS) + 1)]
        for name, r in report["eval"].items():
            lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in r["per_task"].values()) + f" | {r['avg']:.3f} |")
    if report.get("plot"):
        lines += ["", f"![gamma heatmaps]({os.path.basename(report['plot'])})"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("own", help="run dir of the own-weight-fit run")
    parser.add_argument("--paper", default=None, help="run dir of the paper-fit run, if it was trained")
    parser.add_argument("--replay_paper", action="store_true", help="refit the own run's last (D, L) with the paper fit")
    parser.add_argument("--heldout", action="store_true", help="held-out error of both fits on the own run's last refit")
    parser.add_argument("--eval", default=None, help="eval_results.json from src/eval.py, for the Table 1 comparison")
    parser.add_argument("--out", default=None, help="output dir (default: the own run dir)")
    args = parser.parse_args(argv)

    own = load_run(args.own)
    tasks, K = own["tasks"], len(own["tasks"])
    runs = {"own fit": own}
    gammas = {"own fit": own["refits"][-1]["gamma"]}
    if args.paper:
        paper = load_run(args.paper)
        assert paper["tasks"] == tasks, "the two runs train on different domains"
        runs["paper fit"] = paper
        gammas["paper fit"] = paper["refits"][-1]["gamma"]

    last = own["refits"][-1]
    if args.replay_paper or args.heldout:
        from scripts.compare_fits import fit_own, fit_paper, held_out_error
    if args.replay_paper:
        print(f"Replaying the paper fit on the own run's step-{last['step']} data ({K} domains)...")
        gammas["paper fit, replayed on own run"] = gamma_from_params(fit_paper(last["D"], last["L"])["theta"], K)

    report = {
        "K": K,
        "tasks": tasks,
        "runs": {name: run_summary(r) for name, r in runs.items()},
        "gamma_stats": {name: gamma_stats(G, tasks) for name, G in gammas.items()},
        "pairwise": {
            f"own fit vs {name}": compare_gammas(gammas["own fit"], G) for name, G in gammas.items() if name != "own fit"
        },
        "gamma": {name: G.tolist() for name, G in gammas.items()},
    }
    if args.heldout:
        report["heldout"] = {
            name: {str(f): held_out_error(fn, last["D"], last["L"], f) for f in (0.5, 0.7)}
            for name, fn in (("own", fit_own), ("paper", fit_paper))
        }
    if args.eval:
        report["eval"] = eval_table(args.eval)

    out = args.out or args.own
    os.makedirs(out, exist_ok=True)
    report["plot"] = plot(gammas, tasks, os.path.join(out, "gamma_heatmaps.png"))
    with open(os.path.join(out, "gamma_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = to_markdown(report)
    with open(os.path.join(out, "gamma_report.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}/gamma_report.{{md,json}}")
    return report


if __name__ == "__main__":
    main()

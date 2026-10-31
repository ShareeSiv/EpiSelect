import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf

from src.lr_schedule import stable_end_step

ARMS = {  # name -> loader config
    "natural": "mlx_natural",
    "ado": "mlx_ado",
    "episelect_paper": "mlx_episelect_paper",
    "episelect_own": "mlx_episelect",
    "gamma_identity": "mlx_episelect_identity",
}


def commands(root, arms, seeds, overrides):
    # one scripts/train_mlx.py call per (arm, seed), each in <root>/<arm>/seed<seed>
    out = []
    for arm in arms:
        for seed in seeds:
            rundir = os.path.join(root, arm, f"seed{seed}")
            cmd = [sys.executable, "scripts/train_mlx.py", f"loader={ARMS[arm]}", f"seed={seed}", f"rundir={rundir}"]
            out.append((rundir, cmd + list(overrides)))
    return out


def run_summary(rundir):
    # final and stable-phase (warmup end to lr decay start) balanced and worst-domain val loss of one run
    cfg = OmegaConf.load(os.path.join(rundir, "config.yaml"))
    evals = [r for r in map(json.loads, open(os.path.join(rundir, "metrics.jsonl"))) if "val_balanced_loss" in r]
    steps = np.array([r["step"] for r in evals])
    decay_start = stable_end_step(cfg) or cfg.max_steps
    stable = (steps >= cfg.warmup_steps) & (steps < decay_start)
    balanced = np.array([r["val_balanced_loss"] for r in evals])
    worst = np.array([r["val_worst_loss"] for r in evals])
    K = len(json.load(open(os.path.join(rundir, "run_info.json")))["domains"])
    stable_mean = lambda x: float(x[stable].mean()) if stable.any() else float("nan")  # noqa: E731
    return {
        "final_balanced": float(balanced[-1]),
        "final_worst": float(worst[-1]),
        "stable_balanced": stable_mean(balanced),
        "stable_worst": stable_mean(worst),
        "final_per_domain": [evals[-1].get(f"val_loss/task{i}", float("nan")) for i in range(K)],
        "finished": int(steps[-1]) == cfg.max_steps,  # the last eval is at max_steps only once the run ends
    }


def summarize(root, arms=tuple(ARMS)):
    # mean +- std (ddof=1) over the seeds of each arm that finished
    report = {}
    for arm in arms:
        arm_dir = os.path.join(root, arm)
        if not os.path.isdir(arm_dir):
            continue
        runs = []
        for seed_dir in sorted(os.listdir(arm_dir)):
            path = os.path.join(arm_dir, seed_dir)
            if all(os.path.exists(os.path.join(path, f)) for f in ("metrics.jsonl", "config.yaml", "run_info.json")):
                summary = run_summary(path)
                if summary["finished"]:
                    runs.append(summary)
        if not runs:
            continue
        entry = {"n_seeds": len(runs)}
        for key in ("final_balanced", "final_worst", "stable_balanced", "stable_worst"):
            vals = np.array([r[key] for r in runs])
            entry[key] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"))
        entry["final_per_domain"] = np.mean([r["final_per_domain"] for r in runs], axis=0).tolist()
        report[arm] = entry
    return report


def tie_flags(report, a="episelect_own", b="gamma_identity"):
    # small models may transfer little (gamma ~ I); then the own-weight fit and its gamma = I ablation should tie
    flags = []
    if a not in report or b not in report:
        return flags
    for key in ("final_balanced", "final_worst", "stable_balanced", "stable_worst"):
        (ma, sa), (mb, sb) = report[a][key], report[b][key]
        if abs(ma - mb) <= np.nanmax([sa, sb, 0.0]):
            flags.append(f"{a} and {b} tie on {key} ({ma:.4f} vs {mb:.4f}, within one seed std): transfer looks weak")
    return flags


def print_summary(report):
    keys = ("final_balanced", "final_worst", "stable_balanced", "stable_worst")
    print(f"{'arm':18s} {'seeds':>5s} " + " ".join(f"{k:>19s}" for k in keys))
    for arm, entry in report.items():
        cells = " ".join(f"{entry[k][0]:9.4f} ± {entry[k][1]:7.4f}" for k in keys)
        print(f"{arm:18s} {entry['n_seeds']:5d} {cells}")
    for flag in tie_flags(report):
        print(f"\nNOTE: {flag}")


def main():
    parser = argparse.ArgumentParser(description="the tier-3 mlx sweep: selectors x seeds, then a summary")
    parser.add_argument("action", choices=["plan", "run", "summarize"], help="plan only prints the commands")
    parser.add_argument("--root", default="outputs/mlx_sweep")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("overrides", nargs="*", help="passed to every run, e.g. data_dir=... model=mlx_5M")
    args = parser.parse_args()

    if args.action == "summarize":
        report = summarize(args.root, args.arms)
        print_summary(report)
        with open(os.path.join(args.root, "summary.json"), "w") as f:
            json.dump({"arms": report, "flags": tie_flags(report)}, f, indent=2)
        return

    if "natural" in args.arms and any(o.lstrip("+").startswith("loader.start_step") for o in args.overrides):
        # mlx_natural is the ado loader with start_step past the end of the run; moving it turns the arm into ado
        print("# WARNING: loader.start_step also applies to the natural arm, which then refits like ado", flush=True)
    for rundir, cmd in commands(args.root, args.arms, args.seeds, args.overrides):
        if os.path.exists(os.path.join(rundir, "model.safetensors")):
            print(f"# done, skipping: {rundir}")
            continue
        print(shlex.join(cmd), flush=True)
        if args.action == "run":  # one run at a time: each one saturates the gpu
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

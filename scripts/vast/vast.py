#!/usr/bin/env python
"""Rent a Vast.ai GPU for the 124M run and drive it from your own machine, with the `vastai` SDK.

    pip install vastai                 # or: uv pip install -e ".[vast]"
    export VAST_API_KEY=...            # https://console.vast.ai/manage-keys/ (or `vastai set api-key`)
    vastai create ssh-key ~/.ssh/id_ed25519.pub   # once, before renting: status/fetch use ssh

    python scripts/vast/vast.py offers                         # cheapest suitable on-demand GPUs + cost estimate
    python scripts/vast/vast.py launch OFFER_ID                # rent it; the instance runs scripts/vast/pipeline.sh
    python scripts/vast/vast.py status                         # stage, progress, $ so far
    python scripts/vast/vast.py watch --destroy                # wait for DONE, copy the run back, destroy
    python scripts/vast/vast.py fetch                          # copy /workspace/runs back to ./vast_runs
    python scripts/vast/vast.py destroy                        # stop all billing (irreversible)

Commands that take an instance id default to the last one `launch` created (kept in .vast_instance).
Instances are always on-demand: an interruptible one can be stopped at any time, and EpiSelect does not resume
cleanly (see the README).
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from urllib.parse import urlparse

REPO_URL = "https://github.com/ShareeSiv/EpiSelect.git"
BRANCH = "124M-run-NVIDIA"
TOKENS = 60_000 * 256 * 1024  # the paper's 124M run
STATE_FILE = ".vast_instance"
DEFAULT_IMAGE = "vastai/pytorch:@vastai-automatic-tag"  # cuda userspace + git; our venv brings its own torch
DEFAULT_GPUS = ["H100 SXM", "H100 NVL", "H100 PCIE", "A100 SXM4", "A100 PCIE", "L40S", "RTX 5090", "RTX 4090"]

# rough training throughput (tokens/s) of src/torch_train.py per GPU, for the cost column only. these are
# guesses until a run logs perf/tps; replace them with measured numbers
TPS_GUESS = {
    "H100 SXM": 400_000,
    "H100 NVL": 330_000,
    "H100 PCIE": 300_000,
    "A100 SXM4": 180_000,
    "A100 PCIE": 160_000,
    "L40S": 150_000,
    "RTX 6000Ada": 140_000,
    "RTX 5090": 190_000,
    "RTX 4090": 130_000,
    "RTX A6000": 80_000,
    "RTX 3090": 60_000,
}
SETUP_HOURS = 0.5  # install, first compile, the Table 1 eval
BUILD_DATA_HOURS = 1.5  # tokenizing on the instance when no DATA_SRC is given
# traffic the job puts through the host's metered bandwidth (rough): pulling the ~95 GB tokenized build, or
# streaming the raw Common Pile text to tokenize it there, plus the harness datasets; and the results going out
DOWN_GB = {"pull": 110, "build": 250}
UP_GB = 5  # run directory without checkpoints; checkpoints add ~2 GB each
DEFAULT_MAX_BW_COST = 2.0  # $/TB, up or down

# forwarded from your shell to the instance when set (values never printed)
FORWARD_ENV = [
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "HF_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
]
FORWARD_PREFIXES = ["RCLONE_CONFIG_"]


def client():
    try:
        from vastai import VastAI
    except ImportError:
        sys.exit("needs the vastai SDK: pip install vastai")
    try:
        return VastAI()  # key from VAST_API_KEY, then ~/.config/vastai/vast_api_key
    except RuntimeError as e:  # no key
        sys.exit(f"{e}\nCreate one at https://console.vast.ai/manage-keys/")


def offer_query(gpus, min_reliability, disk, max_price=None, min_cuda=12.6, max_bw_cost=None):
    gpu_list = ",".join(g.replace(" ", "_") for g in gpus)
    q = (
        f"num_gpus=1 gpu_name in [{gpu_list}] reliability>={min_reliability} cuda_max_good>={min_cuda} "
        f"disk_space>={disk} compute_cap>=800 inet_down>=200 verified=true rentable=true"
    )
    if max_price is not None:
        q += f" dph_total<={max_price}"
    if max_bw_cost is not None:  # $/TB on the command line, $/GB in the api
        q += f" inet_down_cost<={max_bw_cost / 1024:.6f} inet_up_cost<={max_bw_cost / 1024:.6f}"
    return q


def bandwidth_cost(offer, build_data):
    down = DOWN_GB["build" if build_data else "pull"] * float(offer.get("inet_down_cost") or 0)
    return down + UP_GB * float(offer.get("inet_up_cost") or 0)


def estimate(offer, build_data):
    # (hours, $) for the whole job on this offer, gpu time plus metered bandwidth, or (None, None) for a gpu
    # without a throughput guess
    tps = TPS_GUESS.get(offer.get("gpu_name"))
    if not tps:
        return None, None
    hours = TOKENS / tps / 3600 + SETUP_HOURS + (BUILD_DATA_HOURS if build_data else 0)
    return hours, hours * float(offer["dph_total"]) + bandwidth_cost(offer, build_data)


def format_offers(offers, build_data):
    rows = [
        f"{'offer':>10}  {'gpu':<11} {'$/h':>6} {'rel':>6} {'cuda':>5} {'down':>6} {'disk':>5} {'bw $':>5}  {'est h':>5} {'est $':>6}  location"
    ]
    for o in offers:
        hours, cost = estimate(o, build_data)
        rows.append(
            f"{o['id']:>10}  {o.get('gpu_name', '?'):<11} {float(o['dph_total']):>6.3f} "
            f"{float(o.get('reliability', o.get('reliability2', 0))):>6.3f} {float(o.get('cuda_max_good', 0)):>5.1f} "
            f"{float(o.get('inet_down', 0)):>6.0f} {float(o.get('disk_space', 0)):>5.0f} "
            f"{bandwidth_cost(o, build_data):>5.2f}  "
            f"{(f'{hours:.1f}' if hours else '?'):>5} {(f'{cost:.0f}' if cost else '?'):>6}  {o.get('geolocation', '')}"
        )
    return "\n".join(rows)


def forwarded_env(environ):
    env = {k: environ[k] for k in FORWARD_ENV if environ.get(k)}
    env.update({k: v for k, v in environ.items() if any(k.startswith(p) for p in FORWARD_PREFIXES) and v})
    return env


def onstart_cmd(sha):
    # cloned fresh at a pinned commit, so a container restart reruns the same code
    return (
        "command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git); "
        "mkdir -p /workspace && cd /workspace && "
        f"(test -d EpiSelect/.git || git clone -q --branch {BRANCH} {REPO_URL} EpiSelect) && "
        f"cd EpiSelect && git fetch -q origin {BRANCH} && git checkout -q {sha} && "
        "bash scripts/vast/onstart.sh"
    )


def pushed_head():
    # the instance clones from github, so the commit it runs must be pushed
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True).stdout
    if f"origin/{BRANCH}" not in remote:
        sys.exit(f"HEAD {sha[:8]} is not on origin/{BRANCH}; push it first (the instance clones from GitHub)")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    if dirty.stdout.strip():
        print("note: uncommitted local changes are not sent to the instance")
    return sha


def save_id(iid):
    with open(STATE_FILE, "w") as f:
        f.write(str(iid))


def resolve_id(iid):
    if iid is not None:
        return int(iid)
    if os.path.exists(STATE_FILE):
        return int(open(STATE_FILE).read().strip())
    sys.exit("no instance id given and no .vast_instance from a previous launch")


def ssh_target(vast, iid):
    url = vast.ssh_url(iid)
    if not url:
        sys.exit(f"instance {iid} has no ssh address yet (still starting?)")
    u = urlparse(url)
    return u.username or "root", u.hostname, u.port or 22


def ssh_args(vast, iid):
    user, host, port = ssh_target(vast, iid)
    return ["-p", str(port), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", f"{user}@{host}"]


def ssh_run(vast, iid, command, check=False):
    return subprocess.run(["ssh", *ssh_args(vast, iid), command], capture_output=True, text=True, check=check)


def remote_status(vast, iid):
    out = ssh_run(vast, iid, "cat /workspace/STATUS 2>/dev/null || echo starting").stdout.strip()
    return out or "unknown"


def cmd_offers(args):
    vast = client()
    q = offer_query(args.gpus, args.min_reliability, args.disk, args.max_price, max_bw_cost=args.max_bw_cost)
    offers = vast.search_offers(query=q, type="on-demand", order="dph_total", limit=args.limit, storage=args.disk)
    if not offers:
        print(f"no offers match: {q}")
        return
    if not args.all:  # only offers with a throughput guess, sorted by estimated total cost
        offers = [o for o in offers if estimate(o, args.build_data)[1] is not None]
        offers.sort(key=lambda o: estimate(o, args.build_data)[1])
    print(format_offers(offers, args.build_data))
    print(
        f"\nest $ = (15.7B tokens / guessed tokens/s + {SETUP_HOURS} h setup/eval"
        f"{f' + {BUILD_DATA_HOURS} h tokenizing' if args.build_data else ''}) x $/h, plus bw $ (the host's metered\n"
        f"bandwidth for ~{DOWN_GB['build' if args.build_data else 'pull']} GB in, ~{UP_GB} GB out). Guesses, not measurements."
    )


def cmd_launch(args):
    vast = client()
    sha = pushed_head()
    if not args.skip_ssh_check and not vast.show_ssh_keys():
        sys.exit("no ssh key on your vast account: `vastai create ssh-key ~/.ssh/id_ed25519.pub` first")
    offers = vast.search_offers(
        query=f"id={args.offer_id}", type="on-demand", limit=1, storage=args.disk, no_default=True
    )
    if not offers:
        sys.exit(f"offer {args.offer_id} is gone; run `offers` again")
    offer = offers[0]

    env = forwarded_env(os.environ)
    env["RUN_NAME"] = args.run_name
    for key, val in (
        ("DATA_SRC", args.data_src),
        ("RESULTS_DST", args.results_dst),
        ("TRAIN_OVERRIDES", args.overrides),
    ):
        if val:
            env[key] = val
    hours, cost = estimate(offer, build_data=not args.data_src)
    print(
        f"offer {offer['id']}: {offer.get('gpu_name')} at ${float(offer['dph_total']):.3f}/h on-demand, "
        f"reliability {float(offer.get('reliability', 0)):.3f}, {args.disk} GB disk"
    )
    if hours:
        print(f"estimate: ~{hours:.0f} h, ~${cost:.0f} (a guess; check perf/tps after a few hundred steps)")
    print(f"code: {BRANCH} @ {sha[:8]}   data: {args.data_src or 'tokenized on the instance'}")
    print(f"env sent to the instance: {', '.join(sorted(env))}")
    if "WANDB_API_KEY" not in env:
        print("note: no WANDB_API_KEY in your shell; metrics only go to metrics.jsonl on the instance")
    if not args.yes and input("rent it? [y/N] ").strip().lower() != "y":
        sys.exit("not rented")

    res = vast.create_instance(
        int(offer["id"]),
        image=args.image,
        disk=args.disk,
        env=env,
        label=f"{BRANCH}:{args.run_name}",
        onstart_cmd=onstart_cmd(sha),
        ssh=True,
        direct=True,
        cancel_unavail=True,
    )  # no bid_price: on-demand, never interruptible
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not iid:
        sys.exit(f"create failed: {res}")
    save_id(iid)
    print(f"instance {iid} created; billing starts when it is running. Next: python scripts/vast/vast.py status")


def cmd_status(args):
    vast = client()
    iid = resolve_id(args.id)
    inst = vast.show_instance(iid)
    if not inst:
        sys.exit(f"instance {iid} not found (destroyed?)")
    hours = float(inst.get("duration") or 0) / 3600
    dph = float(inst.get("dph_total") or 0)
    print(
        f"instance {iid}: {inst.get('actual_status')} | {inst.get('gpu_name')} | ${dph:.3f}/h | "
        f"up {hours:.1f} h | ~${dph * hours:.2f} so far"
    )
    if inst.get("actual_status") != "running":
        if inst.get("actual_status") in ("exited", "unknown", "offline"):
            print("the container is not coming back on its own; check `vastai logs` and destroy it if stuck")
        return
    print(f"stage: {remote_status(vast, iid)}")
    tail = ssh_run(
        vast,
        iid,
        "tail -c 3000 /workspace/runs/*/train.log 2>/dev/null | tr '\\r' '\\n' | "
        "grep -v '^$' | tail -n 3; tail -n 3 /workspace/pipeline.log 2>/dev/null",
    ).stdout
    if tail.strip():
        print(tail.rstrip())


def cmd_ssh(args):
    vast = client()
    print("ssh " + " ".join(shlex.quote(a) for a in ssh_args(vast, resolve_id(args.id))))


def fetch(vast, iid, dst, checkpoints=False):
    os.makedirs(dst, exist_ok=True)
    user, host, port = ssh_target(vast, iid)
    cmd = ["rsync", "-az", "--info=progress2", "-e", f"ssh -p {port} -o StrictHostKeyChecking=accept-new"]
    if not checkpoints:
        cmd += ["--exclude", "ckpt_*.pt"]
    cmd += [f"{user}@{host}:/workspace/runs/", dst.rstrip("/") + "/"]
    subprocess.run(cmd, check=True)
    print(f"copied /workspace/runs -> {dst}")


def cmd_fetch(args):
    fetch(client(), resolve_id(args.id), args.dst, args.checkpoints)


def destroy(vast, iid):
    res = vast.destroy_instance(iid)
    print(f"destroyed {iid}: {res}")
    if os.path.exists(STATE_FILE) and open(STATE_FILE).read().strip() == str(iid):
        os.remove(STATE_FILE)


def cmd_destroy(args):
    vast = client()
    iid = resolve_id(args.id)
    if not args.yes and input(f"destroy instance {iid}? its disk is deleted [y/N] ").strip().lower() != "y":
        sys.exit("kept")
    destroy(vast, iid)


def cmd_watch(args):
    # poll until the pipeline writes DONE, then copy the run back and (with --destroy) stop billing
    vast = client()
    iid = resolve_id(args.id)
    while True:
        inst = vast.show_instance(iid)
        if not inst:
            sys.exit(f"instance {iid} is gone")
        state = inst.get("actual_status")
        stage = remote_status(vast, iid) if state == "running" else state
        hours = float(inst.get("duration") or 0) / 3600
        print(
            f"{time.strftime('%H:%M')} {state} | {stage} | ~${float(inst.get('dph_total') or 0) * hours:.2f} so far",
            flush=True,
        )
        if stage.startswith("DONE"):
            fetch(vast, iid, args.dst)
            if args.destroy:
                destroy(vast, iid)
            return
        if stage.startswith("FAILED") or state in ("exited", "offline"):
            sys.exit(
                "stopped watching: the job failed or the instance is down. It is still billing; "
                "inspect with `ssh`/`status`, then `destroy`"
            )
        time.sleep(args.interval)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("offers", help="list suitable on-demand offers with a cost estimate")
    o.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    o.add_argument("--min-reliability", type=float, default=0.98)
    o.add_argument("--disk", type=int, default=200, help="GB: ~95 data + checkpoints + venv")
    o.add_argument("--max-price", type=float)
    o.add_argument(
        "--max-bw-cost",
        type=float,
        default=DEFAULT_MAX_BW_COST,
        help="max $/TB the host charges for download or upload (0 = free bandwidth only)",
    )
    o.add_argument("--limit", type=int, default=40)
    o.add_argument("--build-data", action="store_true", help="include tokenizing on the instance in the estimate")
    o.add_argument("--all", action="store_true", help="also show GPUs without a throughput guess")
    o.set_defaults(fn=cmd_offers)

    la = sub.add_parser("launch", help="rent an offer (on-demand) and start the pipeline on it")
    la.add_argument("offer_id", type=int)
    la.add_argument("--disk", type=int, default=200)
    la.add_argument("--image", default=DEFAULT_IMAGE)
    la.add_argument("--run-name", default="124M_EpiSelect_own")
    la.add_argument("--data-src", help="tokenized data to pull (rclone remote, s3://, gs://); unset: tokenize there")
    la.add_argument("--results-dst", help="upload the run directory here when done (rclone remote, s3://, gs://)")
    la.add_argument("--overrides", help='extra hydra overrides, e.g. "loader.fit_method=paper"')
    la.add_argument("--skip-ssh-check", action="store_true")
    la.add_argument("--yes", action="store_true")
    la.set_defaults(fn=cmd_launch)

    for name, fn, helptext in (
        ("status", cmd_status, "instance state, pipeline stage, progress, cost so far"),
        ("ssh", cmd_ssh, "print the ssh command for the instance"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("id", nargs="?")
        s.set_defaults(fn=fn)

    f = sub.add_parser("fetch", help="rsync /workspace/runs back (checkpoints excluded unless --checkpoints)")
    f.add_argument("id", nargs="?")
    f.add_argument("--dst", default="vast_runs")
    f.add_argument("--checkpoints", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    w = sub.add_parser("watch", help="poll until DONE, fetch the run, optionally destroy the instance")
    w.add_argument("id", nargs="?")
    w.add_argument("--interval", type=int, default=600, help="seconds between polls")
    w.add_argument("--dst", default="vast_runs")
    w.add_argument("--destroy", action="store_true", help="destroy the instance after a successful fetch")
    w.set_defaults(fn=cmd_watch)

    d = sub.add_parser("destroy", help="destroy the instance (stops all billing, deletes its disk)")
    d.add_argument("id", nargs="?")
    d.add_argument("--yes", action="store_true")
    d.set_defaults(fn=cmd_destroy)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

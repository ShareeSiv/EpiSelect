import json
import os

import numpy as np
from omegaconf import OmegaConf

from scripts.sweep_mlx import ARMS, commands, run_summary, summarize, tie_flags
from src.data_selectors.episelect import EpiSelect
from src.scaling_law_paper import _log_loss as paper_log_loss
from tests.test_scaling_law import _synthetic

CFG = dict(max_steps=100, warmup_steps=10, lr_schedule="wsd", wsd_decay_frac=0.2)


def _fake_run(rundir, balanced, worst, cfg=CFG):
    # eval rows every 10 steps; balanced/worst: callables of the step
    os.makedirs(rundir)
    OmegaConf.save(OmegaConf.create(cfg), os.path.join(rundir, "config.yaml"))
    with open(os.path.join(rundir, "run_info.json"), "w") as f:
        json.dump({"domains": ["a", "b"]}, f)
    with open(os.path.join(rundir, "metrics.jsonl"), "w") as f:
        for step in range(0, cfg["max_steps"] + 1, 10):
            b, w = balanced(step), worst(step)
            row = {
                "step": step,
                "val_balanced_loss": b,
                "val_worst_loss": w,
                "val_loss/task0": w,
                "val_loss/task1": 2 * b - w,
            }
            f.write(json.dumps(row) + "\n")


def test_run_summary_stable_phase(tmp_path):
    _fake_run(tmp_path / "r", balanced=lambda s: 3.0 - s / 100, worst=lambda s: 4.0)
    s = run_summary(str(tmp_path / "r"))
    # wsd decay starts at step 80: the stable phase is evals at 10..70
    assert np.isclose(s["stable_balanced"], 3.0 - 0.4) and s["stable_worst"] == 4.0
    assert np.isclose(s["final_balanced"], 2.0) and s["finished"]
    assert s["final_per_domain"] == [4.0, 0.0]


def test_summarize_mean_std_and_tie_flag(tmp_path):
    for seed, off in enumerate([0.0, 0.1, 0.2]):
        _fake_run(tmp_path / "episelect_own" / f"seed{seed}", lambda s, o=off: 3.0 + o, lambda s, o=off: 4.0 + o)
        _fake_run(tmp_path / "gamma_identity" / f"seed{seed}", lambda s, o=off: 3.05 + o, lambda s, o=off: 5.0 + o)
    # an unfinished run is left out
    _fake_run(tmp_path / "natural" / "seed0", lambda s: 3.0, lambda s: 4.0, cfg={**CFG, "max_steps": 200})
    os.remove(tmp_path / "natural" / "seed0" / "metrics.jsonl")
    with open(tmp_path / "natural" / "seed0" / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"step": 0, "val_balanced_loss": 3.0, "val_worst_loss": 4.0}) + "\n")

    report = summarize(str(tmp_path))
    assert set(report) == {"episelect_own", "gamma_identity"}
    mean, std = report["episelect_own"]["final_balanced"]
    assert np.isclose(mean, 3.1) and np.isclose(std, 0.1)
    flags = tie_flags(report)
    assert any("final_balanced" in f for f in flags) and not any("final_worst" in f for f in flags)


def test_commands_cover_every_arm_and_seed():
    cmds = commands("out", list(ARMS), [0, 1], ["data_dir=/d"])
    assert len(cmds) == 10
    rundir, cmd = cmds[0]
    assert rundir == os.path.join("out", "natural", "seed0")
    assert "loader=mlx_natural" in cmd and "seed=0" in cmd and cmd[-1] == "data_dir=/d"


def _selector(fit_method):
    return EpiSelect(
        n_task=4,
        task_token_count=np.ones(4),
        block_size=8,
        ignore_steps=0,
        start_step=0,
        update_interval=1,
        use_prior=False,
        use_credit_assignment=False,
        do_global_avg=True,
        ema=0.9,
        fit_freq=1,
        window_length=5,
        polyorder=2,
        tau=1.0,
        y_axis="val_loss",
        fit_method=fit_method,
    )


def test_paper_fit_method_drives_the_selector():
    # with val-loss fitting, feed a run's evals and check the paper fit is used end to end
    D, L, *_ = _synthetic(T=30)
    sel = _selector("paper")
    sel.task_cnt_TxK = np.diff(np.concatenate([np.zeros((4, 1)), D], 1), axis=1).T.astype(np.int64)
    sel.val_loss_SxK = L.T.astype(np.float32)
    sel.val_step_S = np.arange(1, 31) - 1
    params = sel.compute_scaling_law_params()
    assert np.isnan(sel.gamma_spread) and np.isfinite(params).all()
    # the paper fit's own parameterisation: rows of gamma on the simplex
    pred = np.exp(np.asarray(paper_log_loss(params, np.cumsum(sel.task_cnt_TxK, 0)[sel.val_step_S].astype(np.float32))))
    assert np.abs(np.log(pred / L.T)).max() < 0.15  # loose: its eps >= e^0.5 barrier binds on two domains here
    sel.current_params = params
    metrics = {}
    sel._log_param_metrics(metrics)
    assert np.isclose(metrics["param_gamma/task0_from0"], 1.0)
    p = sel.curr_dist(itr=10)
    assert np.isclose(p.sum(), 1.0) and (p > 0).all()

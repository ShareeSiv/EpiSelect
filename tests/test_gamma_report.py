import json

import numpy as np

from scripts.gamma_report import compare_gammas, gamma_from_params, gamma_stats, main, to_simplex

TASKS = ["github", "pubmed", "wikimedia"]


def _params(G):
    K = G.shape[0]
    return np.concatenate([np.log(np.maximum(G, 1e-12)).reshape(-1), np.full(K, 0.3), np.zeros(K), np.zeros(K)])


def _write_run(path, G, method):
    (path / "fit_data").mkdir(parents=True)
    json.dump({"seed": 0, "loader": {"fit_method": method, "tasks_sampler_cfg": {"tasks": TASKS}}}, open(path / "cfg.json", "w"))
    rng = np.random.default_rng(0)
    for step in (100, 200):
        D = np.cumsum(rng.uniform(1, 2, (3, 20)), 1)
        np.savez(path / "fit_data" / f"fit_data_{step}.npz", D=D, L=3 + 1 / D, params=_params(G), params_own=_params(G),
                 fit_method=method, gamma_guarded=step == 100)


def test_stats_on_a_known_gamma():
    G = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.25, 0.0, 1.0]])
    assert np.allclose(gamma_from_params(_params(G), 3), G)
    S = to_simplex(G)
    assert np.allclose(S.sum(1), 1) and np.isclose(S[0, 1], 1 / 3)
    s = gamma_stats(G, TASKS)
    assert s["diag_dominant_rows"] == 3 and np.isclose(s["offdiag_frac_gt_0.1"], 2 / 6)
    assert s["top_edges"][0] == {"to": "github", "from": "pubmed", "simplex": 1 / 3, "own_weight": 0.5}
    assert len(s["top_edges"]) == 2 and s["asymmetry"] > 1.0  # no reciprocal edges
    same = compare_gammas(G, G)
    assert np.isclose(same["spearman_offdiag_simplex"], 1.0) and same["mean_abs_diff_simplex"] == 0


def test_report_end_to_end(tmp_path):
    G_own = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.25, 0.0, 1.0]])
    G_paper = np.array([[1.0, 0.4, 0.05], [0.0, 1.0, 0.1], [0.3, 0.0, 1.0]])
    _write_run(tmp_path / "own", G_own, "own")
    _write_run(tmp_path / "paper", G_paper, "paper")
    runs = {"runs": {"a": {"label": "EpiSelect own fit", "harness": {"arc_challenge": {"acc": 0.2}, "boolq": {"acc": 0.6}}}}}
    json.dump(runs, open(tmp_path / "eval.json", "w"))
    report = main([str(tmp_path / "own"), "--paper", str(tmp_path / "paper"), "--eval", str(tmp_path / "eval.json"),
                   "--out", str(tmp_path / "out")])
    assert report["runs"]["own fit"]["guarded_refits"] == 1 and report["runs"]["paper fit"]["fit_method"] == "paper"
    assert np.allclose(report["gamma"]["paper fit"], G_paper)
    assert np.isclose(report["eval"]["EpiSelect (paper)"]["avg"], 0.3938)  # Table 1 reports 0.394
    assert np.isclose(report["eval"]["EpiSelect own fit (ours)"]["per_task"]["BoolQ"], 0.6)
    assert (tmp_path / "out" / "gamma_report.md").exists() and (tmp_path / "out" / "gamma_report.json").exists()

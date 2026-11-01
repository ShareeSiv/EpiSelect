import json
import os
import pickle

import numpy as np
import pytest

from scripts import build_compact_commonpile as build
from src import common_pile_domains
from src.dataloader.compact_loader import CompactEvalSet, CompactTasksSampler, domain_tokens

DOMAINS = ["github", "pubmed", "wikimedia"]


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[1 + ord(c) % 200 for c in t] for t in texts]}


def fake_dataset(name):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for i in range(100_000):
        yield {"text": "" if i % 7 == 3 else "".join(rng.choice(list("abcdefgh"), size=int(rng.integers(50, 400))))}


@pytest.fixture
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(common_pile_domains, "_tokenizer", FakeTokenizer())


def test_build_writes_split_streams_with_eos(tmp_path, fake_tokenizer):
    out = tmp_path / "compact"
    build.main(["--out", str(out), "--cap", "2e5", "--domains", *DOMAINS, "--workers", "2"], dataset_fn=fake_dataset)
    top = json.load(open(out / "meta.json"))
    assert top["domains"] == sorted(DOMAINS) and top["eos_token_id"] == 0
    for d in DOMAINS:
        meta = json.load(open(out / d / "meta.json"))
        val_rows, train_rows = common_pile_domains.split_rows(d, token_cap=200_000)
        for split, rows in (("val", val_rows), ("train", train_rows)):
            toks = np.fromfile(out / d / f"{split}.bin", dtype=np.uint16)
            assert toks.size == meta[split]["tokens"]
            assert (toks == 0).sum() == meta[split]["docs"]  # one eos per non-empty doc
            assert meta[split]["docs"] <= rows and toks[-1] == 0
    # rerun skips finished domains
    mtime = os.path.getmtime(out / "github" / "train.bin")
    build.main(["--out", str(out), "--cap", "2e5", "--domains", "github", "--workers", "1"], dataset_fn=fake_dataset)
    assert os.path.getmtime(out / "github" / "train.bin") == mtime


@pytest.fixture
def synthetic_corpus(tmp_path):
    # split over two "datasets", as when uploading parts to kaggle
    build.main(["--out", str(tmp_path / "part1"), "--domains", *DOMAINS[:2], "--synthetic_tokens", "3e4"])
    build.main(["--out", str(tmp_path / "part2"), "--domains", DOMAINS[2], "--synthetic_tokens", "3e4"])
    return str(tmp_path)


def test_sampler_finds_domains_across_parts_and_walks_epochs(synthetic_corpus):
    s = CompactTasksSampler("train", 63, synthetic_corpus, tasks=DOMAINS, seed=1)
    assert np.allclose(s.total_token_n, domain_tokens(DOMAINS, "train", synthetic_corpus))
    assert np.isclose(s.init_dist.sum(), 1.0) and s.init_dist[0] > s.init_dist[1]
    n_win = s.windows[2].shape[0]
    x, y, ids = s.get_batch(np.array([0, 0, n_win]), n_win)
    assert x.shape == (n_win, 63) and (ids == 2).all() and np.array_equal(x[:, 1:], y[:, :-1])
    # one epoch visits every window exactly once
    seen = {w.tobytes() for w in np.concatenate([x, y[:, -1:]], 1).astype(np.uint16)}
    assert len(seen) == n_win and s.epoch[2] == 1 and s.cursor[2] == 0


def test_sampler_resume_continues_the_same_sequence(synthetic_corpus):
    a = CompactTasksSampler("train", 63, synthetic_corpus, tasks=DOMAINS, seed=3)
    counts = np.array([5, 3, 7])
    for _ in range(40):
        a.get_batch(counts, 15)
    state = pickle.loads(pickle.dumps(a.state_dict()))
    b = CompactTasksSampler("train", 63, synthetic_corpus, tasks=DOMAINS, seed=3)
    b.load_state_dict(state)
    for _ in range(5):
        xa, _, _ = a.get_batch(counts, 15)
        xb, _, _ = b.get_batch(counts, 15)
        assert np.array_equal(xa, xb)


def test_eval_set_is_fixed_and_covers_domains(synthetic_corpus):
    e1 = CompactEvalSet(8, 63, synthetic_corpus, tasks=DOMAINS, n_batches=5)
    e2 = CompactEvalSet(8, 63, synthetic_corpus, tasks=DOMAINS, n_batches=5)
    batches = list(e1)
    assert 0 < len(batches) <= 5 and all(ids.shape == (8,) and seq.shape == (8, 64) for ids, seq in batches)
    assert set(np.concatenate([ids for ids, _ in batches]).tolist()) == {0, 1, 2}
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(batches, e2))

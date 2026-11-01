"""Loader for the compact corpus written by scripts/build_compact_commonpile.py (uint16 memmaps, no tensorflow).

Each domain's stream is cut into non-overlapping windows of block_size + 1 tokens (as the tfds path's
rebatch(block_size + 1)). Training visits the windows of each domain in a fresh random order every epoch;
the position in that order is the whole sampler state, so a resumed run continues the same sequence.
"""

import functools
import glob
import json
import os

import jax
import numpy as np

from .adaptive_loader import TasksSampler

TOKEN_BYTES = 2  # uint16


@functools.lru_cache(maxsize=None)
def _domain_dirs(data_dir):
    # domain -> directory, searching data_dir and up to 6 levels below it, so several datasets mounted side by side
    # (e.g. /kaggle/input/commonpile-part1/github, /kaggle/input/commonpile-part2/pubmed) form one corpus
    found = {}
    for depth in range(7):
        pattern = os.path.join(data_dir, *(["*"] * depth), "meta.json")
        for meta in sorted(glob.glob(pattern)):
            ddir = os.path.dirname(meta)
            if os.path.exists(os.path.join(ddir, "train.bin")):
                found.setdefault(os.path.basename(ddir), ddir)
    return found


def domain_path(data_dir, domain, split):
    dirs = _domain_dirs(os.path.abspath(data_dir))
    if domain not in dirs:
        raise FileNotFoundError(
            f"compact domain {domain!r} not found under {data_dir} (found {sorted(dirs)}) -- "
            "build it with scripts/build_compact_commonpile.py"
        )
    return os.path.join(dirs[domain], f"{split}.bin")


def domain_tokens(tasks, split, data_dir):
    return np.array([os.path.getsize(domain_path(data_dir, t, split)) / TOKEN_BYTES for t in tasks], dtype=np.float64)


def open_windows(data_dir, domain, split, window):
    # [n_windows, window] read-only view of one domain's token stream
    mm = np.memmap(domain_path(data_dir, domain, split), dtype=np.uint16, mode="r")
    n = mm.size // window
    if n == 0:
        raise ValueError(f"{domain}/{split}: {mm.size} tokens, fewer than one window of {window}")
    return mm[: n * window].reshape(n, window)


def read_domain_meta(data_dir, domain):
    with open(os.path.join(os.path.dirname(domain_path(data_dir, domain, "train")), "meta.json")) as f:
        return json.load(f)


class CompactTasksSampler(TasksSampler):
    def __init__(self, split: str, block_size: int, data_dir: str, tasks=None, seed: int = 0):
        from src.common_pile_domains import TASKS

        self.proc_idx = jax.process_index()
        self.n_proc = jax.process_count()
        self.block_size = block_size
        self.seed = seed
        self.tasks = list(tasks) if tasks is not None else list(TASKS)
        self.windows = [open_windows(data_dir, t, split, block_size + 1) for t in self.tasks]
        self.total_token_n = domain_tokens(self.tasks, split, data_dir)
        self._init_dist = self.total_token_n / self.total_token_n.sum()
        # per domain: epoch and position in that epoch's permutation (this host's share of the windows)
        self.epoch = np.zeros(len(self.tasks), dtype=np.int64)
        self.cursor = np.zeros(len(self.tasks), dtype=np.int64)
        self._perm = [None] * len(self.tasks)
        print(f"{self.k} tasks. Order:")
        for idx, (tk, w) in enumerate(zip(self.tasks, self.windows)):
            print(f"{idx:,d}: {tk} ({w.shape[0]:,d} windows)")

    def _permutation(self, k):
        if self._perm[k] is None or self._perm[k][0] != self.epoch[k]:
            # hosts split each domain's windows, then shuffle their share every epoch
            own = np.arange(self.proc_idx, self.windows[k].shape[0], self.n_proc, dtype=np.int64)
            rng = np.random.default_rng((self.seed, k, int(self.epoch[k])))
            self._perm[k] = (int(self.epoch[k]), own[rng.permutation(own.size)])
        return self._perm[k][1]

    def _take(self, k, n):
        idx = []
        while n > 0:
            perm = self._permutation(k)
            got = perm[self.cursor[k] : self.cursor[k] + n]
            idx.append(got)
            n -= got.size
            self.cursor[k] += got.size
            if self.cursor[k] >= perm.size:
                self.epoch[k] += 1
                self.cursor[k] = 0
        idx = np.concatenate(idx)
        order = np.argsort(idx)  # sorted reads from the memmap, then back to the shuffled order
        out = np.empty((idx.size, self.block_size + 1), dtype=np.int32)
        out[order] = self.windows[k][idx[order]]
        return out

    def get_batch(self, num_per_task, B):
        tokens, task_ids = [], []
        for k in range(self.k):
            n = int(num_per_task[k])
            if n == 0:
                continue
            tokens.append(self._take(k, n))
            task_ids.append(np.full(n, k, dtype=np.int32))
        tokens = np.concatenate(tokens, axis=0)
        return tokens[:, :-1], tokens[:, 1:], np.concatenate(task_ids)

    def state_dict(self):
        return {"epoch": self.epoch.copy(), "cursor": self.cursor.copy()}

    def load_state_dict(self, state):
        self.epoch = np.asarray(state["epoch"], dtype=np.int64).copy()
        self.cursor = np.asarray(state["cursor"], dtype=np.int64).copy()
        self._perm = [None] * len(self.tasks)

    @property
    def init_dist(self):
        return self._init_dist

    @property
    def k(self):
        return len(self.tasks)


class CompactEvalSet:
    """Held-out val windows, the same every call: up to per_domain windows from each domain's val stream,
    shuffled together with a fixed seed and cut into batches of batch_size // n_proc.
    Yields (taskid_B, seq_BxTp1) like commonpile_loader.make_eval_dset."""

    def __init__(self, batch_size, block_size, data_dir, tasks=None, n_batches=150):
        from src.common_pile_domains import TASKS

        tasks = list(tasks) if tasks is not None else list(TASKS)
        n_proc, proc_idx = jax.process_count(), jax.process_index()
        per_domain = -(-n_batches * batch_size // len(tasks))
        seqs, ids = [], []
        for i, t in enumerate(tasks):
            w = open_windows(data_dir, t, "val", block_size + 1)[:per_domain]
            seqs.append(np.asarray(w))
            ids.append(np.full(w.shape[0], i, dtype=np.int32))
        seqs, ids = np.concatenate(seqs), np.concatenate(ids)
        order = np.random.default_rng(0).permutation(ids.size)
        n_total = min(n_batches, ids.size // batch_size) * batch_size
        order = order[:n_total].reshape(-1, n_proc, batch_size // n_proc)[:, proc_idx]
        self.batches = [(ids[o], seqs[o].astype(np.int32)) for o in order]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

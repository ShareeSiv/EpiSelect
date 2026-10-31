import numpy as np

from .adaptive_loader import TasksSampler


class SyntheticTasksSampler(TasksSampler):
    # stand-in for CommonPileTasksSampler that needs no data on disk, for smoke runs and tests.
    # each domain is a bigram (first-order markov) source over the first `vocab` token ids. domains in the
    # same group share most of their transition logits, so training on one lowers the loss on the others
    def __init__(
        self,
        split,
        block_size,
        data_dir=None,
        tasks=None,
        k=6,
        groups=None,  # group id per domain; default pairs (0,1), (2,3), ...
        vocab=256,
        shared=0.8,  # fraction of logit variance shared within a group
        temperature=0.5,
        seed=0,
    ):
        self.tasks = list(tasks) if tasks is not None else [f"synthetic{i}" for i in range(k)]
        self._k = len(self.tasks)
        self.block_size = block_size
        groups = np.asarray(groups if groups is not None else [i // 2 for i in range(self._k)])
        assert len(groups) == self._k

        rng = np.random.default_rng(seed)  # same sources for every split
        shared_GxVxV = rng.standard_normal((groups.max() + 1, vocab, vocab))
        own_KxVxV = rng.standard_normal((self._k, vocab, vocab))
        logits_KxVxV = (np.sqrt(shared) * shared_GxVxV[groups] + np.sqrt(1 - shared) * own_KxVxV) / temperature
        probs_KxVxV = np.exp(logits_KxVxV - logits_KxVxV.max(-1, keepdims=True))
        self.cdf_KxVxV = np.cumsum(probs_KxVxV / probs_KxVxV.sum(-1, keepdims=True), axis=-1)
        self.vocab = vocab

        self.total_token_n = np.linspace(2.0, 1.0, self._k) * 1e8  # unequal, so the natural mixture is not uniform
        self._init_dist = self.total_token_n / self.total_token_n.sum()
        self.rng = np.random.default_rng((seed, ["train", "val"].index(split)))

    def _sample(self, task_idx, n):
        seq_NxTp1 = np.empty((n, self.block_size + 1), dtype=np.int32)
        seq_NxTp1[:, 0] = self.rng.integers(self.vocab, size=n)
        cdf_VxV = self.cdf_KxVxV[task_idx]
        for t in range(self.block_size):
            u_Nx1 = self.rng.random((n, 1))
            seq_NxTp1[:, t + 1] = np.minimum((cdf_VxV[seq_NxTp1[:, t]] < u_Nx1).sum(-1), self.vocab - 1)
        return seq_NxTp1

    def get_batch(self, num_per_task, B):
        tokens, task_ids = [], []
        for task_idx in range(self.k):
            num_samples = int(num_per_task[task_idx])
            if num_samples == 0:
                continue
            tokens.append(self._sample(task_idx, num_samples))
            task_ids.append(np.full(num_samples, task_idx, dtype=np.int32))
        tokens = np.concatenate(tokens)
        return tokens[..., :-1], tokens[..., 1:], np.concatenate(task_ids)

    @property
    def init_dist(self):
        return self._init_dist

    @property
    def k(self):
        return self._k

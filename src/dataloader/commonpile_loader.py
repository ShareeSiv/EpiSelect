import collections
from types import SimpleNamespace as sn

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental import multihost_utils

from src.sharding import get_data_sharding, get_shard_fn
from src.tfds import common_pile_grouped

from .abstract import Loader
from .adaptive_loader import TasksSampler


def domain_tokens(tasks, split, data_dir):
    # actual built token counts (int32 -> ~4 bytes/token), used as mixture weights
    return np.array(
        [tfds.builder(f"common_pile_grouped/{t}", data_dir=data_dir).info.splits[split].num_bytes / 4 for t in tasks],
        dtype=np.float64,
    )


def make_tfds_dataset(
    task_name,
    split,
    block_size,
    n_proc,
    proc_idx,
    shuffle=False,
    data_dir=None,
    repeat=True,
):
    EOS_TOKEN_ID = common_pile_grouped.tokenizer.eos_token_id
    read_config = tfds.ReadConfig(skip_prefetch=True)
    builder = tfds.builder(f"common_pile_grouped/{task_name}", data_dir=data_dir)
    if split not in builder.info.splits:
        raise RuntimeError(
            f"common_pile_grouped/{task_name} v{common_pile_grouped.CommonPileGrouped.VERSION} "
            f"not found in {data_dir} -- run scripts/build_commonpile_tfds.py"
        )
    val_rows, train_rows = common_pile_grouped.split_rows(task_name)
    expected = {"train": train_rows, "val": val_rows}[split]
    n_built = builder.info.splits[split].num_examples
    assert 0.98 * expected <= n_built <= expected, (
        f"{task_name}/{split}: built {n_built} rows, expected {expected} -- stale build or upstream drift"
    )
    sharded_split = split
    shard_files = n_built >= n_proc
    if shard_files:
        sharded_split = tfds.even_splits(split, n_proc, drop_remainder=False)[proc_idx]
    dset = tfds.load(
        f"common_pile_grouped/{task_name}",
        split=sharded_split,
        shuffle_files=shuffle,
        read_config=read_config,
        data_dir=data_dir,
    )
    if repeat:
        dset = dset.repeat()
    dset = dset.map(
        lambda x: tf.concat([x["tokens"], [EOS_TOKEN_ID]], axis=0),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dset = dset.rebatch(block_size, drop_remainder=True)
    if not shard_files:
        assert not shuffle
        dset = dset.shard(n_proc, proc_idx)
    if shuffle:
        shuffle_size = 1_000
        if common_pile_grouped.DOMAIN_STATS[task_name][1] > 10_000:
            shuffle_size = 10_000
        dset = dset.shuffle(shuffle_size)
    return dset


def make_eval_dset(batch_size, block_size, data_dir):
    n_proc = jax.process_count()
    proc_idx = jax.process_index()
    proc_bs = batch_size // n_proc
    dsets = []
    for i, task in enumerate(common_pile_grouped.TASKS):
        dset = make_tfds_dataset(
            task,
            "val",
            block_size + 1,
            n_proc,
            proc_idx,
            shuffle=False,
            data_dir=data_dir,
            repeat=False,
        )
        dset = dset.map(lambda x, _i=i: (_i, x))
        dsets.append(dset)
    return (
        tf.data.Dataset.sample_from_datasets(
            dsets,
            stop_on_empty_dataset=False,
            rerandomize_each_iteration=False,
            seed=0,
        )
        .batch(proc_bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


class CommonPileBalanced(Loader):
    def __init__(
        self,
        batch_size: int,
        split: str,
        mesh,
        block_size: int,
        g_accum_iters: int,
        data_dir: str,
        balancing_strategy: str = "task_balanced",  # | "empirical"
        seed: int = 0,
    ):
        self.n_proc = jax.process_count()
        self.proc_idx = jax.process_index()
        self.proc_bs = batch_size // self.n_proc
        self.g_accum_iters = g_accum_iters
        self.block_size = block_size
        self.tasks = common_pile_grouped.TASKS
        self.tasks_sampler = sn(tasks=common_pile_grouped.TASKS)

        dsets = []
        for i, task in enumerate(common_pile_grouped.TASKS):
            dset = make_tfds_dataset(
                task,
                split,
                block_size + 1,
                self.n_proc,
                self.proc_idx,
                shuffle=split == "train",
                data_dir=data_dir,
            )
            dset = dset.map(lambda x, _i=i: (_i, x))
            dsets.append(dset)

        weights = None
        if balancing_strategy == "empirical":
            tokens = domain_tokens(self.tasks, split, data_dir)
            weights = (tokens / tokens.sum()).tolist()
        elif balancing_strategy != "task_balanced":
            raise ValueError(f"Unknown balancing_strategy: {balancing_strategy}")

        self.dset = (
            tf.data.Dataset.sample_from_datasets(
                dsets,
                stop_on_empty_dataset=True,
                rerandomize_each_iteration=True,
                weights=weights,
                seed=seed + self.proc_idx,  # hosts draw different domains from the same weights
            )
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )
        self.sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))
        self.cumulative_cnts = collections.Counter()

    def get_batch(self, itr):
        task_idcs, samples = [], []
        for _ in range(self.g_accum_iters * self.proc_bs):
            task_idx, x = next(self.dset)
            samples.append(x)
            task_idcs.append(task_idx.item())
        samples_BGxTp1 = np.stack(samples)
        samples_GxBxTp1 = samples_BGxTp1.reshape(self.g_accum_iters, self.proc_bs, -1)
        x_GxBxT = samples_GxBxTp1[..., :-1]
        y_GxBxT = samples_GxBxTp1[..., 1:]
        task_id_GxB = np.array(task_idcs).reshape(self.g_accum_iters, self.proc_bs)
        info = {"task_ids": task_id_GxB}
        x_GxBxT, y_GxBxT, info = jax.tree_util.tree_map(self.sharding_fn, (x_GxBxT, y_GxBxT, info))
        return x_GxBxT, y_GxBxT, info

    def update(self, loss_GxB, info, itr, **kwargs):
        loss_GxB, task_id_GxB = multihost_utils.process_allgather((loss_GxB, info["task_ids"]), tiled=True)
        self.cumulative_cnts.update(task_id_GxB.reshape(-1).tolist())
        metrics = {}
        for task_id in range(len(common_pile_grouped.TASKS)):
            metrics[f"loss_cnts_cum/task{task_id}"] = self.cumulative_cnts[task_id]
            mask = task_id_GxB == task_id
            if mask.sum() > 0:
                metrics[f"loss/task{task_id}"] = loss_GxB[mask].mean().item()
        for task_id in range(len(common_pile_grouped.TASKS)):
            metrics[f"tokens/domain{task_id}"] = self.cumulative_cnts[task_id] * self.block_size
        metrics["tokens/total"] = sum(self.cumulative_cnts.values()) * self.block_size
        return metrics

    def update_val_loss(self, loss_np_N, taskid_np_N, step):
        pass


class CommonPileTasksSampler(TasksSampler):
    def __init__(self, split: str, block_size: int, data_dir: str, tasks=None):
        self.n_proc = jax.process_count()
        self.proc_idx = jax.process_index()
        self.block_size = block_size
        dsets = []
        self.tasks = list(tasks) if tasks is not None else common_pile_grouped.TASKS  # a subset, e.g. for mlx runs
        self.total_token_n = domain_tokens(self.tasks, split, data_dir)
        for task in self.tasks:
            dset = make_tfds_dataset(
                task,
                split,
                block_size + 1,
                self.n_proc,
                self.proc_idx,
                shuffle=split == "train",
                data_dir=data_dir,
            )
            dset = dset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
            dsets.append(dset)
        self.dsets = dsets
        self._k = len(dsets)
        print(f"{self.k} tasks. Order:")
        for idx, tk in enumerate(self.tasks):
            print(f"{idx:,d}: {tk}")

        self._init_dist = self.total_token_n / self.total_token_n.sum()

    def get_batch(self, num_per_task, B):
        tokens, task_ids = [], []
        for task_idx in range(self.k):
            num_samples = num_per_task[task_idx].item()
            if num_samples == 0:
                continue
            for _ in range(num_samples):
                tokens.append(next(self.dsets[task_idx]))
            task_ids.append(np.ones(num_samples, dtype=np.int32) * task_idx)
        tokens = np.stack(tokens)
        x_BxT, y_BxT = tokens[..., :-1], tokens[..., 1:]
        task_id_B = np.concatenate(task_ids, axis=0)
        return x_BxT, y_BxT, task_id_B

    @property
    def init_dist(self):
        return self._init_dist

    @property
    def k(self):
        return self._k

# training loop for small models on the Mac GPU with MLX. mirrors src/train.py: the same loaders and data
# selectors (fed per-sequence losses each step), lr schedule and per-domain evaluation. no checkpoint resume
import json
import os
import time

import hydra
import jax
import mlx.core as mx
import numpy as np
from jax.sharding import Mesh
from tqdm import trange

from src.lr_schedule import make_lr_schedule, stable_end_step
from src.mlx_model import GPT, count_params, make_eval_fn, make_optimizer, make_train_step


def domain_metrics(loss_N, taskid_N):
    # same keys as src/train.py's evaluate, plus the worst domain
    metrics = {"empirical_loss": float(loss_N.mean())}
    losses = []
    for task in np.unique(taskid_N).tolist():
        task_loss = float(loss_N[taskid_N == task].mean())
        metrics[f"loss/task{task}"] = task_loss
        losses.append(task_loss)
    metrics["balanced_loss"] = float(np.mean(losses))
    metrics["worst_loss"] = float(np.max(losses))
    return metrics


def build_val_set(val_sampler, n_per_domain):
    # a fixed held-out set: the first n_per_domain sequences of each domain's val split
    x_NxT, y_NxT, taskid_N = val_sampler.get_batch(np.full(val_sampler.k, n_per_domain), val_sampler.k * n_per_domain)
    return np.asarray(x_NxT), np.asarray(y_NxT), np.asarray(taskid_N)


def evaluate(eval_fn, val_set, batch_size):
    start_time = time.time()
    x_NxT, y_NxT, taskid_N = val_set
    loss_N = np.concatenate(
        [
            np.array(eval_fn(mx.array(x_NxT[i : i + batch_size]), mx.array(y_NxT[i : i + batch_size])))
            for i in range(0, len(x_NxT), batch_size)
        ]
    ).astype(np.float64)
    metrics = domain_metrics(loss_N, taskid_N)
    metrics["eval_time"] = time.time() - start_time
    metrics["tot_val_exs"] = loss_N.shape[0]
    return loss_N, taskid_N, metrics


def train(config):
    assert config.g_accum_iters == 1, "the mlx loop takes one microbatch per step"
    # mlx reads this at the first matmul; tf32 is ~1.9x faster than full fp32 on the M5 GPU
    os.environ["MLX_ENABLE_TF32"] = "1" if config.tf32 else "0"
    mx.random.seed(config.seed)
    # the loaders put batches on a jax mesh; here that is one cpu device, and batches are copied to mlx
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("replica", "data"))

    model = GPT(config.model)
    n_params = count_params(model)
    print(f"Model has {n_params:,d} parameters.")
    optimizer = make_optimizer(config)
    step, state = make_train_step(model, optimizer)
    eval_fn = make_eval_fn(model)
    lr_T = np.asarray(make_lr_schedule(config)(np.arange(config.max_steps)), dtype=np.float64)

    # with a stable phase, the scaling law is only fit on it so the lr decay does not fake loss drops
    fit_max_step = config.loader.get("fit_max_step") or stable_end_step(config)
    extra = {"fit_max_step": fit_max_step} if "fit_max_step" in config.loader else {}
    train_sampler = hydra.utils.instantiate(
        config.loader,
        **extra,
        batch_size=config.batch_size,
        split="train",
        mesh=mesh,
        block_size=config.block_size,
        g_accum_iters=1,
        seed=config.seed,
    )
    val_sampler = hydra.utils.instantiate(config.sampler, "val", config.block_size, config.data_dir)
    val_set = build_val_set(val_sampler, config.val_seqs_per_domain)
    domains = list(train_sampler.tasks_sampler.tasks)
    print(f"Training over {domains} domains.")

    with open(os.path.join(config.rundir, "run_info.json"), "w") as f:
        json.dump({"domains": domains, "n_params": n_params}, f, indent=2)
    log_f = open(os.path.join(config.rundir, "metrics.jsonl"), "w")

    def log(metrics, itr):
        log_f.write(json.dumps({"step": itr, **{k: float(v) for k, v in metrics.items()}}) + "\n")
        log_f.flush()

    def run_eval(itr):
        loss_N, taskid_N, val_metrics = evaluate(eval_fn, val_set, config.batch_size)
        train_sampler.update_val_loss(loss_N, taskid_N, itr)
        return {f"val_{k}": v for k, v in val_metrics.items()}

    pbar = trange(config.max_steps)
    val_loss = float("nan")
    for itr in pbar:
        metrics = {}
        if itr % config.eval_interval == 0:
            metrics.update(run_eval(itr))
            val_loss = metrics["val_balanced_loss"]

        x_GxBxT, y_GxBxT, info = train_sampler.get_batch(itr)
        optimizer.learning_rate = lr_T[itr]
        loss, loss_B = step(mx.array(np.asarray(x_GxBxT)[0]), mx.array(np.asarray(y_GxBxT)[0]))
        mx.eval(loss, loss_B, state)
        loss_GxB = np.array(loss_B)[None]
        metrics.update({f"opt_{k}": v for k, v in train_sampler.update(loss_GxB, info, itr, lr=lr_T[itr]).items()})
        metrics["loss/opt"] = loss.item()
        metrics["lr"] = lr_T[itr]

        if itr % config.log_interval == 0 or "val_balanced_loss" in metrics:
            if pbar.format_dict["rate"] is not None:
                metrics["perf/tps"] = pbar.format_dict["rate"] * config.batch_size * config.block_size
            log(metrics, itr)
        pbar.set_postfix(loss=metrics["loss/opt"], val=val_loss, lr=metrics["lr"])

    pbar.close()
    log(run_eval(config.max_steps), config.max_steps)
    log_f.close()
    model.save_weights(os.path.join(config.rundir, "model.safetensors"))

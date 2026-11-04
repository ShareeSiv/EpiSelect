# PyTorch training loop for one NVIDIA GPU. mirrors src/train.py: the same configs, loaders and data selectors
# (fed per-sequence losses each step), lr schedule, optimizer update, validation set and checkpoint cadence.
# jax stays on the host cpu, where it builds the paper's init and runs the selectors' scaling-law fits
import glob
import json
import os
import time

# isort: split
from src.torch_model import (  # first: loads triton before tensorflow imports
    GPT,
    count_params,
    init_from_jax,
    seq_losses,
)

# isort: split

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from tqdm import trange  # noqa: E402

from src.dataloader import commonpile_loader  # noqa: E402
from src.lr_schedule import make_lr_schedule, stable_end_step  # noqa: E402
from src.train import _save_token_stats  # noqa: E402


def make_optimizer(model, config):
    # src/train.py's optax chain: clip to global norm 1, adam, then decoupled weight decay of
    # weight_decay / learning_rate * p added before scaling by the lr schedule. torch's AdamW multiplies its
    # decay by the current lr, so the same update is weight_decay=weight_decay / learning_rate. every
    # parameter decays, as with optax.add_decayed_weights without a mask
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, config.beta2),
        eps=1e-8,
        weight_decay=config.weight_decay / config.learning_rate,
        fused=fused,
    )


def optimizer_step(model, optimizer, lr):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def evaluate(loss_fn, dset, chunk, device):
    # src/train.py's evaluate: the first 150 batches of the val stream, per-domain mean losses
    start_time = time.time()
    losses, taskids = [], []
    with torch.no_grad():
        for taskid_B, seq_BxDp1 in dset.take(150).as_numpy_iterator():
            for s in range(0, seq_BxDp1.shape[0], chunk):
                seq = torch.from_numpy(seq_BxDp1[s : s + chunk].astype(np.int64)).to(device, non_blocking=True)
                losses.append(loss_fn(seq[:, :-1], seq[:, 1:]).float().cpu().numpy())
            taskids.append(taskid_B)
    loss_N = np.concatenate(losses).astype(np.float64)
    taskid_N = np.concatenate(taskids).astype(np.int32)
    metrics = {"empirical_loss": loss_N.mean()}
    domain_losses = []
    for task in np.unique(taskid_N).tolist():
        metrics[f"loss/task{task}"] = loss_N[taskid_N == task].mean()
        domain_losses.append(metrics[f"loss/task{task}"])
    metrics["balanced_loss"] = np.mean(domain_losses)
    metrics["eval_time"] = time.time() - start_time
    metrics["tot_val_exs"] = loss_N.shape[0]
    return loss_N, taskid_N, metrics


def _checkpoints(rundir):
    paths = glob.glob(os.path.join(rundir, "ckpt_*.pt"))
    return sorted(paths, key=lambda p: int(os.path.basename(p)[5:-3]))


def save_checkpoint(model, optimizer, itr, config):
    path = os.path.join(config.rundir, f"ckpt_{itr}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": itr}, path + ".tmp")
    os.replace(path + ".tmp", path)  # a crash mid-save never leaves a truncated latest checkpoint
    keep = config.get("ckpt_max_to_keep")
    if keep:
        for old in _checkpoints(config.rundir)[:-keep]:
            os.remove(old)


def train(config):
    assert torch.cuda.is_available() or config.get("allow_cpu"), "no cuda device visible to torch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True  # only fp32 matmuls outside autocast (e.g. nothing in the model)
    torch.backends.cudnn.allow_tf32 = True
    compute_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.compute_dtype]
    assert config.param_dtype == "float32", "params are kept in fp32 and autocast to compute_dtype"
    G = config.g_accum_iters

    model = GPT(config.model, max_len=max(4096, config.block_size)).to(device)
    model.load_state_dict(init_from_jax(config.model, config.seed))  # the paper's init for this seed
    print(
        f"Model has {count_params(model):,d} non-embedding parameters ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'})."
    )
    optimizer = make_optimizer(model, config)
    lr_T = np.asarray(make_lr_schedule(config)(np.arange(config.max_steps + 1)), dtype=np.float64)

    def loss_fn(x_BxT, y_BxT):
        with torch.autocast(device.type, dtype=compute_dtype, enabled=compute_dtype != torch.float32):
            return seq_losses(model, x_BxT, y_BxT)

    if config.get("compile", True) and device.type == "cuda":
        loss_fn = torch.compile(loss_fn)

    first_step = 0
    ckpts = _checkpoints(config.rundir)
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_step = state["step"] + 1
        print(f"Resumed from {ckpts[-1]}. The data selector and data streams restart from scratch.")

    eval_bs = config.get("eval_batch_size") or config.batch_size
    val_dset = commonpile_loader.make_eval_dset(eval_bs, config.block_size, data_dir=config.loader.data_dir)
    eval_chunk = config.get("eval_microbatch") or eval_bs

    # with a stable phase, the scaling law is only fit on it so the lr decay does not fake loss drops
    fit_max_step = config.loader.get("fit_max_step") or stable_end_step(config)
    extra = {"fit_max_step": fit_max_step} if "fit_max_step" in config.loader else {}
    import jax  # cpu only: the loaders put batches on a one-device host mesh, copied to the gpu below
    from jax.sharding import Mesh

    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), axis_names=("replica", "data"))
    train_sampler = hydra.utils.instantiate(
        config.loader,
        **extra,
        batch_size=config.batch_size,
        split="train",
        mesh=mesh,
        block_size=config.block_size,
        g_accum_iters=G,
        seed=config.seed,
    )
    print(f"Training over {train_sampler.tasks_sampler.tasks} domains.")

    log_f = open(os.path.join(config.rundir, "metrics.jsonl"), "a")
    use_wandb = wandb.run is not None
    tok_per_step = config.batch_size * G * config.block_size
    pbar = trange(first_step, config.max_steps, initial=first_step, total=config.max_steps)
    for itr in pbar:
        metrics = {}
        if itr % config.eval_interval == 0:
            loss_N, taskid_N, val_metrics = evaluate(loss_fn, val_dset, eval_chunk, device)
            train_sampler.update_val_loss(loss_N, taskid_N, itr)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            _save_token_stats(train_sampler, config, itr)

        x_GxBxT, y_GxBxT, info = train_sampler.get_batch(itr)
        x_GxBxT = torch.from_numpy(np.asarray(x_GxBxT).astype(np.int64)).to(device, non_blocking=True)
        y_GxBxT = torch.from_numpy(np.asarray(y_GxBxT).astype(np.int64)).to(device, non_blocking=True)
        loss_B_list = []
        for g in range(G):
            loss_B = loss_fn(x_GxBxT[g], y_GxBxT[g])
            (loss_B.mean() / G).backward()  # the mean over all G x B sequences, as src/train.py
            loss_B_list.append(loss_B.detach())
        optimizer_step(model, optimizer, lr_T[itr])
        loss_GxB = torch.stack(loss_B_list).float().cpu().numpy()

        sel = train_sampler.update(loss_GxB, info, itr, lr=float(lr_T[itr]))
        metrics.update({f"opt_{k}": v for k, v in sel.items()})
        metrics["loss/opt"] = float(loss_GxB.mean())
        metrics["lr"] = float(lr_T[itr])

        if (itr + 1) % config.eval_interval == 0 or itr + 1 == config.max_steps:
            save_checkpoint(model, optimizer, itr, config)
        if itr % 20 == 0 or "val_balanced_loss" in metrics:
            if pbar.format_dict["rate"] is not None:
                metrics["perf/tps"] = pbar.format_dict["rate"] * tok_per_step
            if use_wandb:
                wandb.log(metrics, step=itr)
            log_f.write(json.dumps({"step": itr, **{k: float(v) for k, v in metrics.items()}}) + "\n")
            log_f.flush()
        pbar.set_postfix(loss=metrics["loss/opt"], lr=metrics["lr"])
    pbar.close()

    # one more eval on the final weights (src/train.py's last eval is at max_steps - eval_interval)
    _, _, val_metrics = evaluate(loss_fn, val_dset, eval_chunk, device)
    final = {f"val_{k}": v for k, v in val_metrics.items()}
    if use_wandb:
        wandb.log(final, step=config.max_steps)
    log_f.write(json.dumps({"step": config.max_steps, **{k: float(v) for k, v in final.items()}}) + "\n")
    log_f.close()
    torch.save(model.state_dict(), os.path.join(config.rundir, "model.pt"))
    print(f"Final val balanced loss {final['val_balanced_loss']:.4f}. Weights: {config.rundir}/model.pt")

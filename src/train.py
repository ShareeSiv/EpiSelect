import glob
import json
import os
import pickle
import time
import typing as tp
from functools import partial

import equinox as eqx
import gcsfs
import hydra
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jax.experimental import mesh_utils, multihost_utils
from tqdm import trange

from src.dataloader.abstract import Loader
from src.lr_schedule import make_lr_schedule, stable_end_step

from .model import GPT, count_params, shard_gpt
from .sharding import get_data_sharding, get_shard_fn, reshard

jax.config.update("jax_threefry_partitionable", True)

jnp, jrandom, vmap, scan, jtu = jax.numpy, jax.random, jax.vmap, jax.lax.scan, jax.tree_util
Array = jax.Array
KeyArray = tp.Any
Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def cast_pytree(pytree: tp.Any, dtype: jnp.dtype) -> tp.Any:
    # cast every array leaf to dtype, leaving non-arrays untouched

    def cast(x):
        return x.astype(dtype) if eqx.is_array(x) else x

    return jtu.tree_map(cast, pytree)


def make_training_fns(
    config, optimizer: optax.GradientTransformationExtraArgs, mesh: Mesh
) -> tp.Tuple[tp.Callable, tp.Callable]:
    def loss_fn(
        model_params: GPT,
        model_static: GPT,
        x_BxT: Array,
        y_BxT: Array,
        key: tp.Optional[KeyArray],
    ) -> Array:
        model = eqx.combine(model_params, model_static)
        if key is not None:
            key = jrandom.split(key, x_BxT.shape[0])
        logits_BxTxV = vmap(model)(x_BxT, key=key).astype(jnp.float32)
        vector_loss_BxT = optax.softmax_cross_entropy_with_integer_labels(logits_BxTxV, y_BxT)
        vector_loss_B = vector_loss_BxT.mean(axis=-1)
        return vector_loss_B.mean(), vector_loss_B

    @partial(eqx.filter_jit, donate="all")
    def step(model: GPT, opt_state, x_GxBxT: Array, y_GxBxT: Array, key: KeyArray):
        G = config.g_accum_iters
        params, static = eqx.partition(model, eqx.is_array)
        params_cpt = cast_pytree(params, jnp.dtype(config.compute_dtype))

        def microstep(grad_so_far, xykey_g: tp.Tuple[Array, Array, Array, KeyArray]):
            grad, loss_vector = jax.grad(loss_fn, has_aux=True)(params_cpt, static, *xykey_g)
            grad = shard_gpt(grad, mesh, config.shard_model)
            return jtu.tree_map(lambda x, y: x + y, grad, grad_so_far), loss_vector

        all_keys = jrandom.split(key, G)
        init_grad = jtu.tree_map(jnp.zeros_like, params)
        grad, loss_GxB = scan(microstep, init_grad, (x_GxBxT, y_GxBxT, all_keys))
        # Gradients are summed over the G microbatches, so rescale
        loss, grad = jnp.mean(loss_GxB), jtu.tree_map(lambda x: x / G, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        model = eqx.combine(optax.apply_updates(params, updates), static)
        return model, opt_state, loss, loss_GxB

    @eqx.filter_jit
    def simple_loss(model, x_1xBxD: Array, y_1xBxD: Array, key: tp.Optional[KeyArray]) -> Array:
        x_BxD, y_BxD = x_1xBxD.squeeze(0), y_1xBxD.squeeze(0)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        return loss_fn(model_params, model_static, x_BxD, y_BxD, key)

    sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))

    def evaluate(model: GPT, dset) -> tp.Dict[int, float]:
        start_time = time.time()
        eval_model = eqx.Partial(cast_pytree(model, jnp.dtype(config.compute_dtype)), inference=True)
        metrics = {}
        loss_np_N = np.zeros((0,), dtype=np.float64)
        taskid_np_N = np.zeros((0,), dtype=np.int32)
        # Capped at 150 batches: dataset sharding is uneven across hosts, and letting some
        # workers exhaust their shard before others deadlocks the allgather below
        batches = dset.take(150).as_numpy_iterator() if hasattr(dset, "take") else dset
        for taskid_B, seq_BxDp1 in batches:
            x_1xBxD, y_1xBxD = seq_BxDp1[:, :-1][None], seq_BxDp1[:, 1:][None]
            x_1xBxD, y_1xBxD, taskid_1xB = jtu.tree_map(sharding_fn, (x_1xBxD, y_1xBxD, taskid_B[None]))
            _, loss_B = simple_loss(eval_model, x_1xBxD, y_1xBxD, None)
            loss_np_B, taskid_np_1xB = multihost_utils.process_allgather((loss_B, taskid_1xB), tiled=True)
            loss_np_N = np.concatenate((loss_np_N, np.asarray(loss_np_B).reshape(-1)))
            taskid_np_N = np.concatenate((taskid_np_N, np.asarray(taskid_np_1xB).reshape(-1)))

        metrics["empirical_loss"] = loss_np_N.mean()
        losses = []
        for task in np.unique(taskid_np_N).tolist():
            task_loss = loss_np_N[taskid_np_N == task].mean()
            metrics[f"loss/task{task}"] = task_loss
            losses.append(task_loss)
        metrics["balanced_loss"] = np.mean(losses)
        metrics["eval_time"] = time.time() - start_time
        metrics["tot_val_exs"] = loss_np_N.shape[0]
        return loss_np_N, taskid_np_N, metrics

    return step, evaluate


def _save_token_stats(train_sampler, config, itr):
    domain_names = train_sampler.tasks_sampler.tasks
    stats = {
        "step": itr,
        "total_tokens": int(sum(train_sampler.cumulative_cnts.values()) * config.block_size),
        "per_domain": {
            domain_names[task_id] if task_id < len(domain_names) else str(task_id): int(cnt * config.block_size)
            for task_id, cnt in train_sampler.cumulative_cnts.items()
        },
    }
    path = os.path.join(config.rundir, "token_stats.json")
    if config.rundir.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(path, "w") as f:
            json.dump(stats, f, indent=2)
    else:
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def make_eval_set(config, tasks):
    # data_format: tfds (common_pile_grouped, needs tensorflow) | compact (uint16 memmaps, scripts/build_compact_commonpile.py)
    batch_size = config.get("eval_batch_size") or config.batch_size
    if config.get("data_format", "tfds") == "compact":
        from src.dataloader.compact_loader import CompactEvalSet

        return CompactEvalSet(batch_size, config.block_size, config.loader.data_dir, tasks=tasks)
    from src.dataloader import commonpile_loader

    return commonpile_loader.make_eval_dset(batch_size, config.block_size, data_dir=config.loader.data_dir)


def _selector_state_path(rundir, step):
    return os.path.join(rundir, "selector_state", f"state_{step}_p{jax.process_index()}.pkl")


def save_selector_state(train_sampler, rundir, step, keep):
    # written next to each model checkpoint, so a resumed run continues the same curriculum. one file per host:
    # hosts draw batches with their own rng. local rundirs only (the kaggle path); gs:// runs resume without it
    if not hasattr(train_sampler, "state_dict") or rundir.startswith("gs://"):
        return
    path = _selector_state_path(rundir, step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(train_sampler.state_dict(), f)
    os.replace(path + ".tmp", path)
    old = sorted(
        glob.glob(os.path.join(os.path.dirname(path), f"state_*_p{jax.process_index()}.pkl")),
        key=lambda q: int(os.path.basename(q).split("_")[1]),
    )
    for q in old[: -(keep + 1)] if keep else []:
        os.remove(q)


def load_selector_state(train_sampler, rundir, step):
    if not hasattr(train_sampler, "load_state_dict") or rundir.startswith("gs://"):
        return False
    path = _selector_state_path(rundir, step)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"model checkpoint {step} has no selector state at {path}: resuming would restart the curriculum. "
            "Delete the checkpoint to resume from an earlier one, or set allow_selector_reset=true."
        )
    with open(path, "rb") as f:
        train_sampler.load_state_dict(pickle.load(f))
    print(f"Restored selector state from step {step}.")
    return True


def _write_status(config, step, finished, reason):
    if config.debug or jax.process_index() != 0 or config.rundir.startswith("gs://"):
        return
    with open(os.path.join(config.rundir, "run_status.json"), "w") as f:
        json.dump({"last_step": step, "max_steps": config.max_steps, "finished": finished, "reason": reason}, f)


def train(config):
    n_devices = jax.device_count()
    local_cnt = jax.local_device_count()
    assert local_cnt in {1, 4, 8}, f"Unexpected local device count: {local_cnt}."
    mesh = Mesh(
        mesh_utils.create_device_mesh((n_devices // local_cnt, local_cnt)),
        axis_names=("replica", "data"),
    )

    start_time = time.time()
    time_budget_s = config.get("time_budget_h") and 3600 * config.time_budget_h
    ckpt_keep = config.get("ckpt_keep")  # null keeps every checkpoint
    if not config.debug:
        mngr = ocp.CheckpointManager(
            config.rundir,
            ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
            options=ocp.CheckpointManagerOptions(max_to_keep=ckpt_keep, save_interval_steps=config.eval_interval),
        )

    scheduler = make_lr_schedule(config)

    @jax.jit
    def get_lr(_opt_state):
        return scheduler(_opt_state[3].count)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay / config.learning_rate),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    step, evaluate = make_training_fns(config, optimizer, mesh)

    key = jrandom.PRNGKey(config.seed)

    def init_model(model_key):
        model = GPT(config.model, model_key)
        model = cast_pytree(model, config.param_dtype)
        return shard_gpt(model, mesh, config.shard_model)

    key, key1 = jrandom.split(key)
    model = eqx.filter_jit(init_model)(key1)
    print(f"Model has {count_params(model):,d} parameters.")

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Scalars must be explicitly replicated across the mesh
    opt_state = jtu.tree_map(lambda x: reshard(x, NamedSharding(mesh, P())) if x.ndim == 0 else x, opt_state)

    first_step = 0
    if not config.debug and mngr.latest_step() is not None:
        ex_state = (jtu.tree_leaves(model), jtu.tree_leaves(opt_state))
        ex_shardings = jtu.tree_map(lambda x: x.sharding if eqx.is_array(x) else None, ex_state)
        restore_args = ocp.checkpoint_utils.construct_restore_args(ex_state, ex_shardings)
        model_leaves, opt_state_leaves = mngr.restore(mngr.latest_step(), restore_kwargs={"restore_args": restore_args})
        model = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)
        opt_state = jtu.tree_unflatten(jtu.tree_structure(opt_state), opt_state_leaves)
        first_step = mngr.latest_step() + 1

    # with a stable phase, the scaling law is only fit on it so the lr decay does not fake loss drops
    # (only the adaptive loaders take it, so it is not passed to the natural baseline)
    fit_max_step = config.loader.get("fit_max_step") or stable_end_step(config)
    extra = {"fit_max_step": fit_max_step} if "fit_max_step" in config.loader else {}
    train_sampler: Loader = hydra.utils.instantiate(
        config.loader,
        **extra,
        batch_size=config.batch_size,
        split="train",
        mesh=mesh,
        block_size=config.block_size,
        g_accum_iters=config.g_accum_iters,
        seed=config.seed,
    )
    print(f"Training over {train_sampler.tasks_sampler.tasks} domains.")
    val_dset = make_eval_set(config, train_sampler.tasks_sampler.tasks)

    if first_step > 0:
        try:
            load_selector_state(train_sampler, config.rundir, first_step - 1)
        except FileNotFoundError:
            if not config.get("allow_selector_reset", False):
                raise
            print("WARNING: no selector state; the curriculum restarts from the initial distribution.")

    pbar = trange(
        first_step,
        config.max_steps,
        initial=first_step,
        total=config.max_steps,
        disable=jax.process_index() != 0,
        mininterval=config.get("pbar_mininterval", 0.1),
    )
    is_main = jax.process_index() == 0
    timed_from = first_step + 20  # throughput summary skips compilation and the first steps
    t_timed = None
    for itr in pbar:
        if itr == timed_from:
            t_timed = time.time()
        metrics = {}
        if itr % config.eval_interval == 0:
            loss_np_N, taskid_np_N, val_metrics = evaluate(model, val_dset)
            if hasattr(train_sampler, "update_val_loss"):
                train_sampler.update_val_loss(loss_np_N, taskid_np_N, itr)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if not config.debug and is_main:
                _save_token_stats(train_sampler, config, itr)

        key, key1 = jrandom.split(key)
        x_GxBxD, y_GxBxD, info = train_sampler.get_batch(itr)
        model, opt_state, loss, loss_GxB = step(model, opt_state, x_GxBxD, y_GxBxD, key1)
        metrics.update({f"opt_{k}": v for k, v in train_sampler.update(loss_GxB, info, itr, lr=get_lr(opt_state).item()).items()})

        # always save the last step, and stop cleanly (saved) once the wall-clock budget is spent, e.g. at the
        # end of a kaggle session; rerunning with the same rundir resumes from here
        last = itr == config.max_steps - 1
        out_of_time = bool(time_budget_s) and time.time() - start_time > time_budget_s and not last
        if not config.debug:
            force = last or out_of_time
            if force or mngr.should_save(itr):
                mngr.save(itr, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)), force=force)
                save_selector_state(train_sampler, config.rundir, itr, ckpt_keep)
        metrics["loss/opt"] = loss.item()
        metrics["lr"] = get_lr(opt_state).item()

        if not config.debug and is_main and itr % 20 == 0:
            if pbar.format_dict["rate"] is not None:
                tok_per_step = config.batch_size * config.g_accum_iters * config.block_size
                metrics["perf/tps"] = pbar.format_dict["rate"] * tok_per_step
            wandb.log(metrics, step=itr)
        if config.get("pbar_full_metrics", True):
            pbar.set_postfix(**metrics)
        else:  # a few hundred metrics per update floods long notebook logs
            pbar.set_postfix({k: metrics[k] for k in ("loss/opt", "lr", "val_balanced_loss") if k in metrics})
        if out_of_time:
            print(f"Time budget of {config.time_budget_h} h spent: stopping after step {itr} (checkpointed).")
            break

    pbar.close()
    if t_timed is not None and is_main:
        n = itr - timed_from + 1
        sec = (time.time() - t_timed) / n
        tok = config.batch_size * config.g_accum_iters * config.block_size
        mem = jax.local_devices()[0].memory_stats() or {}
        peak = mem.get("peak_bytes_in_use", 0) / 2**30
        limit = mem.get("bytes_limit", 0) / 2**30
        print(
            f"Throughput over {n} steps: {sec:.3f} s/step, {tok / sec:,.0f} tokens/s; "
            f"{config.max_steps} steps would take {config.max_steps * sec / 3600:.2f} h. "
            f"Peak memory on device 0: {peak:.2f} / {limit:.2f} GiB."
        )
    if not config.debug:
        mngr.wait_until_finished()
        finished = mngr.latest_step() == config.max_steps - 1
        _write_status(config, mngr.latest_step(), finished, "done" if finished else "time_budget")

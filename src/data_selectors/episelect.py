import functools

import gcsfs
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from scipy.signal import savgol_filter

from src.data_selectors.ado import AdoSelector, clip_min_probability
from src.scaling_law import epiplexity_grads, fit_scaling_law, readout_params


class EpiSelect(AdoSelector):
    def __init__(
        self,
        use_prior,
        use_credit_assignment,
        do_global_avg,
        ema,
        fit_freq,
        window_length,
        polyorder,
        tau,
        fit_max_frac=1.0,
        fit_max_step=None,
        gamma_l1=0.0,
        gamma_smooth=0.0,
        gamma_identity=False,
        shared_alpha=False,
        alpha_range=(0.05, 0.8),
        alpha_shrink=0.7,
        alpha_min_halfwidth=0.05,
        fit_data_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_prior = use_prior
        self.use_credit_assignment = use_credit_assignment
        self.do_global_avg = do_global_avg
        self.ema = ema
        self.fit_freq = fit_freq
        self.window_length = window_length
        self.polyorder = polyorder
        self.tau = tau
        self.fit_max_frac = fit_max_frac  # fit only the first fraction of points (e.g. before lr decay)
        self.fit_max_step = fit_max_step  # only fit steps below this (end of the lr stable phase)
        self.gamma_l1 = gamma_l1  # l1 penalty on off-diagonal weights in the inner regression
        self.gamma_smooth = gamma_smooth  # ema weight on the previous refit's gamma, in [0,1)
        self.gamma_identity = gamma_identity  # ablation: ignore transfer, gamma = I
        self.shared_alpha = shared_alpha  # one alpha for all domains
        # per-domain alpha search range; each refit narrows it around the last alpha by alpha_shrink
        self.alpha_lo = np.full(self.k, float(alpha_range[0]))
        self.alpha_hi = np.full(self.k, float(alpha_range[1]))
        self.alpha_shrink = alpha_shrink
        self.alpha_min_halfwidth = alpha_min_halfwidth
        self.fit_data_path = fit_data_path  # if set, dump each refit's (D, L) for offline (bayesian) fits

        self.fit_val = None
        self.gamma_spread = None
        self.fit_cond = None

    def curr_dist(self, itr):
        if itr < self.start_step or self.current_params is None:
            return None

        D = jnp.sum(self.task_cnt_TxK, axis=0, keepdims=True).T  # [K,1]
        weighted_slope_K = self._get_S_grads(D=D, current_params=self.current_params)  # [K,]
        self.p_scaling = weighted_slope_K / weighted_slope_K.sum()

        if self.use_prior:
            weighted_slope_K = weighted_slope_K * self.empirical_dist
        if self.use_credit_assignment:
            weighted_slope_K = weighted_slope_K * self.p_history**0.5

        dist_K = weighted_slope_K / weighted_slope_K.sum()

        self.p = clip_min_probability(self.ema * self.p_avg + (1.0 - self.ema) * dist_K, 1e-8)
        if self.do_global_avg:
            self.p_avg = (1 - 1 / (itr + 1)) * self.p_avg + (1 / (itr + 1)) * self.p
        else:
            self.p_avg = self.p
        return self.p

    def _log_param_metrics(self, metrics):
        K = self.k
        metrics["fit_val"] = float(self.fit_val)
        metrics["gamma_spread"] = float(self.gamma_spread)
        metrics["fit_cond"] = float(self.fit_cond)
        metrics["alpha_range_width"] = float(np.mean(self.alpha_hi - self.alpha_lo))
        gamma_KxK, alpha_K, beta_K, irred_K = readout_params(self.current_params, K)
        for i in range(K):
            metrics[f"param_alpha/task{i}"] = float(alpha_K[i])
            metrics[f"param_beta/task{i}"] = float(beta_K[i])
            metrics[f"param_irred/task{i}"] = float(irred_K[i])
            metrics[f"remaining_dist/task{i}"] = self.remain_dist[i]
            for j in range(K):
                metrics[f"param_gamma/task{i}_from{j}"] = float(gamma_KxK[i, j])

    def compute_scaling_law_params(self):
        cum_cnt_TxK = np.cumsum(self.task_cnt_TxK, axis=0)
        task_loss_TxK = self.task_loss_TxK
        if self.fit_max_step is not None:
            cum_cnt_TxK, task_loss_TxK = cum_cnt_TxK[: self.fit_max_step], task_loss_TxK[: self.fit_max_step]
        if self.y_axis == "train_loss":
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.ignore_steps :: self.fit_freq].T)
            # per-step training loss is noisy; smooth before fitting
            task_loss_KxT = savgol_filter(task_loss_TxK.T, self.window_length, self.polyorder, axis=-1)
            task_loss_KxT = jnp.asarray(task_loss_KxT[:, self.ignore_steps :: self.fit_freq])
        else:
            mask = self.val_step_S >= self.ignore_steps
            if self.fit_max_step is not None:
                mask = mask & (self.val_step_S < self.fit_max_step)
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.val_step_S[mask], :].T)
            task_loss_KxT = jnp.asarray(self.val_loss_SxK[mask].T)
        T = max(2, int(np.ceil(self.fit_max_frac * cum_cnt_KxT.shape[-1])))
        cum_cnt_KxT, task_loss_KxT = cum_cnt_KxT[:, :T], task_loss_KxT[:, :T]
        # columns of N must move independently for gamma to be identified; log how bad it is
        cols = np.asarray(cum_cnt_KxT, dtype=np.float64).T
        self.fit_cond = np.linalg.cond(cols / np.maximum(cols.max(axis=0), 1.0))
        cum_cnt_KxT, task_loss_KxT = multihost_utils.broadcast_one_to_all((cum_cnt_KxT, task_loss_KxT))
        if self.fit_data_path and jax.process_index() == 0:
            self._dump_fit_data(np.asarray(cum_cnt_KxT), np.asarray(task_loss_KxT))

        sharding = jax.sharding.NamedSharding(self.mesh, P())
        cum_cnt_KxT = self._make_global_array(cum_cnt_KxT, sharding)
        task_loss_KxT = self._make_global_array(task_loss_KxT, sharding)

        params_Kx3pK, state = self._fit_scaling_law_fn()(cum_cnt_KxT, task_loss_KxT)
        self.fit_val, self.gamma_spread = state[0], state[1]
        params = np.array(multihost_utils.process_allgather(params_Kx3pK, tiled=True))
        K2 = self.k * self.k
        log_gamma = params[:K2].reshape(self.k, self.k)
        if self.gamma_identity:
            log_gamma = np.where(np.eye(self.k, dtype=bool), 0.0, np.log(1e-12))
        elif self.gamma_smooth > 0 and self.current_params is not None:
            prev = np.exp(np.asarray(self.current_params)[:K2].reshape(self.k, self.k))
            new = self.gamma_smooth * prev + (1 - self.gamma_smooth) * np.exp(log_gamma)
            log_gamma = np.log(np.maximum(new, 1e-12))
        params[:K2] = log_gamma.reshape(-1)
        self._narrow_alpha_range(params[K2 : K2 + self.k])
        return params

    def _narrow_alpha_range(self, alpha_K):
        if not np.isfinite(alpha_K).all():
            return
        half = np.maximum(self.alpha_shrink * (self.alpha_hi - self.alpha_lo) / 2, self.alpha_min_halfwidth)
        self.alpha_lo = np.clip(alpha_K - half, 0.01, None)
        self.alpha_hi = np.clip(alpha_K + half, None, 1.0)

    def _dump_fit_data(self, cum_cnt_KxT, task_loss_KxT):
        # path may contain {step}, e.g. gs://bucket/run/fit_data_{step}.npz
        path = self.fit_data_path.format(step=self.task_cnt_TxK.shape[0])
        opener = gcsfs.GCSFileSystem().open if path.startswith("gs://") else open
        with opener(path, "wb") as f:
            np.savez(f, D=cum_cnt_KxT, L=task_loss_KxT)

    def _fit_scaling_law_fn(self):
        return functools.partial(
            fit_scaling_law,
            mesh=self.mesh,
            K=self.k,
            l1=self.gamma_l1,
            alpha_lo=jnp.asarray(self.alpha_lo, dtype=jnp.float32),
            alpha_hi=jnp.asarray(self.alpha_hi, dtype=jnp.float32),
            shared_alpha=self.shared_alpha,
        )

    def _get_S_grads(self, D, current_params):
        return jnp.exp(epiplexity_grads(D, current_params) / self.tau)

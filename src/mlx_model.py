# MLX port of src/model.py, for training small models on the Mac GPU. same architecture: tied embedding,
# interleaved rotary embeddings, QK LayerNorm, SwiGLU and weightless RMSNorm. no dropout and no mup
import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


def _linear(in_features, out_features):
    # bias-free linear with the repo's truncated-normal init
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = mx.random.truncated_normal(-2, 2, (out_features, in_features)) / math.sqrt(in_features)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(2 * (4 * n_embd) / 3)  # same rounding as src/model.py
        hidden_dim = 8 * ((hidden_dim + 7) // 8)
        self.w1 = _linear(n_embd, hidden_dim)
        self.w2 = _linear(hidden_dim, n_embd)
        self.w3 = _linear(n_embd, hidden_dim)

    def __call__(self, x_BxTxD):
        return self.w2(nn.silu(self.w1(x_BxTxD)) * self.w3(x_BxTxD))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.C = n_head, n_embd // n_head
        self.c_attn, self.c_proj = _linear(n_embd, 3 * n_embd), _linear(n_embd, n_embd)
        self.q_ln = nn.LayerNorm(self.C, eps=1e-6, bias=False)
        self.k_ln = nn.LayerNorm(self.C, eps=1e-6, bias=False)
        self.rope = nn.RoPE(self.C, traditional=True)  # pairs (0,1), (2,3), ... like rotate_every_two

    def __call__(self, x_BxTxD):
        B, T, D = x_BxTxD.shape
        q, k, v = mx.split(self.c_attn(x_BxTxD), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.n_head, self.C).transpose(0, 2, 1, 3) for t in (q, k, v))  # [B,H,T,C]
        q, k = self.rope(self.q_ln(q)), self.rope(self.k_ln(k))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(self.C), mask="causal")
        return self.c_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn, self.mlp = CausalSelfAttention(n_embd, n_head), SwiGLU(n_embd)
        self.ln1, self.ln2 = RMSNorm(), RMSNorm()

    def __call__(self, x_BxTxD):
        x_BxTxD = x_BxTxD + self.attn(self.ln1(x_BxTxD))
        return x_BxTxD + self.mlp(self.ln2(x_BxTxD))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dropout == 0.0 and not config.mup, "the mlx port has no dropout or mup"
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wte.weight = mx.random.normal((config.vocab_size, config.n_embd)) / math.sqrt(config.n_embd)
        self.blocks = [Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        self.ln_f = RMSNorm(eps=1e-5)

    def __call__(self, x_BxT):
        x_BxTxD = self.wte(x_BxT)
        for block in self.blocks:
            x_BxTxD = block(x_BxTxD)
        return self.wte.as_linear(self.ln_f(x_BxTxD))  # shared embed and final weights


def count_params(model):
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def loss_fn(model, x_BxT, y_BxT):
    # mean loss and the per-sequence loss the data selectors take
    logits_BxTxV = model(x_BxT).astype(mx.float32)
    loss_B = nn.losses.cross_entropy(logits_BxTxV, y_BxT).mean(axis=-1)
    return loss_B.mean(), loss_B


def make_optimizer(config):
    # same update as src/train.py's optax chain: clip, adam, decoupled weight decay scaled with the lr schedule.
    # optax adds weight_decay / learning_rate * p before scaling by the lr; mlx multiplies its decay by the lr
    return optim.AdamW(
        learning_rate=config.learning_rate,
        betas=[0.9, config.beta2],
        eps=1e-8,
        weight_decay=config.weight_decay / config.learning_rate,
        bias_correction=True,
    )


def make_train_step(model, optimizer, max_grad_norm=1.0):
    # compiled step. set optimizer.learning_rate before each call; it lives in the optimizer state
    state = [model.state, optimizer.state]
    grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x_BxT, y_BxT):
        (loss, loss_B), grads = grad_fn(model, x_BxT, y_BxT)
        grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
        optimizer.update(model, grads)
        return loss, loss_B

    return step, state


def make_eval_fn(model):
    @partial(mx.compile, inputs=model.state)
    def eval_loss(x_BxT, y_BxT):
        return loss_fn(model, x_BxT, y_BxT)[1]

    return eval_loss

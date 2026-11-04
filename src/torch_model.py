# PyTorch port of src/model.py for NVIDIA GPUs. same architecture and numerics where it matters:
# weightless RMSNorm, QK LayerNorm (weight, no bias), interleaved rotary embeddings, SwiGLU, bias-free linears.
# the embedding and output head start from the same matrix but are separate parameters, as in the equinox
# model, whose wte and lm_head are two pytree leaves the optimizer updates independently.
# init_from_jax copies the equinox model's initial weights, so a seed gives the same init as the paper's runs
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # triton (torch.compile) and tensorflow (tf.data) both carry llvm: loading triton after tensorflow aborts
    import triton  # noqa: F401
except ImportError:
    pass


def swiglu_hidden(n_embd):
    hidden_dim = int(2 * (4 * n_embd) / 3)  # same rounding as src/model.py
    return 8 * ((hidden_dim + 7) // 8)


def rms_norm(x, eps):
    # src/layers.RMSNorm without a weight, computed in fp32
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rope_tables(C, T, device):
    # src/layers.fixed_pos_embedding, with each frequency repeated for its (even, odd) pair
    inv_freq = 1.0 / (10000 ** (np.arange(0, C, 2) / C))
    ang_TxCp = np.einsum("i,j->ij", np.arange(T), inv_freq)
    sin = torch.tensor(np.repeat(np.sin(ang_TxCp), 2, axis=-1), dtype=torch.float32, device=device)
    cos = torch.tensor(np.repeat(np.cos(ang_TxCp), 2, axis=-1), dtype=torch.float32, device=device)
    return sin, cos


def apply_rope(x_BxHxTxC, sin_TxC, cos_TxC):
    # src/layers.apply_rotary_pos_emb: rotate_every_two maps [a b c d] -> [-b a -d c]
    x1, x2 = x_BxHxTxC[..., ::2], x_BxHxTxC[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x_BxHxTxC * cos_TxC.to(x_BxHxTxC.dtype) + rot * sin_TxC.to(x_BxHxTxC.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.C = n_head, n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q_ln = nn.LayerNorm(self.C, eps=1e-6, bias=False)
        self.k_ln = nn.LayerNorm(self.C, eps=1e-6, bias=False)

    def forward(self, x_BxTxD, sin, cos):
        B, T, D = x_BxTxD.shape
        q, k, v = self.c_attn(x_BxTxD).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, self.C).transpose(1, 2) for t in (q, k, v))  # [B,H,T,C]
        q = apply_rope(self.q_ln(q).to(v.dtype), sin[:T], cos[:T])
        k = apply_rope(self.k_ln(k).to(v.dtype), sin[:T], cos[:T])
        # flash / cudnn attention; the softmax scale is 1/sqrt(C), as with QK LayerNorm in src/model.py
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1 / math.sqrt(self.C))
        return self.c_proj(out.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        h = swiglu_hidden(n_embd)
        self.w1 = nn.Linear(n_embd, h, bias=False)
        self.w2 = nn.Linear(h, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = SwiGLU(n_embd)

    def forward(self, x, sin, cos):
        x = x + self.attn(rms_norm(x, 1e-6), sin, cos)
        return x + self.mlp(rms_norm(x, 1e-6))


class GPT(nn.Module):
    def __init__(self, config, max_len=4096):
        super().__init__()
        assert config.dropout == 0.0 and not config.mup, "the torch port has no dropout or mup"
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        sin, cos = rope_tables(config.n_embd // config.n_head, max_len, None)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x_BxT):
        x = self.wte(x_BxT)
        for block in self.blocks:
            x = block(x, self.rope_sin, self.rope_cos)
        return self.lm_head(rms_norm(x, 1e-5))


def count_params(model):
    # non-embedding count, as src/model.count_params (one of the two vocab matrices is left out)
    return sum(p.numel() for p in model.parameters()) - model.lm_head.weight.numel()


def seq_losses(model, x_BxT, y_BxT):
    # per-sequence mean cross entropy (what the data selectors take), with fp32 logits as in src/train.py
    logits = model(x_BxT).float()
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y_BxT.reshape(B * T), reduction="none").view(B, T).mean(-1)


def jax_state_dict(eqx_model):
    # map an equinox src/model.GPT onto this module's parameter names
    import jax

    def a(x):
        return torch.from_numpy(np.array(jax.device_get(x), dtype=np.float32))

    b = eqx_model.blocks
    sd = {"wte.weight": a(eqx_model.wte.weight_VxD), "lm_head.weight": a(eqx_model.lm_head.weight_MxN)}
    for i in range(eqx_model.n_layer):
        p = f"blocks.{i}."
        sd[p + "attn.c_attn.weight"] = a(b.attn.c_attn.weight_MxN[i])
        sd[p + "attn.c_proj.weight"] = a(b.attn.c_proj.weight_MxN[i])
        sd[p + "attn.q_ln.weight"] = a(b.attn.q_ln.weight[i])
        sd[p + "attn.k_ln.weight"] = a(b.attn.k_ln.weight[i])
        sd[p + "mlp.w1.weight"] = a(b.mlp.w1.weight_MxN[i])
        sd[p + "mlp.w2.weight"] = a(b.mlp.w2.weight_MxN[i])
        sd[p + "mlp.w3.weight"] = a(b.mlp.w3.weight_MxN[i])
    return sd


def init_from_jax(model_config, seed):
    # the paper's init for this seed: src/train.py splits PRNGKey(seed) once and builds the model from the second
    # key. runs on the host cpu, so it needs no accelerator
    import jax

    from src.model import GPT as EqxGPT

    jax.config.update("jax_threefry_partitionable", True)  # as in src/train.py
    with jax.default_device(jax.devices("cpu")[0]):
        _, key1 = jax.random.split(jax.random.PRNGKey(seed))
        return jax_state_dict(EqxGPT(model_config, key1))

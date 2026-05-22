"""Fixed-point forward pass for the kolmo transformer.

This is Stage B of the bulletproof Rung 2 path: the same model architecture
as `KolmoTransformer`, but all forward math is Q15 integer arithmetic. It is
not wired into compression yet; first we validate the forward pass against
PyTorch and verify hashes across machines.
"""

from __future__ import annotations

import math

import numpy as np

from kolmo import fixed


def extract_fixed_weights(model) -> dict[str, np.ndarray]:
    """Quantize a PyTorch model's parameters into Q15 numpy arrays."""
    return {
        name: fixed.quantize(param.detach().cpu().numpy().astype(np.float64))
        for name, param in model.named_parameters()
    }


def _block_weights(weights: dict[str, np.ndarray], layer: int) -> dict[str, np.ndarray]:
    prefix = f"blocks.{layer}."
    return {
        name[len(prefix):]: value
        for name, value in weights.items()
        if name.startswith(prefix)
    }


def _causal_mask_q15(t: int) -> np.ndarray:
    """Upper-triangular mask for positions a token cannot attend to."""
    return np.triu(np.ones((t, t), dtype=bool), k=1)


def _attention_q15(
    x_q: np.ndarray,
    qkv_w_q: np.ndarray,
    proj_w_q: np.ndarray,
    n_heads: int,
) -> np.ndarray:
    """Causal self-attention for a full sequence.

    x_q is (T, D). qkv_w_q is (3D, D). proj_w_q is (D, D). Return is (T, D).
    """
    T, D = x_q.shape
    d_head = D // n_heads

    qkv = fixed.linear_q15(x_q, qkv_w_q)
    qkv = qkv.reshape(T, 3, n_heads, d_head)
    q = qkv[:, 0].transpose(1, 0, 2)  # (H, T, d_head)
    k = qkv[:, 1].transpose(1, 0, 2)
    v = qkv[:, 2].transpose(1, 0, 2)

    scale_q = np.int32(round((1.0 / math.sqrt(d_head)) * fixed.SCALE))
    heads = []
    mask = _causal_mask_q15(T)
    very_negative = np.int32(-30 * fixed.SCALE)

    for h in range(n_heads):
        # Q15 scores: q @ k.T then multiply by 1/sqrt(d_head).
        scores = fixed.matmul(q[h], k[h].T)
        scores = fixed.mul(scores, np.full_like(scores, scale_q))
        scores = np.where(mask, very_negative, scores).astype(np.int32)
        attn = fixed.softmax_q15(scores)
        heads.append(fixed.matmul(attn, v[h]))

    out = np.stack(heads, axis=1).reshape(T, D)
    return fixed.linear_q15(out, proj_w_q)


def _block_q15(
    x_q: np.ndarray,
    weights: dict[str, np.ndarray],
    n_heads: int,
) -> np.ndarray:
    """One pre-norm transformer block."""
    h = fixed.layernorm_q15(x_q, weights["ln1.weight"], weights["ln1.bias"])
    h = _attention_q15(
        h,
        weights["attn.qkv.weight"],
        weights["attn.proj.weight"],
        n_heads,
    )
    x_q = fixed.add(x_q, h)

    h = fixed.layernorm_q15(x_q, weights["ln2.weight"], weights["ln2.bias"])
    h = fixed.linear_q15(h, weights["ffn.0.weight"], weights["ffn.0.bias"])
    h = fixed.gelu_q15(h)
    h = fixed.linear_q15(h, weights["ffn.2.weight"], weights["ffn.2.bias"])
    return fixed.add(x_q, h)


def fixed_forward(
    input_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    n_heads: int = 8,
    n_layers: int = 4,
    pos_offset: int = 0,
) -> np.ndarray:
    """Fixed-point full-sequence forward.

    input_ids is a 1-D array of byte/token ids. Returns logits as Q15 int32
    with shape (T, vocab_size).
    """
    if input_ids.ndim != 1:
        raise ValueError("fixed_forward expects a 1-D token array")
    T = len(input_ids)
    positions = np.arange(pos_offset, pos_offset + T, dtype=np.int64)

    h = fixed.add(
        weights["token_emb.weight"][input_ids],
        weights["pos_emb.weight"][positions],
    )

    for layer in range(n_layers):
        h = _block_q15(h, _block_weights(weights, layer), n_heads)

    h = fixed.layernorm_q15(h, weights["ln_f.weight"], weights["ln_f.bias"])
    return fixed.linear_q15(h, weights["head.weight"])

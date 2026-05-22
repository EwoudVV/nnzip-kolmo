"""Fixed-point KV cache for incremental inference.

`fixed_forward` re-projects every token through every layer every time it
runs. Compression encodes one byte at a time between training steps, so the
total work per block scales as O(T^2 * BLOCK_SIZE) — quadratic in context.
This module mirrors the PyTorch path's KV cache: full warm-up forward saves
each layer's K/V; subsequent step forwards push a single new token through
the network against those caches.

Determinism note: this is bit-identical to running `fixed_forward` over the
extended history. In a full forward, row T-1 of `scores = q @ k.T` is just
`q[T-1] @ k.T` — exactly the dot product the step path computes against the
appended K cache. Q15 integer matmul is associative across the contraction
axis (integer addition is associative), so the value is unaffected by whether
the row is computed alone or as part of a larger matmul. Same logic holds for
softmax (input row is identical) and attn @ v.
"""

from __future__ import annotations

import math

import numpy as np

from kolmo import fixed
from kolmo.fixed_model import _block_weights


def _attention_warm_q15(
    x_q: np.ndarray,
    qkv_w_q: np.ndarray,
    proj_w_q: np.ndarray,
    n_heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-sequence causal attention that also exposes K/V for caching.

    Mathematically identical to `kolmo.fixed_model._attention_q15`; the only
    difference is the second/third return values (per-head K and V, each of
    shape `(n_heads, T, d_head)`).
    """
    T, D = x_q.shape
    d_head = D // n_heads

    qkv = fixed.linear_q15(x_q, qkv_w_q)
    qkv = qkv.reshape(T, 3, n_heads, d_head)
    q = qkv[:, 0].transpose(1, 0, 2)  # (H, T, d_head)
    k = qkv[:, 1].transpose(1, 0, 2)
    v = qkv[:, 2].transpose(1, 0, 2)

    scale_q = np.int32(round((1.0 / math.sqrt(d_head)) * fixed.SCALE))
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    very_negative = np.int32(-30 * fixed.SCALE)

    heads = []
    for h in range(n_heads):
        scores = fixed.matmul(q[h], k[h].T)
        scores = fixed.mul(scores, np.full_like(scores, scale_q))
        scores = np.where(mask, very_negative, scores).astype(np.int32)
        attn = fixed.softmax_q15(scores)
        heads.append(fixed.matmul(attn, v[h]))

    out = np.stack(heads, axis=1).reshape(T, D)
    return fixed.linear_q15(out, proj_w_q), k, v


def _attention_step_q15(
    x_new_q: np.ndarray,
    qkv_w_q: np.ndarray,
    proj_w_q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    n_heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-token causal attention against cached K/V.

    `x_new_q` is shape `(1, D)`; caches are `(H, T_prev, d_head)`. Appends the
    new token's K/V, then runs a one-row attention against the extended cache
    (no causal mask: the new token can attend to all of history plus itself).

    Returns `(output, k_full, v_full)` where output is `(1, D)`.
    """
    _, D = x_new_q.shape
    d_head = D // n_heads

    qkv = fixed.linear_q15(x_new_q, qkv_w_q)
    qkv = qkv.reshape(1, 3, n_heads, d_head)
    q = qkv[:, 0].transpose(1, 0, 2)  # (H, 1, d_head)
    k_new = qkv[:, 1].transpose(1, 0, 2)
    v_new = qkv[:, 2].transpose(1, 0, 2)

    if k_cache.size == 0:
        k_full = k_new
        v_full = v_new
    else:
        k_full = np.concatenate([k_cache, k_new], axis=1)
        v_full = np.concatenate([v_cache, v_new], axis=1)

    scale_q = np.int32(round((1.0 / math.sqrt(d_head)) * fixed.SCALE))
    heads = []
    for h in range(n_heads):
        scores = fixed.matmul(q[h], k_full[h].T)  # (1, T_total)
        scores = fixed.mul(scores, np.full_like(scores, scale_q))
        attn = fixed.softmax_q15(scores)
        heads.append(fixed.matmul(attn, v_full[h]))

    out = np.stack(heads, axis=1).reshape(1, D)
    return fixed.linear_q15(out, proj_w_q), k_full, v_full


def _block_warm_q15(
    x_q: np.ndarray,
    weights: dict[str, np.ndarray],
    n_heads: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One pre-norm transformer block, full sequence, returning the layer's KV."""
    h = fixed.layernorm_q15(x_q, weights["ln1.weight"], weights["ln1.bias"])
    attn_out, k, v = _attention_warm_q15(
        h,
        weights["attn.qkv.weight"],
        weights["attn.proj.weight"],
        n_heads,
    )
    x_q = fixed.add(x_q, attn_out)

    h = fixed.layernorm_q15(x_q, weights["ln2.weight"], weights["ln2.bias"])
    h = fixed.linear_q15(h, weights["ffn.0.weight"], weights["ffn.0.bias"])
    h = fixed.gelu_q15(h)
    h = fixed.linear_q15(h, weights["ffn.2.weight"], weights["ffn.2.bias"])
    return fixed.add(x_q, h), {"k": k, "v": v}


def _block_step_q15(
    x_new_q: np.ndarray,
    weights: dict[str, np.ndarray],
    layer_cache: dict[str, np.ndarray],
    n_heads: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One pre-norm transformer block, single new token, using cached KV."""
    h = fixed.layernorm_q15(x_new_q, weights["ln1.weight"], weights["ln1.bias"])
    attn_out, k_full, v_full = _attention_step_q15(
        h,
        weights["attn.qkv.weight"],
        weights["attn.proj.weight"],
        layer_cache["k"],
        layer_cache["v"],
        n_heads,
    )
    x_new_q = fixed.add(x_new_q, attn_out)

    h = fixed.layernorm_q15(x_new_q, weights["ln2.weight"], weights["ln2.bias"])
    h = fixed.linear_q15(h, weights["ffn.0.weight"], weights["ffn.0.bias"])
    h = fixed.gelu_q15(h)
    h = fixed.linear_q15(h, weights["ffn.2.weight"], weights["ffn.2.bias"])
    return fixed.add(x_new_q, h), {"k": k_full, "v": v_full}


def fixed_warm(
    input_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    n_heads: int = 8,
    n_layers: int = 4,
    pos_offset: int = 0,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Full-sequence forward that populates a fresh KV cache.

    Returns `(last_logits_q, caches)` where `last_logits_q` has shape
    `(vocab_size,)` — equal to `fixed_forward(input_ids)[-1]` bit-for-bit —
    and `caches` is a list of one `{"k", "v"}` dict per transformer block.
    """
    if input_ids.ndim != 1:
        raise ValueError("fixed_warm expects a 1-D token array")
    T = len(input_ids)
    positions = np.arange(pos_offset, pos_offset + T, dtype=np.int64)

    h = fixed.add(
        weights["token_emb.weight"][input_ids],
        weights["pos_emb.weight"][positions],
    )

    caches: list[dict[str, np.ndarray]] = []
    for layer in range(n_layers):
        h, layer_cache = _block_warm_q15(
            h, _block_weights(weights, layer), n_heads
        )
        caches.append(layer_cache)

    # Only the last row is needed for prediction; LN + head are per-row ops,
    # so restricting to the final position is bit-identical to processing all
    # T rows and indexing the last one.
    h_last = h[-1:]
    h_last = fixed.layernorm_q15(
        h_last, weights["ln_f.weight"], weights["ln_f.bias"]
    )
    logits = fixed.linear_q15(h_last, weights["head.weight"])
    return logits[0], caches


def fixed_step(
    token_id: int,
    caches: list[dict[str, np.ndarray]],
    weights: dict[str, np.ndarray],
    n_heads: int = 8,
    n_layers: int = 4,
    pos_offset: int = 0,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Push one new token through every layer using the existing cache.

    `pos_offset` is the absolute position of this new token (so position
    embeddings stay monotonic across step calls, even if the cache has been
    trimmed). Returns `(last_logits_q, new_caches)`.
    """
    token_arr = np.array([token_id], dtype=np.int64)
    pos_arr = np.array([pos_offset], dtype=np.int64)
    h = fixed.add(
        weights["token_emb.weight"][token_arr],
        weights["pos_emb.weight"][pos_arr],
    )

    new_caches: list[dict[str, np.ndarray]] = []
    for layer in range(n_layers):
        h, layer_cache = _block_step_q15(
            h, _block_weights(weights, layer), caches[layer], n_heads
        )
        new_caches.append(layer_cache)

    h = fixed.layernorm_q15(h, weights["ln_f.weight"], weights["ln_f.bias"])
    logits = fixed.linear_q15(h, weights["head.weight"])
    return logits[0], new_caches


def trim_caches(
    caches: list[dict[str, np.ndarray]],
    max_len: int,
) -> list[dict[str, np.ndarray]]:
    """Drop early K/V rows so each layer's cache stays within `max_len`.

    Position embeddings are baked into K/V at the input layer, so trimming the
    front of the cache doesn't change what survives — and new tokens added by
    `fixed_step` keep getting their own (monotonically increasing) position
    embedding because we thread `pos_offset` separately.
    """
    out: list[dict[str, np.ndarray]] = []
    for c in caches:
        if c["k"].shape[1] > max_len:
            out.append(
                {
                    "k": c["k"][:, -max_len:, :].copy(),
                    "v": c["v"][:, -max_len:, :].copy(),
                }
            )
        else:
            out.append(c)
    return out

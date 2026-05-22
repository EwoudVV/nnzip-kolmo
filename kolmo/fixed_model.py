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
    """Quantize a PyTorch model's parameters into Q15 numpy arrays.

    Weight-tied parameters are handled here: `nn.Module.named_parameters()`
    deduplicates by Parameter id, so a tied pair (e.g. token_emb.weight ===
    head.weight) only shows up under one name. We then walk every
    `(name, attribute)` of the model and re-add any aliased keys that
    `named_parameters()` skipped, pointing them at the same array.
    """
    seen_ids: dict[int, str] = {}
    weights: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        seen_ids[id(param)] = name
        weights[name] = fixed.quantize(
            param.detach().cpu().numpy().astype(np.float64)
        )

    # Look for additional names that point at parameters we already extracted.
    # `named_parameters(remove_duplicate=False)` yields every tensor reference,
    # not just the canonical one, so duplicates show up explicitly.
    for name, param in model.named_parameters(remove_duplicate=False):
        canonical = seen_ids.get(id(param))
        if canonical is not None and name not in weights:
            weights[name] = weights[canonical]
    return weights


def tied_param_pairs(model) -> list[tuple[str, str]]:
    """Return `[(canonical, alias)]` pairs for every parameter referenced under
    more than one name.

    Canonical = the name `named_parameters()` (with dedup) yields.
    Alias = any other name that points at the same `Parameter`.

    Used by the fixed-point training path to sum gradients across tied names
    before the optimizer step.
    """
    canonical_by_id: dict[int, str] = {}
    for name, param in model.named_parameters():
        canonical_by_id[id(param)] = name

    pairs: list[tuple[str, str]] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        canonical = canonical_by_id[id(param)]
        if name != canonical:
            pairs.append((canonical, name))
    return pairs


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


def _attention_forward_parts_q15(
    x_q: np.ndarray,
    qkv_w_q: np.ndarray,
    n_heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return packed attention intermediates needed for backward.

    Returns `(qkv_flat, q, k, v, attn)` where q/k/v/attn are head-major:
    q/k/v have shape (H, T, d_head), attn has shape (H, T, T).
    """
    T, D = x_q.shape
    d_head = D // n_heads

    qkv_flat = fixed.linear_q15(x_q, qkv_w_q)
    qkv = qkv_flat.reshape(T, 3, n_heads, d_head)
    q = qkv[:, 0].transpose(1, 0, 2)
    k = qkv[:, 1].transpose(1, 0, 2)
    v = qkv[:, 2].transpose(1, 0, 2)

    scale_q = np.int32(round((1.0 / math.sqrt(d_head)) * fixed.SCALE))
    mask = _causal_mask_q15(T)
    very_negative = np.int32(-30 * fixed.SCALE)
    attn = []
    for h in range(n_heads):
        scores = fixed.matmul(q[h], k[h].T)
        scores = fixed.mul(scores, np.full_like(scores, scale_q))
        scores = np.where(mask, very_negative, scores).astype(np.int32)
        attn.append(fixed.softmax_q15(scores))
    return qkv_flat, q, k, v, np.stack(attn, axis=0)


def _attention_backward_q15(
    x_q: np.ndarray,
    qkv_w_q: np.ndarray,
    proj_w_q: np.ndarray,
    grad_y_q: np.ndarray,
    n_heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for full-sequence causal self-attention.

    Forward graph:
        q, k, v = linear(x, qkv_w).split()
        attn = softmax(mask((q @ k.T) / sqrt(d_head)))
        heads = attn @ v
        y = linear(concat(heads), proj_w)

    Returns `(grad_x, grad_qkv_w, grad_proj_w)`, all Q15.
    """
    if x_q.dtype != np.int32 or qkv_w_q.dtype != np.int32 or proj_w_q.dtype != np.int32:
        raise TypeError("_attention_backward_q15 expects int32 inputs")
    if x_q.ndim != 2 or grad_y_q.ndim != 2:
        raise ValueError("_attention_backward_q15 expects 2-D x and grad_y")

    T, D = x_q.shape
    d_head = D // n_heads
    scale_q = np.int32(round((1.0 / math.sqrt(d_head)) * fixed.SCALE))
    mask = _causal_mask_q15(T)

    _qkv_flat, q, k, v, attn = _attention_forward_parts_q15(x_q, qkv_w_q, n_heads)
    heads = np.stack([fixed.matmul(attn[h], v[h]) for h in range(n_heads)], axis=1)
    out = heads.reshape(T, D)

    grad_out, grad_proj_w, _ = fixed.linear_backward_q15(
        out,
        proj_w_q,
        grad_y_q,
        has_bias=False,
    )
    grad_heads = grad_out.reshape(T, n_heads, d_head).transpose(1, 0, 2)

    grad_q = np.zeros_like(q)
    grad_k = np.zeros_like(k)
    grad_v = np.zeros_like(v)

    for h in range(n_heads):
        # head_out = attn @ v
        grad_attn = fixed.matmul(grad_heads[h], v[h].T)
        grad_v[h] = fixed.matmul(attn[h].T, grad_heads[h])

        # attn = softmax(masked_scores). Masked positions are constants, so
        # their gradients do not flow back to q/k.
        grad_scores_scaled = fixed.softmax_backward_q15(attn[h], grad_attn)
        grad_scores_scaled = np.where(mask, 0, grad_scores_scaled).astype(np.int32)

        # scores_scaled = scores * (1/sqrt(d_head)).
        grad_scores = fixed.mul(grad_scores_scaled, np.full_like(grad_scores_scaled, scale_q))

        # scores = q @ k.T
        grad_q[h] = fixed.matmul(grad_scores, k[h])
        grad_k[h] = fixed.matmul(grad_scores.T, q[h])

    grad_qkv = np.zeros((T, 3, n_heads, d_head), dtype=np.int32)
    grad_qkv[:, 0] = grad_q.transpose(1, 0, 2)
    grad_qkv[:, 1] = grad_k.transpose(1, 0, 2)
    grad_qkv[:, 2] = grad_v.transpose(1, 0, 2)
    grad_qkv_flat = grad_qkv.reshape(T, 3 * D)

    grad_x, grad_qkv_w, _ = fixed.linear_backward_q15(
        x_q,
        qkv_w_q,
        grad_qkv_flat,
        has_bias=False,
    )
    return grad_x, grad_qkv_w, grad_proj_w


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


def _block_backward_q15(
    x_q: np.ndarray,
    weights: dict[str, np.ndarray],
    grad_y_q: np.ndarray,
    n_heads: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Backward pass for one pre-norm transformer block.

    Forward:
        h1 = ln1(x)
        x_res = x + attention(h1)
        h2 = ln2(x_res)
        y = x_res + ffn(h2)

    Returns `(grad_x, grad_weights)`. Weight keys match `weights`.
    """
    h1 = fixed.layernorm_q15(x_q, weights["ln1.weight"], weights["ln1.bias"])
    attn_out = _attention_q15(
        h1,
        weights["attn.qkv.weight"],
        weights["attn.proj.weight"],
        n_heads,
    )
    x_res = fixed.add(x_q, attn_out)
    h2 = fixed.layernorm_q15(x_res, weights["ln2.weight"], weights["ln2.bias"])

    # y = x_res + ffn(h2)
    grad_h2, grad_ffn0_w, grad_ffn0_b, grad_ffn2_w, grad_ffn2_b = fixed.ffn_backward_q15(
        h2,
        weights["ffn.0.weight"],
        weights["ffn.0.bias"],
        weights["ffn.2.weight"],
        weights["ffn.2.bias"],
        grad_y_q,
    )
    grad_x_res_from_ln2, grad_ln2_w, grad_ln2_b = fixed.layernorm_backward_q15(
        x_res,
        weights["ln2.weight"],
        grad_h2,
    )
    grad_x_res = fixed.add(grad_y_q, grad_x_res_from_ln2)

    # x_res = x + attention(ln1(x))
    grad_h1, grad_qkv_w, grad_proj_w = _attention_backward_q15(
        h1,
        weights["attn.qkv.weight"],
        weights["attn.proj.weight"],
        grad_x_res,
        n_heads,
    )
    grad_x_from_ln1, grad_ln1_w, grad_ln1_b = fixed.layernorm_backward_q15(
        x_q,
        weights["ln1.weight"],
        grad_h1,
    )
    grad_x = fixed.add(grad_x_res, grad_x_from_ln1)

    grads = {
        "ln1.weight": grad_ln1_w,
        "ln1.bias": grad_ln1_b,
        "attn.qkv.weight": grad_qkv_w,
        "attn.proj.weight": grad_proj_w,
        "ln2.weight": grad_ln2_w,
        "ln2.bias": grad_ln2_b,
        "ffn.0.weight": grad_ffn0_w,
        "ffn.0.bias": grad_ffn0_b,
        "ffn.2.weight": grad_ffn2_w,
        "ffn.2.bias": grad_ffn2_b,
    }
    return grad_x, grads


def _scatter_add_rows_q15(
    shape: tuple[int, ...],
    indices: np.ndarray,
    grad_rows_q: np.ndarray,
) -> np.ndarray:
    """Scatter-add row gradients into an embedding matrix shape."""
    out = np.zeros(shape, dtype=np.int64)
    for idx, grad in zip(indices.tolist(), grad_rows_q, strict=True):
        out[idx] += grad.astype(np.int64)
    return np.clip(out, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)


def fixed_backward(
    input_ids: np.ndarray,
    target_positions: np.ndarray,
    targets: np.ndarray,
    weights: dict[str, np.ndarray],
    n_heads: int = 8,
    n_layers: int = 4,
    pos_offset: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Full fixed-point forward + backward for a cross-entropy training slice.

    `input_ids` is the full context sequence. `target_positions` selects which
    logit rows participate in the loss, and `targets` gives the byte target
    for each selected row. This mirrors PyTorch's `cross_entropy` over a
    subset of sequence positions.

    Returns `(logits, grads)` where logits are Q15 and grads has one entry per
    model parameter.
    """
    if input_ids.ndim != 1:
        raise ValueError("fixed_backward expects a 1-D input_ids array")
    target_positions = np.asarray(target_positions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if target_positions.shape != targets.shape:
        raise ValueError("target_positions and targets must have the same shape")

    T = len(input_ids)
    positions = np.arange(pos_offset, pos_offset + T, dtype=np.int64)

    h = fixed.add(
        weights["token_emb.weight"][input_ids],
        weights["pos_emb.weight"][positions],
    )
    block_inputs = []
    for layer in range(n_layers):
        block_inputs.append(h)
        h = _block_q15(h, _block_weights(weights, layer), n_heads)

    h_before_ln_f = h
    h_norm = fixed.layernorm_q15(h, weights["ln_f.weight"], weights["ln_f.bias"])
    logits = fixed.linear_q15(h_norm, weights["head.weight"])

    grad_logits = np.zeros_like(logits)
    grad_logits[target_positions] = fixed.cross_entropy_grad_q15(
        logits[target_positions],
        targets,
    )

    grads: dict[str, np.ndarray] = {}
    grad_h_norm, grads["head.weight"], _ = fixed.linear_backward_q15(
        h_norm,
        weights["head.weight"],
        grad_logits,
        has_bias=False,
    )
    grad_h, grads["ln_f.weight"], grads["ln_f.bias"] = fixed.layernorm_backward_q15(
        h_before_ln_f,
        weights["ln_f.weight"],
        grad_h_norm,
    )

    for layer in reversed(range(n_layers)):
        grad_h, block_grads = _block_backward_q15(
            block_inputs[layer],
            _block_weights(weights, layer),
            grad_h,
            n_heads,
        )
        prefix = f"blocks.{layer}."
        for name, grad in block_grads.items():
            grads[prefix + name] = grad

    grads["token_emb.weight"] = _scatter_add_rows_q15(
        weights["token_emb.weight"].shape,
        input_ids,
        grad_h,
    )
    grads["pos_emb.weight"] = _scatter_add_rows_q15(
        weights["pos_emb.weight"].shape,
        positions,
        grad_h,
    )
    return logits, grads


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

"""Tests for the fixed-point transformer forward pass."""

import numpy as np
import torch

from kolmo.fixed import dequantize, quantize
from kolmo.fixed_model import (
    _attention_backward_q15,
    _block_backward_q15,
    extract_fixed_weights,
    fixed_backward,
    fixed_forward,
    tied_param_pairs,
)
from kolmo.fixed_train import fixed_train_block
from kolmo.model import KolmoTransformer, TransformerBlock
from kolmo.stable_init import stable_init_model


def test_fixed_forward_has_expected_shape_and_is_deterministic():
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
        tie_weights=False,  # tests per-parameter parity with PyTorch; tying
        # would route head + embedding gradients into the same Parameter and
        # break the per-name comparison
    )
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)
    x = np.array([1, 2, 3, 100, 200, 42], dtype=np.int64)

    out1 = fixed_forward(x, weights, n_heads=4, n_layers=2)
    out2 = fixed_forward(x, weights, n_heads=4, n_layers=2)

    assert out1.shape == (6, 256)
    assert out1.dtype == np.int32
    assert np.array_equal(out1, out2)


def test_fixed_forward_tracks_torch_argmax_on_tiny_transformer():
    """The fixed forward should preserve the main prediction structure."""
    torch.manual_seed(7)
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
        tie_weights=False,  # tests per-parameter parity with PyTorch; tying
        # would route head + embedding gradients into the same Parameter and
        # break the per-name comparison
    )
    stable_init_model(model, seed=42)
    model.eval()

    x_t = torch.tensor([[1, 2, 3, 100, 200, 42]], dtype=torch.long)
    with torch.no_grad():
        torch_logits, _ = model(x_t)

    fixed_logits = dequantize(
        fixed_forward(
            x_t.numpy()[0],
            extract_fixed_weights(model),
            n_heads=4,
            n_layers=2,
        )
    )

    ref = torch_logits[0].numpy()
    torch_argmax = ref.argmax(axis=-1)
    fixed_argmax = fixed_logits.argmax(axis=-1)
    assert np.array_equal(torch_argmax, fixed_argmax)
    # Absolute bound: Q15 resolution is ~3e-5 per quantization step, and
    # error accumulates linearly through a small transformer. Logit
    # magnitudes are now ~1 (Linear-scale embeddings) so an absolute
    # tolerance of 0.005 is still ~0.5% relative; argmax + correlation
    # checks are the stronger structural claims.
    assert np.max(np.abs(ref - fixed_logits)) < 0.005
    assert np.corrcoef(ref.flatten(), fixed_logits.flatten())[0, 1] > 0.99999


def test_fixed_rope_forward_has_no_pos_emb_and_is_deterministic():
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
        tie_weights=False,
        use_rope=True,
    )
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)
    x = np.array([1, 2, 3, 100, 200, 42], dtype=np.int64)

    assert "pos_emb.weight" not in weights
    assert "rope.cos" in weights
    assert "rope.sin" in weights

    out1 = fixed_forward(x, weights, n_heads=4, n_layers=2, use_rope=True)
    out2 = fixed_forward(x, weights, n_heads=4, n_layers=2, use_rope=True)

    assert out1.shape == (6, 256)
    assert out1.dtype == np.int32
    assert np.array_equal(out1, out2)


def test_attention_backward_q15_matches_torch():
    """Fixed attention backward should track PyTorch autograd."""
    import math
    import torch.nn.functional as F

    rng = np.random.default_rng(18)
    T = 5
    D = 16
    n_heads = 4
    d_head = D // n_heads

    x = rng.normal(size=(T, D)).astype(np.float64) * 0.25
    qkv_w = rng.normal(size=(3 * D, D)).astype(np.float64) * 0.12
    proj_w = rng.normal(size=(D, D)).astype(np.float64) * 0.12
    grad_y = rng.normal(size=(T, D)).astype(np.float64) * 0.015

    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    qkv_w_t = torch.tensor(qkv_w, dtype=torch.float64, requires_grad=True)
    proj_w_t = torch.tensor(proj_w, dtype=torch.float64, requires_grad=True)

    qkv = (x_t @ qkv_w_t.T).view(T, 3, n_heads, d_head)
    q = qkv[:, 0].permute(1, 0, 2)
    k = qkv[:, 1].permute(1, 0, 2)
    v = qkv[:, 2].permute(1, 0, 2)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d_head))
    mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    attn = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    heads = attn @ v
    out = heads.permute(1, 0, 2).contiguous().view(T, D)
    y = out @ proj_w_t.T
    y.backward(torch.tensor(grad_y, dtype=torch.float64))

    grad_x, grad_qkv_w, grad_proj_w = _attention_backward_q15(
        quantize(x),
        quantize(qkv_w),
        quantize(proj_w),
        quantize(grad_y),
        n_heads=n_heads,
    )

    assert np.max(np.abs(dequantize(grad_x) - x_t.grad.numpy())) < 0.004
    assert np.max(np.abs(dequantize(grad_qkv_w) - qkv_w_t.grad.numpy())) < 0.004
    assert np.max(np.abs(dequantize(grad_proj_w) - proj_w_t.grad.numpy())) < 0.004


def test_fixed_rope_backward_produces_expected_gradient_keys():
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
        tie_weights=False,
        use_rope=True,
    )
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)
    input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    target_positions = np.array([1, 2, 3], dtype=np.int64)
    targets = np.array([2, 3, 4], dtype=np.int64)

    logits, grads = fixed_backward(
        input_ids,
        target_positions,
        targets,
        weights,
        n_heads=4,
        n_layers=2,
        use_rope=True,
    )

    assert logits.shape == (5, 256)
    assert "pos_emb.weight" not in grads
    assert "rope.cos" not in grads
    assert "rope.sin" not in grads
    assert "token_emb.weight" in grads
    assert "blocks.0.attn.qkv.weight" in grads


def test_block_backward_q15_matches_torch():
    """A full transformer block backward should track PyTorch autograd."""
    rng = np.random.default_rng(20)
    T = 5
    D = 16
    n_heads = 4

    block = TransformerBlock(D, n_heads).double()
    arrays = {
        "ln1.weight": rng.normal(size=D).astype(np.float64) * 0.1 + 1.0,
        "ln1.bias": rng.normal(size=D).astype(np.float64) * 0.02,
        "attn.qkv.weight": rng.normal(size=(3 * D, D)).astype(np.float64) * 0.08,
        "attn.proj.weight": rng.normal(size=(D, D)).astype(np.float64) * 0.08,
        "ln2.weight": rng.normal(size=D).astype(np.float64) * 0.1 + 1.0,
        "ln2.bias": rng.normal(size=D).astype(np.float64) * 0.02,
        "ffn.0.weight": rng.normal(size=(4 * D, D)).astype(np.float64) * 0.08,
        "ffn.0.bias": rng.normal(size=4 * D).astype(np.float64) * 0.02,
        "ffn.2.weight": rng.normal(size=(D, 4 * D)).astype(np.float64) * 0.08,
        "ffn.2.bias": rng.normal(size=D).astype(np.float64) * 0.02,
    }
    with torch.no_grad():
        block.ln1.weight.copy_(torch.tensor(arrays["ln1.weight"], dtype=torch.float64))
        block.ln1.bias.copy_(torch.tensor(arrays["ln1.bias"], dtype=torch.float64))
        block.attn.qkv.weight.copy_(torch.tensor(arrays["attn.qkv.weight"], dtype=torch.float64))
        block.attn.proj.weight.copy_(torch.tensor(arrays["attn.proj.weight"], dtype=torch.float64))
        block.ln2.weight.copy_(torch.tensor(arrays["ln2.weight"], dtype=torch.float64))
        block.ln2.bias.copy_(torch.tensor(arrays["ln2.bias"], dtype=torch.float64))
        block.ffn[0].weight.copy_(torch.tensor(arrays["ffn.0.weight"], dtype=torch.float64))
        block.ffn[0].bias.copy_(torch.tensor(arrays["ffn.0.bias"], dtype=torch.float64))
        block.ffn[2].weight.copy_(torch.tensor(arrays["ffn.2.weight"], dtype=torch.float64))
        block.ffn[2].bias.copy_(torch.tensor(arrays["ffn.2.bias"], dtype=torch.float64))

    x = rng.normal(size=(T, D)).astype(np.float64) * 0.25
    grad_y = rng.normal(size=(T, D)).astype(np.float64) * 0.01

    x_t = torch.tensor(x[None, :, :], dtype=torch.float64, requires_grad=True)
    y, _ = block(x_t)
    y.backward(torch.tensor(grad_y[None, :, :], dtype=torch.float64))

    fixed_weights = {name: quantize(value) for name, value in arrays.items()}
    grad_x, grads = _block_backward_q15(
        quantize(x),
        fixed_weights,
        quantize(grad_y),
        n_heads=n_heads,
    )

    expected_grads = {
        "ln1.weight": block.ln1.weight.grad.numpy(),
        "ln1.bias": block.ln1.bias.grad.numpy(),
        "attn.qkv.weight": block.attn.qkv.weight.grad.numpy(),
        "attn.proj.weight": block.attn.proj.weight.grad.numpy(),
        "ln2.weight": block.ln2.weight.grad.numpy(),
        "ln2.bias": block.ln2.bias.grad.numpy(),
        "ffn.0.weight": block.ffn[0].weight.grad.numpy(),
        "ffn.0.bias": block.ffn[0].bias.grad.numpy(),
        "ffn.2.weight": block.ffn[2].weight.grad.numpy(),
        "ffn.2.bias": block.ffn[2].bias.grad.numpy(),
    }

    assert np.max(np.abs(dequantize(grad_x) - x_t.grad.numpy()[0])) < 0.012
    for name, expected in expected_grads.items():
        assert np.max(np.abs(dequantize(grads[name]) - expected)) < 0.012, name


def test_fixed_backward_matches_torch_on_tiny_transformer():
    """Full model backward should track PyTorch autograd on a tiny model."""
    rng = np.random.default_rng(21)
    D = 16
    n_heads = 4
    n_layers = 1
    max_context = 32
    model = KolmoTransformer(
        d_model=D,
        n_heads=n_heads,
        n_layers=n_layers,
        max_context=max_context,
        tie_weights=False,  # see comment on the other tie_weights=False above
    ).double()

    arrays = {}
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        if name.endswith("ln1.weight") or name.endswith("ln2.weight") or name == "ln_f.weight":
            arr = rng.normal(size=shape).astype(np.float64) * 0.1 + 1.0
        elif name.endswith(".bias") or name == "ln_f.bias":
            arr = rng.normal(size=shape).astype(np.float64) * 0.02
        elif name.endswith("emb.weight"):
            arr = rng.normal(size=shape).astype(np.float64) * 0.05
        else:
            arr = rng.normal(size=shape).astype(np.float64) * 0.05
        arrays[name] = arr

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.tensor(arrays[name], dtype=torch.float64))

    input_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    target_positions = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    targets = np.array([2, 3, 4, 5, 6], dtype=np.int64)

    x_t = torch.tensor(input_ids[None, :], dtype=torch.long)
    logits_t, _ = model(x_t)
    selected = logits_t[0, target_positions]
    loss = torch.nn.functional.cross_entropy(
        selected,
        torch.tensor(targets, dtype=torch.long),
    )
    loss.backward()

    weights = {name: quantize(value) for name, value in arrays.items()}
    logits_q, grads = fixed_backward(
        input_ids,
        target_positions,
        targets,
        weights,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    assert logits_q.shape == (len(input_ids), 256)
    # Check all parameters, but allow the full stack a little more room than
    # the single-op tests because quantization error compounds through the
    # whole model.
    for name, param in model.named_parameters():
        expected = param.grad.detach().numpy()
        got = dequantize(grads[name])
        assert got.shape == expected.shape
        assert np.max(np.abs(got - expected)) < 0.03, name


def test_extract_fixed_weights_aliases_tied_head():
    """A tied model produces a weights dict where head.weight and
    token_emb.weight share the same underlying array.

    `named_parameters()` deduplicates by Parameter id, so the head wouldn't
    normally appear in the dict — `extract_fixed_weights` walks the model
    once more to re-add aliased names.
    """
    model = KolmoTransformer(d_model=32, n_heads=4, n_layers=1, max_context=32)
    assert model.tie_weights is True
    weights = extract_fixed_weights(model)
    assert "head.weight" in weights
    assert "token_emb.weight" in weights
    assert weights["head.weight"] is weights["token_emb.weight"]
    assert tied_param_pairs(model) == [("token_emb.weight", "head.weight")]


def test_extract_fixed_weights_uses_float64_rope_cos_sin():
    """The Q15 RoPE cos/sin tables must be computed from a float64 freqs
    table, not from the PyTorch model's float32 buffer.

    Why: torch.cos / torch.sin on the float32 freqs buffer was producing
    Q15 cos values that disagreed in 2 entries between Mac's Accelerate
    libm and Windows' UCRT libm. One flipped Q15 cos cascaded through
    attention into completely different trained weights after the first
    training step, breaking Rung 4 cross-OS determinism. The fix routes
    extraction through a freshly-recomputed float64 freqs table whose
    libm output has ~38 bits of headroom over Q15.

    Structural check: the Q15 cos table extracted from the model must
    equal `fixed.quantize(np.cos(freqs))` computed from the same
    architecture parameters in pure float64. If they differ, someone
    reintroduced the float32 path.
    """
    from kolmo import fixed
    from kolmo.fixed_model import _rope_freqs_for

    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
        tie_weights=False,
        use_rope=True,
    )
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)

    freqs = _rope_freqs_for(model)
    assert freqs.dtype == np.float64, (
        "_rope_freqs_for must return float64 — the entire point of the "
        "Rung-4 fix is to have ~38 bits of headroom over Q15"
    )
    expected_cos_q = fixed.quantize(np.cos(freqs))
    expected_sin_q = fixed.quantize(np.sin(freqs))
    assert np.array_equal(weights["rope.cos"], expected_cos_q), (
        "rope.cos Q15 table doesn't match np.cos(float64 freqs) — "
        "someone reintroduced a path that uses the PyTorch float32 buffer"
    )
    assert np.array_equal(weights["rope.sin"], expected_sin_q), (
        "rope.sin Q15 table doesn't match np.sin(float64 freqs)"
    )


def test_fixed_train_block_respects_weight_tying():
    """One training step in fixed mode must:
    1. Sum head + token_emb gradients before Adam (matching PyTorch's autograd
       behavior for shared Parameters).
    2. Leave weights["head.weight"] === weights["token_emb.weight"] after the
       step, so the next forward sees a consistent shared tensor.
    """
    model = KolmoTransformer(d_model=32, n_heads=4, n_layers=1, max_context=64)
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)
    tied = tied_param_pairs(model)
    assert tied  # sanity: model is actually tied

    history = list(range(1, 17))
    block = list(range(17, 33))
    state = fixed_train_block(
        weights,
        None,
        history,
        block,
        n_heads=4,
        n_layers=1,
        context=64,
        tied_params=tied,
    )

    # Re-aliased after Adam.
    assert weights["head.weight"] is weights["token_emb.weight"]

    # Adam only ran once for the canonical name — no m/v entries for the alias.
    assert "token_emb.weight" in state.m
    assert "head.weight" not in state.m
    assert "head.weight" not in state.v

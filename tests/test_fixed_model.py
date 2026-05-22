"""Tests for the fixed-point transformer forward pass."""

import numpy as np
import torch

from kolmo.fixed import dequantize, quantize
from kolmo.fixed_model import (
    _attention_backward_q15,
    _block_backward_q15,
    extract_fixed_weights,
    fixed_forward,
)
from kolmo.model import KolmoTransformer, TransformerBlock
from kolmo.stable_init import stable_init_model


def test_fixed_forward_has_expected_shape_and_is_deterministic():
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
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
    assert np.max(np.abs(ref - fixed_logits)) < 0.001
    assert np.corrcoef(ref.flatten(), fixed_logits.flatten())[0, 1] > 0.99999


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

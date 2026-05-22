"""Tests for the fixed-point transformer forward pass."""

import numpy as np
import torch

from kolmo.fixed import dequantize
from kolmo.fixed_model import extract_fixed_weights, fixed_forward
from kolmo.model import KolmoTransformer
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

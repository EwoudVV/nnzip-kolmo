"""Tests for the pure-NumPy transformer forward pass."""

import numpy as np
import torch

from kolmo.model import KolmoTransformer
from kolmo.np_model import extract_weights, kolmo_forward


def test_numpy_forward_matches_torch_tiny_transformer():
    """NumPy forward should numerically match the PyTorch model it replaces."""
    torch.manual_seed(123)
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
    )
    model.eval()

    x = torch.tensor([[1, 2, 3, 100, 200, 42]], dtype=torch.long)
    with torch.no_grad():
        torch_logits, _ = model(x)

    np_logits = kolmo_forward(
        x.numpy()[0],
        extract_weights(model),
        n_heads=4,
        n_layers=2,
    )

    assert np_logits.shape == (6, 256)
    assert np.allclose(torch_logits[0].numpy(), np_logits, atol=1e-5)


def test_numpy_forward_respects_position_offset():
    """Position offsets matter when rebuilding caches mid-stream."""
    torch.manual_seed(456)
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
    )
    model.eval()

    x = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    with torch.no_grad():
        torch_logits, _ = model(x, pos_offset=7)

    np_logits = kolmo_forward(
        x.numpy()[0],
        extract_weights(model),
        n_heads=4,
        n_layers=2,
        pos_offset=7,
    )

    assert np.allclose(torch_logits[0].numpy(), np_logits, atol=1e-5)

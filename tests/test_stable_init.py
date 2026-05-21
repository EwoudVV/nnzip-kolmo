"""Tests for platform-stable model initialization."""

import torch

from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model


def test_stable_init_does_not_depend_on_torch_rng_state():
    torch.manual_seed(1)
    a = KolmoTransformer(d_model=64, n_heads=4, n_layers=2, max_context=128)
    stable_init_model(a, seed=42)

    torch.manual_seed(999)
    b = KolmoTransformer(d_model=64, n_heads=4, n_layers=2, max_context=128)
    stable_init_model(b, seed=42)

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_stable_init_seed_changes_weights():
    a = KolmoTransformer(d_model=64, n_heads=4, n_layers=2, max_context=128)
    b = KolmoTransformer(d_model=64, n_heads=4, n_layers=2, max_context=128)
    stable_init_model(a, seed=42)
    stable_init_model(b, seed=43)

    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))

"""Sanity checks on the model: shapes, parameter count, valid distribution."""

import torch

from kolmo import KolmoTransformer


def test_forward_returns_logits_with_expected_shape():
    model = KolmoTransformer()
    x = torch.randint(0, 256, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 256)


def test_param_count_is_in_target_range():
    """Default config should land near 3M parameters. If a tweak suddenly
    pushes us outside ~2-5M, we want a test to flag it."""
    model = KolmoTransformer()
    n = model.num_parameters()
    assert 2_000_000 < n < 5_000_000, (
        f"unexpected param count {n:,} (target ~3M)"
    )


def test_softmax_of_logits_sums_to_one():
    model = KolmoTransformer()
    x = torch.randint(0, 256, (1, 10))
    probs = torch.softmax(model(x), dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_context_length_check():
    model = KolmoTransformer(max_context=64)
    too_long = torch.randint(0, 256, (1, 65))
    try:
        model(too_long)
    except ValueError as e:
        assert "context length" in str(e)
    else:
        raise AssertionError("expected ValueError for oversized context")


def test_seeded_init_is_reproducible():
    """Two models created with the same seed should have identical weights.
    This is the foundation for Rung 1 round-trip correctness."""
    torch.manual_seed(42)
    a = KolmoTransformer()
    torch.manual_seed(42)
    b = KolmoTransformer()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)

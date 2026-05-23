"""Sanity checks on the model: shapes, parameter count, valid distribution,
and KV-cache equivalence with a single full forward."""

import pytest
import torch

from kolmo import KolmoTransformer
from kolmo.model import GeGLUFFN
from kolmo.stable_init import stable_init_model


def test_forward_returns_logits_with_expected_shape():
    model = KolmoTransformer()
    x = torch.randint(0, 256, (1, 10))
    logits, caches = model(x)
    assert logits.shape == (1, 10, 256)
    assert len(caches) == 4  # one cache per layer (default 4 layers)


def test_param_count_is_in_target_range():
    """Default config should land near 3.3M parameters.

    Previously this asserted 5-12M because max_context defaulted to 16384,
    making pos_emb a 4.2M-param tensor — most of it dead weight Adam still
    updated every step. The new default max_context=512 brings pos_emb down
    to ~131K. Bulk of remaining params is the 4 transformer blocks
    (~800K each via FFN's 4*d_model expansion).
    """
    model = KolmoTransformer()
    n = model.num_parameters()
    assert 3_000_000 < n < 4_000_000, (
        f"unexpected param count {n:,} (target 3-4M)"
    )


def test_softmax_of_logits_sums_to_one():
    model = KolmoTransformer()
    x = torch.randint(0, 256, (1, 10))
    logits, _ = model(x)
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_seeded_init_is_reproducible():
    torch.manual_seed(42)
    a = KolmoTransformer()
    torch.manual_seed(42)
    b = KolmoTransformer()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_kv_cache_matches_full_forward():
    """A single forward over [a, b, c, d] should produce the same logits as
    feeding [a, b] then [c, d] with a KV cache. This is the core
    correctness condition for the cache implementation."""
    torch.manual_seed(7)
    model = KolmoTransformer(max_context=64)
    stable_init_model(model, seed=42)  # stable scale; default torch init for
    # nn.Embedding is N(0,1), which with weight tying makes head outputs ~28x
    # bigger than the linear-init scale and amplifies ULP-level float drift.
    model.eval()

    x_full = torch.randint(0, 256, (1, 8))

    with torch.no_grad():
        # Reference: single forward
        full_logits, _ = model(x_full)

        # Incremental: two halves with cache
        first_logits, caches = model(x_full[:, :4], kv_caches=None, pos_offset=0)
        second_logits, _ = model(x_full[:, 4:], kv_caches=caches, pos_offset=4)

    # First half matches positions 0..3 of full
    assert torch.allclose(first_logits, full_logits[:, :4], atol=1e-5)
    # Second half matches positions 4..7 of full
    assert torch.allclose(second_logits, full_logits[:, 4:], atol=1e-5)


def test_kv_cache_one_token_at_a_time():
    """Token-by-token incremental forward should also match a single full
    forward — this is the actual usage pattern in decompress."""
    torch.manual_seed(7)
    model = KolmoTransformer(max_context=64)
    stable_init_model(model, seed=42)
    model.eval()

    x_full = torch.randint(0, 256, (1, 6))

    with torch.no_grad():
        full_logits, _ = model(x_full)

        caches = None
        incr_logits = []
        for i in range(6):
            l, caches = model(x_full[:, i:i + 1], kv_caches=caches, pos_offset=i)
            incr_logits.append(l)
        incr_logits = torch.cat(incr_logits, dim=1)

    assert torch.allclose(incr_logits, full_logits, atol=1e-5)


def test_geglu_constructs_and_forwards():
    """Opt-in GeGLU FFN should construct, forward, and produce same-shape
    output as the default GELU FFN."""
    torch.manual_seed(7)
    model = KolmoTransformer(ffn_type="geglu")
    stable_init_model(model, seed=42)
    x = torch.randint(0, 256, (1, 16))
    logits, _ = model(x)
    assert logits.shape == (1, 16, 256)
    assert isinstance(model.blocks[0].ffn, GeGLUFFN)


def test_geglu_param_count_close_to_gelu():
    """GeGLU's d_ff = ceil(8*d_model/3 / 32) * 32 keeps total params within
    a few percent of the standard 4*d_model GELU FFN."""
    gelu_model = KolmoTransformer(ffn_type="gelu")
    geglu_model = KolmoTransformer(ffn_type="geglu")
    g = gelu_model.num_parameters()
    gg = geglu_model.num_parameters()
    assert abs(gg - g) < 0.05 * g, (
        f"GeGLU param count {gg:,} too far from GELU {g:,}"
    )


def test_unknown_ffn_type_rejected():
    with pytest.raises(ValueError, match="unknown ffn_type"):
        KolmoTransformer(ffn_type="banana")

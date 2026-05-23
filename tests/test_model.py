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

    Tried scaling to d_model=384 n_layers=6 (~10M params); ratio benefit
    at 1-4KB was negligible (-0.0 to -0.4 pp), and the model was 2.6x
    slower. Reverted: bigger model wins only with bigger data. Will revisit
    after Cython kernels + RoPE let us test at scales where scaling
    actually pays off.
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


def test_rope_model_constructs_and_drops_pos_emb():
    """use_rope=True should remove the pos_emb table and reclaim its 131K
    params (vocab=256, max_context=512, d_model=256 -> 512*256 = 131,072).
    """
    abs_model = KolmoTransformer(use_rope=False)
    rope_model = KolmoTransformer(use_rope=True)
    delta = abs_model.num_parameters() - rope_model.num_parameters()
    expected = 512 * 256  # max_context * d_model
    assert delta == expected, (
        f"expected RoPE to save {expected:,} params, got {delta:,}"
    )
    assert rope_model.pos_emb is None
    assert rope_model.rope is not None
    assert abs_model.pos_emb is not None
    assert abs_model.rope is None


def test_rope_kv_cache_matches_full_forward():
    """The trickiest invariant for any positional scheme: feeding tokens
    incrementally through the KV cache must produce the same logits as
    a single full forward. With RoPE, K is rotated by its position when
    first computed and then cached, so concatenated cached-K + new-K
    must give the same attention scores as a fresh full-forward K.
    """
    torch.manual_seed(7)
    model = KolmoTransformer(use_rope=True, max_context=64)
    stable_init_model(model, seed=42)
    model.eval()

    x_full = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        full_logits, _ = model(x_full)
        first, caches = model(x_full[:, :4], kv_caches=None, pos_offset=0)
        second, _ = model(x_full[:, 4:], kv_caches=caches, pos_offset=4)

    assert torch.allclose(first, full_logits[:, :4], atol=1e-5)
    assert torch.allclose(second, full_logits[:, 4:], atol=1e-5)


def test_rope_round_trip_via_engine_env(monkeypatch, tmp_path):
    """End-to-end: enabling RoPE via the engine should not break round-trip.

    Uses KOLMO_USE_ROPE=1 to flip the default; skip-prime to keep test fast.
    """
    monkeypatch.setenv("KOLMO_USE_ROPE", "1")
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    from kolmo import compress, decompress
    data = b"rope round-trip smoke test"
    blob = compress(data)
    assert decompress(blob) == data

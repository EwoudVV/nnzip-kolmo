import numpy as np

import kolmo._engine as engine
from kolmo._engine import LiteralModel, literal_context_bucket


def test_literal_model_probs_are_normalized_and_nonzero():
    model = LiteralModel()
    neural = np.ones(256, dtype=np.float64) / 256.0
    probs = model.probs(neural)
    assert probs.shape == (256,)
    assert np.all(probs > 0.0)
    assert np.isclose(probs.sum(), 1.0)


def test_literal_model_order2_learns_observed_transition(monkeypatch):
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "mix")
    monkeypatch.setattr(engine, "LITERAL_ORDER2_WEIGHT", 0.5)
    monkeypatch.setattr(engine, "LITERAL_ORDER1_WEIGHT", 0.03)
    monkeypatch.setattr(engine, "LITERAL_ORDER0_WEIGHT", 0.005)

    model = LiteralModel()
    neural = np.ones(256, dtype=np.float64) / 256.0

    # Teach context ("a", "b") -> "c" repeatedly.
    for _ in range(20):
        model.observe(ord("a"))
        model.observe(ord("b"))
        model.observe(ord("c"))

    # Re-enter the ("a", "b") context and query the next-byte distribution.
    model.observe(ord("a"))
    model.observe(ord("b"))
    probs = model.probs(neural)

    assert probs[ord("c")] > probs[ord("x")]
    assert np.isclose(probs.sum(), 1.0)


def test_literal_model_proxy_bits_reflect_learned_context(monkeypatch):
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "mix")
    monkeypatch.setattr(engine, "LITERAL_ORDER2_WEIGHT", 0.5)
    monkeypatch.setattr(engine, "LITERAL_ORDER2_CONFIDENCE", 1.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER1_WEIGHT", 0.03)
    monkeypatch.setattr(engine, "LITERAL_ORDER0_WEIGHT", 0.005)

    model = LiteralModel()
    for _ in range(20):
        model.observe(ord("a"))
        model.observe(ord("b"))
        model.observe(ord("c"))

    model.observe(ord("a"))
    model.observe(ord("b"))
    expected = model.proxy_bits(b"c", neural_bpb=2.75)
    unexpected = model.proxy_bits(b"x", neural_bpb=2.75)

    assert expected < unexpected


def test_literal_model_order4_learns_observed_transition(monkeypatch):
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "mix")
    monkeypatch.setattr(engine, "LITERAL_ORDER4_WEIGHT", 0.25)
    monkeypatch.setattr(engine, "LITERAL_ORDER4_CONFIDENCE", 1.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER4_BUCKETS", 1 << 12)
    monkeypatch.setattr(engine, "LITERAL_ORDER3_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER2_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER1_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER0_WEIGHT", 0.0)

    model = LiteralModel()
    neural = np.ones(256, dtype=np.float64) / 256.0

    # Teach context ("a", "b", "c", "d") -> "e" repeatedly.
    for _ in range(20):
        for ch in b"abcde":
            model.observe(ch)

    for ch in b"abcd":
        model.observe(ch)
    probs = model.probs(neural)

    assert probs[ord("e")] > probs[ord("x")]
    assert np.isclose(probs.sum(), 1.0)


def test_literal_model_order5_learns_observed_transition(monkeypatch):
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "mix")
    monkeypatch.setattr(engine, "LITERAL_ORDER5_WEIGHT", 0.25)
    monkeypatch.setattr(engine, "LITERAL_ORDER5_CONFIDENCE", 1.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER5_BUCKETS", 1 << 12)
    monkeypatch.setattr(engine, "LITERAL_ORDER4_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER3_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER2_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER1_WEIGHT", 0.0)
    monkeypatch.setattr(engine, "LITERAL_ORDER0_WEIGHT", 0.0)

    model = LiteralModel()
    neural = np.ones(256, dtype=np.float64) / 256.0

    # Teach context ("a", "b", "c", "d", "e") -> "f" repeatedly.
    for _ in range(20):
        for ch in b"abcdef":
            model.observe(ch)

    for ch in b"abcde":
        model.observe(ch)
    probs = model.probs(neural)

    assert probs[ord("f")] > probs[ord("x")]
    assert np.isclose(probs.sum(), 1.0)


def test_literal_model_ppm_learns_observed_transition(monkeypatch):
    """Default PPM strategy should pick up an observed (a,b)->c transition.

    Order 2 in PPM-C will see distinct=1, count[c]=N, so p(c) ≈ N/(N+1)
    times the cumulative escape from any higher orders. Even with the
    neural mix at 0.5, the c probability should dominate.
    """
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT", 0.5)
    model = LiteralModel()
    neural = np.ones(256, dtype=np.float64) / 256.0

    for _ in range(20):
        model.observe(ord("a"))
        model.observe(ord("b"))
        model.observe(ord("c"))

    model.observe(ord("a"))
    model.observe(ord("b"))
    probs = model.probs(neural)

    assert probs[ord("c")] > probs[ord("x")]
    assert np.isclose(probs.sum(), 1.0)


def test_literal_model_ppm_pure_no_neural(monkeypatch):
    """LITERAL_NEURAL_WEIGHT=0 should ignore the neural distribution
    entirely. Feeding a degenerate neural prob mass on byte 'x' must
    not push p(x) up at all if PPM is the sole signal."""
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT", 0.0)
    model = LiteralModel()
    # No learning — just check that the neural input has no effect.
    pinned = np.zeros(256, dtype=np.float64)
    pinned[ord("x")] = 1.0
    p0 = model.probs(np.ones(256, dtype=np.float64) / 256.0)
    p1 = model.probs(pinned)
    assert np.allclose(p0, p1)


def test_literal_context_bucket_avalanches_shared_suffix_contexts():
    """High-order byte contexts often share suffix bytes on real text.

    Bucket counts are powers of two, so a plain multiplicative hash mostly
    preserves low-bit structure; contexts with the same final byte can collapse
    into a tiny fraction of the table. The bucket mixer should avalanche those
    contexts across most of the table instead.
    """
    buckets = 1 << 18
    contexts = [(i << 8) | ord(" ") for i in range(4096)]
    occupied = {literal_context_bucket(ctx, buckets) for ctx in contexts}
    assert len(occupied) > 4000

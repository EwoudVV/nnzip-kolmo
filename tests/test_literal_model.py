import numpy as np

import kolmo._engine as engine
from kolmo._engine import LiteralModel


def test_literal_model_probs_are_normalized_and_nonzero():
    model = LiteralModel()
    neural = np.ones(256, dtype=np.float64) / 256.0
    probs = model.probs(neural)
    assert probs.shape == (256,)
    assert np.all(probs > 0.0)
    assert np.isclose(probs.sum(), 1.0)


def test_literal_model_order2_learns_observed_transition(monkeypatch):
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

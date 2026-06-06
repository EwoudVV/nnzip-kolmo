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
    """LITERAL_NEURAL_WEIGHT=0 in the *legacy* fixed-weight path should
    ignore the neural distribution entirely.

    Has to explicitly opt out of the cost-aware adaptive blend (now
    default), because adaptive ignores LITERAL_NEURAL_WEIGHT and
    computes the weight from PPM peak instead — so with adaptive on
    and an untrained model (uniform PPM, peak=1/256) the neural
    distribution still gets through with weight ≈ HIGH = 0.70.
    """
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "_ADAPTIVE_WEIGHT", False)
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT", 0.0)
    model = LiteralModel()
    # No learning — just check that the neural input has no effect.
    pinned = np.zeros(256, dtype=np.float64)
    pinned[ord("x")] = 1.0
    p0 = model.probs(np.ones(256, dtype=np.float64) / 256.0)
    p1 = model.probs(pinned)
    assert np.allclose(p0, p1)


def test_adaptive_blend_shifts_weight_with_ppm_confidence(monkeypatch):
    """The cost-aware blend (KOLMO_ADAPTIVE_WEIGHT=1, default) should put
    relatively MORE weight on PPM when PPM is sharply peaked, and LESS
    when PPM is uniform. We can't directly read the per-call weight, but
    we can engineer two cases and assert the output distribution shifts
    in the expected direction.

    Setup: pin LITERAL_NEURAL_WEIGHT_LOW=0.0, LITERAL_NEURAL_WEIGHT_HIGH=1.0
    — these are extreme but make the dependence loud.
    Case A: PPM is uniform (untrained model, all-prior order-0). Adaptive
            weight ≈ HIGH = 1.0, so output ≈ p_neural exactly.
    Case B: train PPM with a strong (a,b)->c transition. p_ppm.max() will be
            close to 1.0, adaptive weight ≈ LOW = 0.0, so output ≈ p_ppm.
    """
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "_ADAPTIVE_WEIGHT", True)
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT_LOW", 0.0)
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT_HIGH", 1.0)

    # A non-uniform neural — pin mass on byte 'x' — so we can detect when
    # adaptive lets it through vs damps it.
    pinned_neural = np.full(256, (1.0 - 0.9) / 255, dtype=np.float64)
    pinned_neural[ord("x")] = 0.9

    # Case A: uniform PPM (no training). Adaptive weight → HIGH = 1.0, so
    # output should be ≈ pinned_neural (modulo PPM's tiny order-0 prior).
    fresh = LiteralModel()
    out_uniform = fresh.probs(pinned_neural)
    assert out_uniform[ord("x")] > 0.5, (
        "uniform PPM should let neural's 0.9 mass on 'x' come through"
    )

    # Case B: sharply trained PPM. (a,b)->c repeated many times so order-2
    # picks up p(c|a,b) ≈ 1.0. Adaptive weight → LOW = 0.0, so output
    # should be ≈ p_ppm, dominated by 'c'.
    trained = LiteralModel()
    for _ in range(40):
        trained.observe(ord("a"))
        trained.observe(ord("b"))
        trained.observe(ord("c"))
    trained.observe(ord("a"))
    trained.observe(ord("b"))
    out_trained = trained.probs(pinned_neural)
    assert out_trained[ord("c")] > out_trained[ord("x")], (
        "sharp PPM should suppress neural's 'x' bias and let PPM's 'c' win"
    )
    assert out_trained[ord("c")] > 0.8, (
        "with adaptive LOW=0.0, sharp PPM should dominate the blend"
    )


def test_adaptive_blend_can_be_disabled(monkeypatch):
    """Setting KOLMO_ADAPTIVE_WEIGHT=0 must fall back to the static
    LITERAL_NEURAL_WEIGHT — preserves the legacy code path for
    bisection and historical comparison."""
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "_ADAPTIVE_WEIGHT", False)
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT", 0.5)
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT_LOW", 0.0)
    monkeypatch.setattr(engine, "LITERAL_NEURAL_WEIGHT_HIGH", 1.0)

    pinned_neural = np.full(256, (1.0 - 0.9) / 255, dtype=np.float64)
    pinned_neural[ord("x")] = 0.9

    model = LiteralModel()
    out = model.probs(pinned_neural)
    # With static weight=0.5 and uniform PPM, output[x] ≈ 0.5*0.9 + 0.5*(1/256)
    # ≈ 0.452. With adaptive (LOW=0, HIGH=1, uniform PPM → w=1) it would be
    # ≈ 0.9. The static-path assertion is the lower one.
    assert 0.4 < out[ord("x")] < 0.55, (
        f"static weight=0.5 + 'x'=0.9 in neural should give ~0.45, got {out[ord('x')]}"
    )


def test_post_copy_predictor_records_and_uses_transitions(monkeypatch):
    """The third predictor: after `mark_copy_end(last_byte)`, the next
    observe() must increment post_copy_counts[last_byte, observed_byte],
    and the next probs() call should reflect that transition via a
    3-way blend (only when KOLMO_POST_COPY is on).

    Three-step assertion:
    1. Without mark_copy_end, probs() ignores post_copy_counts entirely.
    2. After mark_copy_end + observe, the count for (last, observed)
       is +1 from its prior.
    3. With KOLMO_POST_COPY enabled, the next probs() after mark_copy_end
       shifts mass toward the bytes commonly seen post-copy.
    """
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "_POST_COPY", True)
    monkeypatch.setattr(engine, "LITERAL_POST_COPY_WEIGHT", 0.5)

    model = LiteralModel()

    # Step 1: pristine state — post_copy_counts is all uniform priors.
    assert np.all(model.post_copy_counts == 1.0)

    # Teach the post-copy transition: after a copy ending in 'g', byte 'X'
    # follows. Repeat enough times for the post-copy distribution to
    # noticeably favor 'X'.
    for _ in range(40):
        model.mark_copy_end(ord("g"))
        model.observe(ord("X"))

    # Step 2: the count was incremented exactly 40 times.
    assert model.post_copy_counts[ord("g"), ord("X")] == 1.0 + 40

    # Step 3: probs() right after mark_copy_end blends in the post-copy
    # row, which now heavily favors 'X'. Compare to a control where we
    # didn't mark a copy end — that case should NOT have the bias.
    neural = np.ones(256, dtype=np.float64) / 256.0
    model.mark_copy_end(ord("g"))
    probs_post_copy = model.probs(neural)

    # Re-create a fresh model with the same trained post-copy counts, but
    # don't mark_copy_end this time — control case.
    control = LiteralModel()
    control.post_copy_counts = model.post_copy_counts.copy()
    probs_no_flag = control.probs(neural)

    # The post-copy probs should put more mass on 'X' than the control.
    assert probs_post_copy[ord("X")] > probs_no_flag[ord("X")] + 0.05


def test_post_copy_disabled_by_default(monkeypatch):
    """KOLMO_POST_COPY default is 0; mark_copy_end is still safe to call,
    but probs() must produce the same output as if the call hadn't
    happened. This is the bisection / rollback path."""
    monkeypatch.setattr(engine, "_LITERAL_STRATEGY", "ppm")
    monkeypatch.setattr(engine, "_POST_COPY", False)

    model_a = LiteralModel()
    model_b = LiteralModel()
    for _ in range(40):
        model_a.mark_copy_end(ord("g"))
        model_a.observe(ord("X"))
        model_b.observe(ord("X"))

    neural = np.ones(256, dtype=np.float64) / 256.0
    model_a.mark_copy_end(ord("g"))
    out_a = model_a.probs(neural)
    out_b = model_b.probs(neural)
    # Should be very close — model_a observed extra ordering data
    # (alternating mark_copy_end + observe) but the BLEND in probs()
    # ignores it when _POST_COPY is False.
    assert np.allclose(out_a, out_b, atol=1e-12)


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

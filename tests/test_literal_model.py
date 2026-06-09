import numpy as np
import pytest

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


def test_linear_ensemble_mixer_blends_predictors_and_neural():
    """LinearEnsembleMixer with equal weights on neural and ppm should
    average them. The output must be normalized."""
    from kolmo._predictors import LinearEnsembleMixer

    mixer = LinearEnsembleMixer([("neural", 0.5), ("ppm", 0.5)])
    # Make neural strongly favor byte 'x', ppm strongly favor byte 'y'.
    neural = np.full(256, 0.001, dtype=np.float64)
    neural[ord("x")] = 0.745
    neural = neural / neural.sum()
    ppm = np.full(256, 0.001, dtype=np.float64)
    ppm[ord("y")] = 0.745
    ppm = ppm / ppm.sum()

    out = mixer.combine({"ppm": ppm}, neural)
    assert np.isclose(out.sum(), 1.0)
    # With equal weights, x and y should be roughly tied.
    assert np.isclose(out[ord("x")], out[ord("y")], atol=1e-9)
    # Both should dominate over a random other byte.
    assert out[ord("x")] > 10 * out[ord("a")]


def test_linear_ensemble_mixer_drops_none_outputs_and_renormalizes():
    """A predictor that returns None should be dropped from the blend
    and its weight redistributed across remaining contributors."""
    from kolmo._predictors import LinearEnsembleMixer

    mixer = LinearEnsembleMixer(
        [("neural", 0.5), ("ppm", 0.3), ("post_copy", 0.2)]
    )
    neural = np.full(256, 1.0 / 256.0, dtype=np.float64)
    ppm = np.full(256, 0.001, dtype=np.float64)
    ppm[ord("e")] = 0.745
    ppm = ppm / ppm.sum()

    # post_copy returns None: weights should be neural=0.5/0.8=0.625,
    # ppm=0.3/0.8=0.375 after renorm.
    out = mixer.combine({"ppm": ppm, "post_copy": None}, neural)
    assert np.isclose(out.sum(), 1.0)
    # Result should be a 62.5% neural + 37.5% ppm blend, so 'e' gets
    # ~0.375 * 0.745 + 0.625 * (1/256) ≈ 0.28 + 0.0024 ≈ 0.28.
    assert 0.27 < out[ord("e")] < 0.29


def test_linear_ensemble_mixer_handles_all_silent_predictors():
    """If every predictor in the weights is silent AND neural isn't
    listed, fall back to uniform rather than crash. Defensive."""
    from kolmo._predictors import LinearEnsembleMixer

    mixer = LinearEnsembleMixer([("post_copy", 1.0)])
    neural = np.full(256, 1.0 / 256.0, dtype=np.float64)
    out = mixer.combine({"post_copy": None}, neural)
    assert np.isclose(out.sum(), 1.0)
    assert np.allclose(out, 1.0 / 256.0)


def test_parse_linear_weights_accepts_standard_spec():
    from kolmo._predictors import parse_linear_weights

    result = parse_linear_weights("neural:0.4,ppm:0.5,post_copy:0.1")
    assert result == [("neural", 0.4), ("ppm", 0.5), ("post_copy", 0.1)]

    # Whitespace tolerance.
    result = parse_linear_weights("  neural : 0.4 , ppm : 0.5 ")
    assert result == [("neural", 0.4), ("ppm", 0.5)]


def test_parse_linear_weights_rejects_malformed_input():
    from kolmo._predictors import parse_linear_weights

    with pytest.raises(ValueError, match="missing ':'"):
        parse_linear_weights("neural=0.4")
    with pytest.raises(ValueError, match="non-numeric"):
        parse_linear_weights("neural:high")
    with pytest.raises(ValueError, match="empty name"):
        parse_linear_weights(":0.5")
    with pytest.raises(ValueError, match="at least one entry"):
        parse_linear_weights("")


def test_register_predictor_forwards_observe_and_mark_copy_end():
    """The predictor framework's extension point: a custom Predictor
    registered via `register_predictor()` must receive `observe(byte)`
    on every literal-model observation and `mark_copy_end(byte)` on
    every copy-event end. The current default mixer doesn't blend
    extra predictors into the output yet (that's a follow-up commit),
    but the dataflow plumbing is exercised here so it's caught early
    if a future refactor breaks the wiring.
    """
    from kolmo._predictors import Predictor

    class _RecordingPredictor(Predictor):
        name = "test_recording"

        def __init__(self):
            self.observed = []
            self.copy_ends = []

        def probs(self):
            return None  # silent — mixer ignores it

        def observe(self, byte):
            self.observed.append(byte)

        def mark_copy_end(self, last_byte):
            self.copy_ends.append(last_byte)

    rec = _RecordingPredictor()
    model = LiteralModel()
    model.register_predictor(rec)

    # Drive a literal-byte sequence.
    for ch in b"hello":
        model.observe(ch)
    assert rec.observed == list(b"hello"), (
        "register_predictor() must wire observe() forwarding"
    )

    # And a copy-event end.
    model.mark_copy_end(ord(" "))
    assert rec.copy_ends == [ord(" ")], (
        "register_predictor() must wire mark_copy_end() forwarding"
    )


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


def test_word_fragment_predictor_learns_word_internal_transitions():
    """WordFragmentPredictor should learn that after 'he' the next byte
    is likely 'l' (from 'hello'), even though both are separate words
    separated by delimiters. The fragment resets on whitespace so the
    statistics are word-internal only."""
    from kolmo._predictors import WordFragmentPredictor

    p = WordFragmentPredictor(max_context_len=2, table_buckets=1 << 8)

    # Train on "hello world" — teaches transitional patterns inside words.
    for b in b"hello world ":
        p.observe(b)

    # After space, fragment is []. Now feed "he".
    p.observe(ord("h"))  # no learning (fragment was empty)
    p.observe(ord("e"))  # records: 'h' -> 'e'
    probs = p.probs()
    # Context 'he' was seen during 'hello' where 'l' followed.
    assert probs[ord("l")] > probs[ord("x")], (
        "context 'he' should favor 'l' from 'hello' training"
    )

    # After space, feed "wor".
    p.observe(ord(" "))  # records 'he' -> ' ', fragment cleared
    p.observe(ord("w"))
    p.observe(ord("o"))
    p.observe(ord("r"))
    probs = p.probs()
    # Context 'or' was seen during 'world' where 'l' followed.
    assert probs[ord("l")] > probs[ord("x")], (
        "context 'or' should favor 'l' from 'world' training"
    )


def test_word_fragment_predictor_returns_none_between_words():
    """Probs() returns None when the fragment is empty (between words)."""
    from kolmo._predictors import WordFragmentPredictor

    p = WordFragmentPredictor()
    assert p.probs() is None, "fresh predictor has empty fragment"

    p.observe(ord("h"))
    assert p.probs() is not None, "inside a word, probs is not None"

    p.observe(ord(" "))  # delimiter
    assert p.probs() is None, "after space, fragment is empty"

    p.observe(ord("\n"))  # consecutive delimiters
    assert p.probs() is None, "after newline, fragment is empty"


def test_balanced_delimiter_tracks_curly_brace_nesting():
    from kolmo._predictors import BalancedDelimiterPredictor

    p = BalancedDelimiterPredictor()
    assert p.curly_depth == 0

    p.observe(ord("{"))  # depth → 1
    assert p.curly_depth == 1
    p.observe(ord("{"))  # depth → 2
    assert p.curly_depth == 2

    p.observe(ord("}"))  # depth → 1
    assert p.curly_depth == 1
    p.observe(ord("}"))  # depth → 0
    assert p.curly_depth == 0

    # Extra closes should clamp to 0.
    p.observe(ord("}"))
    assert p.curly_depth == 0


def test_balanced_delimiter_tracks_multiple_bracket_types():
    from kolmo._predictors import BalancedDelimiterPredictor

    p = BalancedDelimiterPredictor()
    p.observe(ord("["))   # square 1
    p.observe(ord("{"))   # curly 1, square 1
    assert p.square_depth == 1
    assert p.curly_depth == 1
    assert p.paren_depth == 0
    assert p.angle_depth == 0

    p.observe(ord("]"))   # square 0, curly 1
    assert p.square_depth == 0
    assert p.curly_depth == 1

    p.observe(ord("}"))   # all 0
    assert p.curly_depth == 0
    assert p.square_depth == 0


def test_balanced_delimiter_depth_clamps_at_three():
    from kolmo._predictors import BalancedDelimiterPredictor

    p = BalancedDelimiterPredictor()
    for _ in range(5):
        p.observe(ord("{"))
    # state method uses min(depth, 3), so the
    # _state-encoded depth should be 3.
    assert p.curly_depth == 5  # raw can exceed 3
    assert (p._state() & 3) == 3  # clamped


def test_balanced_delimiter_learns_template_patterns():
    from kolmo._predictors import BalancedDelimiterPredictor

    p = BalancedDelimiterPredictor()
    # Train on a template: {{Infobox | name = Foo }}
    for b in b"{{Abox | name = Foo }} ":
        p.observe(b)

    # Now re-enter {{ and check the prediction.
    p.observe(ord("{"))
    p.observe(ord("{"))
    probs = p.probs()
    # At depth=2 (inside {{), | and } should be more likely than a typical
    # letter like 'z'.
    assert probs[ord("|")] > probs[ord("z")], (
        "inside {{, pipe should be favored over 'z'"
    )
    assert probs[ord("}")] > probs[ord("z")], (
        "inside {{, closing brace should be favored over 'z'"
    )


def test_balanced_delimiter_state_encoding_lower_bits_for_curly():
    """State key should use lower 2 bits for curly depth, next 2 for square,
    etc. This is a property test on the encoding scheme, not on the learned
    distribution."""
    from kolmo._predictors import BalancedDelimiterPredictor

    p = BalancedDelimiterPredictor()
    # At no depth, state = 0.
    assert p._state() == 0

    # One { -> curly=1, state should be 1 (binary 00000001, bits 0-1).
    p.observe(ord("{"))
    assert p._state() == 1

    # Add one [ -> square=1, state should be 1 | (1 << 2) = 5.
    p.observe(ord("["))
    assert p._state() == 5


def test_balanced_delimiter_integrates_with_literal_model():
    """Registered BalancedDelimiterPredictor receives observe() forwarding
    and appears in the predictor_outputs dict. The default mixer ignores
    extra predictors, so this doesn't change output — it validates
    framework plumbing end-to-end."""
    from kolmo._predictors import BalancedDelimiterPredictor

    bd = BalancedDelimiterPredictor()
    model = LiteralModel()
    model.register_predictor(bd)

    # Feed a string with nested brackets.
    for b in b"{{Infobox | name = Foo }} [[link]]":
        model.observe(b)

    # After all closes, depth should be back to 0.
    assert bd.curly_depth == 0
    assert bd.square_depth == 0


def test_balanced_delimiter_probs_always_non_none():
    """Unlike post_copy or word_fragment, balanced_delimiter should always
    return a distribution — even at state 0 the table has a uniform prior,
    and the distribution is always informative."""
    from kolmo._predictors import BalancedDelimiterPredictor

    p = BalancedDelimiterPredictor()
    probs = p.probs()
    assert probs is not None
    assert np.isclose(probs.sum(), 1.0)
    assert probs.shape == (256,)


def test_balanced_delimiter_integrates_with_literal_model():
    """Registered WordFragmentPredictor receives observe() forwarding
    and appears in the predictor_outputs dict. The default mixer ignores
    extra predictors, so this doesn't change output — it validates
    framework plumbing end-to-end."""
    from kolmo._predictors import WordFragmentPredictor

    wf = WordFragmentPredictor()
    model = LiteralModel()
    model.register_predictor(wf)

    # Feed a sentence — the predictor should build and reset fragments.
    for b in b"hello world ":
        model.observe(b)

    # After space, fragment is empty.
    assert wf._fragment == [], "fragment should be empty after space"

    # Feed 't' and 'e' — fragment should be ['t', 'e'].
    model.observe(ord("t"))
    model.observe(ord("e"))
    assert wf._fragment == [ord("t"), ord("e")], (
        f"expected [t, e], got {wf._fragment}"
    )

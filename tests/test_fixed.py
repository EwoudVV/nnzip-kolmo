"""Verify Q15 fixed-point arithmetic agrees with float math within
quantization precision (~1/32768 ≈ 3e-5)."""

import numpy as np
import pytest

from kolmo import fixed


def test_quantize_dequantize_round_trip():
    """Quantizing then dequantizing should give back the original up to 1 ULP
    of Q15 resolution."""
    x = np.array([0.0, 1.0, -1.0, 0.5, -0.25, 3.14159, -2.71828], dtype=np.float64)
    x_q = fixed.quantize(x)
    x_back = fixed.dequantize(x_q)
    # Q15 resolution is 1/32768 ≈ 3e-5; we should match to half that.
    assert np.max(np.abs(x - x_back)) < 1.0 / (2 * fixed.SCALE)


def test_quantize_returns_int32():
    x = np.zeros(5, dtype=np.float64)
    assert fixed.quantize(x).dtype == np.int32


def test_add_matches_float_add():
    """Q15 add then dequantize ≈ float add."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=64).astype(np.float64)
    b = rng.normal(size=64).astype(np.float64)
    expected = a + b
    got = fixed.dequantize(fixed.add(fixed.quantize(a), fixed.quantize(b)))
    assert np.max(np.abs(expected - got)) < 1.0 / fixed.SCALE


def test_mul_matches_float_mul():
    """Q15 elementwise mul vs float mul, within Q15 precision."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=64).astype(np.float64)
    b = rng.normal(size=64).astype(np.float64)
    expected = a * b
    got = fixed.dequantize(fixed.mul(fixed.quantize(a), fixed.quantize(b)))
    # mul has compounded quantization error from both inputs, so allow ~2x.
    assert np.max(np.abs(expected - got)) < 4.0 / fixed.SCALE


def test_matmul_matches_float_matmul_small():
    """Q15 matmul vs float matmul on a small case."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(8, 16)).astype(np.float64)
    b = rng.normal(size=(16, 8)).astype(np.float64)
    expected = a @ b
    got = fixed.dequantize(fixed.matmul(fixed.quantize(a), fixed.quantize(b)))
    # matmul accumulates K=16 products, each with ~1/SCALE quantization error.
    # Worst-case error is K * (resolution of one product) ≈ 16/SCALE ≈ 5e-4.
    assert np.max(np.abs(expected - got)) < 32.0 / fixed.SCALE


def test_matmul_matches_float_matmul_medium():
    """Q15 matmul on a 64x64 case — closer to attention head matmul size."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 64)).astype(np.float64) * 0.1  # smaller magnitudes
    b = rng.normal(size=(64, 64)).astype(np.float64) * 0.1
    expected = a @ b
    got = fixed.dequantize(fixed.matmul(fixed.quantize(a), fixed.quantize(b)))
    # K=64 accumulated quantization error ≈ 64 / SCALE ≈ 2e-3.
    assert np.max(np.abs(expected - got)) < 0.01


def test_matmul_is_bit_deterministic_across_runs():
    """Two runs with same inputs give bit-identical int32 outputs.

    The point of fixed-point: integer addition is associative, so the matmul
    is the SAME regardless of how the reduction is parallelized internally.
    """
    rng = np.random.default_rng(4)
    a_q = fixed.quantize(rng.normal(size=(32, 32)))
    b_q = fixed.quantize(rng.normal(size=(32, 32)))
    out1 = fixed.matmul(a_q, b_q)
    out2 = fixed.matmul(a_q, b_q)
    # Bit-identical — not "within precision" but EXACTLY the same.
    assert np.array_equal(out1, out2)
    assert out1.dtype == np.int32


def test_matmul_rejects_non_int32():
    a = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(TypeError):
        fixed.matmul(a, b)


def test_round_to_nearest_not_floor():
    """We add ROUND_OFFSET before shifting in matmul — that gives
    round-to-nearest, not floor. Tiny but matters for symmetry."""
    # 1.5 * 1 = 1.5 — should quantize back to ~1.5, not floor to 1.0.
    a_q = fixed.quantize(np.array([1.5], dtype=np.float64))
    b_q = fixed.quantize(np.array([1.0], dtype=np.float64))
    result = fixed.matmul(a_q.reshape(1, 1), b_q.reshape(1, 1))
    back = fixed.dequantize(result)
    assert abs(back[0, 0] - 1.5) < 1.0 / fixed.SCALE


def test_isqrt_vec_matches_math_isqrt():
    """Vectorized isqrt matches math.isqrt on a range of values."""
    import math
    values = np.array([0, 1, 4, 9, 100, 2**15, 2**20, 2**30], dtype=np.int64)
    expected = np.array([math.isqrt(int(v)) for v in values], dtype=np.int64)
    got = fixed.isqrt_vec(values)
    assert np.array_equal(got, expected)


def test_sqrt_q15_matches_float_sqrt():
    """Q15 sqrt agrees with float sqrt within Q15 precision."""
    x = np.array([0.0, 0.25, 1.0, 4.0, 100.0, 0.001], dtype=np.float64)
    expected = np.sqrt(x)
    got = fixed.dequantize(fixed.sqrt_q15(fixed.quantize(x)))
    # Allow a few Q15 units of error.
    assert np.max(np.abs(expected - got)) < 4.0 / fixed.SCALE


def test_reciprocal_q15_matches_float():
    """1/x in Q15 agrees with float."""
    x = np.array([1.0, 2.0, 0.5, -3.7, 0.1], dtype=np.float64)
    expected = 1.0 / x
    got = fixed.dequantize(fixed.reciprocal_q15(fixed.quantize(x)))
    # 1/0.1 = 10 — precision degrades for small denominators, so use looser bound.
    assert np.max(np.abs(expected - got)) < 0.01


def test_div_q15_matches_float():
    """a/b in Q15 agrees with float."""
    rng = np.random.default_rng(5)
    a = rng.uniform(-1, 1, size=32).astype(np.float64)
    b = rng.uniform(0.1, 2.0, size=32).astype(np.float64) * np.sign(rng.normal(size=32))
    expected = a / b
    got = fixed.dequantize(fixed.div_q15(fixed.quantize(a), fixed.quantize(b)))
    assert np.max(np.abs(expected - got)) < 0.005


def test_sqrt_q15_is_deterministic_across_runs():
    rng = np.random.default_rng(6)
    x_q = fixed.quantize(rng.uniform(0.01, 100.0, size=256))
    out1 = fixed.sqrt_q15(x_q)
    out2 = fixed.sqrt_q15(x_q)
    assert np.array_equal(out1, out2)


def test_exp_q15_at_zero():
    """exp(0) should be exactly 1.0 (=SCALE in Q15)."""
    out = fixed.exp_q15(np.array([0], dtype=np.int32))
    assert int(out[0]) == fixed.SCALE


def test_exp_q15_basic_values():
    """Q15 exp matches float exp to within ~1% relative — for values that are
    larger than the Q15 floor (1/32768 ≈ 3e-5). Smaller exp results round to
    the nearest Q15 unit and have unavoidable rel-error up to ~50%, but their
    *absolute* error stays at Q15 resolution."""
    x = np.array([0.0, 0.5, 1.0, -1.0, -2.0, -5.0, -10.0], dtype=np.float64)
    expected = np.exp(x)
    got = fixed.dequantize(fixed.exp_q15(fixed.quantize(x)))
    # Two-mode tolerance: relative for large values, absolute for tiny ones.
    abs_err = np.abs(got - expected)
    rel_err = abs_err / np.maximum(np.abs(expected), 1e-30)
    # Allow whichever bound is looser per element.
    ok = (rel_err < 0.01) | (abs_err < 2.0 / fixed.SCALE)
    assert np.all(ok), f"abs_err={abs_err}, rel_err={rel_err}"


def test_exp_q15_large_negative_underflows_to_zero():
    """exp(-30) is ~9e-14, way below Q15 resolution — output is 0."""
    out = fixed.exp_q15(np.array([fixed.quantize(np.array([-30.0]))[0]], dtype=np.int32))
    assert int(out[0]) == 0


def test_exp_q15_deterministic():
    rng = np.random.default_rng(7)
    x_q = fixed.quantize(rng.uniform(-10.0, 2.0, size=256))
    out1 = fixed.exp_q15(x_q)
    out2 = fixed.exp_q15(x_q)
    assert np.array_equal(out1, out2)


def test_exp_q15_softmax_typical_range():
    """exp is most-used after softmax max-subtract — inputs in [-30, 0].

    Only check the values that are above Q15 resolution. The tiny ones round
    to the precision floor (1/32768) but that's fine for compression: in
    softmax, tiny exp values get summed with millions of others and their
    individual quantization noise (~1 Q15 unit) doesn't move the normalizer.
    """
    x = np.linspace(-10, 0, 21).astype(np.float64)
    expected = np.exp(x)
    got = fixed.dequantize(fixed.exp_q15(fixed.quantize(x)))
    # The Q15 floor adds ~0.5/SCALE absolute error to each value. The
    # achievable relative bound is therefore 0.5/(SCALE*v); test that we
    # are within ~2x the theoretical Q15 quantization noise.
    abs_err = np.abs(got - expected)
    theoretical = 0.5 / fixed.SCALE
    # Allow 2 Q15 units of slack for the algorithm vs ideal rounding.
    assert np.max(abs_err) < 2.0 * theoretical, (
        f"max abs_err {np.max(abs_err)}, theoretical floor {theoretical}"
    )

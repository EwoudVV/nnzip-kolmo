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

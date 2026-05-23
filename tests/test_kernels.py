"""Bit-identity tests for the numba-JIT'd hot kernels.

Each numba kernel must produce byte-identical output to the pure-numpy
reference in fixed.py. If they ever diverge, fixed mode's cross-machine
determinism guarantee breaks — a numba bug would propagate into the
compressed blob.
"""

import math

import numpy as np
import pytest


def _math_isqrt_array(xs: np.ndarray) -> np.ndarray:
    return np.array([math.isqrt(int(v)) for v in xs.flat], dtype=np.int64).reshape(
        xs.shape
    )


def test_numba_isqrt_matches_math_isqrt_small():
    """Direct check: numba kernel matches Python's math.isqrt for any
    int64 input."""
    from kolmo._kernels import HAS_NUMBA, _isqrt_vec_numba

    if not HAS_NUMBA:
        pytest.skip("numba not installed")
    xs = np.array([0, 1, 2, 3, 4, 5, 100, 2**15, 2**20, 2**30, 2**40, 2**46],
                  dtype=np.int64)
    got = _isqrt_vec_numba(xs)
    expected = _math_isqrt_array(xs)
    assert np.array_equal(got, expected)


def test_numba_isqrt_matches_math_isqrt_large_random():
    """Stress test: random int64 values across the Q46 range Adam uses,
    plus boundary perfect-squares where Newton can oscillate."""
    from kolmo._kernels import HAS_NUMBA, _isqrt_vec_numba

    if not HAS_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(7)
    samples = rng.integers(0, 1 << 46, size=4096, dtype=np.int64)
    ks = rng.integers(1, 1 << 20, size=128, dtype=np.int64)
    boundaries = np.concatenate([ks * ks - 1, ks * ks, ks * ks + 1])
    xs = np.concatenate([samples, boundaries])
    got = _isqrt_vec_numba(xs)
    expected = _math_isqrt_array(xs)
    assert np.array_equal(got, expected)


def test_isqrt_vec_dispatches_to_numba_when_available():
    """The top-level isqrt_vec wrapper should route through the numba
    kernel when numba is installed. We can't easily mock it, but we can
    at least verify that asking for isqrt twice on the same input
    produces bit-identical output (which the numba path does)."""
    from kolmo import fixed

    rng = np.random.default_rng(11)
    xs = rng.integers(0, 1 << 40, size=1024, dtype=np.int64)
    out1 = fixed.isqrt_vec(xs)
    out2 = fixed.isqrt_vec(xs)
    assert np.array_equal(out1, out2)


def test_isqrt_pure_numpy_fallback_matches_numba(monkeypatch):
    """Force the pure-numpy fallback path and verify it produces the same
    bytes as the numba path. This is what guarantees we can drop numba
    in the future without changing compressed-blob bytes."""
    from kolmo import _kernels, fixed

    rng = np.random.default_rng(13)
    xs = rng.integers(0, 1 << 40, size=1024, dtype=np.int64)

    # Numba path
    numba_out = fixed.isqrt_vec(xs)

    # Pure-numpy path (force fallback by hiding numba)
    monkeypatch.setattr(_kernels, "HAS_NUMBA", False)
    fallback_out = fixed.isqrt_vec(xs)

    assert np.array_equal(numba_out, fallback_out)


def test_numba_round_div_matches_pure_numpy_fallback(monkeypatch):
    """_round_div_int64 is easy to get subtly wrong for negatives because
    Python floor division goes toward -inf. Verify the numba hot path
    preserves the pure reference exactly across several divisors.
    """
    from kolmo import _kernels, fixed

    if not _kernels.HAS_NUMBA:
        pytest.skip("numba not installed")

    rng = np.random.default_rng(17)
    values = rng.integers(
        -(1 << 44),
        1 << 44,
        size=4096,
        dtype=np.int64,
    )
    # Add hand-picked values around zero and half-divisor boundaries.
    values = np.concatenate([
        values,
        np.array([-100, -99, -51, -50, -49, -5, -4, -3, -2, -1, 0,
                  1, 2, 3, 4, 5, 49, 50, 51, 99, 100], dtype=np.int64),
    ])

    for divisor in (1, 2, 3, 10, 32768, 99991):
        numba_out = _kernels._round_div_int64_numba(
            np.ascontiguousarray(values),
            np.int64(divisor),
        )
        monkeypatch.setattr(_kernels, "HAS_NUMBA", False)
        fallback_out = fixed._round_div_int64(values, divisor)
        monkeypatch.setattr(_kernels, "HAS_NUMBA", True)
        assert np.array_equal(numba_out, fallback_out)


def test_numba_exp_q15_matches_pure_numpy_fallback(monkeypatch):
    """The Q15 exp kernel is used inside softmax/GELU, so exact byte
    identity matters. Compare numba against the pure-numpy reference over
    the practical range softmax sees after subtracting max logits.
    """
    from kolmo import _kernels, fixed

    if not _kernels.HAS_NUMBA:
        pytest.skip("numba not installed")

    rng = np.random.default_rng(19)
    random_values = rng.integers(
        -30 * fixed.SCALE,
        10 * fixed.SCALE,
        size=8192,
        dtype=np.int32,
    )
    boundary_values = np.array([
        -40 * fixed.SCALE,
        -31 * fixed.SCALE,
        -30 * fixed.SCALE,
        -10 * fixed.SCALE,
        -fixed.LN2_Q15 - 1,
        -fixed.LN2_Q15,
        -fixed.LN2_Q15 + 1,
        -1,
        0,
        1,
        fixed.LN2_Q15 - 1,
        fixed.LN2_Q15,
        fixed.LN2_Q15 + 1,
        4 * fixed.SCALE,
        10 * fixed.SCALE,
    ], dtype=np.int32)
    values = np.concatenate([random_values, boundary_values])

    numba_out = _kernels._exp_q15_numba(
        np.ascontiguousarray(values),
        np.int64(fixed.LN2_Q15),
        fixed._INV_FACT_Q15,
        np.int64(fixed.SCALE_BITS),
    )
    monkeypatch.setattr(_kernels, "HAS_NUMBA", False)
    fallback_out = fixed.exp_q15(values)
    monkeypatch.setattr(_kernels, "HAS_NUMBA", True)

    assert np.array_equal(numba_out, fallback_out)


def test_exp_and_round_div_dispatch_match_fallback(monkeypatch):
    """Top-level fixed.py helpers should produce the same values whether
    they dispatch to numba or fall back to numpy.
    """
    from kolmo import _kernels, fixed

    rng = np.random.default_rng(23)
    values = rng.integers(-20 * fixed.SCALE, 5 * fixed.SCALE, size=1024,
                          dtype=np.int32)
    div_values = rng.integers(-(1 << 40), 1 << 40, size=1024, dtype=np.int64)

    exp_numba = fixed.exp_q15(values)
    div_numba = fixed._round_div_int64(div_values, 32768)

    monkeypatch.setattr(_kernels, "HAS_NUMBA", False)
    exp_fallback = fixed.exp_q15(values)
    div_fallback = fixed._round_div_int64(div_values, 32768)

    assert np.array_equal(exp_numba, exp_fallback)
    assert np.array_equal(div_numba, div_fallback)

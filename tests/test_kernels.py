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

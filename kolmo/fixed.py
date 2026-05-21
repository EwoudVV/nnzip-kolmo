"""Q15 fixed-point arithmetic for deterministic neural network ops.

Every value is represented as int32 holding `round(x * 2^15)`. To get back
the float, divide by 2^15. Range is roughly ±65536 with ~3e-5 resolution.

Why this exists:
  PyTorch's float matmul is non-deterministic across CPU architectures because
  float addition isn't associative — different SIMD widths produce different
  reduction orders, and Mac M1 / Windows AVX disagree by ~1 ULP per intermediate.
  After many training steps these ULP differences compound past any rounding
  workaround.

  Integer addition IS associative. So the same input ints always produce the
  same output ints, no matter how the reduction is parallelized or which CPU
  runs it. Q15 fixed-point gives us float-like math with this guarantee.

Conventions:
  * x_q is the int32 encoded value: x_q = round(x * 2^15)
  * matmul of two Q15 tensors uses int64 accumulators to avoid overflow,
    then shifts right by 15 to bring the result back to Q15
  * we expose helpers (quantize, dequantize, matmul, add, mul_elementwise)
    that each leave their output in Q15 — the scale is invariant across the
    whole pipeline
"""

from __future__ import annotations

import numpy as np

# Q15 scale: shifting an integer left by SCALE_BITS converts it to fixed-point;
# shifting right brings it back. We choose 15 so int32 holds values up to
# 2^16 = 65536 in magnitude with 2^-15 ≈ 3e-5 resolution.
SCALE_BITS: int = 15
SCALE: int = 1 << SCALE_BITS  # 32768

# When we multiply two Q15 ints, the result has 2*SCALE_BITS bits of fractional
# precision. We shift right by SCALE_BITS to bring it back to Q15.
ROUND_OFFSET: int = 1 << (SCALE_BITS - 1)  # add this before shifting → round, not floor


def quantize(x: np.ndarray) -> np.ndarray:
    """Convert a float array to its Q15 int32 representation.

    Saturating: values outside ±2^16 get clipped to int32 min/max. In
    practice our neural net values stay in [-10, 10] so this never fires;
    the clip is defensive.
    """
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"quantize expects a float array, got {x.dtype}")
    scaled = np.rint(x.astype(np.float64) * SCALE)
    return np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)


def dequantize(x_q: np.ndarray) -> np.ndarray:
    """Convert a Q15 int32 array back to float64."""
    return x_q.astype(np.float64) / SCALE


def matmul(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """Q15 matmul: a_q @ b_q, returning Q15.

    a_q is (..., M, K) int32 in Q15. b_q is (..., K, N) int32 in Q15.
    Result is (..., M, N) int32 in Q15.

    Uses int64 accumulators to avoid overflow. NumPy's @ operator handles
    int64 matmul via BLAS — that's deterministic for integers regardless of
    threading because integer addition is associative.
    """
    if a_q.dtype != np.int32 or b_q.dtype != np.int32:
        raise TypeError(f"matmul expects int32 inputs, got {a_q.dtype}, {b_q.dtype}")
    acc = a_q.astype(np.int64) @ b_q.astype(np.int64)
    # Round-to-nearest before shift (instead of floor):
    rounded = (acc + ROUND_OFFSET) >> SCALE_BITS
    return rounded.astype(np.int32)


def add(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """Q15 + Q15 = Q15. Both inputs already same scale, so just add."""
    if a_q.dtype != np.int32 or b_q.dtype != np.int32:
        raise TypeError("add expects int32 inputs")
    return a_q + b_q


def mul(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """Elementwise Q15 * Q15 = Q15. Same trick as matmul: int64 product,
    shift back by SCALE_BITS."""
    if a_q.dtype != np.int32 or b_q.dtype != np.int32:
        raise TypeError("mul expects int32 inputs")
    prod = a_q.astype(np.int64) * b_q.astype(np.int64)
    rounded = (prod + ROUND_OFFSET) >> SCALE_BITS
    return rounded.astype(np.int32)


def neg(a_q: np.ndarray) -> np.ndarray:
    """Q15 unary negation."""
    return -a_q


def sub(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """Q15 subtraction."""
    return a_q - b_q

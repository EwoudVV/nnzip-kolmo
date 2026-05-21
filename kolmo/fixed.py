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


def isqrt_vec(x: np.ndarray) -> np.ndarray:
    """Integer square root of each element of x (int64), via Newton's method.

    Returns floor(sqrt(x)) as int64. Used by sqrt_q15 to compute LayerNorm
    denominator. Math.isqrt-equivalent but vectorized.

    Algorithm: classic Newton iteration s := (s + x/s) // 2, starting from
    a power-of-2 upper bound. Converges in <= 30 iterations for any int64.
    """
    x = np.asarray(x, dtype=np.int64)
    if np.any(x < 0):
        raise ValueError("isqrt_vec requires non-negative integers")

    # Initial estimate: shift right by half the bit-width of x.
    # For Q15 LayerNorm denominators, x is typically small enough that
    # a few iterations suffice, but we run a fixed number for determinism.
    s = np.where(x == 0, np.int64(0), np.int64(1))
    # Bring s to roughly sqrt(x) magnitude. floor(log2(x))//2 leftshift.
    # For typical values up to 2^31 (Q15 squared values), 16 iterations always
    # converges. Use a deterministic fixed-iteration count.
    for _ in range(32):
        # Avoid division by zero by guarding x > 0.
        # s_new = (s + x/s) // 2, but only for x > 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            quotient = np.where(s > 0, x // np.maximum(s, 1), x)
            s = np.where(x > 0, (s + quotient) // 2, np.int64(0))
    # After convergence, the iteration may oscillate by 1 between two values;
    # the smaller one is floor(sqrt(x)).
    # Final correction: while s*s > x, decrement.
    too_big = s * s > x
    s = np.where(too_big, s - 1, s)
    return s


def sqrt_q15(x_q: np.ndarray) -> np.ndarray:
    """Q15 square root.

    If x is in Q15 (so int x_q represents x = x_q / 2^15), then sqrt(x)
    in Q15 is sqrt(x_q / 2^15) * 2^15 = sqrt(x_q * 2^15). Compute the
    integer sqrt of (x_q << SCALE_BITS).

    Requires x_q >= 0 (caller's responsibility).
    """
    if x_q.dtype != np.int32:
        raise TypeError("sqrt_q15 expects int32")
    if np.any(x_q < 0):
        raise ValueError("sqrt_q15 requires non-negative inputs")
    scaled = x_q.astype(np.int64) << SCALE_BITS
    return isqrt_vec(scaled).astype(np.int32)


def reciprocal_q15(x_q: np.ndarray) -> np.ndarray:
    """Q15 reciprocal: returns 1/x in Q15.

    1/x in Q15 = (1/x) * 2^15 = 2^15 / x. To keep precision when x is in Q15
    (x_q = x * 2^15), compute (2^30) / x_q which gives (2^30) / (x * 2^15) =
    2^15 / x — exactly Q15 of 1/x. Use int64 to avoid 2^30 overflow.
    """
    if x_q.dtype != np.int32:
        raise TypeError("reciprocal_q15 expects int32")
    if np.any(x_q == 0):
        raise ValueError("reciprocal_q15 requires non-zero inputs")
    numerator = np.int64(1) << (2 * SCALE_BITS)
    return (numerator // x_q.astype(np.int64)).astype(np.int32)


def div_q15(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """Q15 division: a / b returned in Q15.

    (a / b) in Q15 = (a / b) * 2^15. If a, b are in Q15 (a_q = a*2^15, b_q = b*2^15),
    then (a_q / b_q) is dimensionless (no scale), so (a_q * 2^15) / b_q gives
    Q15 of (a/b). Use int64 for the (a_q << 15) product.
    """
    if a_q.dtype != np.int32 or b_q.dtype != np.int32:
        raise TypeError("div_q15 expects int32 inputs")
    if np.any(b_q == 0):
        raise ValueError("div_q15: zero denominator")
    numerator = a_q.astype(np.int64) << SCALE_BITS
    return (numerator // b_q.astype(np.int64)).astype(np.int32)


# Hard-coded constants in Q15 — these are exactly representable so they're
# the same on every machine. ln(2) and reciprocal-factorials for Taylor exp.
LN2_Q15: int = 22713  # round(0.6931471805599453 * 2^15) = 22713
# Reciprocals 1/k! stored in Q15. Used by exp Taylor series — pre-computed
# from math, then quantized via round(). Since these are constants in code,
# they're identical on every machine regardless of where they were derived.
_INV_FACT_Q15 = np.array([
    32768,   # 1/0! = 1.0
    32768,   # 1/1! = 1.0
    16384,   # 1/2! = 0.5
    5461,    # 1/3! ≈ 0.16667
    1365,    # 1/4! ≈ 0.04167
    273,     # 1/5! ≈ 0.00833
    46,      # 1/6! ≈ 0.001389
    7,       # 1/7! ≈ 0.0001984
    1,       # 1/8! ≈ 2.48e-5  (rounds to 1 in Q15)
], dtype=np.int64)


def exp_q15(x_q: np.ndarray) -> np.ndarray:
    """Q15 exp(x). Works for any real x (positive or negative) within
    practical range. For very negative x (below ~-22), output saturates to 0.
    For very positive x, output may overflow int32 — caller is responsible
    for keeping inputs sensible (typically softmax subtracts max first).

    Algorithm:
      1. Range-reduce: x = n*ln(2) + r where r is in [0, ln(2)).
      2. exp(r) ≈ sum_{k=0..8} r^k / k!  computed in fixed-point.
      3. exp(x) = 2^n * exp(r). For n >= 0 left-shift, for n < 0 right-shift.

    All operations are int64 → int32 with explicit scale management. No
    floats touched. Bit-identical on any machine.
    """
    if x_q.dtype != np.int32:
        raise TypeError("exp_q15 expects int32")

    x = x_q.astype(np.int64)

    # Range reduce: n = floor(x / ln(2)). Python's // already floors toward
    # -infinity, which is what we want.
    n = x // LN2_Q15
    r_q = (x - n * LN2_Q15).astype(np.int64)  # r in [0, ln(2)) in Q15

    # Compute exp(r) via Horner-like Taylor: result_q = 1 + r + r²/2 + ...
    # Each term is r^k / k! in Q15. To compute r^k cheaply, we iterate:
    #   term_k = (term_{k-1} * r_q) >> SCALE_BITS  -- new r^k in Q15
    # Then add term_k * (1/k!_q) >> SCALE_BITS.
    # To keep precision through the sum, accumulate in int64 with Q30 scale,
    # then shift back at end.
    acc = np.int64(1) << (2 * SCALE_BITS)  # 1.0 in Q30
    pow_r = np.full_like(r_q, np.int64(1) << SCALE_BITS)  # r^0 = 1 in Q15
    for k in range(1, len(_INV_FACT_Q15)):
        # pow_r := pow_r * r_q (Q15 * Q15 = Q30, shift back to Q15)
        # Use rounding shift for symmetry.
        prod = pow_r * r_q
        pow_r = (prod + (1 << (SCALE_BITS - 1))) >> SCALE_BITS
        # Term in Q30 = pow_r (Q15) * inv_fact (Q15)
        term = pow_r * _INV_FACT_Q15[k]
        acc = acc + term

    # acc is now exp(r) in Q30. Shift down to Q15 (with rounding).
    exp_r_q15 = (acc + (1 << (SCALE_BITS - 1))) >> SCALE_BITS

    # Now multiply by 2^n.
    # For n >= 0: result_q15 = exp_r_q15 << n. May overflow int32 — saturate.
    # For n < 0: result_q15 = round(exp_r_q15 / 2^|n|). For round-to-nearest,
    # add 2^(|n|-1) before the shift instead of plain floor-shift.
    max_int32 = np.int64(np.iinfo(np.int32).max)
    shift_n = np.minimum(np.abs(n), 31).astype(np.int64)
    round_offset = np.where(n < 0, np.int64(1) << np.maximum(shift_n - 1, 0), np.int64(0))
    result = np.where(
        n >= 0,
        np.where(
            n > 31,  # any left-shift past 31 bits overflows int32 — saturate
            max_int32,
            exp_r_q15 << shift_n,
        ),
        np.where(
            n < -31,  # very small exp values round to 0
            np.int64(0),
            (exp_r_q15 + round_offset) >> shift_n,
        ),
    )
    return np.clip(result, 0, max_int32).astype(np.int32)

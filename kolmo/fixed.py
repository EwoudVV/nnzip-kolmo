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


# ---------------------------------------------------------------------------
# Composite ops: softmax, layernorm, GELU, linear.
# These build on the primitives above (matmul, add, mul, exp, sqrt, div).
# All inputs and outputs are int32 in Q15 unless noted. Every op is
# bit-deterministic across machines because the underlying primitives are.
# ---------------------------------------------------------------------------


def softmax_q15(x_q: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis, in Q15.

    Steps (all in int32):
      1. subtract per-row max so exp argument is <= 0 (no overflow)
      2. exp_q15 of the shifted values (each in (0, 1] when represented in Q15)
      3. sum each row (int64 accumulator since 256 values * 32768 = 8.4M)
      4. divide each entry by the row sum, returning Q15 probabilities
    """
    if x_q.dtype != np.int32:
        raise TypeError("softmax_q15 expects int32")
    # Subtract row max — keeps exp argument <= 0.
    row_max = x_q.max(axis=-1, keepdims=True)
    shifted = (x_q - row_max).astype(np.int32)
    e = exp_q15(shifted)
    # Sum each row in int64 to prevent overflow on large alphabets.
    row_sum = e.astype(np.int64).sum(axis=-1, keepdims=True)
    # Guard against all-zero rows (shouldn't happen since max contributes 1.0).
    row_sum = np.maximum(row_sum, 1)
    # Per-entry: result_q15 = (e * SCALE) / row_sum. e is already Q15, multiply
    # by SCALE (left-shift 15) then divide → Q15 result.
    numerator = e.astype(np.int64) << SCALE_BITS
    result = (numerator + row_sum // 2) // row_sum  # round-to-nearest
    return result.astype(np.int32)


def layernorm_q15(
    x_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    eps_q: int = 1,
) -> np.ndarray:
    """LayerNorm over the last axis, in Q15.

    Computes: (x - mean) / sqrt(var + eps) * weight + bias
    All operations in integer math. eps_q is the epsilon in Q15 units
    (default 1 ≈ 3e-5, matches PyTorch's default 1e-5).
    """
    if x_q.dtype != np.int32 or weight_q.dtype != np.int32 or bias_q.dtype != np.int32:
        raise TypeError("layernorm_q15 expects int32 inputs")

    D = x_q.shape[-1]
    # Mean: sum then integer-divide by D. Use int64 to avoid overflow on
    # large rows.
    summed = x_q.astype(np.int64).sum(axis=-1, keepdims=True)
    mean = (summed + (D // 2)) // D  # round-to-nearest
    centered = x_q - mean.astype(np.int32)

    # Variance: mean of squared centered values. centered is Q15; squaring
    # gives Q30. We divide by D in Q30, then take sqrt to get back to Q15.
    sq = centered.astype(np.int64) ** 2  # Q30
    var_q30 = (sq.sum(axis=-1, keepdims=True) + (D // 2)) // D
    # Add epsilon (in Q15 → shift to Q30 by left-shift 15).
    var_with_eps_q30 = var_q30 + (np.int64(eps_q) << SCALE_BITS)
    # sqrt(var_q30) ≈ sqrt(var) * 2^15 = Q15. (Because sqrt(x * 2^30) = sqrt(x) * 2^15.)
    # Use isqrt_vec on the int64 array directly.
    stddev_q15 = isqrt_vec(var_with_eps_q30).astype(np.int32)
    stddev_q15 = np.maximum(stddev_q15, 1)  # avoid div-by-zero

    # Normalize: (x - mean) / stddev, then * weight + bias. All in Q15.
    # Use div_q15 element-wise (but stddev_q15 broadcasts).
    # We do this in chunks because div_q15 expects same-shape inputs.
    centered_shape = centered.shape
    stddev_broadcast = np.broadcast_to(stddev_q15, centered_shape).copy()
    normed = div_q15(centered, stddev_broadcast)
    scaled = mul(normed, np.broadcast_to(weight_q, centered_shape).copy())
    return add(scaled, np.broadcast_to(bias_q, centered_shape).copy())


# Polynomial coefficients for erf approximation (Abramowitz & Stegun 7.1.26).
# 1 - (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵) * exp(-x²) where t = 1/(1 + p*x).
# Stored as Q15 constants — exactly the same bytes on every machine.
_ERF_P_Q15 = int(round(0.3275911 * SCALE))  # 10737
_ERF_A1_Q15 = int(round(0.254829592 * SCALE))  # 8347
_ERF_A2_Q15 = int(round(-0.284496736 * SCALE))  # -9322
_ERF_A3_Q15 = int(round(1.421413741 * SCALE))  # 46577
_ERF_A4_Q15 = int(round(-1.453152027 * SCALE))  # -47617
_ERF_A5_Q15 = int(round(1.061405429 * SCALE))  # 34772

# Pre-computed 1/sqrt(2) in Q15 for GELU input scaling.
_INV_SQRT2_Q15 = int(round(0.7071067811865476 * SCALE))  # 23170


def erf_q15(x_q: np.ndarray) -> np.ndarray:
    """erf(x) in Q15 via Abramowitz & Stegun 7.1.26 polynomial.

    Output is in (-1, 1) so always fits in Q15 with room. Uses absolute
    value, then negates if input was negative.
    """
    if x_q.dtype != np.int32:
        raise TypeError("erf_q15 expects int32")
    sign = np.where(x_q < 0, -1, 1).astype(np.int32)
    abs_x = np.abs(x_q).astype(np.int32)
    # t = 1 / (1 + p * |x|)
    one_q15 = np.int32(SCALE)
    px = mul(abs_x, np.full_like(abs_x, _ERF_P_Q15))
    one_plus_px = add(np.full_like(abs_x, one_q15), px)
    t = div_q15(np.full_like(abs_x, one_q15), one_plus_px)

    # Polynomial: a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵, Horner form.
    poly = np.full_like(t, _ERF_A5_Q15)
    poly = add(mul(poly, t), np.full_like(t, _ERF_A4_Q15))
    poly = add(mul(poly, t), np.full_like(t, _ERF_A3_Q15))
    poly = add(mul(poly, t), np.full_like(t, _ERF_A2_Q15))
    poly = add(mul(poly, t), np.full_like(t, _ERF_A1_Q15))
    poly = mul(poly, t)

    # exp(-x²): square x then negate then exp.
    x_sq = mul(abs_x, abs_x)
    exp_neg_x_sq = exp_q15((-x_sq).astype(np.int32))

    # 1 - poly * exp(-x²)
    inner = mul(poly, exp_neg_x_sq)
    result = sub(np.full_like(abs_x, one_q15), inner)
    return (sign * result).astype(np.int32)


def gelu_q15(x_q: np.ndarray) -> np.ndarray:
    """GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))), in Q15.

    Matches torch.nn.GELU (exact, not the tanh approximation).
    """
    if x_q.dtype != np.int32:
        raise TypeError("gelu_q15 expects int32")
    # Scale input by 1/sqrt(2)
    scaled = mul(x_q, np.full_like(x_q, _INV_SQRT2_Q15))
    erf_val = erf_q15(scaled)
    # 1 + erf
    one_plus_erf = add(np.full_like(x_q, np.int32(SCALE)), erf_val)
    # 0.5 * x * (1 + erf) — use mul, then divide by 2 via right-shift with rounding.
    prod = mul(x_q, one_plus_erf)
    return ((prod + np.int32(1)) >> 1).astype(np.int32)


def linear_q15(
    x_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray | None = None,
) -> np.ndarray:
    """Linear layer: y = x @ weight.T + bias, in Q15.

    x_q: (..., in_features). weight_q: (out_features, in_features). bias_q:
    (out_features,) or None.

    Returns (..., out_features) int32 in Q15.
    """
    if x_q.dtype != np.int32 or weight_q.dtype != np.int32:
        raise TypeError("linear_q15 expects int32 inputs")
    y = matmul(x_q, weight_q.T)
    if bias_q is not None:
        if bias_q.dtype != np.int32:
            raise TypeError("linear_q15 bias must be int32")
        y = y + bias_q  # broadcasts over batch dims
    return y


def cross_entropy_grad_q15(logits_q: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Gradient of mean cross-entropy loss w.r.t. logits, in Q15.

    For each token row: grad = (softmax(logits) - one_hot(target)) / T
    where T is the number of rows/tokens. This matches PyTorch's default
    `F.cross_entropy(..., reduction="mean")` gradient.

    Returns an int32 Q15 array with the same shape as logits_q.
    """
    if logits_q.dtype != np.int32:
        raise TypeError("cross_entropy_grad_q15 expects int32 logits")
    if logits_q.ndim != 2:
        raise ValueError("cross_entropy_grad_q15 expects a 2-D logits array")
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (logits_q.shape[0],):
        raise ValueError("targets shape must match logits rows")

    grad = softmax_q15(logits_q).astype(np.int64)
    rows = np.arange(logits_q.shape[0])
    grad[rows, targets] -= SCALE

    # Mean reduction over token rows, with symmetric rounding.
    t = logits_q.shape[0]
    signs = np.where(grad >= 0, 1, -1)
    rounded = (np.abs(grad) + (t // 2)) // t
    return (signs * rounded).astype(np.int32)

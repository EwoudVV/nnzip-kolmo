"""Numba-JIT'd hot kernels for the fixed-point engine.

Why numba: profiling showed the remaining fixed-mode hotspots after the
float64-BLAS matmul win are isqrt_vec, exp_q15, and _round_div_int64 —
all element-wise integer math that numpy's vectorized ops handle with
significant per-iteration Python overhead. Numba compiles these inner
loops to LLVM, eliminating the overhead while keeping the integer-only
semantics (bit-determinism preserved).

Determinism:
- All kernels use ONLY integer arithmetic (no float math)
- numba's @njit on int64 ops behaves like C int64 — same answer on every
  platform, every thread count, every SIMD width
- We verify cross-machine bit-identity via the existing CI hash probes
  (hash_q15_matmul, hash_fixed_forward, hash_fixed_training,
  hash_fixed_compress); they must produce byte-identical SHAs before
  and after each kernel lands

Fallback:
- If numba isn't installed (or the JIT fails to compile), we fall back
  to the pure-numpy implementations in fixed.py. The fast path is opt-in
  via `HAS_NUMBA = True`. Tests verify both paths give bit-identical
  output.
"""

from __future__ import annotations

import numpy as np

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @numba.njit(cache=True, boundscheck=False)
    def _isqrt_vec_numba(x: np.ndarray) -> np.ndarray:
        """Element-wise integer floor(sqrt(x)) for int64 inputs.

        Per-element Newton's method, seeded with `2^ceil(bit_length(x)/2)`
        — within sqrt(2) of true sqrt for any x, so 8 iterations always
        converge. Pure integer arithmetic, no float, no library calls.
        """
        out = np.empty_like(x)
        flat = x.ravel()
        flat_out = out.ravel()
        for i in range(flat.size):
            v = flat[i]
            if v <= 0:
                flat_out[i] = 0
                continue
            # bit length: count leading bits
            bl = 0
            v2 = v
            while v2 > 0:
                v2 >>= 1
                bl += 1
            s = np.int64(1) << ((bl + 1) // 2)
            # Newton iterations
            for _ in range(8):
                s = (s + v // s) // 2
            # Correction: Newton may land at floor(sqrt(x)) + 1
            if s * s > v:
                s -= 1
            flat_out[i] = s
        return out

    def isqrt_vec(x: np.ndarray) -> np.ndarray:
        """Numba-JIT'd integer sqrt. Drop-in replacement for the pure-numpy
        version in fixed.py."""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
        if np.any(x < 0):
            raise ValueError("isqrt_vec requires non-negative integers")
        return _isqrt_vec_numba(x)


if HAS_NUMBA:

    @numba.njit(cache=True, boundscheck=False)
    def _round_div_int64_numba(values: np.ndarray, divisor: np.int64) -> np.ndarray:
        """Round-half-away-from-zero integer division. See the gotcha doc
        on fixed._round_div_int64 — the seemingly natural sign+offset
        rewrite is wrong for negatives because of floor-div semantics.
        We do the division on |v| (always positive) and reapply sign,
        which avoids the gotcha by construction.
        """
        out = np.empty_like(values)
        flat = values.ravel()
        flat_out = out.ravel()
        half = divisor // 2
        for i in range(flat.size):
            v = flat[i]
            if v >= 0:
                flat_out[i] = (v + half) // divisor
            else:
                flat_out[i] = -(((-v) + half) // divisor)
        return out


if HAS_NUMBA:

    @numba.njit(cache=True, boundscheck=False)
    def _exp_q15_numba(
        x_q: np.ndarray,
        ln2_q15: np.int64,
        inv_fact_q15: np.ndarray,
        scale_bits: np.int64,
    ) -> np.ndarray:
        """Per-element Q15 exp via Taylor series with range reduction.

        Bit-identical to the pure-numpy version: same range reduction
        (`n = x // ln2_q15` with Python/numba floor div semantics), same
        Taylor coefficients, same intermediate Q30 accumulator, same
        round-half-up shifts.
        """
        out = np.empty(x_q.shape, dtype=np.int32)
        flat = x_q.ravel()
        flat_out = out.ravel()
        max_int32 = np.int64((1 << 31) - 1)
        round_q15 = np.int64(1) << (scale_bits - 1)
        round_q30 = np.int64(1) << (scale_bits - 1)
        one_q15 = np.int64(1) << scale_bits
        one_q30 = np.int64(1) << (2 * scale_bits)
        n_terms = inv_fact_q15.size
        for i in range(flat.size):
            x = np.int64(flat[i])
            # Range reduce: n = floor(x / ln2_q15)
            n = x // ln2_q15
            r_q = x - n * ln2_q15
            # Taylor: acc in Q30, pow_r in Q15
            acc = one_q30
            pow_r = one_q15
            for k in range(1, n_terms):
                prod = pow_r * r_q
                pow_r = (prod + round_q15) >> scale_bits
                acc += pow_r * inv_fact_q15[k]
            exp_r_q15 = (acc + round_q30) >> scale_bits
            # Multiply by 2^n with overflow saturation
            if n >= 0:
                if n > 31:
                    result = max_int32
                else:
                    result = exp_r_q15 << n
            elif n < -31:
                result = np.int64(0)
            else:
                shift = -n
                offset = np.int64(1) << (shift - 1)
                result = (exp_r_q15 + offset) >> shift
            if result > max_int32:
                result = max_int32
            if result < 0:
                result = np.int64(0)
            flat_out[i] = np.int32(result)
        return out


if not HAS_NUMBA:
    # Keep these names importable so the optional-numba fallback path can
    # check HAS_NUMBA without tripping over missing attributes.
    _isqrt_vec_numba = None
    isqrt_vec = None
    _round_div_int64_numba = None
    _exp_q15_numba = None


def warmup() -> None:
    """Trigger numba JIT compile up front so the first compression call
    doesn't pay the ~800ms compile cost. Idempotent — calling twice is
    cheap after the first.
    """
    if not HAS_NUMBA:
        return
    _isqrt_vec_numba(np.array([1, 4, 100], dtype=np.int64))
    _round_div_int64_numba(np.array([1, -3, 100], dtype=np.int64), np.int64(10))
    # exp warmup needs the LN2 constant + inv_fact table; defer to caller
    # that already has them (kolmo.fixed module-level).

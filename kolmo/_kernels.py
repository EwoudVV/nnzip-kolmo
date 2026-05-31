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


if HAS_NUMBA:

    @numba.njit(cache=True, boundscheck=False)
    def _fused_adam_step_numba(
        params: np.ndarray,            # int32, modified in place
        grads: np.ndarray,             # int32
        m: np.ndarray,                  # int64, modified in place
        v: np.ndarray,                  # int64, modified in place
        beta1_num: np.int64,
        beta1_den: np.int64,
        beta2_num: np.int64,
        beta2_den: np.int64,
        one_minus_b1: np.int64,
        one_minus_b2: np.int64,
        lr_num: np.int64,
        lr_den: np.int64,
        eps_q15: np.int32,
    ) -> None:
        """Fused per-element Adam update.

        Bit-for-bit equivalent to the pure-numpy `fixed_adam_step`. Eliminates
        the ~17 temporary numpy arrays the python path allocates per tensor
        per step, and the corresponding per-op dispatch overhead. The math
        is intentionally NOT simplified — every operation uses the same
        rounding (round-half-away-from-zero), the same int32 wraparound at
        the same points, and the same Q-scale conventions as the numpy
        version, so cross-machine bit-identity is preserved.

        Conventions:
          M_EXTRA_BITS = 16   m stored as Q31 (Q15 + 16 guard)
          V_EXTRA_BITS = 16   v stored as Q46 (Q30 + 16 guard)
          divisors of round_div are powers of two when possible, but the
          beta1_den / beta2_den / one_minus_b{1,2} / lr_den paths require
          general division
        """
        # Constants — kept inside the kernel so numba constant-folds them.
        M_EXTRA = np.int64(16)
        V_EXTRA = np.int64(16)
        V_HALF_DIV_BITS = np.int64(8)  # V_EXTRA // 2 — used for denom shrink
        max_shiftable = np.int64((1 << 63) - 1) >> V_EXTRA
        max_bias_correctable = np.int64((1 << 63) - 1) >> 30
        int32_max = np.int64((1 << 31) - 1)
        int32_min = np.int64(-(1 << 31))

        # Pre-compute round-half offsets (constant for the loop)
        half_b1d = beta1_den // np.int64(2)
        half_b2d = beta2_den // np.int64(2)
        half_omb1 = one_minus_b1 // np.int64(2)
        half_omb2 = one_minus_b2 // np.int64(2)
        half_lrden = lr_den // np.int64(2)
        half_m_extra = np.int64(1) << (M_EXTRA - np.int64(1))     # 1 << 15
        half_v_half = np.int64(1) << (V_HALF_DIV_BITS - np.int64(1))  # 1 << 7

        flat_p = params.ravel()
        flat_g = grads.ravel()
        flat_m = m.ravel()
        flat_v = v.ravel()
        n = flat_p.size
        eps_i64 = np.int64(eps_q15)

        for i in range(n):
            g_i = np.int64(flat_g[i])
            m_old_i = flat_m[i]
            v_old_i = flat_v[i]

            # --- m_new = round_div(beta1_num * m_old + (1-beta1)*g_q31, beta1_den)
            g_q31 = g_i << M_EXTRA
            m_num = beta1_num * m_old_i + (beta1_den - beta1_num) * g_q31
            if m_num >= 0:
                m_new = (m_num + half_b1d) // beta1_den
            else:
                m_new = -(((-m_num) + half_b1d) // beta1_den)

            # --- v_new = round_div(beta2_num * v_old + (1-beta2)*g2_q46, beta2_den)
            g2 = g_i * g_i  # non-negative
            if g2 > max_shiftable:
                g2 = max_shiftable
            g2_q46 = g2 << V_EXTRA
            v_num = beta2_num * v_old_i + (beta2_den - beta2_num) * g2_q46
            if v_num >= 0:
                v_new = (v_num + half_b2d) // beta2_den
            else:
                v_new = -(((-v_num) + half_b2d) // beta2_den)

            # --- m_hat (Q15 int32)
            m_for_hat = m_new
            if m_for_hat > max_bias_correctable:
                m_for_hat = max_bias_correctable
            elif m_for_hat < -max_bias_correctable:
                m_for_hat = -max_bias_correctable
            m_hat_num = m_for_hat << 30
            if m_hat_num >= 0:
                m_hat_q31 = (m_hat_num + half_omb1) // one_minus_b1
            else:
                m_hat_q31 = -(((-m_hat_num) + half_omb1) // one_minus_b1)
            # round_div(m_hat_q31, 1 << 16) then int32 cast (wraps on overflow)
            if m_hat_q31 >= 0:
                m_hat_i64 = (m_hat_q31 + half_m_extra) >> M_EXTRA
            else:
                m_hat_i64 = -(((-m_hat_q31) + half_m_extra) >> M_EXTRA)
            m_hat = np.int32(m_hat_i64)

            # --- v_hat / denom
            v_for_hat = v_new
            if v_for_hat > max_bias_correctable:
                v_for_hat = max_bias_correctable
            # v is non-negative; no lower clamp matches np.minimum behaviour.
            v_hat_num = v_for_hat << 30
            if v_hat_num >= 0:
                v_hat_q46 = (v_hat_num + half_omb2) // one_minus_b2
            else:
                v_hat_q46 = -(((-v_hat_num) + half_omb2) // one_minus_b2)

            # isqrt(max(v_hat_q46, 0))  — same algorithm as _isqrt_vec_numba
            sqrt_in = v_hat_q46 if v_hat_q46 > 0 else np.int64(0)
            if sqrt_in == 0:
                denom_q23 = np.int64(0)
            else:
                bl = np.int64(0)
                v2 = sqrt_in
                while v2 > 0:
                    v2 >>= 1
                    bl += 1
                s = np.int64(1) << ((bl + 1) // 2)
                # 8 Newton iterations always converge for any int64 v
                for _ in range(8):
                    s = (s + sqrt_in // s) // 2
                if s * s > sqrt_in:
                    s -= 1
                denom_q23 = s

            if denom_q23 >= 0:
                denom_i64 = (denom_q23 + half_v_half) >> V_HALF_DIV_BITS
            else:
                denom_i64 = -(((-denom_q23) + half_v_half) >> V_HALF_DIV_BITS)
            # int32 cast then add eps_q15 — matches the numpy version's two
            # casts. Both wrap on int32 overflow; for Adam values this never
            # fires in practice.
            denom_i32 = np.int32(denom_i64) + np.int32(eps_i64)

            # --- ratio = div_q15(m_hat, denom): (m_hat << 15) // denom, int32
            # denom_i32 != 0 because eps_q15 >= 1 in fixed mode.
            ratio_i64 = (np.int64(m_hat) << 15) // np.int64(denom_i32)
            ratio = np.int32(ratio_i64)

            # --- update = round_div(ratio * lr_num, lr_den) as int32
            update_num = np.int64(ratio) * lr_num
            if update_num >= 0:
                update_i64 = (update_num + half_lrden) // lr_den
            else:
                update_i64 = -(((-update_num) + half_lrden) // lr_den)
            update = np.int32(update_i64)

            # --- params <- params - update, cast back to int32 (wraps)
            new_p = np.int64(flat_p[i]) - np.int64(update)
            flat_p[i] = np.int32(new_p)
            flat_m[i] = m_new
            flat_v[i] = v_new


if not HAS_NUMBA:
    # Keep these names importable so the optional-numba fallback path can
    # check HAS_NUMBA without tripping over missing attributes.
    _isqrt_vec_numba = None
    isqrt_vec = None
    _round_div_int64_numba = None
    _exp_q15_numba = None
    _fused_adam_step_numba = None


def warmup() -> None:
    """Trigger numba JIT compile up front so the first compression call
    doesn't pay the ~800ms compile cost. Idempotent — calling twice is
    cheap after the first.
    """
    if not HAS_NUMBA:
        return
    _isqrt_vec_numba(np.array([1, 4, 100], dtype=np.int64))
    _round_div_int64_numba(np.array([1, -3, 100], dtype=np.int64), np.int64(10))
    # Tiny Adam warmup so the first real fixed-mode training step doesn't
    # pay the JIT compile cost. All scalars matter for type-inference; use
    # the same shapes/types as the real call.
    _p = np.zeros(8, dtype=np.int32)
    _g = np.zeros(8, dtype=np.int32)
    _m = np.zeros(8, dtype=np.int64)
    _v = np.zeros(8, dtype=np.int64)
    _fused_adam_step_numba(
        _p, _g, _m, _v,
        np.int64(9), np.int64(10),
        np.int64(999), np.int64(1000),
        np.int64((1 << 30) - (1 << 30) * 9 // 10),
        np.int64((1 << 30) - (1 << 30) * 999 // 1000),
        np.int64(1), np.int64(1000),
        np.int32(1),
    )
    # exp warmup needs the LN2 constant + inv_fact table; defer to caller
    # that already has them (kolmo.fixed module-level).

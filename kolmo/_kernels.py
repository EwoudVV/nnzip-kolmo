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


def warmup() -> None:
    """Trigger numba JIT compile up front so the first compression call
    doesn't pay the ~800ms compile cost. Idempotent — calling twice is
    cheap after the first.
    """
    if not HAS_NUMBA:
        return
    _isqrt_vec_numba(np.array([1, 4, 100], dtype=np.int64))

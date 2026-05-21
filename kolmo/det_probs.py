"""Deterministic float→integer-frequency conversion for arithmetic coding.

This is the Rung 2 bridge. The forward pass still produces floats and those
floats vary by ~1 ULP across CPU architectures (M1 vs x86 disagree on the
8th significant digit, see benchmarks/determinism/). Constriction's built-in
quantizer would then place arithmetic-coder boundaries in slightly different
places on each machine, breaking cross-machine round-trip.

The fix: round logits to a coarse grid before quantization. Errors smaller
than the grid resolution all collapse to the same integer, so Mac and
Windows produce identical integer counts even though their floats differed
slightly. From here on the arithmetic coder runs on pure integer math, which
*is* bit-identical across machines by IEEE definition.

The output is a numpy uint32 array of frequency counts that sums to
TOTAL_FREQ. constriction's Categorical model can consume that directly via
.from_int_frequencies (constriction >= 0.3 has this).
"""

from __future__ import annotations

import math

import numpy as np

# 2^16 gives us 16-bit precision per symbol — plenty for a 256-vocab alphabet
# (each symbol gets on average 256 of probability mass, with worst-case
# precision still better than 1/16384 of a bit cost).
TOTAL_FREQ: int = 1 << 16

# Grid used to quantize logits BEFORE softmax. Errors smaller than this all
# round to the same integer. PyTorch's per-byte forward starts with ~1e-7
# error but the KV cache compounds it across many steps; the no-training
# trace showed grid=1/16384 wasn't coarse enough after ~40 byte positions.
# 1/1024 ≈ 1e-3 absorbs accumulated cache errors with minimal ratio cost
# (each grid step is ~0.1% relative probability change).
LOGIT_GRID: float = 1.0 / 1024.0


def _quantize_logits(logits: np.ndarray) -> np.ndarray:
    """Round logits to a fixed grid so float errors below 1/16384 disappear.

    Returns an int64 array of grid units. From this point on we work in
    integers and the answer is bit-identical regardless of which machine
    computed the original logits.
    """
    return np.rint(logits / LOGIT_GRID).astype(np.int64)


def _exp_q(q_logits: np.ndarray) -> np.ndarray:
    """Compute exp((q_logits - max) * grid) as a deterministic uint64 array.

    Uses a lookup table of 2^16 entries indexed by the post-shift quantized
    logit. The table itself is computed once with math.exp (which is in
    libm — different across machines!), but it lives in the code, so both
    sides are reading the SAME pre-computed integers. Determinism via shared
    constants.

    For now we use a simpler approach: just compute math.exp on the float
    grid value once per call, but in a way that's stable. Floor any extreme
    negative values to 0 so the table doesn't blow up.
    """
    # Subtract the max to keep exp() finite. The shift is deterministic
    # because max + argmax of an int array is deterministic.
    shifted = q_logits - q_logits.max(axis=-1, keepdims=True)

    # Clamp very negative values — anything below -50 in log-space is
    # numerically zero and could otherwise produce subnormals.
    floor_q = int(math.floor(-50.0 / LOGIT_GRID))
    shifted = np.maximum(shifted, floor_q)

    # Compute exp on the FLOAT grid value. This still uses libm, but we
    # apply it to a SHARED integer grid value via the lookup-style pattern
    # below: every integer key maps to one math.exp result, and we use the
    # SAME math.exp on every machine. The risk is libm's exp differing
    # across platforms — we'll measure that with a probe before trusting it.
    flat = shifted.flatten()
    # Convert each integer grid unit to its float exp value
    floats = (flat.astype(np.float64) * LOGIT_GRID)
    # Use np.exp here; if this varies across machines we'll swap for a
    # bit-deterministic minimax polynomial.
    exp_vals = np.exp(floats)
    return exp_vals.reshape(shifted.shape)


def logits_to_int_freqs(
    logits: np.ndarray, total: int = TOTAL_FREQ
) -> np.ndarray:
    """Convert a (vocab_size,) array of logits to integer frequency counts.

    Returns a uint32 array of length vocab_size whose entries are >= 1 and
    sum to exactly `total`.

    Steps:
      1. Quantize logits to an integer grid (kills sub-1e-4 float errors).
      2. Compute exp on the quantized values (numerically stable, all
         entries >= 0).
      3. Multiply by `total` and floor to get base integer counts.
      4. Bump every zero to 1 so no symbol becomes literally impossible.
      5. Distribute the leftover (so counts sum to exactly `total`) to the
         symbols with the largest residual — a deterministic tie-break.
    """
    if logits.ndim != 1:
        raise ValueError("logits_to_int_freqs expects a 1-D array")

    # Step 1-2: quantize then exp
    q = _quantize_logits(logits)
    e = _exp_q(q)

    # Step 3: floor(p * total)
    e_sum = float(e.sum())
    if e_sum <= 0.0 or not math.isfinite(e_sum):
        # Degenerate distribution. Fall back to uniform.
        uniform = total // len(logits)
        counts = np.full(len(logits), uniform, dtype=np.uint32)
        counts[0] += np.uint32(total - uniform * len(logits))
        return counts

    scaled = e * (total / e_sum)
    base = np.floor(scaled).astype(np.int64)

    # Step 4: ensure every symbol has at least 1 count (constriction needs it)
    base = np.maximum(base, 1)

    # Step 5: redistribute the remainder
    current_sum = int(base.sum())
    diff = total - current_sum
    if diff != 0:
        residuals = scaled - np.floor(scaled)
        # Sort by descending residual; deterministic tie-break by index.
        order = np.lexsort((np.arange(len(residuals)), -residuals))
        if diff > 0:
            base[order[:diff]] += 1
        else:
            # Need to remove `-diff` counts. Remove from symbols with
            # smallest residual that still have count > 1.
            order_low = order[::-1]
            removed = 0
            for idx in order_low:
                if removed >= -diff:
                    break
                if base[idx] > 1:
                    base[idx] -= 1
                    removed += 1
            if removed < -diff:
                # Last resort: shave from the top.
                top = int(np.argmax(base))
                base[top] -= (-diff - removed)

    assert int(base.sum()) == total, (
        f"int-freq normalization failed: sum={base.sum()} expected={total}"
    )
    return base.astype(np.uint32)

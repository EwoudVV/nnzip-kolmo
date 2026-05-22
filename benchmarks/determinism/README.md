# Determinism testing

Tools for verifying cross-machine bit-identity. This is the Rung 2 acceptance
test — when kolmo produces the same bytes on Mac and Windows (and Linux) for
the same input, we have a deterministic compressor.

## Current state — Rung 2

The PyTorch path can't get there alone. Per-machine SIMD differences inside
matmul/softmax produce ~1-ULP intermediate values even when the input weights
are byte-identical, and over hundreds of training steps that drift compounds
past any gradient-rounding grid we tried.

The fix was Path 2 (fixed-point arithmetic): swap the entire forward + backward
+ optimizer for Q15 integer math. Integer addition is associative, so
multi-threaded reductions produce the same answer regardless of SIMD width.
The work landed across `kolmo/fixed.py`, `kolmo/fixed_model.py`,
`kolmo/fixed_optim.py`, `kolmo/fixed_train.py`, and `kolmo/fixed_kv_cache.py`,
gated behind `KOLMO_FIXED=1`.

| Probe | Status |
|---|---|
| Stable initializer (cross-machine) | ✅ |
| Q15 integer matmul | ✅ (`hash_q15_matmul.py`) |
| Q15 transformer forward | ✅ (`hash_fixed_forward.py`) |
| Q15 manual backward — matches PyTorch autograd within 0.001 | ✅ |
| Q15 Adam optimizer with Q31/Q46 guard bits | ✅ |
| Multi-block fixed-point training trajectory | ✅ (`hash_fixed_training.py`) |
| Fixed-mode KV cache (warm + step) | ✅ bit-identical to `fixed_forward` |
| End-to-end fixed-mode compress blob | ✅ (`hash_fixed_compress.py`) |
| Cross-OS verification on every commit | ✅ (`.github/workflows/determinism.yml`) |

GitHub Actions runs all four hashes (`hash_q15_matmul`, `hash_fixed_forward`,
`hash_fixed_training`, `hash_fixed_compress`) on macos-latest, windows-latest,
and ubuntu-latest. The `compare-determinism` job fails the workflow if any
runner's transcript doesn't match the others. That's the cross-machine
acceptance test, automated.

## Cost of bulletproof

| Metric | PyTorch path | Fixed-point path |
|---|---|---|
| Cross-machine bit-identity | ❌ | ✅ |
| Inference per byte (post KV-cache) | ~0.04s | ~0.2s |
| Training step (per block) | ~0.05s | ~2s |
| Compression ratio on warmed model | baseline | within ~0.01 of baseline |
| Compression ratio with random init | poor (overconfident misses) | better (Q15 clamps confidence) |

The fixed path is ~40× slower than PyTorch — most of that is the Python
overhead of numpy int64 ops without a JIT. Going faster needs C extensions,
which would add a build step.

## Scripts

- `hash_q15_matmul.py` — hash an int matmul. Foundation claim for fixed-point.
- `hash_fixed_forward.py` — hash int32 logits from a Q15 transformer forward.
- `hash_fixed_training.py` — hash weights + Adam state after a few training blocks.
- `hash_fixed_compress.py` — hash a `KOLMO_FIXED=1` compress blob end-to-end.
- `hash_compress.py` — hash a PyTorch-path compress blob (will not match cross-OS).
- `make_blob.py PATH` — save a compressed blob to PATH.
- `try_decompress.py PATH` — try to decompress, report whether output matches expected.

The older probes (`hash_after_warmup`, `hash_model_state`, `hash_no_training`,
`hash_numpy_forward`, `hash_pytorch_freqs`, `hash_int_freqs`, `measure_train_drift`)
were diagnostic artifacts from finding the original fault line — kept for the
historical record.

## Manual acceptance test

CI handles this on every push, but the manual version is still:

```
# On machine A:
KOLMO_FIXED=1 python benchmarks/determinism/make_blob.py blob.kmo

# Copy blob.kmo to machine B (any architecture, any OS).

# On machine B:
KOLMO_FIXED=1 python benchmarks/determinism/try_decompress.py blob.kmo
# Should print: ROUND-TRIP OK
```

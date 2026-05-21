# Determinism testing

Tools for verifying cross-machine bit-identity. This is the Rung 2 acceptance
test — when kolmo produces the same bytes on Mac and Windows for the same
input, we have a deterministic compressor.

## Current state (Rung 1)

| Test | Mac CPU | Windows CPU | Windows CUDA |
|---|---|---|---|
| Within-machine | ✅ | ✅ | ✅ |
| Match Mac CPU | — | ❌ | ❌ |

The original fault line was **CPU architecture** (Apple Silicon vs x86), not
the device. PyTorch CPU and CUDA matched on Windows, but Mac CPU differed.

Rung 2 progress:

| Probe | Result |
|---|---|
| Stable initializer | ✅ Mac and Windows produce identical initial weights |
| PyTorch seed warmup | ❌ weights diverge after training on the built-in seed corpus |
| NumPy forward pass | ❌ raw logits are close but not byte-identical across Mac and Windows |
| Quantized NumPy logits | ✅ logits rounded to 1e-4 match on the tiny probe |

This means NumPy is useful as a readable stepping stone, but raw floating-point
NumPy is not enough for prize-grade determinism. The promising intermediate
path is quantized activations/logits plus deterministic integer probability
conversion. If that fails, the final path is full fixed-point
forward/backward/optimizer math.

## Scripts

- `hash_compress.py` — compress a fixed input, print sha256. Run on multiple machines, compare hashes.
- `hash_model_state.py` — hash PyTorch default init, stable init, and post-seed-warmup weights.
- `hash_numpy_forward.py` — hash a pure-NumPy forward pass from stable weights.
- `make_blob.py PATH` — save a compressed blob to PATH.
- `try_decompress.py PATH` — try to decompress, report whether output matches expected.

## Acceptance test

When Rung 2 is done, this round-trip should succeed:

```
# On machine A:
python benchmarks/determinism/make_blob.py blob.kmo

# Copy blob.kmo to machine B (any architecture, any OS).

# On machine B:
python benchmarks/determinism/try_decompress.py blob.kmo
# Should print: ROUND-TRIP OK
```

Today it prints `DECOMPRESS RAN but output DIFFERS` because PyTorch float
math diverges between machines.

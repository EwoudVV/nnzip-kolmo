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
| NumPy forward pass | ❌ raw logits close (~1e-7 max diff) but not byte-identical |
| Quantized NumPy logits at 1e-4 grid | ✅ logits match on tiny model |
| Quantized PyTorch logits at 1e-4 grid | ❌ accumulated KV-cache drift exceeds grid by token ~40 |
| Quantized PyTorch logits at **1/64 grid** | ✅ holds for full inference path |
| **No-training compress** with 1/64 grid | ✅ Mac and Windows produce identical compressed bytes |
| Single-step training with grad+weight rounding | ✅ identical weights after one step |
| Multi-step seed warmup | ❌ state diverges over ~348 steps |
| Full compress (with training) | ❌ cross-decompress fails |

**Today's wins:**

1. **Deterministic inference path.** `kolmo/det_probs.py` converts logits to integer frequencies on a 1/64 grid. The grid is much coarser than PyTorch's accumulated float drift, so Mac and Windows round to identical integer counts. The arithmetic coder then runs on pure integer math (`Categorical(probs, perfect=True)`), giving bit-identical encoding. Verified end-to-end with a no-training compress probe.
2. **Deterministic single-step training.** `train_block` rounds gradients to 1/8192 and weights+Adam-state to 1/16384 after each step. Starting from identical state, both machines produce identical post-step state.

**What still diverges:**

Multi-step training. Even though each step is internally deterministic given identical input state, PyTorch's forward pass has SIMD-width-dependent float ops that produce ~1-ULP differences in intermediate values (logits, softmax, attention) even when the model weights themselves are byte-identical. Over hundreds of training steps, the rare cases where this ULP drift lands near a gradient-rounding boundary cause divergence to compound.

**Next path:** replace the PyTorch forward in `train_block` with a fully-deterministic implementation. Two options:

1. **NumPy forward with single-thread BLAS.** `kolmo/np_model.py` already exists and matches PyTorch within float precision; pin BLAS to one thread and verify cross-machine bit-identity at the intermediate-value level (not just logits).
2. **Fixed-point forward.** Represent weights as int32 with a known scale factor; reimplement matmul/layernorm/attention in integer math. Bulletproof but several days of work.

Path 1 is the cheaper first attempt.

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

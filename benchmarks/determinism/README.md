# Determinism testing

Tools for verifying cross-machine bit-identity. This is the Rung 2 acceptance
test — when kolmo produces the same bytes on Mac and Windows for the same
input, we have a deterministic compressor.

## Current state (Rung 1)

| Test | Mac CPU | Windows CPU | Windows CUDA |
|---|---|---|---|
| Within-machine | ✅ | ✅ | ✅ |
| Match Mac CPU | — | ❌ | ❌ |

The fault line is **CPU architecture** (Apple Silicon vs x86), not the device.
PyTorch float operations diverge between architectures.

## Scripts

- `hash_compress.py` — compress a fixed input, print sha256. Run on multiple machines, compare hashes.
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

"""Measure the compression-ratio cost of running kolmo in fixed-point mode.

The Q15 fixed-point engine is bit-deterministic across machines but uses
coarser arithmetic than PyTorch float32. That coarseness should cost some
compression ratio. This benchmark quantifies how much.

Both modes run with KOLMO_SKIP_PRIME=1 — same random starting weights, same
payload, just different arithmetic. (With prime, fixed mode would spend
several minutes warming up on the seed corpus before touching the actual
input, and the comparison would be dominated by prime time.)

Observed result on a 246-byte English snippet (May 2026):
    pytorch  246 -> 432  ratio 1.76   (model is wildly miscalibrated;
                                       confident wrong predictions cost
                                       many bits each.)
    fixed    246 -> 184  ratio 0.75   (Q15 quantization clamps the worst
                                       miscalibrations, so random-init
                                       failures cost less.)

Counterintuitive read: in the *skip-prime* regime, the coarser arithmetic
actually helps — it bounds how confident the random-init model can be about
wrong predictions, so arithmetic coding pays less for misses. With prime the
gap inverts (PyTorch's accurate predictions outperform the slightly-noisy
quantized version), but the delta in either direction stays small.

The numbers here aren't the absolute compression ratio — they're the
*delta* between the two arithmetic backends.
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(line_buffering=True)


def run_mode(label: str, fixed: bool, data: bytes) -> None:
    # Each run forks a fresh subprocess so KOLMO_FIXED can flip without
    # carrying cached state from the previous mode.
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["KOLMO_FIXED"] = "1" if fixed else "0"
    env["KOLMO_SKIP_PRIME"] = "1"

    script = """
import sys, time, hashlib
from kolmo import compress, decompress

raw = sys.stdin.buffer.read()
t = time.perf_counter()
blob = compress(raw)
enc = time.perf_counter() - t
t = time.perf_counter()
out = decompress(blob)
dec = time.perf_counter() - t
assert out == raw, "round-trip failed"
print(f"input={len(raw)} output={len(blob)} ratio={len(blob)/len(raw):.4f} enc={enc:.2f}s dec={dec:.2f}s sha={hashlib.sha256(blob).hexdigest()[:16]}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=data,
        env=env,
        capture_output=True,
        check=True,
    )
    print(f"{label:8}: {result.stdout.decode().strip()}")


def main() -> None:
    # A short payload — fixed mode at ~0.4s per byte makes longer runs
    # impractical until the training step gets vectorized further.
    payload = (
        b"online-trained neural compression aims to spend fewer bits each "
        b"time a pattern returns. the model learns from the same bytes the "
        b"compressor has already emitted, so encoder and decoder stay in "
        b"lockstep without storing learned weights in the blob.\n"
    )
    print(f"payload: {len(payload)} bytes")
    print()

    t = time.perf_counter()
    run_mode("pytorch", fixed=False, data=payload)
    print(f"  wall: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    run_mode("fixed", fixed=True, data=payload)
    print(f"  wall: {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()

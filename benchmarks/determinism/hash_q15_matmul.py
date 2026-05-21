"""Hash a Q15 integer matmul to verify cross-machine bit-identity.

This is the foundation claim of fixed-point: integer addition is associative,
so the same int inputs produce the same int outputs on any machine, with any
threading. If this fails, our whole Rung 2 plan is wrong.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo import fixed  # noqa: E402


def main() -> None:
    # Deterministic inputs (same on every machine — no float ops involved).
    # Use a fixed seed and integer generation to be extra safe.
    rng = np.random.default_rng(0xC0FFEE)
    # Generate floats deterministically (numpy's RNG is bit-identical across
    # machines for the same seed), then quantize.
    a_f = rng.normal(size=(256, 256)).astype(np.float64)
    b_f = rng.normal(size=(256, 256)).astype(np.float64)
    a_q = fixed.quantize(a_f)
    b_q = fixed.quantize(b_f)

    # Hash the inputs to confirm both machines start identical.
    print(f"a_q sha256: {hashlib.sha256(a_q.tobytes()).hexdigest()}")
    print(f"b_q sha256: {hashlib.sha256(b_q.tobytes()).hexdigest()}")

    # The actual test.
    c_q = fixed.matmul(a_q, b_q)
    print(f"a_q @ b_q (256x256, Q15): sha256={hashlib.sha256(c_q.tobytes()).hexdigest()}")
    print(f"  shape={c_q.shape}, dtype={c_q.dtype}")
    print(f"  c_q[0,:5]={c_q[0, :5]}")
    print(f"  c_q[127,:5]={c_q[127, :5]}")


if __name__ == "__main__":
    main()

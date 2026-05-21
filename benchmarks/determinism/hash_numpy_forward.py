"""Hash a pure-NumPy forward pass from stable weights.

This checks whether NumPy matmul/layernorm/softmax produce byte-identical
logits across machines for the same stable-initialized model. If this differs
between Mac and Windows, Rung 2 cannot stop at NumPy and must move to explicit
fixed-point or otherwise controlled arithmetic.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo.model import KolmoTransformer
from kolmo.np_model import extract_weights, kolmo_forward
from kolmo.stable_init import stable_init_model


def main() -> None:
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
    )
    stable_init_model(model, seed=42)

    input_ids = np.array([1, 2, 3, 100, 200, 42], dtype=np.int64)
    logits = kolmo_forward(
        input_ids,
        extract_weights(model),
        n_heads=4,
        n_layers=2,
    )

    print(f"shape {logits.shape}")
    print(f"dtype {logits.dtype}")
    print(f"sha256 {hashlib.sha256(logits.tobytes()).hexdigest()}")
    for scale in (1_000_000, 100_000, 10_000):
        quantized = np.rint(logits * scale).astype(np.int64)
        digest = hashlib.sha256(quantized.tobytes()).hexdigest()
        print(f"q{scale} {digest}")
    print("first8 " + " ".join(f"{x:.9f}" for x in logits[0, :8]))


if __name__ == "__main__":
    main()

"""Hash the fixed-point transformer forward pass.

This is the Stage B cross-machine probe. Unlike the NumPy/PyTorch probes,
fixed_forward should be byte-identical across Mac and Windows because every
operation is integer math.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo.fixed_model import extract_fixed_weights, fixed_forward
from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model


def run_probe(label: str, kwargs: dict, n_heads: int, n_layers: int) -> None:
    model = KolmoTransformer(**kwargs)
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)
    input_ids = np.array([0, 84, 104, 101, 32, 113, 117, 105, 99, 107], dtype=np.int64)
    logits_q = fixed_forward(input_ids, weights, n_heads=n_heads, n_layers=n_layers)
    digest = hashlib.sha256(logits_q.tobytes()).hexdigest()
    print(f"{label:5} shape={logits_q.shape} sha256={digest}")
    print(f"      last[:8]={logits_q[-1, :8].tolist()}")


def main() -> None:
    run_probe(
        "tiny",
        dict(d_model=64, n_heads=4, n_layers=2, max_context=128),
        n_heads=4,
        n_layers=2,
    )
    run_probe(
        "prod",
        dict(d_model=256, n_heads=8, n_layers=4, max_context=512),
        n_heads=8,
        n_layers=4,
    )


if __name__ == "__main__":
    main()

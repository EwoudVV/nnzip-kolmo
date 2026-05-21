"""Hash model weights before and after the deterministic seed warmup.

This isolates one Rung 2 question:

1. Do seeded initial weights match across machines?
2. Do weights still match after the built-in seed-corpus training pass?

If (1) fails, initialization must be replaced. If (1) passes and (2) fails,
the first cross-machine divergence is in forward/backward/optimizer math.
"""

import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo._engine import new_model_and_optimizer
from kolmo.model import KolmoTransformer


def hash_model(model: KolmoTransformer) -> str:
    h = hashlib.sha256()
    for name, param in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    torch.manual_seed(42)
    initial = KolmoTransformer()
    print(f"initial {hash_model(initial)}")

    primed, _ = new_model_and_optimizer()
    print(f"after_prime {hash_model(primed)}")


if __name__ == "__main__":
    main()

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
from kolmo.stable_init import stable_init_model


def hash_model(model: KolmoTransformer) -> str:
    h = hashlib.sha256()
    for name, param in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    torch.manual_seed(42)
    pytorch_default = KolmoTransformer()
    print(f"pytorch_default {hash_model(pytorch_default)}")

    stable_initial = KolmoTransformer()
    stable_init_model(stable_initial, 42)
    print(f"stable_initial {hash_model(stable_initial)}")

    primed, _ = new_model_and_optimizer()
    print(f"after_prime {hash_model(primed)}")


if __name__ == "__main__":
    main()

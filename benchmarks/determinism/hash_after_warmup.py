"""Hash model state after seed warmup completes.

Single-step training is now deterministic across machines. But the full
compress does many training steps (warmup + per-block). If Mac and Windows
agree on the post-warmup model, the divergence is in per-block training of
the user input. If they disagree here, warmup itself accumulates drift even
with grid rounding.
"""

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from kolmo._engine import new_model_and_optimizer  # noqa: E402


def hash_params(model) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def hash_optimizer_state(optimizer) -> str:
    h = hashlib.sha256()
    # Sort by id for stable order across machines.
    for param_id in sorted(optimizer.state.keys(), key=id):
        state = optimizer.state[param_id]
        for key in sorted(state.keys()):
            v = state[key]
            if isinstance(v, torch.Tensor):
                h.update(key.encode())
                h.update(v.detach().cpu().numpy().tobytes())
            else:
                h.update(key.encode())
                h.update(repr(v).encode())
    return h.hexdigest()


def main() -> None:
    torch.set_num_threads(1)
    model, optimizer = new_model_and_optimizer()
    print(f"post-warmup params sha256:    {hash_params(model)}")
    print(f"post-warmup adam state sha256: {hash_optimizer_state(optimizer)}")


if __name__ == "__main__":
    main()

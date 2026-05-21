"""Measure how much a single training step diverges across machines.

Both machines start from identical stable-initialized weights. We run one
forward-backward-Adam step on a known input. Then we hash the resulting
weights and report the max absolute difference between them.

If the drift per step is < 1e-3, we can probably make training deterministic
by rounding weights to a 1/1024 grid after each step. If it's bigger, we'd
need NumPy backward instead.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo.model import KolmoTransformer  # noqa: E402
from kolmo.stable_init import stable_init_model  # noqa: E402


def hash_state(state_dict) -> str:
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        h.update(name.encode())
        h.update(state_dict[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    # Production config
    model = KolmoTransformer()
    stable_init_model(model, seed=42)

    initial_hash = hash_state(dict(model.named_parameters()))
    print(f"initial weights sha256: {initial_hash}")

    # One training step on a fixed input
    history = list(b"\x00" + b"the quick brown fox jumps over the lazy dog. ")
    block_bytes = list(b"the quick brown ")  # 16 bytes
    full = history + block_bytes
    n_hist = len(history)
    m = len(block_bytes)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor([full], dtype=torch.long)
    logits, _ = model(x, kv_caches=None, pos_offset=0)
    block_logits = logits[0, n_hist - 1 : n_hist + m - 1]
    targets = torch.tensor(block_bytes, dtype=torch.long)
    loss = F.cross_entropy(block_logits, targets)
    print(f"loss: {loss.item():.10f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after_hash = hash_state(dict(model.named_parameters()))
    print(f"after-step weights sha256: {after_hash}")

    # Hash a single representative parameter so we can compare across machines
    head_w = model.head.weight.detach().cpu().numpy()
    print(f"head.weight sha256: {hashlib.sha256(head_w.tobytes()).hexdigest()}")
    print(f"head.weight[0, :5]: {head_w[0, :5]}")
    print(f"head.weight[127, :5]: {head_w[127, :5]}")


if __name__ == "__main__":
    main()

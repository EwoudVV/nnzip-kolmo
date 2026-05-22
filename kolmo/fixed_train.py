"""Fixed-point training helpers for the compressor.

This is the bridge between the mathematical fixed-point components and the
online compressor training loop. It trains on one byte block using:

  fixed_backward -> deterministic Q15 gradients
  fixed_adam_step -> deterministic integer optimizer update
"""

from __future__ import annotations

import numpy as np

from kolmo.fixed_model import fixed_backward
from kolmo.fixed_optim import FixedAdamState, fixed_adam_step, init_fixed_adam_state


def fixed_train_block(
    weights: dict[str, np.ndarray],
    optimizer_state: FixedAdamState | None,
    history: list[int],
    block_bytes: list[int],
    *,
    n_heads: int = 8,
    n_layers: int = 4,
    context: int = 256,
) -> FixedAdamState:
    """Train fixed-point weights on one compressor block, in-place.

    This mirrors `kolmo._engine.train_block`: the model sees
    `history + block_bytes`, and the logits immediately before each block byte
    predict that byte. Both encoder and decoder can call this with the same
    observed bytes and get identical updated weights.
    """
    if not block_bytes:
        return optimizer_state if optimizer_state is not None else init_fixed_adam_state()

    full = (history + block_bytes)[-context:]
    m = len(block_bytes)
    n_hist = len(full) - m
    if n_hist <= 0:
        raise ValueError("fixed_train_block requires at least one history token")

    target_positions = np.arange(n_hist - 1, n_hist + m - 1, dtype=np.int64)
    targets = np.array(block_bytes, dtype=np.int64)
    input_ids = np.array(full, dtype=np.int64)

    _logits, grads = fixed_backward(
        input_ids,
        target_positions,
        targets,
        weights,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    return fixed_adam_step(weights, grads, optimizer_state)

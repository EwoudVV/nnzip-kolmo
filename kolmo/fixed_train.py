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
    tied_params: list[tuple[str, str]] | None = None,
    use_rope: bool = False,
) -> FixedAdamState:
    """Train fixed-point weights on one compressor block, in-place.

    This mirrors `kolmo._engine.train_block`: the model sees
    `history + block_bytes`, and the logits immediately before each block byte
    predict that byte. Both encoder and decoder can call this with the same
    observed bytes and get identical updated weights.

    `tied_params` lists (canonical, alias) name pairs. Gradients for aliases
    are summed into the canonical entry before the Adam step (matches what
    PyTorch's autograd does automatically for shared Parameters), and the
    alias is re-pointed at the canonical weight after the step so that the
    next forward sees a consistent shared tensor.
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
        use_rope=use_rope,
    )

    # Sum tied gradients into their canonical entry, drop the alias gradient
    # so Adam updates a single tensor. Using int64 for the sum since two Q15
    # gradients can overflow int32 in the worst case.
    if tied_params:
        for canonical, alias in tied_params:
            if alias in grads and canonical in grads:
                summed = grads[canonical].astype(np.int64) + grads[alias].astype(np.int64)
                grads[canonical] = np.clip(
                    summed,
                    np.iinfo(np.int32).min,
                    np.iinfo(np.int32).max,
                ).astype(np.int32)
                del grads[alias]

    new_state = fixed_adam_step(weights, grads, optimizer_state)

    # Re-alias: Adam created a fresh array for `canonical`, but `alias` in the
    # weights dict still points at the pre-step array. Point it at the new
    # canonical so subsequent forward passes use the same shared tensor.
    if tied_params:
        for canonical, alias in tied_params:
            if canonical in weights:
                weights[alias] = weights[canonical]

    return new_state

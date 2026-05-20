"""Shared logic for compress and decompress.

Both directions need to walk the model through the *same* training trajectory:
build identical weights, predict identical probabilities, take identical
optimizer steps. The KV cache lets each direction do most of the predictions
incrementally (O(T) per byte instead of O(T²)); the training step still needs
a full forward over the recent history with gradient tracking, but that's
only done once per block.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kolmo.model import KolmoTransformer

SEED = 42
LR = 1e-3
CONTEXT = 256  # sliding-window cap (max tokens kept in KV cache)
BLOCK_SIZE = 16  # bytes between optimizer steps
BOS = 0  # implicit start-of-stream byte, never written to disk


def new_model_and_optimizer() -> tuple[KolmoTransformer, torch.optim.Optimizer]:
    """Build a model with deterministic init. Both compress and decompress
    must call this and get bit-identical starting weights."""
    torch.manual_seed(SEED)
    model = KolmoTransformer()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, optimizer


def _trim_caches(caches: list, max_len: int) -> list:
    """Slide the KV cache window: keep only the last `max_len` positions."""
    out = []
    for c in caches:
        if c["k"].shape[2] > max_len:
            out.append({
                "k": c["k"][:, :, -max_len:],
                "v": c["v"][:, :, -max_len:],
            })
        else:
            out.append(c)
    return out


def warm_cache(model: KolmoTransformer, history: list[int]) -> tuple[np.ndarray, list, int]:
    """Run a fresh forward over `history` (no grad) to rebuild the KV cache
    and get the prediction for the next byte. Used at the start of each block,
    after a training step has invalidated the previous cache.

    Returns (probs over next byte as float64 numpy, kv_caches, pos_after).
    """
    x = torch.tensor([history], dtype=torch.long)
    with torch.no_grad():
        logits, caches = model(x, kv_caches=None, pos_offset=0)
    probs = torch.softmax(logits[0, -1], dim=-1).numpy().astype(np.float64)
    return probs, caches, len(history)


def step_cache(
    model: KolmoTransformer,
    byte: int,
    caches: list,
    pos_offset: int,
) -> tuple[np.ndarray, list, int]:
    """Feed one new byte using the cache. Returns (probs over next byte,
    updated caches, new pos_offset)."""
    x = torch.tensor([[byte]], dtype=torch.long)
    with torch.no_grad():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    probs = torch.softmax(logits[0, -1], dim=-1).numpy().astype(np.float64)
    return probs, caches, pos_offset + 1


def train_block(
    model: KolmoTransformer,
    optimizer: torch.optim.Optimizer,
    history: list[int],
    block_bytes: list[int],
) -> None:
    """Run a full forward with gradient over `history + block_bytes`, compute
    cross-entropy loss against the block targets, backward + optimizer step.

    Both compress and decompress call this with the same arguments at the
    same step, so weights stay in lockstep.
    """
    full = (history + block_bytes)[-CONTEXT:]
    m = len(block_bytes)
    n_hist = len(full) - m

    x = torch.tensor([full], dtype=torch.long)
    logits, _ = model(x, kv_caches=None, pos_offset=0)
    # Predictions for block bytes come from logits at positions [n_hist-1 .. n_hist+m-2]
    block_logits = logits[0, n_hist - 1 : n_hist + m - 1]

    targets = torch.tensor(block_bytes, dtype=torch.long)
    loss = F.cross_entropy(block_logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_history(history: list[int], new_bytes: list[int]) -> list[int]:
    """Append new bytes to the sliding-window history."""
    history = history + new_bytes
    if len(history) > CONTEXT:
        history = history[-CONTEXT:]
    return history

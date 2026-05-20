"""Compress raw bytes by online-training a transformer and arithmetic-coding
the resulting probability stream.

Per-block algorithm:
  1. Warm the KV cache by running a fresh forward over the current history.
     Get probs for the first byte of the block.
  2. For each subsequent byte in the block, feed one token through the model
     using the cache (O(T) per byte instead of O(T²)).
  3. After the block, run a full forward over (history + block) with
     gradient tracking, compute cross-entropy loss, backward + step.
  4. Cache is now stale (weights changed). Next block restarts at step 1.
"""

import struct

from kolmo._engine import (
    BLOCK_SIZE,
    BOS,
    new_model_and_optimizer,
    step_cache,
    train_block,
    update_history,
    warm_cache,
)
from kolmo.codec import RangeEncoder

MAGIC = b"KMO1"


def compress(data: bytes) -> bytes:
    if len(data) == 0:
        raise ValueError("cannot compress empty data")

    model, optimizer = new_model_and_optimizer()
    encoder = RangeEncoder()

    history = [BOS]
    pos = 0
    while pos < len(data):
        block_end = min(pos + BLOCK_SIZE, len(data))
        block = list(data[pos:block_end])
        m = len(block)

        probs, caches, pos_offset = warm_cache(model, history)
        encoder.encode(block[0], probs)

        for i in range(1, m):
            probs, caches, pos_offset = step_cache(
                model, block[i - 1], caches, pos_offset
            )
            encoder.encode(block[i], probs)

        train_block(model, optimizer, history, block)
        history = update_history(history, block)
        pos += m

    payload = encoder.finish()
    return MAGIC + struct.pack(">I", len(data)) + payload

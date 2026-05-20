"""Decompress kolmo blobs by re-running the same training trajectory the
compressor used and reading the arithmetic-coded probability stream.

Mirrors compress.py exactly: warm cache, step cache per byte to get probs,
decode each byte using the same probs the encoder used, then run the same
training step at the block boundary.
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
from kolmo.codec import RangeDecoder
from kolmo.compress import MAGIC


def decompress(blob: bytes) -> bytes:
    if len(blob) < 8 or blob[:4] != MAGIC:
        raise ValueError(f"not a kolmo blob (expected magic {MAGIC!r})")
    n_bytes = struct.unpack(">I", blob[4:8])[0]
    payload = blob[8:]

    model, optimizer = new_model_and_optimizer()
    decoder = RangeDecoder(payload)

    history = [BOS]
    output = bytearray()
    decoded_total = 0
    while decoded_total < n_bytes:
        m = min(BLOCK_SIZE, n_bytes - decoded_total)
        block: list[int] = []

        probs, caches, pos_offset = warm_cache(model, history)
        byte = decoder.decode(probs)
        block.append(byte)
        output.append(byte)

        for _ in range(1, m):
            probs, caches, pos_offset = step_cache(
                model, block[-1], caches, pos_offset
            )
            byte = decoder.decode(probs)
            block.append(byte)
            output.append(byte)

        train_block(model, optimizer, history, block)
        history = update_history(history, block)
        decoded_total += m

    return bytes(output)

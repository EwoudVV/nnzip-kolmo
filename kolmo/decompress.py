"""Decompress kolmo blobs by re-running the same training trajectory the
compressor used and reading the arithmetic-coded probability stream."""

import struct

from kolmo._engine import (
    BLOCK_SIZE,
    BOS,
    new_model_and_optimizer,
    predict,
    slide,
    train_block,
)
from kolmo.codec import RangeDecoder
from kolmo.compress import MAGIC


def decompress(blob: bytes) -> bytes:
    """Decompress a blob produced by `compress`. Returns the original bytes."""
    if len(blob) < 8 or blob[:4] != MAGIC:
        raise ValueError(f"not a kolmo blob (expected magic {MAGIC!r})")
    n_bytes = struct.unpack(">I", blob[4:8])[0]
    payload = blob[8:]

    model, optimizer = new_model_and_optimizer()
    decoder = RangeDecoder(payload)

    context = [BOS]
    output = bytearray()
    block_logits: list = []
    block_bytes: list[int] = []

    for _ in range(n_bytes):
        probs, last_logits = predict(model, context)
        byte = decoder.decode(probs)
        output.append(byte)
        block_logits.append(last_logits)
        block_bytes.append(byte)

        if len(block_bytes) >= BLOCK_SIZE:
            train_block(optimizer, block_logits, block_bytes)
            block_logits = []
            block_bytes = []

        context = slide(context, byte)

    if block_bytes:
        train_block(optimizer, block_logits, block_bytes)

    return bytes(output)

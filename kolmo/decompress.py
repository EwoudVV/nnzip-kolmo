"""Decompress kolmo blobs by re-running the same training trajectory the
compressor used and reading the arithmetic-coded probability stream."""

import struct

from kolmo._engine import BOS, new_model_and_optimizer, predict, slide, train_step
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
    for _ in range(n_bytes):
        probs, last_logits = predict(model, context)
        byte = decoder.decode(probs)
        output.append(byte)
        train_step(optimizer, last_logits, byte)
        context = slide(context, byte)

    return bytes(output)

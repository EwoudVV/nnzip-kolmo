"""Compress raw bytes by online-training a transformer and arithmetic-coding
the resulting probability stream."""

import struct

from kolmo._engine import BOS, new_model_and_optimizer, predict, slide, train_step
from kolmo.codec import RangeEncoder

MAGIC = b"KMO1"  # 4-byte magic + format version


def compress(data: bytes) -> bytes:
    """Compress `data`. Returns a self-contained blob (magic + length + payload).

    Output format (v1):
        4 bytes  : magic "KMO1"
        4 bytes  : uncompressed length, uint32 big-endian
        rest     : arithmetic-coded payload (uint32 words from constriction)
    """
    if len(data) == 0:
        raise ValueError("cannot compress empty data")

    model, optimizer = new_model_and_optimizer()
    encoder = RangeEncoder()

    context = [BOS]
    for byte in data:
        probs, last_logits = predict(model, context)
        encoder.encode(byte, probs)
        train_step(optimizer, last_logits, byte)
        context = slide(context, byte)

    payload = encoder.finish()
    return MAGIC + struct.pack(">I", len(data)) + payload

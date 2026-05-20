"""Compress raw bytes by online-training a transformer and arithmetic-coding
the resulting probability stream.

The stream is a sequence of literal events and copy events. Literals are
encoded with neural probabilities; copies encode an offset/length pair into
recent already-observed bytes. Either way, the model observes the reconstructed
bytes and trains every BLOCK_SIZE bytes, so decompression can replay the same
trajectory.
"""

import struct

from kolmo._engine import (
    BLOCK_SIZE,
    BOS,
    COPY_MAX,
    COPY_MIN,
    COPY_WINDOW,
    EVENT_PROBS,
    find_copy,
    new_model_and_optimizer,
    step_cache,
    train_block,
    update_history,
    warm_cache,
)
from kolmo.codec import RangeEncoder

MAGIC = b"KMO2"


def compress(data: bytes) -> bytes:
    if len(data) == 0:
        raise ValueError("cannot compress empty data")

    model, optimizer = new_model_and_optimizer()
    encoder = RangeEncoder()

    history = [BOS]
    copy_history = bytearray()
    pending: list[int] = []
    probs = None
    caches = None
    pos_offset = 0

    def ensure_cache():
        nonlocal probs, caches, pos_offset
        if probs is None:
            probs, caches, pos_offset = warm_cache(model, history)

    def observe_byte(byte: int):
        nonlocal history, copy_history, pending, probs, caches, pos_offset
        ensure_cache()
        probs, caches, pos_offset = step_cache(model, byte, caches, pos_offset)
        copy_history.append(byte)
        pending.append(byte)
        if len(pending) == BLOCK_SIZE:
            train_block(model, optimizer, history, pending)
            history = update_history(history, pending)
            pending = []
            probs = None
            caches = None
            pos_offset = 0

    pos = 0
    while pos < len(data):
        copy = find_copy(data, pos, bytes(copy_history))
        if copy is not None:
            offset, length = copy
            encoder.encode(1, EVENT_PROBS)
            encoder.encode_uniform(offset - 1, min(COPY_WINDOW, len(copy_history)))
            encoder.encode_uniform(
                length - COPY_MIN,
                min(COPY_MAX, len(data) - pos) - COPY_MIN + 1,
            )
            start = len(copy_history) - offset
            copied = copy_history[start : start + length]
            for byte in copied:
                observe_byte(byte)
            pos += length
            continue

        ensure_cache()
        encoder.encode(0, EVENT_PROBS)
        byte = data[pos]
        encoder.encode(byte, probs)
        observe_byte(byte)
        pos += 1

    if pending:
        train_block(model, optimizer, history, pending)

    payload = encoder.finish()
    return MAGIC + struct.pack(">I", len(data)) + payload

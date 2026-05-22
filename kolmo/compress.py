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
    EventModel,
    LengthModel,
    OffsetModel,
    RollingCopyMatcher,
    append_copy_history,
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
    offset_model = OffsetModel(COPY_WINDOW)
    event_model = EventModel()
    length_model = LengthModel(COPY_MAX - COPY_MIN + 1)
    copy_matcher = RollingCopyMatcher(data)

    history = [BOS]
    copy_history = bytearray()
    pending: list[int] = []
    probs = None
    caches = None
    pos_offset = 0

    def train_pending_if_full():
        nonlocal history, pending, probs, caches, pos_offset
        if len(pending) != BLOCK_SIZE:
            return
        train_block(model, optimizer, history, pending)
        history = update_history(history, pending)
        pending = []
        probs = None
        caches = None
        pos_offset = 0

    def ensure_cache():
        nonlocal probs, caches, pos_offset
        train_pending_if_full()
        if probs is None:
            probs, caches, pos_offset = warm_cache(model, history)

    def observe_byte(byte: int):
        nonlocal copy_history, pending, probs, caches, pos_offset
        train_pending_if_full()
        ensure_cache()
        probs, caches, pos_offset = step_cache(model, byte, caches, pos_offset)
        append_copy_history(copy_history, byte)
        pending.append(byte)

    pos = 0
    while pos < len(data):
        copy = copy_matcher.find(pos)
        if copy is not None:
            offset, length = copy
            encoder.encode(1, event_model.probs())
            event_model.observe(1)
            max_offset = min(COPY_WINDOW, len(copy_history))
            max_len = min(COPY_MAX, len(data) - pos) - COPY_MIN + 1
            offset_bucket = offset_model.bucket_for(offset)
            encoder.encode(offset_bucket, offset_model.probs_for(max_offset))
            offset_lo, offset_hi = offset_model.bucket_bounds(
                offset_bucket,
                max_offset,
            )
            offset_width = offset_hi - offset_lo + 1
            if offset_width > 1:
                encoder.encode(
                    offset - offset_lo,
                    offset_model.residual_probs_for(offset_bucket, max_offset),
                )
            if max_len > 1:
                encoder.encode(length - COPY_MIN, length_model.probs_for(max_len))
            offset_model.observe(offset)
            length_model.observe(length - COPY_MIN)
            start = len(copy_history) - offset
            copied = copy_history[start : start + length]
            for byte in copied:
                observe_byte(byte)
            pos += length
            continue

        ensure_cache()
        encoder.encode(0, event_model.probs())
        event_model.observe(0)
        byte = data[pos]
        encoder.encode(byte, probs)
        observe_byte(byte)
        pos += 1

    payload = encoder.finish()
    return MAGIC + struct.pack(">I", len(data)) + payload

"""Compress raw bytes by online-training a transformer and arithmetic-coding
the resulting probability stream.

The stream is a sequence of literal events and copy events. Literals are
encoded with neural probabilities; copies encode an offset/length pair into
recent already-observed bytes. Either way, the model observes the reconstructed
bytes and trains every BLOCK_SIZE bytes, so decompression can replay the same
trajectory.
"""

import math
import struct

from kolmo._engine import (
    BOS,
    COPY_LITERAL_BPB,
    COPY_MAX,
    COPY_MIN,
    COPY_USE_LITERAL_MODEL_PROXY,
    COPY_WINDOW,
    EventModel,
    LengthModel,
    LiteralModel,
    OffsetModel,
    RollingCopyMatcher,
    append_copy_history,
    new_model_and_optimizer,
    step_cache,
    step_cache_batch,
    train_block,
    training_block_size_at,
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
    literal_model = LiteralModel()
    copy_matcher = RollingCopyMatcher(data)

    history = [BOS]
    copy_history = bytearray()
    pending: list[int] = []
    bytes_trained_through = 0  # total bytes the model has already trained on
    probs = None
    caches = None
    pos_offset = 0

    def train_pending_if_full():
        nonlocal history, pending, probs, caches, pos_offset
        nonlocal bytes_trained_through
        threshold = training_block_size_at(bytes_trained_through)
        if len(pending) < threshold:
            return
        train_block(model, optimizer, history, pending)
        history = update_history(history, pending)
        bytes_trained_through += len(pending)
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
        literal_model.observe(byte)
        pending.append(byte)

    def observe_byte_sequence(seq):
        """Feed a known sequence (copied bytes) through the model in batches.

        Each chunk runs as one forward pass through the KV cache, which is
        ~10x faster than per-byte forward on PyTorch for non-trivial chunks.
        Chunks are bounded by the next training-block boundary, since
        training invalidates the cache and the schedule depends on
        bytes_trained_through.

        Per-byte probabilities are discarded — the copy event is encoded by
        its (offset, length), not by predicting each copied byte.
        """
        nonlocal copy_history, pending, probs, caches, pos_offset
        i = 0
        seq_len = len(seq)
        while i < seq_len:
            train_pending_if_full()
            ensure_cache()
            threshold = training_block_size_at(bytes_trained_through)
            room = threshold - len(pending)
            chunk_end = min(i + room, seq_len)
            chunk = seq[i:chunk_end]
            probs, caches, pos_offset = step_cache_batch(
                model, chunk, caches, pos_offset
            )
            for b in chunk:
                append_copy_history(copy_history, int(b))
                literal_model.observe(int(b))
                pending.append(int(b))
            i = chunk_end

    def copy_header_bits(offset: int, length: int, pos: int) -> float:
        """Approximate arithmetic-coded bits for a copy header now.

        This mirrors the actual event/offset/length encoding below but only
        reads model probabilities. It lets the encoder reject short/far copies
        whose header costs more than the bytes they replace. Decoder behavior
        is unchanged because the chosen event stream remains explicit.
        """
        probs = event_model.probs()
        bits = -math.log2(max(float(probs[1]), 1e-300))

        max_offset = min(COPY_WINDOW, len(copy_history))
        max_len = min(COPY_MAX, len(data) - pos) - COPY_MIN + 1
        offset_bucket = offset_model.bucket_for(offset)
        bucket_probs = offset_model.probs_for(max_offset)
        bits += -math.log2(max(float(bucket_probs[offset_bucket]), 1e-300))
        offset_lo, offset_hi = offset_model.bucket_bounds(offset_bucket, max_offset)
        offset_width = offset_hi - offset_lo + 1
        if offset_width > 1:
            residual_probs = offset_model.residual_probs_for(
                offset_bucket,
                max_offset,
            )
            bits += -math.log2(
                max(float(residual_probs[offset - offset_lo]), 1e-300)
            )
        if max_len > 1:
            length_offset = length - COPY_MIN
            length_bucket = length_model.bucket_for(length_offset)
            len_probs = length_model.probs_for(max_len)
            bits += -math.log2(max(float(len_probs[length_bucket]), 1e-300))
            len_lo, len_hi = length_model.bucket_bounds(length_bucket, max_len)
            if len_hi > len_lo:
                residual_probs = length_model.residual_probs_for(
                    length_bucket,
                    max_len,
                )
                bits += -math.log2(
                    max(float(residual_probs[length_offset - len_lo]), 1e-300)
                )
        return bits

    def choose_copy(pos: int):
        best = None
        best_savings = 0.0
        for offset, length in copy_matcher.candidates(pos):
            header_bits = copy_header_bits(offset, length, pos)
            if COPY_USE_LITERAL_MODEL_PROXY:
                literal_bits = literal_model.proxy_bits(
                    data[pos : pos + length],
                    COPY_LITERAL_BPB,
                )
            else:
                literal_bits = COPY_LITERAL_BPB * length
            savings = literal_bits - header_bits
            if savings > best_savings:
                best_savings = savings
                best = (offset, length)
        return best

    pos = 0
    while pos < len(data):
        copy = choose_copy(pos)
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
                length_offset = length - COPY_MIN
                length_bucket = length_model.bucket_for(length_offset)
                encoder.encode(length_bucket, length_model.probs_for(max_len))
                len_lo, len_hi = length_model.bucket_bounds(length_bucket, max_len)
                if len_hi > len_lo:
                    encoder.encode(
                        length_offset - len_lo,
                        length_model.residual_probs_for(length_bucket, max_len),
                    )
            offset_model.observe(offset)
            length_model.observe(length - COPY_MIN)
            start = len(copy_history) - offset
            copied = bytes(copy_history[start : start + length])
            observe_byte_sequence(copied)
            # Tell the literal model the next byte will be right after a
            # copy whose last byte was copied[-1]. Used by the optional
            # post-copy predictor in LiteralModel.probs(). No-op overhead
            # if KOLMO_POST_COPY=0.
            if copied:
                literal_model.mark_copy_end(copied[-1])
            pos += length
            continue

        ensure_cache()
        encoder.encode(0, event_model.probs())
        event_model.observe(0)
        byte = data[pos]
        encoder.encode(byte, literal_model.probs(probs))
        observe_byte(byte)
        literal_model.train_on_literal(byte)
        pos += 1

    payload = encoder.finish()
    return MAGIC + struct.pack(">I", len(data)) + payload

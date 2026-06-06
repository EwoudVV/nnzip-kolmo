"""Decompress kolmo blobs by replaying the compressor's trajectory.

The arithmetic stream contains literal events and copy events. Literals decode
one byte from the neural probability distribution; copies reproduce bytes from
recent output history. Both paths still feed the reconstructed bytes through
the model and train every BLOCK_SIZE bytes.
"""

import struct

from kolmo._engine import (
    BOS,
    COPY_MAX,
    COPY_MIN,
    COPY_WINDOW,
    EventModel,
    LengthModel,
    LiteralModel,
    OffsetModel,
    append_copy_history,
    new_model_and_optimizer,
    step_cache,
    step_cache_batch,
    train_block,
    training_block_size_at,
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
    offset_model = OffsetModel(COPY_WINDOW)
    event_model = EventModel()
    length_model = LengthModel(COPY_MAX - COPY_MIN + 1)
    literal_model = LiteralModel()

    history = [BOS]
    copy_history = bytearray()
    pending: list[int] = []
    bytes_trained_through = 0
    output = bytearray()
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
        """Batched mirror of observe_byte for copy-event bytes.
        See compress.py for the rationale; the trajectory must match.
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

    decoded_total = 0
    while decoded_total < n_bytes:
        event = decoder.decode(event_model.probs())
        event_model.observe(event)
        if event == 1:
            max_offset = min(COPY_WINDOW, len(copy_history))
            max_len = min(COPY_MAX, n_bytes - decoded_total)
            if max_offset == 0 or max_len < COPY_MIN:
                raise ValueError("invalid copy event in kolmo blob")
            offset_bucket = decoder.decode(offset_model.probs_for(max_offset))
            offset_lo, offset_hi = offset_model.bucket_bounds(
                offset_bucket,
                max_offset,
            )
            offset_width = offset_hi - offset_lo + 1
            if offset_width > 1:
                offset = offset_lo + decoder.decode(
                    offset_model.residual_probs_for(offset_bucket, max_offset)
                )
            else:
                offset = offset_lo
            len_alpha = max_len - COPY_MIN + 1
            if len_alpha > 1:
                length_bucket = decoder.decode(length_model.probs_for(len_alpha))
                len_lo, len_hi = length_model.bucket_bounds(
                    length_bucket,
                    len_alpha,
                )
                if len_hi > len_lo:
                    length_offset = len_lo + decoder.decode(
                        length_model.residual_probs_for(length_bucket, len_alpha)
                    )
                else:
                    length_offset = len_lo
                length = length_offset + COPY_MIN
            else:
                length = COPY_MIN
            offset_model.observe(offset)
            length_model.observe(length - COPY_MIN)
            start = len(copy_history) - offset
            copied = bytes(copy_history[start : start + length])
            output.extend(copied)
            observe_byte_sequence(copied)
            # Mirror the compress side: the literal model needs to know the
            # next byte is right after a copy so its third predictor stays
            # in lockstep. See compress.py for the rationale.
            if copied:
                literal_model.mark_copy_end(copied[-1])
            decoded_total += length
            continue

        ensure_cache()
        byte = decoder.decode(literal_model.probs(probs))
        output.append(byte)
        observe_byte(byte)
        decoded_total += 1

    return bytes(output)

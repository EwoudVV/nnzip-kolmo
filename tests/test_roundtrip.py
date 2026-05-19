"""Round-trip tests: compress + decompress must return the exact original bytes.

This is the entire correctness criterion for Rung 1. If these pass, the
online-training architecture works end-to-end.
"""

import pytest

from kolmo import compress, decompress


def test_roundtrip_short():
    data = b"hello world"
    blob = compress(data)
    assert decompress(blob) == data


def test_roundtrip_single_byte():
    data = b"A"
    blob = compress(data)
    assert decompress(blob) == data


def test_roundtrip_repeated():
    data = b"abc" * 30  # 90 bytes, very predictable
    blob = compress(data)
    assert decompress(blob) == data


def test_roundtrip_1kb_text():
    data = (b"The quick brown fox jumps over the lazy dog. " * 23)[:1024]
    assert len(data) == 1024
    blob = compress(data)
    assert decompress(blob) == data


def test_roundtrip_binary_bytes():
    """Round-trip should work on arbitrary byte values, not just printable
    ASCII — the codec must not depend on UTF-8 validity."""
    data = bytes(range(256))  # every possible byte value exactly once
    blob = compress(data)
    assert decompress(blob) == data


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        compress(b"")


def test_invalid_magic_rejected():
    with pytest.raises(ValueError, match="kolmo"):
        decompress(b"NOPE" + b"\x00" * 4)

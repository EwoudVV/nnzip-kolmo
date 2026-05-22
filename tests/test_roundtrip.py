"""Round-trip tests: compress + decompress must return the exact original bytes.

This is the entire correctness criterion for Rung 1. If these pass, the
online-training architecture works end-to-end.
"""

import pytest

import kolmo._engine as engine
from kolmo import compress, decompress
from kolmo.compress import MAGIC

DEFAULT_SEED_CORPUS = engine.SEED_CORPUS
FAST_SEED_CORPUS = DEFAULT_SEED_CORPUS[:256]


@pytest.fixture(autouse=True)
def fast_seed_corpus(monkeypatch):
    """Most round-trip tests exercise codec symmetry, not seed quality.

    Keep one explicit full-seed smoke test below; use a short deterministic
    seed everywhere else so the suite stays runnable during iteration.
    """
    monkeypatch.setattr(engine, "SEED_CORPUS", FAST_SEED_CORPUS)


def test_roundtrip_short():
    data = b"hello world"
    blob = compress(data)
    assert blob.startswith(MAGIC)
    assert decompress(blob) == data


def test_roundtrip_single_byte():
    data = b"A"
    blob = compress(data)
    assert decompress(blob) == data


def test_fixed_point_roundtrip_single_byte_skip_prime(monkeypatch):
    """Fixed-point engine path should be codec-symmetric.

    Skipping the seed prime keeps this as a fast integration smoke; the fixed
    training path itself is tested separately in test_fixed_train.py.
    """
    monkeypatch.setenv("KOLMO_FIXED", "1")
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
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


def test_roundtrip_default_seed_smoke(monkeypatch):
    """The production seed is much larger, so keep a direct smoke test for it."""
    monkeypatch.setattr(engine, "SEED_CORPUS", DEFAULT_SEED_CORPUS)
    data = b"default seed smoke"
    blob = compress(data)
    assert decompress(blob) == data


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        compress(b"")


def test_invalid_magic_rejected():
    with pytest.raises(ValueError, match="kolmo"):
        decompress(b"NOPE" + b"\x00" * 4)

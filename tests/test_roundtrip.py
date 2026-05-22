"""Round-trip tests: compress + decompress must return the exact original bytes.

This is the entire correctness criterion for Rung 1. If these pass, the
online-training architecture works end-to-end.
"""

import pytest

import kolmo._engine as engine
from kolmo import compress, decompress
from kolmo.compress import HEADER_SIZE, MAGIC, MODE_FIXED, MODE_PYTORCH

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
    assert blob[4] == MODE_PYTORCH
    assert len(blob) >= HEADER_SIZE
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
    assert blob[4] == MODE_FIXED
    assert decompress(blob) == data


def test_fixed_point_roundtrip_exercises_kv_cache(monkeypatch):
    """Round-trip a payload long enough to exercise warm + step + training
    iterations on the fixed-point path. This is the integration check that
    the KV cache and training-step invalidation stay in lockstep between
    compress and decompress.
    """
    monkeypatch.setenv("KOLMO_FIXED", "1")
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    data = b"the cache must stay in lockstep across compress and decompress."
    blob = compress(data)
    assert blob.startswith(MAGIC)
    assert decompress(blob) == data


def test_fixed_point_roundtrip_with_seed_prime(monkeypatch):
    """Smoke-test the seed-warmup path in fixed mode.

    Other fixed-mode tests use KOLMO_SKIP_PRIME=1 because the full seed
    corpus takes ~3 minutes to prime in fixed-point. Shrink the corpus to
    one block's worth so the prime runs in seconds but still exercises the
    `_prime_model -> train_block` codepath, which is otherwise only hit by
    PyTorch tests.
    """
    monkeypatch.setattr(engine, "SEED_CORPUS", b"prime me deterministically.")
    monkeypatch.setenv("KOLMO_FIXED", "1")
    data = b"hi"
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


def test_fixed_blob_requires_fixed_mode(monkeypatch):
    """A fixed-mode blob should fail fast under PyTorch mode.

    The arithmetic payloads are backend-specific: PyTorch and fixed Q15 compute
    different distributions, so trying to decode under the wrong backend should
    produce a clear mode error instead of wandering into an opaque codec error.
    """
    monkeypatch.setenv("KOLMO_FIXED", "1")
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    blob = compress(b"A")

    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    with pytest.raises(ValueError, match="KOLMO_FIXED=1"):
        decompress(blob)


def test_pytorch_blob_rejects_fixed_mode(monkeypatch):
    """A PyTorch-mode blob should fail fast under fixed mode."""
    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    blob = compress(b"A")

    monkeypatch.setenv("KOLMO_FIXED", "1")
    with pytest.raises(ValueError, match="PyTorch mode"):
        decompress(blob)

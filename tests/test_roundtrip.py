"""Round-trip tests: compress + decompress must return the exact original bytes.

This is the entire correctness criterion for Rung 1. If these pass, the
online-training architecture works end-to-end.
"""

import pytest

import importlib
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


def test_fixed_point_roundtrip_with_rope(monkeypatch):
    """Fixed RoPE path should stay codec-symmetric through warm, step, and
    training invalidation just like the absolute-pos path."""
    monkeypatch.setenv("KOLMO_FIXED", "1")
    monkeypatch.setenv("KOLMO_USE_ROPE", "1")
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    data = b"rope must stay in lockstep across fixed compress and decompress."
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


def test_compress_skips_final_useless_training_step(monkeypatch):
    """Training after the last byte cannot affect any future prediction.

    Keep full-block training lazy: a completed block is trained only when the
    next byte/copy observation needs the updated model. This saves one
    train_block call for every file whose last observed bytes are not followed
    by another prediction, including exact BLOCK_SIZE multiples.
    """
    compress_mod = importlib.import_module("kolmo.compress")
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    calls: list[list[int]] = []

    def count_train(model, optimizer, history, pending):
        calls.append(list(pending))

    monkeypatch.setattr(compress_mod, "train_block", count_train)

    calls.clear()
    compress_mod.compress(bytes(range(engine.BLOCK_SIZE - 1)))
    assert calls == []

    calls.clear()
    compress_mod.compress(bytes(range(engine.BLOCK_SIZE)))
    assert calls == []

    calls.clear()
    compress_mod.compress(bytes(range(engine.BLOCK_SIZE + 1)))
    assert calls == [list(range(engine.BLOCK_SIZE))]


def test_invalid_magic_rejected():
    with pytest.raises(ValueError, match="kolmo"):
        decompress(b"NOPE" + b"\x00" * 4)


def test_logistic_mixer_roundtrip(monkeypatch):
    """KOLMO_MIXER=logistic round-trip: the bit-tree mixer trains online
    on both sides, so compressor and decompressor weights must stay in
    lockstep or decoding diverges."""
    monkeypatch.setattr(engine, "_MIXER_NAME", "logistic")
    data = b"the quick brown fox jumps over the lazy dog. " * 6
    blob = compress(data)
    assert decompress(blob) == data


def test_logistic_mixer_roundtrip_with_extra_predictors(monkeypatch):
    """Same, with structural predictors enabled via KOLMO_PREDICTORS.
    Their observe()/mark_copy_end() state must mirror across both sides."""
    monkeypatch.setattr(engine, "_MIXER_NAME", "logistic")
    monkeypatch.setattr(
        engine,
        "_EXTRA_PREDICTOR_NAMES",
        ["balanced_delimiter", "after_number", "in_text", "position_modulo"],
    )
    data = b"<page>[[Wiki link|alias]] {{cite|year=1942}} (note)</page>" * 4
    blob = compress(data)
    assert decompress(blob) == data


def test_logistic_mixer_roundtrip_with_match_predictor(monkeypatch):
    """Match predictor state (pointer, table, confidence counters) must
    mirror exactly across compress/decompress, including across copy
    events that feed observe() in batches."""
    monkeypatch.setattr(engine, "_MIXER_NAME", "logistic")
    monkeypatch.setattr(engine, "_EXTRA_PREDICTOR_NAMES", ["match"])
    data = b"the cat sat on the mat. the cat sat on the hat. " * 5
    blob = compress(data)
    assert decompress(blob) == data

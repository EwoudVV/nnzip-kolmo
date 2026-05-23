"""Compression-ratio regression tests.

`test_training_works.py` catches the case where the model silently stops
training (gradient explosion, vanishing learning, dead Adam). This file
catches the broader case where the compression *pipeline* breaks while
the model is still healthy — e.g. a bug in the arithmetic coder, the
copy mechanism, the offset/length encoding, the event flag, or any
piece that turns model probabilities into bytes.

The thresholds are loose on purpose: the tests should pass even after
modest ratio shifts from architecture tweaks. They only fire if
something gets fundamentally wrong (output near or above input size,
output worse than gzip by a lot, etc.).
"""

import gzip

import pytest  # noqa: F401

import kolmo._engine as engine
from kolmo import compress, decompress

# A small prime corpus used by the with-prime tests below. The real
# SEED_CORPUS is ~5KB and primes for ~70s; this 256-byte slice runs in
# under 10s and still exercises the full prime -> train -> compress
# pipeline. The point of these tests is detecting pipeline regressions,
# not validating seed-quality on the production corpus.
_TINY_PRIME = b"the quick brown fox jumps over the lazy dog. " * 6


REPETITIVE_PAYLOAD = (
    b"the quick brown fox jumps over the lazy dog. " * 12
)  # ~540 bytes, very repetitive — copy mechanism should dominate


def test_pytorch_repetitive_payload_compresses_substantially(monkeypatch):
    """A highly repetitive payload should compress to a small fraction of
    its input even without training (the LZ-style copy mechanism handles
    repetition independent of the neural model).

    Catches: broken copy matching, broken event-flag coding, broken
    offset/length coding, broken arithmetic coder.

    Threshold (output < 50% of input) is loose enough to survive
    architecture tweaks but tight enough that pipeline breakage trips it.
    On a healthy build the actual ratio is ~10-20%.
    """
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    data = REPETITIVE_PAYLOAD
    blob = compress(data)
    assert decompress(blob) == data
    ratio = len(blob) / len(data)
    assert ratio < 0.5, (
        f"compression failed: {len(data)} -> {len(blob)} ratio {ratio:.2f} "
        f"(expected < 0.50 for a highly repetitive payload — copy mechanism "
        f"or arithmetic coder may be broken)"
    )


def test_pytorch_skip_prime_does_not_inflate_dramatically(monkeypatch):
    """Without prime + on non-repetitive text, the model has no signal and
    arithmetic coding pays full cost per literal byte. The output may be
    slightly larger than the input (due to event flags + small overhead)
    but shouldn't be wildly inflated.

    Catches the rounding-bug class of regression: gradient explosion that
    makes the model assign near-zero probability to actual bytes, causing
    output to grow many times the input size.
    """
    monkeypatch.setenv("KOLMO_SKIP_PRIME", "1")
    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    # Non-repetitive payload so the copy mechanism can't save us
    data = b"Pneumonoultramicroscopicsilicovolcanoconiosis."
    blob = compress(data)
    assert decompress(blob) == data
    ratio = len(blob) / len(data)
    # The actual broken case had ratio ~1.8 (output 80% bigger than input);
    # healthy random-init kolmo on this payload sits around 1.3 (~1 bit
    # per byte of overhead for event flags + a near-uniform 8 bpb per byte).
    assert ratio < 1.6, (
        f"compression catastrophically inflated: {len(data)} -> {len(blob)} "
        f"ratio {ratio:.2f} — model probably outputs near-zero probability "
        f"on actual bytes (gradient explosion or similar)"
    )


def test_pytorch_with_prime_beats_uncompressed(monkeypatch):
    """With seed prime + on English-ish text, kolmo should compress to
    well under 100% of input — otherwise the whole compression pipeline
    has lost the plot.

    Uses a tiny prime corpus (monkeypatched) so the test runs in seconds
    instead of running the full production prime.
    """
    monkeypatch.setattr(engine, "SEED_CORPUS", _TINY_PRIME)
    monkeypatch.delenv("KOLMO_SKIP_PRIME", raising=False)
    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    data = (
        b"The library kept rows of shelves with old books, quiet readers, "
        b"and printed forms. The same words returned in nearby sentences.\n"
    )
    blob = compress(data)
    assert decompress(blob) == data
    ratio = len(blob) / len(data)
    assert ratio < 1.0, (
        f"compress regressed badly: {len(data)} -> {len(blob)} "
        f"ratio {ratio:.2f} (expected < 1.0 with prime on English)"
    )


def test_pytorch_competitive_with_gzip_on_short_english(monkeypatch):
    """Sanity check: on text where gzip works well, kolmo should be in
    the same ballpark (not many times worse). Catches the case where
    compression "technically works" (output < input) but is dramatically
    less efficient than the simplest alternative.
    """
    monkeypatch.setattr(engine, "SEED_CORPUS", _TINY_PRIME)
    monkeypatch.delenv("KOLMO_SKIP_PRIME", raising=False)
    monkeypatch.delenv("KOLMO_FIXED", raising=False)
    data = (
        b"Compression reduces the size of data by exploiting patterns. "
        b"A dictionary method stores repeated phrases as pointers, while "
        b"a statistical method assigns shorter codes to likely events.\n"
    )
    blob = compress(data)
    gz = gzip.compress(data, compresslevel=9)
    assert decompress(blob) == data
    # Tiny-prime kolmo on short input won't beat gzip's static codebook
    # (gzip is great on short prose). Loose threshold catches the case
    # where kolmo is many times worse, which would indicate a broken
    # pipeline rather than a small ratio difference.
    assert len(blob) < 3 * len(gz), (
        f"kolmo dramatically worse than gzip: kolmo={len(blob)}B "
        f"gzip={len(gz)}B (expected within 3x)"
    )

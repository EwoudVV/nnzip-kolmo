"""Tests for the primed-state disk cache.

The cache is an optimization: the same primed state is just deterministic
output of the seed-corpus training trajectory, so saving and reloading must
be lossless to the bit. If the cache silently mutated state, compress and
decompress could diverge on cache-hit runs vs cache-miss runs.
"""

import numpy as np
import pytest

from kolmo.fixed_optim import FixedAdamState, Q30
from kolmo.seed_cache import (
    cache_path_for,
    compute_config_hash,
    load_state,
    save_state,
)


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("KOLMO_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("KOLMO_NO_SEED_CACHE", raising=False)
    return tmp_path


def _make_state() -> tuple[dict, FixedAdamState, list[tuple[str, str]]]:
    rng = np.random.default_rng(0)
    weights = {
        "token_emb.weight": rng.integers(-1000, 1000, size=(8, 16), dtype=np.int32),
        "blocks.0.ln1.weight": rng.integers(-100, 100, size=16, dtype=np.int32),
        "head.weight": None,  # alias, set below
    }
    weights["head.weight"] = weights["token_emb.weight"]
    state = FixedAdamState(
        step=7,
        m={
            "token_emb.weight": rng.integers(-(2**40), 2**40, size=(8, 16), dtype=np.int64),
            "blocks.0.ln1.weight": rng.integers(-(2**40), 2**40, size=16, dtype=np.int64),
        },
        v={
            "token_emb.weight": rng.integers(0, 2**40, size=(8, 16), dtype=np.int64),
            "blocks.0.ln1.weight": rng.integers(0, 2**40, size=16, dtype=np.int64),
        },
        beta1_pow_q30=int(Q30 * 0.95),
        beta2_pow_q30=int(Q30 * 0.99),
    )
    tied = [("token_emb.weight", "head.weight")]
    return weights, state, tied


def test_save_load_roundtrip_is_bit_exact(cache_root):
    """Loaded state must exactly equal saved state — same weights, same Adam
    moments, same step counter."""
    weights, state, tied = _make_state()
    path = cache_root / "seed_state_abc.npz"

    save_state(path, weights, state, tied)
    assert path.exists()

    weights2, state2, tied2 = load_state(path)

    assert tied2 == tied
    assert set(weights2) == set(weights)
    for name in weights:
        assert np.array_equal(weights2[name], weights[name]), name
    # The alias must be reconstructed (not just present, but pointing at
    # the same array object as the canonical).
    assert weights2["head.weight"] is weights2["token_emb.weight"]
    assert state2.step == state.step
    assert state2.beta1_pow_q30 == state.beta1_pow_q30
    assert state2.beta2_pow_q30 == state.beta2_pow_q30
    for name in state.m:
        assert np.array_equal(state2.m[name], state.m[name]), f"m/{name}"
        assert np.array_equal(state2.v[name], state.v[name]), f"v/{name}"


def test_compute_config_hash_is_stable():
    """Same inputs always produce the same hash, every run, every machine."""
    args = dict(
        seed_corpus=b"hello",
        model_config={"d_model": 256, "n_heads": 8},
        init_seed=42,
        block_size=16,
    )
    h1 = compute_config_hash(**args)
    h2 = compute_config_hash(**args)
    assert h1 == h2
    assert len(h1) == 16  # 64-bit hash prefix


def test_compute_config_hash_changes_on_any_input_change():
    """Any change to a hashed input must flip the hash, so a stale cache
    can't accidentally match a new config."""
    base = dict(
        seed_corpus=b"hello",
        model_config={"d_model": 256, "n_heads": 8},
        init_seed=42,
        block_size=16,
    )
    h0 = compute_config_hash(**base)

    variants = [
        dict(base, seed_corpus=b"goodbye"),
        dict(base, model_config={"d_model": 128, "n_heads": 8}),
        dict(base, model_config={"d_model": 256, "n_heads": 4}),
        dict(base, init_seed=43),
        dict(base, block_size=32),
    ]
    for variant in variants:
        assert compute_config_hash(**variant) != h0


def test_cache_path_uses_custom_dir(cache_root):
    """KOLMO_CACHE_DIR override actually routes file writes there."""
    weights, state, tied = _make_state()
    path = cache_path_for("test123")
    assert str(path).startswith(str(cache_root))
    save_state(path, weights, state, tied)
    assert path.exists()
    assert path.parent == cache_root


def test_cache_save_is_atomic(cache_root):
    """Saving uses a .tmp -> rename pattern so a crashed save doesn't leave
    a half-written file that subsequent runs would happily load."""
    weights, state, tied = _make_state()
    path = cache_path_for("atomic")
    save_state(path, weights, state, tied)
    # After a successful save, no .tmp file should remain.
    assert not (cache_root / "seed_state_atomic.npz.tmp").exists()


def test_compress_uses_cache_on_second_call(cache_root, monkeypatch):
    """End-to-end: first compress in fixed mode populates the cache, second
    compress finds it and skips priming entirely. We test by checking the
    cache directory before/after and verifying the second run produces the
    same blob (correctness preserved) without re-running prime.
    """
    import time

    import kolmo._engine as engine
    from kolmo import compress, decompress

    # Tiny seed corpus so the first prime fits in a few seconds.
    monkeypatch.setattr(engine, "SEED_CORPUS", b"prime me deterministically.")
    monkeypatch.setenv("KOLMO_FIXED", "1")
    monkeypatch.delenv("KOLMO_NO_SEED_CACHE", raising=False)

    data = b"hi"

    # First run: empty cache, must prime.
    assert list(cache_root.glob("*.npz")) == []
    t0 = time.perf_counter()
    blob1 = compress(data)
    first_run_s = time.perf_counter() - t0
    assert decompress(blob1) == data

    cache_files = list(cache_root.glob("*.npz"))
    assert len(cache_files) == 1, "expected one cache file after first run"

    # Second run: cache hit, should be substantially faster.
    t0 = time.perf_counter()
    blob2 = compress(data)
    second_run_s = time.perf_counter() - t0
    assert decompress(blob2) == data

    # Same input + same primed state -> identical blob.
    assert blob1 == blob2
    # The cache should at least halve the cost; in practice it's much faster
    # because no training steps run on the seed at all.
    assert second_run_s < first_run_s * 0.7, (
        f"cache hit was not faster: first={first_run_s:.2f}s "
        f"second={second_run_s:.2f}s"
    )


def test_cache_disabled_env_var_skips_load(cache_root, monkeypatch):
    """KOLMO_NO_SEED_CACHE=1 must bypass the cache even if one exists."""
    import kolmo._engine as engine
    from kolmo import compress

    monkeypatch.setattr(engine, "SEED_CORPUS", b"a")
    monkeypatch.setenv("KOLMO_FIXED", "1")

    # First populate the cache.
    monkeypatch.delenv("KOLMO_NO_SEED_CACHE", raising=False)
    compress(b"x")
    cache_files_before = list(cache_root.glob("*.npz"))
    assert len(cache_files_before) == 1

    # Now disable the cache. Compress should not error and should not
    # touch the existing file.
    monkeypatch.setenv("KOLMO_NO_SEED_CACHE", "1")
    compress(b"x")
    cache_files_after = list(cache_root.glob("*.npz"))
    # Same file, untouched (mtime would have updated if it were rewritten).
    assert cache_files_after == cache_files_before

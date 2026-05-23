"""Bit-identity tests for the fixed-point KV cache path.

The whole point of the cache is to speed up inference *without changing the
output a single bit*. If warm+step ever diverges from `fixed_forward` over the
extended history, cross-machine determinism breaks silently. These tests pin
the equivalence down at the bit level.
"""

import numpy as np
import pytest

from kolmo.fixed_kv_cache import (
    fixed_step,
    fixed_warm,
    trim_caches,
)
from kolmo.fixed_model import extract_fixed_weights, fixed_forward
from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model


@pytest.fixture
def tiny_weights():
    model = KolmoTransformer(d_model=64, n_heads=4, n_layers=2, max_context=128)
    stable_init_model(model, seed=42)
    return extract_fixed_weights(model), {"n_heads": 4, "n_layers": 2}


def test_fixed_warm_matches_fixed_forward_last_row(tiny_weights):
    """warm should produce bit-identical logits to fixed_forward's last row."""
    weights, cfg = tiny_weights
    history = np.array([1, 2, 3, 100, 200, 42, 7, 9], dtype=np.int64)

    last_full = fixed_forward(history, weights, **cfg)[-1]
    last_warm, _ = fixed_warm(history, weights, **cfg)

    assert last_warm.dtype == np.int32
    assert last_warm.shape == last_full.shape
    assert np.array_equal(last_warm, last_full)


def test_fixed_warm_returns_per_layer_caches(tiny_weights):
    """Cache shape sanity: (n_heads, T, d_head) per layer."""
    weights, cfg = tiny_weights
    history = np.array([5, 6, 7, 8, 9], dtype=np.int64)
    T = len(history)
    d_head = 64 // cfg["n_heads"]

    _, caches = fixed_warm(history, weights, **cfg)

    assert len(caches) == cfg["n_layers"]
    for layer_cache in caches:
        assert layer_cache["k"].shape == (cfg["n_heads"], T, d_head)
        assert layer_cache["v"].shape == (cfg["n_heads"], T, d_head)
        assert layer_cache["k"].dtype == np.int32
        assert layer_cache["v"].dtype == np.int32


def test_fixed_step_matches_full_forward(tiny_weights):
    """One step of the cache should equal a full forward over the extended history."""
    weights, cfg = tiny_weights
    history = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    new_byte = 77

    _, caches = fixed_warm(history, weights, **cfg)
    last_step, _ = fixed_step(
        new_byte, caches, weights, pos_offset=len(history), **cfg
    )

    extended = np.append(history, new_byte)
    last_full = fixed_forward(extended, weights, **cfg)[-1]

    assert np.array_equal(last_step, last_full)


def test_fixed_step_iterated_matches_full_forward(tiny_weights):
    """Stepping N tokens in a row should still match a single full forward.

    This is the realistic compress-path workload: warm once, then stream a
    block of new tokens one at a time. If the cache drifts after a few steps,
    bytes silently mispredict.
    """
    weights, cfg = tiny_weights
    history = np.array([1, 2, 3, 4], dtype=np.int64)
    new_bytes = [11, 22, 33, 44, 55, 66, 77, 88]

    _, caches = fixed_warm(history, weights, **cfg)
    pos = len(history)
    extended = history.copy()
    for byte in new_bytes:
        last_step, caches = fixed_step(
            byte, caches, weights, pos_offset=pos, **cfg
        )
        pos += 1
        extended = np.append(extended, byte)
        last_full = fixed_forward(extended, weights, **cfg)[-1]
        assert np.array_equal(last_step, last_full), (
            f"step diverged after byte {byte} at pos {pos - 1}"
        )


def test_fixed_rope_warm_and_step_match_full_forward():
    """RoPE rotates Q/K before caching; cached K must still match a full
    forward over the same absolute positions exactly."""
    model = KolmoTransformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_context=128,
        use_rope=True,
    )
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)
    cfg = {"n_heads": 4, "n_layers": 2, "use_rope": True}
    history = np.array([1, 2, 3, 4], dtype=np.int64)
    new_bytes = [11, 22, 33, 44]

    last_warm, caches = fixed_warm(history, weights, **cfg)
    last_full = fixed_forward(history, weights, **cfg)[-1]
    assert np.array_equal(last_warm, last_full)

    pos = len(history)
    extended = history.copy()
    for byte in new_bytes:
        last_step, caches = fixed_step(
            byte,
            caches,
            weights,
            pos_offset=pos,
            **cfg,
        )
        extended = np.append(extended, byte)
        last_full = fixed_forward(extended, weights, **cfg)[-1]
        assert np.array_equal(last_step, last_full)
        pos += 1


def test_trim_caches_preserves_tail_rows(tiny_weights):
    """Trim is a pure slice — it must keep the last `max_len` K/V rows
    untouched at the bit level, on every layer.

    Note: a trimmed cache is *not* equivalent to running `fixed_forward` on
    the trimmed history. The surviving K/V rows were produced inside the
    full-history attention context (earlier layers' attention saw all of
    history), so they carry information that a fresh forward over the
    trimmed window wouldn't reproduce. Both compress and decompress trim
    identically though, so determinism survives — that's what matters.
    """
    weights, cfg = tiny_weights
    history = np.arange(20, dtype=np.int64) % 256
    keep = 8

    _, caches = fixed_warm(history, weights, **cfg)
    trimmed = trim_caches(caches, max_len=keep)

    assert len(trimmed) == len(caches)
    for orig, trim in zip(caches, trimmed, strict=True):
        assert trim["k"].shape[1] == keep
        assert trim["v"].shape[1] == keep
        assert np.array_equal(trim["k"], orig["k"][:, -keep:, :])
        assert np.array_equal(trim["v"], orig["v"][:, -keep:, :])


def test_fixed_step_after_trim_is_deterministic(tiny_weights):
    """Trim+step must be bit-identical across runs.

    This is the cross-machine claim in microcosm: same inputs, same trim,
    same step → same logits. Compress and decompress will both follow this
    path with the same arguments, so they stay in lockstep even after the
    cache loses old history.
    """
    weights, cfg = tiny_weights
    history = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.int64)
    keep = 6
    new_byte = 99

    _, c1 = fixed_warm(history, weights, **cfg)
    c1 = trim_caches(c1, keep)
    l1, _ = fixed_step(new_byte, c1, weights, pos_offset=len(history), **cfg)

    _, c2 = fixed_warm(history, weights, **cfg)
    c2 = trim_caches(c2, keep)
    l2, _ = fixed_step(new_byte, c2, weights, pos_offset=len(history), **cfg)

    assert np.array_equal(l1, l2)


def test_trim_caches_no_op_when_below_limit(tiny_weights):
    weights, cfg = tiny_weights
    history = np.array([1, 2, 3], dtype=np.int64)
    _, caches = fixed_warm(history, weights, **cfg)

    out = trim_caches(caches, max_len=10)
    for c_in, c_out in zip(caches, out, strict=True):
        # Shape unchanged.
        assert c_out["k"].shape == c_in["k"].shape
        assert np.array_equal(c_in["k"], c_out["k"])
        assert np.array_equal(c_in["v"], c_out["v"])


def test_fixed_warm_then_step_is_deterministic_across_calls(tiny_weights):
    """Two independent runs over the same input should produce identical
    logits and identical caches — this is the cross-machine claim, just done
    locally (the same code on the same machine is the easiest stress case)."""
    weights, cfg = tiny_weights
    history = np.array([10, 20, 30, 40], dtype=np.int64)

    l1, c1 = fixed_warm(history, weights, **cfg)
    l2, c2 = fixed_warm(history, weights, **cfg)
    assert np.array_equal(l1, l2)
    for a, b in zip(c1, c2, strict=True):
        assert np.array_equal(a["k"], b["k"])
        assert np.array_equal(a["v"], b["v"])

    s1, _ = fixed_step(55, c1, weights, pos_offset=4, **cfg)
    s2, _ = fixed_step(55, c2, weights, pos_offset=4, **cfg)
    assert np.array_equal(s1, s2)

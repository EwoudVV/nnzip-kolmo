"""Integration tests for fixed-point training blocks."""

import numpy as np

from kolmo.fixed_model import extract_fixed_weights
from kolmo.fixed_train import fixed_train_block
from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model


def _copy_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in weights.items()}


def test_fixed_train_block_is_bit_deterministic():
    model = KolmoTransformer(
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_context=64,
    )
    stable_init_model(model, seed=42)
    initial = extract_fixed_weights(model)

    weights_a = _copy_weights(initial)
    weights_b = _copy_weights(initial)
    history = [0] + list(b"the quick brown ")
    blocks = [list(b"fox jumps over "), list(b"the lazy dog.")]

    state_a = None
    state_b = None
    for block in blocks:
        state_a = fixed_train_block(
            weights_a,
            state_a,
            history,
            block,
            n_heads=4,
            n_layers=1,
            context=64,
        )
        state_b = fixed_train_block(
            weights_b,
            state_b,
            history,
            block,
            n_heads=4,
            n_layers=1,
            context=64,
        )
        history = (history + block)[-64:]

    assert state_a is not None and state_b is not None
    assert state_a.step == state_b.step == 2
    for name in weights_a:
        assert np.array_equal(weights_a[name], weights_b[name]), name
    for name in state_a.m:
        assert np.array_equal(state_a.m[name], state_b.m[name]), name
        assert np.array_equal(state_a.v[name], state_b.v[name]), name

    changed = [
        not np.array_equal(initial[name], weights_a[name])
        for name in weights_a
    ]
    assert any(changed)

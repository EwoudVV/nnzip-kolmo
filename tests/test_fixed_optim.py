"""Tests for deterministic fixed-point optimizer."""

import numpy as np
import pytest

from kolmo.fixed import SCALE, dequantize, quantize
from kolmo.fixed_optim import fixed_adam_step, init_fixed_adam_state


def _float_adam_one_step(params: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """Reference Adam update using the same epsilon as fixed_adam_step."""
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0 / SCALE

    m = (1.0 - beta1) * grads
    v = (1.0 - beta2) * grads * grads
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    return params - lr * m_hat / (np.sqrt(v_hat) + eps)


def test_fixed_adam_one_step_tracks_float_reference():
    rng = np.random.default_rng(22)
    params = rng.normal(size=(8, 8)).astype(np.float64) * 0.1
    grads = rng.normal(size=(8, 8)).astype(np.float64) * 0.02

    params_q = {"w": quantize(params)}
    grads_q = {"w": quantize(grads)}
    p0 = dequantize(params_q["w"])
    g0 = dequantize(grads_q["w"])

    fixed_adam_step(params_q, grads_q)
    got = dequantize(params_q["w"])
    expected = _float_adam_one_step(p0, g0)

    assert np.max(np.abs(got - expected)) < 2.0 / SCALE


def test_fixed_adam_is_bit_deterministic_across_runs():
    rng = np.random.default_rng(23)
    initial = {"w": quantize(rng.normal(size=(4, 4)).astype(np.float64) * 0.1)}
    grad_sequence = [
        {"w": quantize(rng.normal(size=(4, 4)).astype(np.float64) * 0.02)}
        for _ in range(5)
    ]

    params_a = {"w": initial["w"].copy()}
    params_b = {"w": initial["w"].copy()}
    state_a = init_fixed_adam_state()
    state_b = init_fixed_adam_state()

    for grads in grad_sequence:
        state_a = fixed_adam_step(params_a, grads, state_a)
        state_b = fixed_adam_step(params_b, grads, state_b)

    assert np.array_equal(params_a["w"], params_b["w"])
    assert state_a.step == state_b.step
    assert state_a.beta1_pow_q30 == state_b.beta1_pow_q30
    assert state_a.beta2_pow_q30 == state_b.beta2_pow_q30
    assert np.array_equal(state_a.m["w"], state_b.m["w"])
    assert np.array_equal(state_a.v["w"], state_b.v["w"])


def test_fixed_adam_rejects_unknown_gradient():
    params = {"w": quantize(np.zeros((2, 2), dtype=np.float64))}
    grads = {"missing": quantize(np.zeros((2, 2), dtype=np.float64))}
    with pytest.raises(KeyError):
        fixed_adam_step(params, grads)


def test_fixed_adam_numba_matches_numpy_fallback(monkeypatch):
    """The fused-numba Adam kernel must produce bit-identical params/m/v as
    the pure-numpy fallback for every step, across a multi-step trajectory.
    This is the contract that lets fixed mode keep its cross-machine
    determinism guarantee — if the kernel and the fallback diverge by even
    one bit, fixed-mode blobs would be platform-dependent again.
    """
    from kolmo import _kernels

    if not _kernels.HAS_NUMBA:
        pytest.skip("numba not installed; nothing to compare against")

    rng = np.random.default_rng(123)
    init = quantize(rng.normal(size=(6, 7)).astype(np.float64) * 0.1)
    grads_seq = [
        {"w": quantize(rng.normal(size=(6, 7)).astype(np.float64) * 0.02)}
        for _ in range(6)
    ]

    # Numba path
    params_n = {"w": init.copy()}
    state_n = init_fixed_adam_state()
    for g in grads_seq:
        state_n = fixed_adam_step(params_n, {"w": g["w"].copy()}, state_n)

    # Numpy fallback path — flip HAS_NUMBA off + null out the kernel symbol
    # so the dispatch in fixed_adam_step falls through to the numpy block.
    monkeypatch.setattr(_kernels, "HAS_NUMBA", False)
    monkeypatch.setattr(_kernels, "_fused_adam_step_numba", None)
    params_p = {"w": init.copy()}
    state_p = init_fixed_adam_state()
    for g in grads_seq:
        state_p = fixed_adam_step(params_p, {"w": g["w"].copy()}, state_p)

    assert np.array_equal(params_n["w"], params_p["w"]), (
        "numba Adam diverged from numpy fallback in params"
    )
    assert np.array_equal(state_n.m["w"], state_p.m["w"]), (
        "numba Adam diverged from numpy fallback in m"
    )
    assert np.array_equal(state_n.v["w"], state_p.v["w"]), (
        "numba Adam diverged from numpy fallback in v"
    )
    assert state_n.beta1_pow_q30 == state_p.beta1_pow_q30
    assert state_n.beta2_pow_q30 == state_p.beta2_pow_q30

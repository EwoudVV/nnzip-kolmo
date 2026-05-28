"""Deterministic fixed-point optimizer.

Stage D of the bulletproof Rung 2 path: update Q15 weights using only integer
math. This module intentionally does not call PyTorch, NumPy float ops, or
platform libm. Given identical weights and gradients, it produces identical
updated weights and optimizer state on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kolmo import fixed

Q30: int = 1 << 30
M_EXTRA_BITS: int = 16
V_EXTRA_BITS: int = 16


@dataclass
class FixedAdamState:
    """Integer Adam state.

    `m` is stored in Q31 (Q15 plus 16 guard bits). `v` is stored in Q46
    (Q30 plus 16 guard bits) because it tracks squared Q15 gradients. The
    guard bits matter: Adam's `(1-beta1)` and `(1-beta2)` factors are smaller
    than one, and without extra precision small gradients would vanish before
    bias correction can restore them.
    """

    step: int
    m: dict[str, np.ndarray]
    v: dict[str, np.ndarray]
    beta1_pow_q30: int
    beta2_pow_q30: int


def init_fixed_adam_state() -> FixedAdamState:
    """Create an empty Adam state. Per-parameter arrays are allocated lazily."""
    return FixedAdamState(
        step=0,
        m={},
        v={},
        beta1_pow_q30=Q30,
        beta2_pow_q30=Q30,
    )


def _round_div(values: np.ndarray, divisor: int) -> np.ndarray:
    return fixed._round_div_int64(values.astype(np.int64), divisor)


def fixed_adam_step(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: FixedAdamState | None = None,
    *,
    lr_num: int = 1,
    lr_den: int = 1000,
    beta1_num: int = 9,
    beta1_den: int = 10,
    beta2_num: int = 999,
    beta2_den: int = 1000,
    eps_q15: int = 1,
) -> FixedAdamState:
    """Apply one deterministic Adam step in-place to `params`.

    Defaults match Adam's usual beta values and lr=0.001, except epsilon is
    one Q15 unit (`1/32768`) because `1e-8` is below Q15 resolution.
    """
    if state is None:
        state = init_fixed_adam_state()
    if lr_den <= 0 or beta1_den <= 0 or beta2_den <= 0:
        raise ValueError("denominators must be positive")

    state.step += 1
    state.beta1_pow_q30 = int((state.beta1_pow_q30 * beta1_num) // beta1_den)
    state.beta2_pow_q30 = int((state.beta2_pow_q30 * beta2_num) // beta2_den)
    one_minus_b1 = Q30 - state.beta1_pow_q30
    one_minus_b2 = Q30 - state.beta2_pow_q30

    for name, grad in grads.items():
        if name not in params:
            raise KeyError(f"gradient for unknown parameter: {name}")
        if params[name].dtype != np.int32 or grad.dtype != np.int32:
            raise TypeError("fixed_adam_step expects int32 params and grads")

        g = grad.astype(np.int64)
        m_old = state.m.get(name)
        v_old = state.v.get(name)
        if m_old is None:
            m_old = np.zeros_like(grad, dtype=np.int64)
        if v_old is None:
            v_old = np.zeros_like(grad, dtype=np.int64)

        # m_old/v_old are stored as int64 (FixedAdamState invariant), so the
        # previous `.astype(int64)` calls were no-op allocations. Skip them.
        g_q31 = g << M_EXTRA_BITS
        m_num = beta1_num * m_old + (beta1_den - beta1_num) * g_q31
        m_new = _round_div(m_num, beta1_den)

        g2_q30 = g * g
        max_shiftable = np.iinfo(np.int64).max >> V_EXTRA_BITS
        g2_q46 = np.minimum(g2_q30, max_shiftable) << V_EXTRA_BITS
        v_num = beta2_num * v_old + (beta2_den - beta2_num) * g2_q46
        v_new = _round_div(v_num, beta2_den)

        # Bias correction. m is Q31, v is Q46.
        # max_*_bias_correctable values bound m/v_for_hat so that the
        # subsequent <<30 shift can't overflow int64 (the shift is the
        # actual numerator going into one_minus_b1/b2 division).
        max_m_bias_correctable = np.iinfo(np.int64).max >> 30
        m_for_hat = np.clip(m_new, -max_m_bias_correctable, max_m_bias_correctable)
        m_hat_q31 = _round_div(m_for_hat << 30, one_minus_b1)
        m_hat = _round_div(m_hat_q31, 1 << M_EXTRA_BITS).astype(np.int32)
        v_for_hat = np.minimum(v_new, max_m_bias_correctable)
        v_hat_q46 = _round_div(v_for_hat << 30, one_minus_b2)
        # sqrt(Q46) is Q23. Shift down by 8 guard bits to get Q15.
        denom_q23 = fixed.isqrt_vec(np.maximum(v_hat_q46, 0))
        denom = _round_div(denom_q23, 1 << (V_EXTRA_BITS // 2)).astype(np.int32)
        denom = denom + np.int32(eps_q15)
        ratio = fixed.div_q15(m_hat, denom)
        update = _round_div(ratio.astype(np.int64) * lr_num, lr_den).astype(np.int32)

        params[name] = (params[name] - update).astype(np.int32)
        state.m[name] = m_new
        state.v[name] = v_new

    return state

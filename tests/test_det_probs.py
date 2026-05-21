"""Sanity tests for the deterministic integer-frequency converter.

These tests check the math on a single machine — they confirm that the
function produces well-formed integer frequencies for arithmetic coding.
Cross-machine determinism is tested separately by
benchmarks/determinism/hash_int_freqs.py.
"""

import numpy as np
import pytest

from kolmo.det_probs import LOGIT_GRID, TOTAL_FREQ, logits_to_int_freqs


def test_uniform_logits_give_uniform_counts():
    """If all logits are equal, counts should be uniform up to rounding."""
    logits = np.zeros(256, dtype=np.float64)
    counts = logits_to_int_freqs(logits)
    assert counts.sum() == TOTAL_FREQ
    # Every symbol gets either floor(2^16/256)=256 or ceil=256, so all equal.
    assert counts.min() >= 256 - 1
    assert counts.max() <= 256 + 1


def test_sharp_logits_concentrate_mass():
    """One huge logit should grab most of the available probability mass.

    Upper bound is TOTAL_FREQ - (vocab_size - 1) because every other symbol
    is floored to at least 1 count.
    """
    logits = np.zeros(256, dtype=np.float64)
    logits[42] = 10.0  # exp(10)/(exp(10)+255) ≈ 98.85% before flooring
    counts = logits_to_int_freqs(logits)
    assert counts.sum() == TOTAL_FREQ
    assert counts[42] > 0.98 * TOTAL_FREQ  # close to the achievable ceiling
    # And every other symbol still got at least 1 (no zeros allowed).
    assert counts.min() >= 1


def test_counts_sum_exactly_to_total():
    """The redistribution step must produce a sum exactly equal to TOTAL_FREQ."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        logits = rng.normal(size=256).astype(np.float64)
        counts = logits_to_int_freqs(logits)
        assert int(counts.sum()) == TOTAL_FREQ


def test_quantization_kills_tiny_perturbations():
    """The whole point: float errors below the grid should not change counts.

    Add ~1e-7 noise (machine-level disagreement) and check the result is
    unchanged.
    """
    rng = np.random.default_rng(1)
    logits = rng.normal(size=256).astype(np.float64)
    counts_a = logits_to_int_freqs(logits)
    perturbed = logits + (rng.standard_normal(256) * 1e-7).astype(np.float64)
    counts_b = logits_to_int_freqs(perturbed)
    # The grid is ~6.1e-5, so 1e-7 noise should never cross a grid boundary.
    assert np.array_equal(counts_a, counts_b)


def test_quantization_eventually_loses_resolution():
    """Sanity: differences much larger than the grid SHOULD change counts.
    This is a regression guard against accidentally building an identity
    function."""
    logits = np.zeros(256, dtype=np.float64)
    counts_a = logits_to_int_freqs(logits)
    logits[0] = 5.0  # much bigger than LOGIT_GRID
    counts_b = logits_to_int_freqs(logits)
    assert not np.array_equal(counts_a, counts_b)


def test_no_count_is_zero():
    """constriction requires every symbol to have at least probability 1/total.
    Even if the float prob is tiny, we floor it to 1."""
    logits = np.full(256, -100.0, dtype=np.float64)
    logits[7] = 0.0  # everything else is effectively zero in float
    counts = logits_to_int_freqs(logits)
    assert counts.min() >= 1
    assert counts[7] > TOTAL_FREQ - 256  # 7 grabs almost all mass


def test_rejects_non_1d_input():
    with pytest.raises(ValueError):
        logits_to_int_freqs(np.zeros((2, 256), dtype=np.float64))


def test_grid_resolution_is_finer_than_observed_error():
    """The grid (1/16384 ≈ 6.1e-5) needs to be larger than the cross-machine
    float error (~1e-7 measured by Codex's probe). Otherwise the whole scheme
    doesn't work."""
    cross_machine_observed_error = 1e-7
    assert LOGIT_GRID > 100 * cross_machine_observed_error

"""Predictor + Mixer framework for the literal model.

The architectural target — eventually — is a PAQ-style ensemble: many cheap
predictors, each producing a (256,) probability distribution over the next
byte, combined by a mixer that learns per-context which predictors to trust.
That's the canonical Hutter-winning shape, but the path there has many
predictors and mixer variants to design, bench, and discard.

This module is the *foundation* for that work. It defines two interfaces —
`Predictor` and `Mixer` — and two concrete classes that wrap the existing
post-copy and cost-aware-adaptive logic exactly as they were inlined in
`LiteralModel`. The behavioral output is bit-for-bit identical after this
refactor; what's new is the **extension point**: a new predictor is now a
class with two methods and a registration call, not a chunk of inlined
logic in the middle of `LiteralModel.probs()`.

Design choices and why:

- `Predictor.probs()` returns `np.ndarray | None`. The None case lets a
  predictor signal "I have nothing to say at this byte" (post-copy is the
  obvious example — it's silent unless the previous event was a copy).
  The mixer then ignores that predictor for the current prediction.

- `Predictor.observe(byte)` and `Predictor.mark_copy_end(byte)` are both
  in the base — the latter is a no-op by default. We accept the small
  cost of a no-op method call per copy event to keep `LiteralModel`'s
  forwarding loop clean and uniform.

- `Mixer.combine()` takes predictor outputs as a `dict[name, probs]` so
  mixers can refer to specific predictors by name (e.g. the cost-aware
  adaptive mixer reads PPM's peak to compute its adaptive weight; a
  generic logistic mixer ignores predictor names and treats them all
  the same). The `neural_probs` argument is special-cased because the
  transformer is outside this framework — it lives in the PyTorch /
  fixed-mode pipeline and gets passed in.

- Cross-OS determinism: every reduction in this module uses `math.fsum`
  for the same reason explained in `_engine.py`'s `_ppm_distribution`.
  numpy's `.sum()` is platform-dependent in the last ULP across SIMD
  widths and numpy versions; `math.fsum` is correctly-rounded and
  order-independent, which keeps the int frequencies passed to the
  arithmetic coder identical on every machine.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class Predictor:
    """A cheap online-updated predictor for the literal byte stream.

    Each concrete predictor owns its own state. The literal model orchestrates
    a collection of them and feeds the orchestration result to the mixer.

    Subclasses must implement `probs()` and `observe()`. `mark_copy_end()`
    has a default no-op implementation — override it only if your predictor
    cares about copy-event boundaries.
    """

    #: A short identifier used by mixers that need to refer to specific
    #: predictors (e.g. the cost-aware mixer looks for "ppm" to compute its
    #: adaptive weight). Subclasses must set this.
    name: str = ""

    def probs(self) -> np.ndarray | None:
        """Return a `(256,) float64` distribution over the next byte, or
        `None` to signal "I have no opinion at this byte" (the mixer will
        ignore this predictor for the current prediction)."""
        raise NotImplementedError

    def observe(self, byte: int) -> None:
        """Update internal state given the byte that was just emitted."""
        raise NotImplementedError

    def mark_copy_end(self, last_byte: int) -> None:
        """Tell the predictor the most recent event was a copy whose final
        byte was `last_byte`. Default: no-op. Override to react to copy
        boundaries."""


class Mixer:
    """Combines predictor outputs and the neural distribution into a final
    probability distribution.

    The neural distribution is passed in separately because the transformer
    lives outside the Predictor framework (it's in PyTorch or fixed-mode
    integer math) and has a different lifecycle.
    """

    def combine(
        self,
        predictor_outputs: dict[str, np.ndarray | None],
        neural_probs: np.ndarray,
    ) -> np.ndarray:
        """Combine predictor distributions and the neural distribution into
        a `(256,) float64` output that sums to ~1.0. The mixer is responsible
        for the final renormalization."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete predictors
# ---------------------------------------------------------------------------


class PostCopyPredictor(Predictor):
    """Conditional distribution over the next byte given that the previous
    event was a copy.

    Maintains `counts[last_byte_of_copy, next_byte_observed]`. After a copy
    event ends, `mark_copy_end(last_byte)` arms the predictor; the next
    `observe(byte)` records the (last, observed) transition and disarms.
    `probs()` returns the row for `last_byte_of_copy` when armed, `None`
    when not — so the mixer naturally ignores it during long literal runs.

    This is the predictor that lived inline in `LiteralModel` before the
    framework refactor. Behavior is unchanged; only the home address moved.
    """

    name = "post_copy"

    def __init__(self):
        # Float64 with prior 1.0 so an unseen (last, next) pair gives a
        # uniform-with-weight-1 distribution before any observations. The
        # prior is identical to the inline version.
        self.counts = np.full((256, 256), 1.0, dtype=np.float64)
        self._armed = False
        self._last_byte = 0

    def probs(self) -> np.ndarray | None:
        if not self._armed:
            return None
        row = self.counts[self._last_byte]
        # fsum, not np.sum: see comment at the top of this module.
        return row / math.fsum(row)

    def observe(self, byte: int) -> None:
        if self._armed:
            self.counts[self._last_byte, byte] += 1.0
            self._armed = False

    def mark_copy_end(self, last_byte: int) -> None:
        self._armed = True
        self._last_byte = int(last_byte)


# ---------------------------------------------------------------------------
# Concrete mixer
# ---------------------------------------------------------------------------


class CostAwareAdaptiveMixer(Mixer):
    """The default mixer (pre-refactor behavior, lifted out of LiteralModel).

    Two- or three-way linear blend of `neural` + `ppm` + optional `post_copy`.
    The neural weight is set by a cost-aware rule: when PPM is sharply peaked
    (one byte highly likely) the mixer trusts PPM more and gives less weight
    to neural; when PPM is near-uniform (cold context) the mixer trusts
    neural more. The post-copy predictor, when it has something to say,
    eats from PPM's share of the blend, not neural's — the intuition is that
    PPM and post-copy are both "structural" signals that partly overlap.

    The constructor takes the weight knobs that used to be module-level
    constants. Defaults match the shipped behavior.
    """

    def __init__(
        self,
        *,
        adaptive: bool,
        static_neural_weight: float,
        neural_weight_low: float,
        neural_weight_high: float,
        post_copy_enabled: bool,
        post_copy_weight: float,
    ):
        self.adaptive = adaptive
        self.static_neural_weight = static_neural_weight
        self.neural_weight_low = neural_weight_low
        self.neural_weight_high = neural_weight_high
        self.post_copy_enabled = post_copy_enabled
        self.post_copy_weight = post_copy_weight

    def combine(
        self,
        predictor_outputs: dict[str, np.ndarray | None],
        neural_probs: np.ndarray,
    ) -> np.ndarray:
        # PPM is required by this mixer — if it's not registered, the
        # configuration is wrong; raising is better than silently producing
        # a degenerate distribution.
        ppm = predictor_outputs.get("ppm")
        if ppm is None:
            raise RuntimeError(
                "CostAwareAdaptiveMixer requires a 'ppm' predictor"
            )

        # Normalize neural (fsum: see top-of-module comment about cross-OS
        # determinism).
        p_neural = neural_probs.astype(np.float64, copy=False)
        n_sum = math.fsum(p_neural)
        if n_sum > 0.0:
            p_neural = p_neural / n_sum

        # Adaptive vs static neural weight.
        if self.adaptive:
            # max() is a per-element comparison (no SIMD reduction), so it's
            # platform-deterministic regardless of numpy version. float() to
            # promote to Python scalar so the subsequent arithmetic uses
            # IEEE-754 scalar ops.
            peak = float(ppm.max())
            peak_norm = (peak - 1.0 / 256.0) / (1.0 - 1.0 / 256.0)
            if peak_norm < 0.0:
                peak_norm = 0.0
            elif peak_norm > 1.0:
                peak_norm = 1.0
            w_neural = (
                self.neural_weight_high * (1.0 - peak_norm)
                + self.neural_weight_low * peak_norm
            )
        else:
            w_neural = self.static_neural_weight

        # Optional 3-way blend with post-copy. The post-copy term eats from
        # PPM's share (1 - w_neural), capped to avoid a negative PPM weight.
        post_copy = predictor_outputs.get("post_copy")
        if self.post_copy_enabled and post_copy is not None:
            pc = self.post_copy_weight
            if pc > 1.0 - w_neural:
                pc = 1.0 - w_neural
            w_ppm = 1.0 - w_neural - pc
            mixed = w_neural * p_neural + w_ppm * ppm + pc * post_copy
        else:
            mixed = w_neural * p_neural + (1.0 - w_neural) * ppm

        return mixed / math.fsum(mixed)


class LinearEnsembleMixer(Mixer):
    """Generic N-way linear blend of predictor outputs and the neural
    distribution. Static weights set at construction time.

    Constructed from a list of `(name, weight)` pairs. The special name
    "neural" refers to the `neural_probs` argument passed to `combine()`;
    every other name must match a key in the `predictor_outputs` dict.

    None-handling: if a predictor returns None at combine time (e.g.
    PostCopy when it's not armed), it is silently dropped and its weight
    is redistributed proportionally across the remaining contributors.
    This matches how the cost-aware mixer collapses to a 2-way blend
    when post-copy is silent.

    Cross-OS determinism: same `math.fsum`-everywhere policy as the
    cost-aware mixer.

    This mixer is opt-in (KOLMO_MIXER=linear). The default for now
    remains the cost-aware adaptive blend — flipping the default would
    change ratio output. Once the linear mixer has been benched against
    the cost-aware one at scale and a winning config identified, we'll
    revisit.
    """

    def __init__(self, weights: list[tuple[str, float]]):
        if not weights:
            raise ValueError("LinearEnsembleMixer needs at least one weight")
        # Allow duplicate names for now — the last one wins, which is
        # surprising but matches how dict() handles duplicate keys.
        # Negative weights are rejected because they aren't meaningful
        # in a probability mixture (the renormalization at the end can't
        # rescue a negative contribution).
        for name, weight in weights:
            if not isinstance(name, str) or not name:
                raise ValueError(f"predictor name must be a non-empty string, got {name!r}")
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(
                    f"weight for {name!r} must be a non-negative number, got {weight!r}"
                )
        self.weights = [(name, float(weight)) for name, weight in weights]

    def combine(
        self,
        predictor_outputs: dict[str, np.ndarray | None],
        neural_probs: np.ndarray,
    ) -> np.ndarray:
        # Normalize neural the same way the cost-aware mixer does.
        p_neural = neural_probs.astype(np.float64, copy=False)
        n_sum = math.fsum(p_neural)
        if n_sum > 0.0:
            p_neural = p_neural / n_sum

        # Collect the active contributors: (probs, weight) pairs whose
        # predictor produced a non-None output (or whose name is "neural").
        active: list[tuple[np.ndarray, float]] = []
        for name, weight in self.weights:
            if weight <= 0.0:
                continue
            if name == "neural":
                active.append((p_neural, weight))
            else:
                probs = predictor_outputs.get(name)
                if probs is not None:
                    active.append((probs, weight))

        if not active:
            # Defensive fallback: a configuration where nothing contributed
            # (e.g., only post_copy is in the weights and post_copy is
            # silent right now). Uniform distribution beats crashing.
            return np.full(256, 1.0 / 256.0, dtype=np.float64)

        # Renormalize the active weights so they sum to 1. fsum to keep
        # platform determinism.
        total = math.fsum(w for _, w in active)
        if total <= 0.0:
            return np.full(256, 1.0 / 256.0, dtype=np.float64)

        # Linear blend — accumulate weighted contributions element-wise
        # (no reduction step, so deterministic per-element).
        result = np.zeros(256, dtype=np.float64)
        for probs, weight in active:
            result += (weight / total) * probs

        # Final renormalization (small ULP drift from the per-element
        # adds and the / total above).
        return result / math.fsum(result)


def parse_linear_weights(spec: str) -> list[tuple[str, float]]:
    """Parse a `KOLMO_LINEAR_WEIGHTS` value into a list of (name, weight)
    pairs. The format is comma-separated `name:weight` entries, e.g.
    `neural:0.4,ppm:0.5,post_copy:0.1`. Whitespace around tokens is
    stripped. Raises ValueError on malformed input — better to surface
    a config error at module load than silently fall back to a default.
    """
    result: list[tuple[str, float]] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"KOLMO_LINEAR_WEIGHTS entry {raw_part!r} is missing ':'; "
                f"expected 'name:weight'"
            )
        name, _, weight_str = part.partition(":")
        name = name.strip()
        weight_str = weight_str.strip()
        if not name:
            raise ValueError(
                f"KOLMO_LINEAR_WEIGHTS entry {raw_part!r} has an empty name"
            )
        try:
            weight = float(weight_str)
        except ValueError:
            raise ValueError(
                f"KOLMO_LINEAR_WEIGHTS entry {raw_part!r} has a non-numeric weight"
            ) from None
        result.append((name, weight))
    if not result:
        raise ValueError("KOLMO_LINEAR_WEIGHTS must contain at least one entry")
    return result

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
# Helpers shared between predictors
# ---------------------------------------------------------------------------


_MASK64 = 0xFFFFFFFFFFFFFFFF


def literal_context_bucket(context: int, buckets: int) -> int:
    """Map a byte context integer to a hashed literal-model bucket.

    SplitMix64 finalizer. Used by the hashed order-3/4/5 PPM tables to
    spread contexts that share suffix bytes across the whole table
    instead of collapsing them into a tiny fraction of buckets the way a
    plain multiplicative hash would. See PPMPredictor for callers.
    """
    if buckets <= 0:
        raise ValueError("bucket count must be positive")
    x = (int(context) + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    x = (x ^ (x >> 31)) & _MASK64
    if buckets & (buckets - 1) == 0:
        return x & (buckets - 1)
    return x % buckets


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


class PPMPredictor(Predictor):
    """PPM-C with full backoff over orders {5, 4, 3, 2, 1, 0}.

    Owns its own state — count tables for each order plus a five-byte
    history. `probs()` walks the orders from longest to shortest and
    accumulates probability mass with escape-on-novel-byte semantics
    (the same algorithm that lived inline in `LiteralModel`
    pre-refactor).

    Memory profile (with default bucket sizes):
      count0  256                           float64    ~2 KB
      count1  256 * 256                     float64    ~512 KB
      count2  65536 * 256                   uint32     ~64 MB
      count3  (1<<16) * 256  if enabled     uint16     ~32 MB
      count4  (1<<18) * 256  if enabled     uint16     ~128 MB
      count5  (1<<18) * 256  if enabled     uint16     ~128 MB

    Order-3 / 4 / 5 are hashed because the dense table size grows like
    256^k and orders past 2 are infeasible dense. `literal_context_bucket`
    (SplitMix64 finalizer) avalanches contexts that share suffix bytes
    across the whole table.

    Cross-OS determinism: every float reduction in this class uses
    `math.fsum` (correctly-rounded, order-independent across SIMD widths
    and numpy versions). Integer reductions and per-element ops are
    deterministic by IEEE-754 / int arithmetic semantics.
    """

    name = "ppm"

    #: Default bucket sizes for the hashed orders. Match what the engine
    #: shipped pre-refactor; can be overridden at construction.
    DEFAULT_ORDER3_BUCKETS = 1 << 16
    DEFAULT_ORDER4_BUCKETS = 1 << 18
    DEFAULT_ORDER5_BUCKETS = 1 << 18

    def __init__(
        self,
        prior: float = 1.0,
        *,
        enable_order3: bool = False,
        enable_order4: bool = True,
        enable_order5: bool = False,
        order3_buckets: int = DEFAULT_ORDER3_BUCKETS,
        order4_buckets: int = DEFAULT_ORDER4_BUCKETS,
        order5_buckets: int = DEFAULT_ORDER5_BUCKETS,
        bos: int = 0,
    ):
        self.count0 = np.full(256, prior, dtype=np.float64)
        self.count1 = np.full((256, 256), prior, dtype=np.float64)
        self.count2 = np.zeros((256 * 256, 256), dtype=np.uint32)
        self.count3 = (
            np.zeros((order3_buckets, 256), dtype=np.uint16)
            if enable_order3
            else None
        )
        self.count4 = (
            np.zeros((order4_buckets, 256), dtype=np.uint16)
            if enable_order4
            else None
        )
        self.count5 = (
            np.zeros((order5_buckets, 256), dtype=np.uint16)
            if enable_order5
            else None
        )
        self._order3_buckets = order3_buckets
        self._order4_buckets = order4_buckets
        self._order5_buckets = order5_buckets
        self.prev5 = bos
        self.prev4 = bos
        self.prev3 = bos
        self.prev2 = bos
        self.prev = bos

    def probs(self) -> np.ndarray:
        """PPM-C backoff walk over orders 5..0.

        At each order: compute the conditional distribution from the
        relevant count row, allocate a chunk of the remaining escape mass
        to the byte values seen at this order, carry the rest forward as
        escape to the next order down. Byte values not seen at any order
        get a small share of the residual escape distributed uniformly.
        """
        p = np.zeros(256, dtype=np.float64)
        accounted = np.zeros(256, dtype=bool)
        escape = 1.0

        def fold(row: np.ndarray) -> None:
            nonlocal escape
            # row is float64 of small exact-integer counts; sum is small
            # enough to be exact regardless of order, but fsum makes that
            # guarantee version-independent. seen.sum() is a bool reduction
            # to an integer count, deterministic by construction.
            s = math.fsum(row)
            if s <= 0.0:
                return
            seen = row > 0.0
            distinct = float(seen.sum())
            denom = s + distinct
            mask = seen & ~accounted
            p[mask] += escape * row[mask] / denom
            accounted[seen] = True
            escape *= distinct / denom

        # Order 5 (hashed) — only walked if enabled.
        if self.count5 is not None:
            ctx5 = (
                (self.prev5 << 32)
                | (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            fold(
                self.count5[
                    literal_context_bucket(ctx5, self._order5_buckets)
                ].astype(np.float64)
            )

        # Order 4 (hashed)
        if self.count4 is not None:
            ctx4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            fold(
                self.count4[
                    literal_context_bucket(ctx4, self._order4_buckets)
                ].astype(np.float64)
            )

        # Order 3 (hashed) — disabled by default.
        if self.count3 is not None:
            ctx3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            fold(
                self.count3[
                    literal_context_bucket(ctx3, self._order3_buckets)
                ].astype(np.float64)
            )

        # Order 2 (dense)
        ctx2 = (self.prev2 << 8) | self.prev
        fold(self.count2[ctx2].astype(np.float64))

        # Order 1 (dense, float counts include prior)
        fold(self.count1[self.prev])

        # Order 0 (float counts always positive due to prior=1.0)
        fold(self.count0)

        # Anything still unaccounted (only possible if all orders had
        # s=0, which can't happen because order 0 always has prior=1.0).
        # Spread remaining escape uniformly as a defensive fallback.
        unaccounted = ~accounted
        n_unaccounted = int(unaccounted.sum())
        if n_unaccounted > 0:
            p[unaccounted] += escape / 256.0

        # fsum, not np.sum, for the same reason as everywhere else.
        return p / math.fsum(p)

    def observe(self, byte: int) -> None:
        """Update count tables for every active order, then shift the
        five-byte history. The history shift comes AFTER the count
        updates so the counts use the pre-shift context."""
        self.count0[byte] += 1.0
        self.count1[self.prev, byte] += 1.0
        context2 = (self.prev2 << 8) | self.prev
        self.count2[context2, byte] += 1
        if self.count3 is not None:
            context3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            bucket3 = literal_context_bucket(context3, self._order3_buckets)
            if self.count3[bucket3, byte] < np.iinfo(np.uint16).max:
                self.count3[bucket3, byte] += 1
        if self.count4 is not None:
            context4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket4 = literal_context_bucket(context4, self._order4_buckets)
            if self.count4[bucket4, byte] < np.iinfo(np.uint16).max:
                self.count4[bucket4, byte] += 1
        if self.count5 is not None:
            context5 = (
                (self.prev5 << 32)
                | (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket5 = literal_context_bucket(context5, self._order5_buckets)
            if self.count5[bucket5, byte] < np.iinfo(np.uint16).max:
                self.count5[bucket5, byte] += 1
        # Shift the five-byte history.
        self.prev5 = self.prev4
        self.prev4 = self.prev3
        self.prev3 = self.prev2
        self.prev2 = self.prev
        self.prev = byte


class WordFragmentPredictor(Predictor):
    """Predicts the next byte given the current word-internal byte fragment.

    Tracks bytes since the last whitespace delimiter (space, newline, tab,
    carriage return). The last K bytes of the current fragment are hashed
    into a bounded count table, producing a distribution over the next byte.

    Why this is novel:
      PPM's order-K contexts span word boundaries, so its counts for a
      byte pair like "th" include both word-internal transitions ("the",
      "that", "there") and cross-boundary noise (period+space, comma+space,
      tag boundaries). WordFragmentPredictor only fires inside words, so
      its statistics are purer — "th" inside a word is almost always
      followed by a vowel, never by space or punctuation. This is genuinely
      complementary signal that cmix-style LSTM mixers don't explicitly
      model.

    The table is deliberately small (1<<12 buckets * 256 bytes ≈ 8 MB
    float64). Collisions smear rare fragments into nearby ones, which acts
    as a soft regularizer.

    Cross-OS determinism: same `math.fsum`-and-SplitMix64 policy as the
    rest of the framework.
    """

    name = "word_fragment"

    # Whitelist of characters that reset the fragment. These are true word
    # boundaries — inside a word we track the raw byte sequence.
    _DELIMS = frozenset({ord(" "), ord("\n"), ord("\t"), ord("\r")})

    def __init__(
        self,
        max_context_len: int = 4,
        table_buckets: int = 1 << 12,
    ):
        self.max_context_len = max_context_len
        self.table_buckets = table_buckets
        # Prior 1.0 per (fragment_hash, next_byte) — unseen combinations
        # get a uniform nudge instead of impossible zero.
        self.counts = np.ones((table_buckets, 256), dtype=np.float64)
        self._fragment: list[int] = []

    def probs(self) -> np.ndarray | None:
        if not self._fragment:
            return None
        ctx = self._fragment[-self.max_context_len :]
        bucket = literal_context_bucket(
            _encode_context(ctx, self.max_context_len),
            self.table_buckets,
        )
        row = self.counts[bucket]
        return row / math.fsum(row)

    def observe(self, byte: int) -> None:
        # Record the transition from the CURRENT fragment state (before
        # updating). If the fragment was non-empty, the last probs() call
        # produced a distribution using it, and now we know which byte
        # actually followed that fragment — that's the learning signal.
        # If the fragment was empty, this predictor was silent (probs()
        # returned None), so there's nothing to learn from this byte for
        # this predictor; we just start building the fragment for future
        # predictions.
        if self._fragment:
            ctx = self._fragment[-self.max_context_len :]
            bucket = literal_context_bucket(
                _encode_context(ctx, self.max_context_len),
                self.table_buckets,
            )
            # Float64 can hold counts up to ~1e308; the 1e200 cap is
            # defensive against any edge-case overflow.
            if self.counts[bucket, int(byte)] < 1e200:
                self.counts[bucket, int(byte)] += 1.0

        # Update the fragment for the NEXT prediction.
        if byte in self._DELIMS:
            self._fragment.clear()
        else:
            self._fragment.append(int(byte))

    def mark_copy_end(self, last_byte: int) -> None:
        # Copy events end mid-word or at a word boundary. The copied bytes
        # have already been observed via observe() calls (from
        # observe_byte_sequence in compress/decompress), so the fragment
        # already reflects the post-copy state. Nothing extra to do.
        pass


class BalancedDelimiterPredictor(Predictor):
    """Predicts the next byte given the current nesting depth of bracket
    characters.

    Tracks four bracket types: curly ({}), square ([]), paren (()), and
    angle (<>). Each depth is clamped to the range 0..3, producing a
    4-bit-deep * 4-bracket-type = 256-state encoding. The count table
    records transitions from each state to the next byte observed.

    Why this is novel:
      PPM's byte-N-gram model loses depth information after ~4 bytes
      inside a [[link]] or {{template}} — at that point the active n-gram
      context is just the last few raw bytes, regardless of whether we are
      1 or 5 levels deep in nested brackets. BalancedDelimiterPredictor
      retains that structural state across arbitrarily long spans, which
      gives it orthogonal signal: inside a template (depth > 0), `|` and
      `}` are far more likely than normal; inside a wiki link, `|` and `]`
      are far more likely.

    The state is updated on every observe() call. Brackets pairs are tracked
    by the open/close character individually: each `{` increments the curly
    counter, each `}` decrements it. This naturally handles `{{...}}` (two
    increments from the two `{` bytes, two decrements from the two `}` bytes)
    without needing explicit multi-character detection.

    Cross-OS determinism: same ``math.fsum``-and-SplitMix64 policy as the
    rest of the framework.
    """

    name = "balanced_delimiter"

    def __init__(self):
        # 4 depth counters, each clamped to 0..3 → 4^4 = 256 states.
        self.curly_depth = 0
        self.square_depth = 0
        self.paren_depth = 0
        self.angle_depth = 0
        # Count table with prior 1.0 — unseen (state, byte) pairs get
        # a uniform nudge instead of zero.
        self.counts = np.ones((256, 256), dtype=np.float64)

    def _state(self) -> int:
        c = min(self.curly_depth, 3)
        s = min(self.square_depth, 3)
        p = min(self.paren_depth, 3)
        a = min(self.angle_depth, 3)
        return c | (s << 2) | (p << 4) | (a << 6)

    def probs(self) -> np.ndarray:
        row = self.counts[self._state()]
        return row / math.fsum(row)

    def observe(self, byte: int) -> None:
        # Record the transition from the state BEFORE this byte, then
        # update depths based on the byte we just observed.
        state_before = self._state()
        self.counts[state_before, int(byte)] += 1.0

        byte_ = int(byte)
        if byte_ == ord("{"):
            self.curly_depth += 1
        elif byte_ == ord("}"):
            self.curly_depth = max(0, self.curly_depth - 1)
        elif byte_ == ord("["):
            self.square_depth += 1
        elif byte_ == ord("]"):
            self.square_depth = max(0, self.square_depth - 1)
        elif byte_ == ord("("):
            self.paren_depth += 1
        elif byte_ == ord(")"):
            self.paren_depth = max(0, self.paren_depth - 1)
        elif byte_ == ord("<"):
            self.angle_depth += 1
        elif byte_ == ord(">"):
            self.angle_depth = max(0, self.angle_depth - 1)

    def mark_copy_end(self, last_byte: int) -> None:
        pass


class AfterNumberPredictor(Predictor):
    """Predicts the next byte given the current digit-run state.

    Tracks whether we are (a) inside a digit sequence, (b) on the first
    non-digit byte immediately after a digit sequence, or (c) in normal
    text. Each state captures a different next-byte distribution:

      NORMAL (0):       broad distribution; digits possible but rare.
      IN_NUMBER (1):    heavily favors digits 0-9; also `.`, `,`, `:`.
      AFTER_NUMBER (2): favors `.`, ` `, `]`, `}`, `\n`, `|`, `,`, `-`.

    Why this is novel:
      PPM's byte-N-gram model memorises specific number sequences (e.g.
      "1024." is a distinct context from "2048.") and cannot generalise
      across them. AfterNumberPredictor treats ALL digit runs identically,
      so the transition "IN_NUMBER → `.`" is learned once and applies to
      every number regardless of its digits. This is genuinely orthogonal
      signal: in enwik9, numbers appear as page IDs, timestamps,
      coordinates, and sizes — each with a distinctive following-byte
      distribution that PPM cannot exploit until it has seen the exact
      same number multiple times.

    The 3-state machine uses 768 float64 entries (~6 KB) with a prior
    of 1.0. Every byte position maps to exactly one state, so probs()
    never returns None.
    """

    name = "after_number"

    #: ASCII codes for digits 0-9.
    _DIGITS = frozenset({ord(str(d)) for d in range(10)})

    # State constants for readability.
    NORMAL = 0
    IN_NUMBER = 1
    AFTER_NUMBER = 2

    def __init__(self):
        self.counts = np.ones((3, 256), dtype=np.float64)
        self._state = self.NORMAL

    def probs(self) -> np.ndarray:
        row = self.counts[self._state]
        return row / math.fsum(row)

    def observe(self, byte: int) -> None:
        # Record the transition from the state BEFORE this byte.
        self.counts[self._state, int(byte)] += 1.0

        # Advance the state machine.
        if int(byte) in self._DIGITS:
            self._state = self.IN_NUMBER
        elif self._state == self.IN_NUMBER:
            self._state = self.AFTER_NUMBER
        else:
            self._state = self.NORMAL

    def mark_copy_end(self, last_byte: int) -> None:
        pass


class InTextPredictor(Predictor):
    """Predicts the next byte given whether we are inside XML text content
    (between ``>`` and ``<``) or inside XML markup (between ``<`` and ``>``).

    The state machine is trivial: ``>`` sets ``in_text = True``, ``<`` sets
    ``in_text = False``. This captures the fundamental XML/document structure
    of enwik9: inside ``<text>...</text>`` we see wiki markup (letters, ``{{``,
    ``[[``, ``|``, ``\n``, ``#``, ``*``, ``'``); outside — in XML tags and
    metadata — we see ``<``, ``>``, ``"``, ``/``, ``=``, and digits from
    IDs and timestamps.

    Why this is novel:
      PPM at order-5 sees a 5-byte sliding window that cannot encode "we are
      100 bytes into a text block." After 100 bytes inside ``<text>...</text>``,
      the 5-byte context is something like ``tent.`` or ``he qu`` — there is no
      trace of the ``<text>`` tag that opened the region. InTextPredictor's
      single bit (``in_text`` vs not) is preserved across arbitrarily long
      runs, and the byte distributions for the two states differ dramatically:
      ``"`` appears only in XML attribute values (markup state), ``'`` appears
      only in wiki text. Near-perfect discrimination at minimal cost.

    The count table is (2, 256) float64 = ~4 KB with prior 1.0. Every byte
    position maps to exactly one state, so ``probs()`` never returns ``None``.
    """

    name = "in_text"

    def __init__(self):
        self.counts = np.ones((2, 256), dtype=np.float64)
        self._in_text = False

    def probs(self) -> np.ndarray:
        row = self.counts[1 if self._in_text else 0]
        return row / math.fsum(row)

    def observe(self, byte: int) -> None:
        state_before = 1 if self._in_text else 0
        self.counts[state_before, int(byte)] += 1.0

        byte_ = int(byte)
        if byte_ == ord(">"):
            self._in_text = True
        elif byte_ == ord("<"):
            self._in_text = False

    def mark_copy_end(self, last_byte: int) -> None:
        pass


class PositionModuloPredictor(Predictor):
    """Predicts the next byte given the distance from the last ``\\n``.

    Tracks the byte position within the current line (0..10) and lumps all
    positions 11+ into a single catch-all bucket. Each line-start bucket
    (0—10) sees a highly distinctive byte distribution shaped by enwik9's
    wiki markup structure:

      bucket 0 (line start):  ``*`` 63x, ``{`` 18x, ``#`` 13x, ``=`` 10x
      bucket 1:               ``{`` 18x, ``=`` 11x, ``[``  9x
      bucket 4:               ``:`` 24x (namespace prefix after ``{{``)
      bucket 5—7:             UTF-8 continuation bytes + ``<`` / ``>``
      bucket 11+ (long-line): ~global average

    Why this is novel:
      PPM at order-5 cannot encode "I am 55 bytes into a line." After 55 bytes
      its context is the last 5 raw bytes (e.g. ``tent.``) — no trace of the
      line-start structure. PositionModuloPredictor retains a byte-position
      counter across arbitrarily long lines and builds per-position statistics.

      **Scale matters.** At 16 KB this predictor sees ~200 line starts — noisy.
      At the Hutter Prize scale (100 MB) it sees ~1.2M line starts with
      converged per-bucket distributions. Bucket 0 alone accumulates 1.2M
      observations of what follows ``\\n``, distinguishing ``*`` (bullets),
      ``{`` (templates), ``=`` (headings), ``\\n`` (blank lines), ``#``
      (numbered lists), ``|`` (table rows), and dozens more at sub-percent
      precision — all invisible to PPM's fixed-order window.

    The count table is (12, 256) float64 = ~24 KB with prior 1.0. Every
    byte maps to exactly one bucket, so ``probs()`` never returns ``None``.
    """

    name = "position_modulo"

    def __init__(self):
        self.counts = np.ones((12, 256), dtype=np.float64)
        self._pos = 0

    def probs(self) -> np.ndarray:
        bucket = self._pos if self._pos < 11 else 11
        return self.counts[bucket] / math.fsum(self.counts[bucket])

    def observe(self, byte: int) -> None:
        bucket = self._pos if self._pos < 11 else 11
        self.counts[bucket, int(byte)] += 1.0

        if int(byte) == ord("\n"):
            self._pos = 0
        else:
            self._pos += 1

    def mark_copy_end(self, last_byte: int) -> None:
        pass


def _encode_context(ctx: list[int], max_len: int) -> int:
    """Pack the last up-to-`max_len` bytes of `ctx` into an integer.

    Zero-pads on the left so that short fragments get a consistent hash.
    For example, with max_len=4 and fragment ['h', 'i'], the context
    becomes 0x00_00_68_69 (little-endian byte order).
    """
    value = 0
    # Iterate from the end so the most recent byte is in the lowest 8 bits.
    for i, b in enumerate(reversed(ctx)):
        if i >= max_len:
            break
        value = (value << 8) | int(b)
    return value


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

"""Shared logic for compress and decompress.

Both directions need to walk the model through the *same* training trajectory:
build identical weights, predict identical probabilities, take identical
optimizer steps. The KV cache lets each direction do most of the predictions
incrementally (O(T) per byte instead of O(T²)); the training step still needs
a full forward over the recent history with gradient tracking, but that's
only done once per block.
"""

import os
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from kolmo.det_probs import TOTAL_FREQ, logits_to_int_freqs
from kolmo.fixed import dequantize
from kolmo.fixed_kv_cache import fixed_step, fixed_warm, trim_caches
from kolmo.fixed_model import extract_fixed_weights, tied_param_pairs
from kolmo.fixed_optim import FixedAdamState
from kolmo.fixed_train import fixed_train_block
from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model

SEED = 42
LR = 1e-3
# Linear warmup ramps LR from 0 to LR over the first N optimizer steps.
# Bigger models need this — without warmup, Adam's first few steps with
# zero-initialized m/v moments cause large updates that can destabilize
# training (a known problem at d_model >= ~384). 100 steps ≈ the first
# 1600 bytes of input, well before any training-block-size doubling.
LR_WARMUP_STEPS = 100
_CONTEXT_ENV = os.environ.get("KOLMO_CONTEXT")
CONTEXT = int(_CONTEXT_ENV) if _CONTEXT_ENV else 256  # sliding-window cap (max tokens kept in KV cache)
BLOCK_SIZE = 16  # base bytes between optimizer steps (early in file)
# Sublinear training schedule: training interval doubles every N bytes of
# input seen, capped by CONTEXT-1 so every training slice still has at least
# one preceding token to predict the first byte of the block. Rationale: the
# model adapts fastest in the first few KB. After that each additional Adam
# step contributes less per byte.
#
# Schedule sweep at 4KB (skip-prime, mixed prose):
#   doubling=4096 (old): 4096->1868  ratio=0.4561  total=107.5s
#   doubling=2048      : 4096->1864  ratio=0.4551  total= 92.5s   (-14%, -0.10pp)
# 2048 is a clear win — slightly better ratio AND faster, because halving the
# doubling distance gets the schedule to bigger block sizes sooner, which
# saves more backward calls without losing meaningful early-file adaptation.
_TRAIN_SCHEDULE_DOUBLING_BYTES = 2048
_TRAIN_SCHEDULE_MAX_MULT = 32


def training_block_size_at(bytes_observed: int) -> int:
    """How many bytes to accumulate before the next optimizer step, given
    that `bytes_observed` bytes have already been processed.

    Both compress and decompress call this with the same argument at the
    same point in the trajectory, so they agree on every training step
    boundary without exchanging any extra state.
    """
    bucket = bytes_observed // _TRAIN_SCHEDULE_DOUBLING_BYTES
    mult = min(1 << bucket, _TRAIN_SCHEDULE_MAX_MULT)
    return min(BLOCK_SIZE * mult, CONTEXT - 1)
BOS = 0  # implicit start-of-stream byte, never written to disk
COPY_PROB = 0.005
COPY_WINDOW = 65536
COPY_MIN = 8
# COPY_MAX=256 was capping ~75% of copy bytes at the ceiling on 16KB English
# (27 of 29 saturated copies in the structural-repetition regime). Bumping to
# 1024 lets long Wikipedia-style template / citation / header blocks collapse
# into a single copy event instead of N adjacent 256-length copies, each
# paying its own event flag + offset + length header.
COPY_MAX = 1024
COPY_CANDIDATES = 64
# Encoder-side heuristic for copy selection. A copy event is used only if its
# adaptive event+offset+length header costs less than spelling the same bytes
# as literals at this proxy bpb. This is deliberately conservative for enwik:
# current RoPE runs are ~3.1 bpb at 32KB, and long-file literals should get
# cheaper as the model adapts, so short/far copies need to clear a real bar.
COPY_LITERAL_BPB = 2.75
COPY_USE_LITERAL_MODEL_PROXY = False
# Adaptive literal side model mixed into neural byte probabilities. This is
# mirrored by the decoder and costs zero blob bytes. It learns file-local byte
# statistics much faster than the transformer's gradient updates, especially
# for wiki markup and punctuation. Strong mixes hurt enwik; the default is a
# order-2 carries the file-local byte structure; keep small order-1/order-0
# backoff nudges for contexts that are still cold.
LITERAL_ORDER2_WEIGHT = 0.40
LITERAL_ORDER1_WEIGHT = 0.02
LITERAL_ORDER0_WEIGHT = 0.005
# 0 means "use the full order-2 weight after the context has been seen once".
# Positive values ramp order-2 trust as count/(count + confidence), useful if
# one-observation contexts overfit.
LITERAL_ORDER2_CONFIDENCE = 2.0
LITERAL_ORDER3_WEIGHT = 0.0
LITERAL_ORDER3_CONFIDENCE = 2.0
LITERAL_ORDER3_BUCKETS = 1 << 16
LITERAL_ORDER4_WEIGHT = 0.20
LITERAL_ORDER4_CONFIDENCE = 2.0
LITERAL_ORDER4_BUCKETS = 1 << 18
LITERAL_ORDER5_WEIGHT = 0.0
LITERAL_ORDER5_CONFIDENCE = 4.0
LITERAL_ORDER5_BUCKETS = 1 << 18
_MASK64 = 0xFFFFFFFFFFFFFFFF


def literal_context_bucket(context: int, buckets: int) -> int:
    """Map a byte context integer to a hashed literal-model bucket.

    The high-order literal tables are bounded and hashed. The old code used
    `context * odd_constant % buckets`; when `buckets` is a power of two, that
    mostly preserves low-bit structure. For byte contexts, low bits are just
    the most recent byte(s), so many distinct order-4 contexts collapsed into
    surprisingly few buckets on enwik prefixes.

    SplitMix64's finalizer gives a cheap avalanche: nearby contexts and
    contexts sharing suffix bytes spread across the whole table. Collisions
    still happen (bounded memory is the point), but they become random noise
    instead of systematic suffix aliasing.
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


# Seed corpus: baked into both encoder and decoder code, costs zero bytes in
# the compressed blob, but trains the model to a useful starting state before
# the user's data is touched. Bigger and more diverse = better prior on common
# English/Wikipedia patterns = fewer bits per literal byte once real data
# starts.
#
# The previous version repeated one paragraph three times, which mostly
# taught the model "this paragraph repeats." Replaced with non-repetitive
# content covering the regimes Hutter (enwik8) actually contains: prose,
# dialogue, lists, tables, references, markup, dates, numbers, code, math.
SEED_CORPUS = (
    # English prose — letter/word/punctuation regularities.
    b"English text is full of small regularities. Letters form words, words "
    b"form phrases, and phrases repeat with punctuation, spacing, and rhythm. "
    b"A compressor that begins from a blank model wastes bits learning that "
    b"spaces are common, vowels follow consonants, and sentences end with a "
    b"period followed by a space and a capital letter. The model should "
    b"already know all of this before the first user byte arrives.\n\n"

    # Wikipedia article-style — title, intro, infobox-ish lines, references.
    b"Compression (information theory)\n\n"
    b"In information theory, data compression is the process of encoding "
    b"information using fewer bits than the original representation. Any "
    b"particular compression is either lossy or lossless. Lossless compression "
    b"reduces bits by identifying and eliminating statistical redundancy, so "
    b"that no information is lost. Lossy compression reduces bits by removing "
    b"unnecessary or less important information.[1]\n\n"
    b"The process of reducing the size of a data file is often referred to as "
    b"data compression. In the context of data transmission, it is called "
    b"source coding: encoding done at the source of the data before it is "
    b"stored or transmitted.[2] Source coding should not be confused with "
    b"channel coding, for error detection and correction, or line coding, "
    b"the means for mapping data onto a signal.\n\n"
    b"See also: entropy (information theory), Kolmogorov complexity, "
    b"arithmetic coding, Huffman coding, Lempel-Ziv-Welch.\n\n"
    b"References\n"
    b"1. Wade, Graham (1994). Signal coding and processing. ISBN 978-0-521-42336-6.\n"
    b"2. Mahdi, O.A.; Mohammed, M.A.; Mohamed, A.J. (November 2012). "
    b"\"Implementing a Novel Approach an Convert Audio Compression to Text Coding "
    b"via Hybrid Technique\". International Journal of Computer Science Issues. 9 "
    b"(6, No. 3): 53-59.\n\n"

    # Wiki markup — links, templates, italics, headers.
    b"== History ==\n\n"
    b"The theoretical basis for compression is provided by [[information theory]] "
    b"and, more specifically, [[Algorithmic information theory|algorithmic "
    b"information theory]] for lossless compression and [[rate-distortion theory]] "
    b"for lossy compression. These fields of study were essentially forged by "
    b"[[Claude Shannon]], who published fundamental papers on the topic in the "
    b"late 1940s and early 1950s. Other topics associated with compression "
    b"include [[coding theory]] and [[statistical inference]].\n\n"
    b"{{Main|Lossless compression}}\n"
    b"Lossless data compression algorithms usually exploit "
    b"[[statistical redundancy]] to represent data without losing any "
    b"[[information]], so that the process is reversible.\n\n"

    # Dialogue — handles colons, names, line breaks.
    b"Dialogue:\n"
    b"Alice: Does the model remember the phrase from earlier in the file?\n"
    b"Ben: It remembered letters and short words, but not the exact sentence.\n"
    b"Alice: Then we need a better prior, a longer context, or an explicit "
    b"copy mechanism for repeated text.\n"
    b"Ben: We already have a copy mechanism. It catches matches above eight "
    b"bytes within a sixty-four-kilobyte window.\n\n"

    # Markdown — lists, code, emphasis.
    b"# Notes on the build\n\n"
    b"- Train deterministically; the encoder and decoder must agree on every "
    b"bit produced.\n"
    b"- Keep the model architecture identical on both sides; any drift in "
    b"weights between compress and decompress breaks the round-trip.\n"
    b"- Measure `gzip`, `kolmo`, ratio, and wall time on every change.\n"
    b"- Revert changes that only help tiny inputs at the cost of large ones.\n"
    b"- The seed corpus is part of the algorithm, not part of the data; it "
    b"costs nothing in the output blob.\n\n"
    b"```python\n"
    b"def compress(data: bytes) -> bytes:\n"
    b"    model = build_model()\n"
    b"    return arithmetic_encode(predict_stream(model, data))\n"
    b"```\n\n"

    # Numbers, dates, units, currencies — common token shapes.
    b"Numbers and dates: 2026-05-22, 1,024 bytes, 2,048 bytes, 4,096 bytes, "
    b"65,536 entries, 10^6 iterations, 3.14159, 2.71828, -273.15 C, 98.6 F, "
    b"$1.99, 49.95 EUR, GBP 12.50, 12:30 PM, 23:59 UTC, 1989-1992, ca. 1850, "
    b"version 1.0.3, RFC 8259, ISO 8601.\n\n"

    # Sentence-level variety — questions, exclamations, parenthetical asides.
    b"Why does a transformer help here? Because text contains both local "
    b"spelling rules (which a small context handles) and long-range reuse "
    b"(which attention captures). A model that handles only local rules will "
    b"plateau; one with useful memory keeps improving as the document grows. "
    b"Note: the model is reset to its seed-warmed state at the start of every "
    b"file, so no information leaks between separate runs.\n\n"

    # Tables — pipes and column structure.
    b"| Algorithm | Type     | Year | Use case               |\n"
    b"|-----------|----------|------|------------------------|\n"
    b"| Huffman   | static   | 1952 | symbol-by-symbol       |\n"
    b"| LZ77      | dictionary | 1977 | general-purpose      |\n"
    b"| Arithmetic | statistical | 1976 | per-bit precision   |\n"
    b"| Neural    | learned  | 2010s | context-sensitive      |\n\n"

    # Math / LaTeX-ish.
    b"Entropy: H(X) = -sum p(x) log p(x), where the base of the logarithm "
    b"determines the unit (bits for log_2, nats for ln). Cross-entropy: "
    b"H(p, q) = -sum p(x) log q(x). The expected code length under arithmetic "
    b"coding equals H(p, q), so a better model q gives shorter blobs. "
    b"KL divergence: D(p || q) = H(p, q) - H(p) >= 0.\n\n"

    # Closing prose — repeats some words from above to reinforce.
    b"A final passage to round out the seed: the city library kept rows of "
    b"shelves, tables, lamps, catalog records, quiet readers, printed forms, "
    b"and old magazines. The same words return in nearby sentences and the "
    b"compressor should pay fewer bits each time a pattern becomes familiar. "
    b"That is the entire point of online learning: shape the distribution to "
    b"match the data as the data arrives.\n"
)
EVENT_PROBS = np.array([1.0 - COPY_PROB, COPY_PROB], dtype=np.float64)


@dataclass
class FixedModelState:
    """Fixed-point model state used when KOLMO_FIXED=1."""

    weights: dict[str, np.ndarray]
    optimizer_state: FixedAdamState | None = None
    n_heads: int = 8
    n_layers: int = 4
    use_rope: bool = False
    # Pairs of (canonical, alias) parameter names that share underlying
    # weights — used to sum gradients before Adam and re-alias after.
    tied_params: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tied_params is None:
            self.tied_params = []


def _use_fixed() -> bool:
    return os.environ.get("KOLMO_FIXED", "").lower() in {"1", "true", "yes"}


def _skip_prime() -> bool:
    return os.environ.get("KOLMO_SKIP_PRIME", "").lower() in {"1", "true", "yes"}


def _use_rope() -> bool:
    value = os.environ.get("KOLMO_USE_ROPE")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes"}


# Model presets for hyperparameter sweeps. The "full" preset is the production
# default. The "draft" preset trades ~1pp of ratio for ~2x speed — useful when
# iterating on copy / literal / schedule tuning where the ratio delta between
# configs is what matters, not absolute ratio. Blobs are NOT interchangeable
# across presets; set KOLMO_MODEL on both sides.
_MODEL_PRESETS = {
    "full": dict(d_model=256, n_heads=8, n_layers=4),
    "draft": dict(d_model=192, n_heads=6, n_layers=3),
    # Scaling-law experiment: bigger than full. Earlier 10M-at-4-KB test
    # showed no benefit because there wasn't enough training data; theory
    # says it should win once the model has seen enough bytes to actually
    # use the extra capacity. ~11 M params.
    "large": dict(d_model=384, n_heads=8, n_layers=6),
}


def _model_preset() -> str:
    name = os.environ.get("KOLMO_MODEL", "full").lower()
    if name not in _MODEL_PRESETS:
        raise ValueError(
            f"unknown KOLMO_MODEL preset {name!r}; "
            f"choices: {sorted(_MODEL_PRESETS)}"
        )
    return name


def offset_probs(n: int) -> np.ndarray:
    """Static prior over offset values 0..n-1 (representing actual offsets
    1..n). Uses 1/sqrt(k) — a reasonable starting point before any events
    are observed. Used by OffsetModel as the initial Laplace prior."""
    if n <= 0:
        return np.array([], dtype=np.float64)
    raw = 1.0 / np.sqrt(np.arange(1, n + 1, dtype=np.float64))
    return raw / raw.sum()


def length_probs(n: int) -> np.ndarray:
    """Probability distribution over length-MIN values 0..n-1. Favors shorter
    matches via 1/k decay — match-length distribution is steeper than offset
    distribution in practice."""
    if n <= 0:
        return np.array([], dtype=np.float64)
    raw = 1.0 / np.arange(1, n + 1, dtype=np.float64)
    return raw / raw.sum()


class LengthModel:
    """Adaptive probability model for copy lengths using log buckets.

    Lengths are represented as offsets from COPY_MIN:

      length_offset = length - COPY_MIN  # 0..COPY_MAX-COPY_MIN

    The old model encoded that offset as one categorical symbol over up to
    1017 choices. That works, but after COPY_MAX=1024 long matches pay for a
    large flat alphabet even though lengths are naturally log-ish: exact 8-byte
    copies are common, then ranges like 9-10, 11-14, 15-22, etc.

    This mirrors OffsetModel:

      1. bucket = floor(log2(length_offset + 1))
      2. residual = length_offset - bucket_lo

    The initial bucket/residual priors are derived from the old 1/k exact
    length prior, so first-copy behavior stays close while the adaptive model
    learns whether a file prefers tiny or long matches.
    """

    def __init__(
        self,
        n: int,
        prior_strength: float = 16.0,
        residual_prior_strength: float = 8.0,
    ):
        self.n = n
        offsets = np.arange(n, dtype=np.int64)
        buckets = np.array(
            [self.bucket_for(int(o)) for o in offsets],
            dtype=np.int64,
        )
        exact_prior = length_probs(n)
        prior = np.bincount(
            buckets,
            weights=exact_prior,
            minlength=n.bit_length(),
        )
        prior = prior / prior.sum() * prior_strength
        self.counts = prior.astype(np.float64)
        self.residual_counts: list[np.ndarray] = []
        for bucket in range(n.bit_length()):
            lo, hi = self.bucket_bounds(bucket, n)
            exact = exact_prior[lo : hi + 1].copy()
            exact = exact / exact.sum() * residual_prior_strength
            self.residual_counts.append(exact.astype(np.float64))

    @staticmethod
    def bucket_for(length_offset: int) -> int:
        if length_offset < 0:
            raise ValueError("length offset must be non-negative")
        return (length_offset + 1).bit_length() - 1

    @staticmethod
    def bucket_bounds(bucket: int, max_n: int) -> tuple[int, int]:
        if bucket < 0:
            raise ValueError("bucket must be non-negative")
        if max_n <= 0:
            raise ValueError("max_n must be positive")
        lo = (1 << bucket) - 1
        hi = min((1 << (bucket + 1)) - 2, max_n - 1)
        if lo > hi:
            raise ValueError("bucket is not legal for max_n")
        return lo, hi

    def probs_for(self, max_n: int) -> np.ndarray:
        """Return normalized probabilities over legal length buckets."""
        if max_n <= 0:
            return np.array([], dtype=np.float64)
        p = self.counts[: max_n.bit_length()].copy()
        return p / p.sum()

    def residual_probs_for(self, bucket: int, max_n: int) -> np.ndarray:
        lo, hi = self.bucket_bounds(bucket, max_n)
        width = hi - lo + 1
        p = self.residual_counts[bucket][:width].copy()
        return p / p.sum()

    def observe(self, length_offset: int) -> None:
        bucket = self.bucket_for(length_offset)
        lo, _ = self.bucket_bounds(bucket, self.n)
        self.counts[bucket] += 1.0
        self.residual_counts[bucket][length_offset - lo] += 1.0


class EventModel:
    """Adaptive probability model for the literal/copy event flag.

    The fixed EVENT_PROBS = [0.995, 0.005] assumes a 0.5% copy rate, but real
    text shows 5-15% rates once enough history is available. This costs ~7.6
    bits per copy flag with the static prior; with adaptation, copies in long
    files cost ~3 bits.

    Both encoder and decoder hold an instance and call `observe` after every
    event, in the same order — distribution evolves bit-identically.
    """

    def __init__(self, prior_copy: float = 0.05, prior_strength: float = 50.0):
        self.copy_count = prior_copy * prior_strength
        self.literal_count = (1.0 - prior_copy) * prior_strength

    def probs(self) -> np.ndarray:
        total = self.copy_count + self.literal_count
        return np.array(
            [self.literal_count / total, self.copy_count / total],
            dtype=np.float64,
        )

    def observe(self, event: int) -> None:
        if event == 1:
            self.copy_count += 1.0
        else:
            self.literal_count += 1.0


class LiteralModel:
    """Adaptive byte-context model mixed with neural literal probabilities.

    The transformer adapts via comparatively expensive optimizer steps. This
    model adapts immediately after every observed byte, including bytes emitted
    by copy events, and captures cheap file-local regularities such as:
    - after '<' in wiki/XML markup, letters and '/' are common
    - after '[' another '[' is common
    - after '\n' markup bullets, headings, and capitals are common

    It is deliberately bounded: order-0 counts, a dense order-1 table, a dense
    order-2 table, and optional hashed order-3/order-4/order-5 tables. Dense
    exact order-3 would be too large (2^24 contexts * 256 next bytes), and
    dense exact order-4/order-5 is completely out. The hashed tables are
    fixed-size and collisions only smear the distribution.
    """

    def __init__(self, prior: float = 1.0):
        self.count0 = np.full(256, prior, dtype=np.float64)
        self.count1 = np.full((256, 256), prior, dtype=np.float64)
        self.count2 = np.zeros((256 * 256, 256), dtype=np.uint32)
        self.count3 = (
            np.zeros((LITERAL_ORDER3_BUCKETS, 256), dtype=np.uint16)
            if LITERAL_ORDER3_WEIGHT > 0.0
            else None
        )
        self.count4 = (
            np.zeros((LITERAL_ORDER4_BUCKETS, 256), dtype=np.uint16)
            if LITERAL_ORDER4_WEIGHT > 0.0
            else None
        )
        self.count5 = (
            np.zeros((LITERAL_ORDER5_BUCKETS, 256), dtype=np.uint16)
            if LITERAL_ORDER5_WEIGHT > 0.0
            else None
        )
        self.prev5 = BOS
        self.prev4 = BOS
        self.prev3 = BOS
        self.prev2 = BOS
        self.prev = BOS

    def probs(self, neural_probs: np.ndarray) -> np.ndarray:
        p = neural_probs.astype(np.float64, copy=True)
        if (
            LITERAL_ORDER0_WEIGHT <= 0.0
            and LITERAL_ORDER1_WEIGHT <= 0.0
            and LITERAL_ORDER2_WEIGHT <= 0.0
            and LITERAL_ORDER3_WEIGHT <= 0.0
            and LITERAL_ORDER4_WEIGHT <= 0.0
            and LITERAL_ORDER5_WEIGHT <= 0.0
        ):
            return p / p.sum()

        order5_w = 0.0
        p5 = p
        if self.count5 is not None:
            context5 = (
                (self.prev5 << 32)
                | (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket5 = literal_context_bucket(context5, LITERAL_ORDER5_BUCKETS)
            row5 = self.count5[bucket5]
            row5_sum = int(row5.sum())
            if row5_sum > 0:
                p5 = row5.astype(np.float64) / float(row5_sum)
                confidence5 = (
                    row5_sum / (row5_sum + LITERAL_ORDER5_CONFIDENCE)
                    if LITERAL_ORDER5_CONFIDENCE > 0.0
                    else 1.0
                )
                order5_w = LITERAL_ORDER5_WEIGHT * confidence5

        order4_w = 0.0
        p4 = p
        if self.count4 is not None:
            context4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket4 = literal_context_bucket(context4, LITERAL_ORDER4_BUCKETS)
            row4 = self.count4[bucket4]
            row4_sum = int(row4.sum())
            if row4_sum > 0:
                p4 = row4.astype(np.float64) / float(row4_sum)
                confidence4 = (
                    row4_sum / (row4_sum + LITERAL_ORDER4_CONFIDENCE)
                    if LITERAL_ORDER4_CONFIDENCE > 0.0
                    else 1.0
                )
                order4_w = LITERAL_ORDER4_WEIGHT * confidence4

        order3_w = 0.0
        p3 = p
        if self.count3 is not None:
            context3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            bucket3 = literal_context_bucket(context3, LITERAL_ORDER3_BUCKETS)
            row3 = self.count3[bucket3]
            row3_sum = int(row3.sum())
            if row3_sum > 0:
                p3 = row3.astype(np.float64) / float(row3_sum)
                confidence3 = (
                    row3_sum / (row3_sum + LITERAL_ORDER3_CONFIDENCE)
                    if LITERAL_ORDER3_CONFIDENCE > 0.0
                    else 1.0
                )
                order3_w = LITERAL_ORDER3_WEIGHT * confidence3

        p0 = self.count0 / self.count0.sum()
        row = self.count1[self.prev]
        p1 = row / row.sum()
        context2 = (self.prev2 << 8) | self.prev
        row2 = self.count2[context2]
        row2_sum = int(row2.sum())
        if row2_sum > 0:
            p2 = row2.astype(np.float64) / float(row2_sum)
            if LITERAL_ORDER2_CONFIDENCE > 0.0:
                confidence = row2_sum / (row2_sum + LITERAL_ORDER2_CONFIDENCE)
                order2_w = LITERAL_ORDER2_WEIGHT * confidence
            else:
                order2_w = LITERAL_ORDER2_WEIGHT
        else:
            p2 = p
            order2_w = 0.0
        neural_w = max(
            0.0,
            1.0
            - LITERAL_ORDER0_WEIGHT
            - LITERAL_ORDER1_WEIGHT
            - order2_w
            - order3_w
            - order4_w
            - order5_w,
        )
        mixed = (
            neural_w * p
            + LITERAL_ORDER0_WEIGHT * p0
            + LITERAL_ORDER1_WEIGHT * p1
            + order2_w * p2
            + order3_w * p3
            + order4_w * p4
            + order5_w * p5
        )
        return mixed / mixed.sum()

    def observe(self, byte: int) -> None:
        self.count0[byte] += 1.0
        self.count1[self.prev, byte] += 1.0
        context2 = (self.prev2 << 8) | self.prev
        self.count2[context2, byte] += 1
        if self.count3 is not None:
            context3 = (self.prev3 << 16) | (self.prev2 << 8) | self.prev
            bucket3 = literal_context_bucket(context3, LITERAL_ORDER3_BUCKETS)
            if self.count3[bucket3, byte] < np.iinfo(np.uint16).max:
                self.count3[bucket3, byte] += 1
        if self.count4 is not None:
            context4 = (
                (self.prev4 << 24)
                | (self.prev3 << 16)
                | (self.prev2 << 8)
                | self.prev
            )
            bucket4 = literal_context_bucket(context4, LITERAL_ORDER4_BUCKETS)
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
            bucket5 = literal_context_bucket(context5, LITERAL_ORDER5_BUCKETS)
            if self.count5[bucket5, byte] < np.iinfo(np.uint16).max:
                self.count5[bucket5, byte] += 1
        self.prev5 = self.prev4
        self.prev4 = self.prev3
        self.prev3 = self.prev2
        self.prev2 = self.prev
        self.prev = byte

    def proxy_bits(self, seq: bytes | bytearray, neural_bpb: float) -> float:
        """Cheap estimate of literal bits for a known byte sequence.

        Used only by the encoder when deciding whether a copy candidate is
        worth its header. We don't have future neural probabilities without
        actually stepping the transformer through the candidate, so this uses
        `neural_bpb` as a constant proxy for the neural component and adds the
        current adaptive byte-context probabilities for the actual bytes.

        The method intentionally does not mutate counts. It advances the local
        context variables while reading the current tables; that is enough to
        distinguish "the byte model expects this sequence" from "copy header is
        probably cheaper" without spending GPU work on rejected candidates.
        """
        if not seq:
            return 0.0
        base_p = 2.0 ** (-neural_bpb)
        count0_sum = self.count0.sum()
        prev5 = self.prev5
        prev4 = self.prev4
        prev3 = self.prev3
        prev2 = self.prev2
        prev = self.prev
        total_bits = 0.0
        for byte in seq:
            p0 = float(self.count0[byte] / count0_sum)
            row1 = self.count1[prev]
            p1 = float(row1[byte] / row1.sum())

            context2 = (prev2 << 8) | prev
            row2 = self.count2[context2]
            row2_sum = int(row2.sum())
            if row2_sum > 0:
                p2 = float(row2[byte] / row2_sum)
                if LITERAL_ORDER2_CONFIDENCE > 0.0:
                    confidence2 = row2_sum / (row2_sum + LITERAL_ORDER2_CONFIDENCE)
                    order2_w = LITERAL_ORDER2_WEIGHT * confidence2
                else:
                    order2_w = LITERAL_ORDER2_WEIGHT
            else:
                p2 = base_p
                order2_w = 0.0

            order5_w = 0.0
            p5 = base_p
            if self.count5 is not None:
                context5 = (
                    (prev5 << 32)
                    | (prev4 << 24)
                    | (prev3 << 16)
                    | (prev2 << 8)
                    | prev
                )
                bucket5 = literal_context_bucket(context5, LITERAL_ORDER5_BUCKETS)
                row5 = self.count5[bucket5]
                row5_sum = int(row5.sum())
                if row5_sum > 0:
                    p5 = float(row5[byte] / row5_sum)
                    confidence5 = (
                        row5_sum / (row5_sum + LITERAL_ORDER5_CONFIDENCE)
                        if LITERAL_ORDER5_CONFIDENCE > 0.0
                        else 1.0
                    )
                    order5_w = LITERAL_ORDER5_WEIGHT * confidence5

            order4_w = 0.0
            p4 = base_p
            if self.count4 is not None:
                context4 = (
                    (prev4 << 24) | (prev3 << 16) | (prev2 << 8) | prev
                )
                bucket4 = literal_context_bucket(context4, LITERAL_ORDER4_BUCKETS)
                row4 = self.count4[bucket4]
                row4_sum = int(row4.sum())
                if row4_sum > 0:
                    p4 = float(row4[byte] / row4_sum)
                    confidence4 = (
                        row4_sum / (row4_sum + LITERAL_ORDER4_CONFIDENCE)
                        if LITERAL_ORDER4_CONFIDENCE > 0.0
                        else 1.0
                    )
                    order4_w = LITERAL_ORDER4_WEIGHT * confidence4

            order3_w = 0.0
            p3 = base_p
            if self.count3 is not None:
                context3 = (prev3 << 16) | (prev2 << 8) | prev
                bucket3 = (
                    literal_context_bucket(context3, LITERAL_ORDER3_BUCKETS)
                )
                row3 = self.count3[bucket3]
                row3_sum = int(row3.sum())
                if row3_sum > 0:
                    p3 = float(row3[byte] / row3_sum)
                    confidence3 = (
                        row3_sum / (row3_sum + LITERAL_ORDER3_CONFIDENCE)
                        if LITERAL_ORDER3_CONFIDENCE > 0.0
                        else 1.0
                    )
                    order3_w = LITERAL_ORDER3_WEIGHT * confidence3

            neural_w = max(
                0.0,
                1.0
                - LITERAL_ORDER0_WEIGHT
                - LITERAL_ORDER1_WEIGHT
                - order2_w
                - order3_w
                - order4_w
                - order5_w,
            )
            p = (
                neural_w * base_p
                + LITERAL_ORDER0_WEIGHT * p0
                + LITERAL_ORDER1_WEIGHT * p1
                + order2_w * p2
                + order3_w * p3
                + order4_w * p4
                + order5_w * p5
            )
            total_bits += -np.log2(max(p, 1e-300))
            prev5 = prev4
            prev4 = prev3
            prev3 = prev2
            prev2 = prev
            prev = int(byte)
        return float(total_bits)


class OffsetModel:
    """Adaptive probability model for copy offset log-buckets.

    Both compress and decompress hold an instance and call `observe` after
    every copy event, in the same order with the same offsets — so the
    distribution evolves bit-identically on both sides.

    Encoding an exact offset in a 64 KB window as one categorical symbol is
    expensive: every copy event builds a 65,536-way model, and rare long
    offsets pay for a giant alphabet. Instead, encode:

      1. bucket = floor(log2(offset)) with adaptive bucket probabilities
      2. residual = offset - 2^bucket with adaptive within-bucket counts

    This is gzip-style distance coding. The initial bucket prior is derived
    from the old 1/sqrt(offset) prior by summing that mass into buckets, so
    the first-copy behavior remains sensible while the alphabet shrinks from
    65,536 symbols to at most 17 for the first stage. Residual priors are also
    initialized from 1/sqrt(offset), so the initial factorized probability is
    close to the old exact-offset prior while still allowing common exact
    offsets to become cheap.
    """

    def __init__(
        self,
        window: int,
        prior_strength: float = 128.0,
        residual_prior_strength: float = 16.0,
    ):
        self.window = window
        offsets = np.arange(1, window + 1, dtype=np.int64)
        buckets = np.array([self.bucket_for(int(o)) for o in offsets], dtype=np.int64)
        raw = 1.0 / np.sqrt(offsets.astype(np.float64))
        prior = np.bincount(buckets, weights=raw, minlength=window.bit_length())
        prior = prior / prior.sum() * prior_strength
        self.counts = prior.astype(np.float64)
        self.residual_counts: list[np.ndarray] = []
        for bucket in range(window.bit_length()):
            lo, hi = self.bucket_bounds(bucket, window)
            bucket_offsets = np.arange(lo, hi + 1, dtype=np.float64)
            residual_prior = 1.0 / np.sqrt(bucket_offsets)
            residual_prior = (
                residual_prior
                / residual_prior.sum()
                * residual_prior_strength
            )
            self.residual_counts.append(residual_prior.astype(np.float64))

    def probs_for(self, max_offset: int) -> np.ndarray:
        """Return normalized probabilities over legal offset buckets."""
        if max_offset <= 0:
            return np.array([], dtype=np.float64)
        p = self.counts[: max_offset.bit_length()].copy()
        return p / p.sum()

    @staticmethod
    def bucket_for(offset: int) -> int:
        if offset <= 0:
            raise ValueError("copy offset must be positive")
        return offset.bit_length() - 1

    @staticmethod
    def bucket_bounds(bucket: int, max_offset: int) -> tuple[int, int]:
        if bucket < 0:
            raise ValueError("bucket must be non-negative")
        lo = 1 << bucket
        hi = min((1 << (bucket + 1)) - 1, max_offset)
        if lo > hi:
            raise ValueError("bucket is not legal for max_offset")
        return lo, hi

    def residual_probs_for(self, bucket: int, max_offset: int) -> np.ndarray:
        lo, hi = self.bucket_bounds(bucket, max_offset)
        width = hi - lo + 1
        p = self.residual_counts[bucket][:width].copy()
        return p / p.sum()

    def observe(self, offset: int) -> None:
        """Record an offset observation by bucket and residual."""
        bucket = self.bucket_for(offset)
        lo, _ = self.bucket_bounds(bucket, self.window)
        self.counts[bucket] += 1.0
        self.residual_counts[bucket][offset - lo] += 1.0


def _select_device() -> torch.device:
    """Pick CUDA when available so per-byte forward/backward runs on GPU.

    Determinism caveat: GPU ops are non-deterministic across machines, so
    cross-machine round-trip will diverge. For Rung 1 (single-machine) this
    is fine; Rung 2 is where we drop PyTorch entirely for bit-identical
    cross-platform output.

    Override with KOLMO_DEVICE=cpu to force CPU.
    """
    forced = os.environ.get("KOLMO_DEVICE", "").lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def new_model_and_optimizer() -> tuple[KolmoTransformer | FixedModelState, torch.optim.Optimizer | None]:
    """Build a model with deterministic init. Both compress and decompress
    must call this and get bit-identical starting weights."""
    torch.manual_seed(SEED)
    use_rope = _use_rope()
    preset_kwargs = dict(_MODEL_PRESETS[_model_preset()])
    # max_context must exceed the highest absolute position ever indexed.
    # After warm_cache (positions 0..len(history)-1, len(history) <= CONTEXT),
    # step_cache increments pos_offset per byte. A training block can grow
    # up to training_block_size_at == min(BLOCK_SIZE * mult, CONTEXT - 1)
    # before firing, so the worst case is 2*CONTEXT - 1 (full history +
    # full pending). Round up to a power of two for clean RoPE tables.
    min_max_context = 2 * CONTEXT
    max_context = 512
    while max_context < min_max_context:
        max_context *= 2
    preset_kwargs.setdefault("max_context", max_context)
    model = KolmoTransformer(use_rope=use_rope, **preset_kwargs)
    stable_init_model(model, SEED)
    if _use_fixed():
        fixed_model = FixedModelState(
            weights=extract_fixed_weights(model),
            tied_params=tied_param_pairs(model),
            use_rope=use_rope,
        )
        if not _skip_prime():
            if _load_primed_state(fixed_model, model):
                return fixed_model, None
            _prime_model(fixed_model, None)
            _save_primed_state(fixed_model, model)
        return fixed_model, None
    model.to(_select_device())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if not _skip_prime():
        _prime_model(model, optimizer)
    return model, optimizer


def _prime_model(
    model: KolmoTransformer | FixedModelState,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    """Train on a tiny built-in corpus before real data starts."""
    history = [BOS]
    for pos in range(0, len(SEED_CORPUS), BLOCK_SIZE):
        block = list(SEED_CORPUS[pos : pos + BLOCK_SIZE])
        train_block(model, optimizer, history, block)
        history = update_history(history, block)


def _seed_cache_config(model: KolmoTransformer) -> dict:
    """The bits of model architecture that affect the primed state."""
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.blocks[0].attn.n_heads,
        "n_layers": len(model.blocks),
        "max_context": model.max_context,
        "tie_weights": model.tie_weights,
        "use_rope": model.use_rope,
        "lr": LR,
        "context": CONTEXT,
        "bos": BOS,
    }


def _load_primed_state(
    fixed_model: FixedModelState,
    pytorch_model: KolmoTransformer,
) -> bool:
    """Try to load the primed state from disk. Returns True on hit."""
    from kolmo.seed_cache import (
        cache_disabled,
        cache_path_for,
        compute_config_hash,
        load_state,
    )

    if cache_disabled():
        return False
    config_hash = compute_config_hash(
        seed_corpus=SEED_CORPUS,
        model_config=_seed_cache_config(pytorch_model),
        init_seed=SEED,
        block_size=BLOCK_SIZE,
    )
    path = cache_path_for(config_hash)
    if not path.exists():
        return False
    weights, state, tied = load_state(path)
    fixed_model.weights = weights
    fixed_model.optimizer_state = state
    fixed_model.tied_params = tied
    return True


def _save_primed_state(
    fixed_model: FixedModelState,
    pytorch_model: KolmoTransformer,
) -> None:
    """Save the primed state to disk. Quiet on failure — the cache is an
    optimization, not a correctness requirement."""
    from kolmo.seed_cache import (
        cache_disabled,
        cache_path_for,
        compute_config_hash,
        save_state,
    )

    if cache_disabled() or fixed_model.optimizer_state is None:
        return
    config_hash = compute_config_hash(
        seed_corpus=SEED_CORPUS,
        model_config=_seed_cache_config(pytorch_model),
        init_seed=SEED,
        block_size=BLOCK_SIZE,
    )
    path = cache_path_for(config_hash)
    try:
        save_state(
            path,
            fixed_model.weights,
            fixed_model.optimizer_state,
            fixed_model.tied_params,
        )
    except OSError:
        # Disk full, permission denied, etc. — we already primed in memory,
        # so the current run succeeds; subsequent runs will just re-prime.
        pass


def _trim_caches(caches: list, max_len: int) -> list:
    """Slide the KV cache window: keep only the last `max_len` positions."""
    out = []
    for c in caches:
        if c["k"].shape[2] > max_len:
            out.append({
                "k": c["k"][:, :, -max_len:],
                "v": c["v"][:, :, -max_len:],
            })
        else:
            out.append(c)
    return out


def warm_cache(model: KolmoTransformer, history: list[int]) -> tuple[np.ndarray, list, int]:
    """Run a fresh forward over `history` (no grad) to rebuild the KV cache
    and get the prediction for the next byte. Used at the start of each block,
    after a training step has invalidated the previous cache.

    Returns (probs over next byte as float64 numpy, kv_caches, pos_after).
    """
    if isinstance(model, FixedModelState):
        last_logits_q, caches = fixed_warm(
            np.array(history, dtype=np.int64),
            model.weights,
            n_heads=model.n_heads,
            n_layers=model.n_layers,
            pos_offset=0,
            use_rope=model.use_rope,
        )
        # If the prime/seed history is already longer than CONTEXT, the
        # warmed cache exceeds the window — trim now so subsequent steps
        # operate on the same window the PyTorch path would.
        if caches and caches[0]["k"].shape[1] > CONTEXT:
            caches = trim_caches(caches, CONTEXT)
        probs = _probs_from_q15_logits(last_logits_q)
        return probs, caches, len(history)

    device = next(model.parameters()).device
    x = torch.tensor([history], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, caches = model(x, kv_caches=None, pos_offset=0)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    # Quantize through deterministic int frequencies so probs derived
    # from float math on different machines collapse to the same values.
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, len(history)


def step_cache(
    model: KolmoTransformer | FixedModelState,
    byte: int,
    caches: list,
    pos_offset: int,
) -> tuple[np.ndarray, list, int]:
    """Feed one new byte using the cache. Returns (probs over next byte,
    updated caches, new pos_offset)."""
    if isinstance(model, FixedModelState):
        last_logits_q, caches = fixed_step(
            byte,
            caches,
            model.weights,
            n_heads=model.n_heads,
            n_layers=model.n_layers,
            pos_offset=pos_offset,
            use_rope=model.use_rope,
        )
        if caches and caches[0]["k"].shape[1] > CONTEXT:
            caches = trim_caches(caches, CONTEXT)
        probs = _probs_from_q15_logits(last_logits_q)
        return probs, caches, pos_offset + 1

    device = next(model.parameters()).device
    x = torch.tensor([[byte]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, pos_offset + 1


def step_cache_batch(
    model: KolmoTransformer | FixedModelState,
    bytes_list: list[int] | bytes | bytearray,
    caches: list,
    pos_offset: int,
) -> tuple[np.ndarray, list, int]:
    """Feed a sequence of new bytes through the KV cache in one forward pass.

    Used for copy events, where the bytes are already known (from the copy's
    offset+length) so per-byte probabilities are not needed for encoding.
    The cache still has to absorb all N bytes so the next prediction is
    accurate. One forward over N tokens is much faster than N forwards over
    1 token because matmul efficiency scales with the batch dim.

    Returns (probs for the byte AFTER the batch, updated caches, new pos_offset).
    The returned `probs` is the same as if the last byte's `step_cache` had
    been called individually.
    """
    if not bytes_list:
        # No-op convenience; callers typically guarantee non-empty.
        return np.zeros(0, dtype=np.float64), caches, pos_offset

    if isinstance(model, FixedModelState):
        # Fixed mode doesn't have a batched step yet; fall back to a per-byte
        # loop. Still saves the function-call overhead vs the outer caller
        # doing the loop, and keeps the interface uniform.
        last_probs = None
        for byte in bytes_list:
            last_probs, caches, pos_offset = step_cache(
                model, int(byte), caches, pos_offset
            )
        return last_probs, caches, pos_offset

    device = next(model.parameters()).device
    x = torch.tensor([list(bytes_list)], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, pos_offset + len(bytes_list)


def _probs_from_q15_logits(last_logits_q: np.ndarray) -> np.ndarray:
    """Dequantize Q15 logits and quantize them through the deterministic
    int-frequency grid so the resulting probs match the PyTorch path."""
    last_logits = dequantize(last_logits_q).astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    return freqs.astype(np.float64) / float(TOTAL_FREQ)


def train_block(
    model: KolmoTransformer | FixedModelState,
    optimizer: torch.optim.Optimizer | None,
    history: list[int],
    block_bytes: list[int],
) -> None:
    """Run a full forward with gradient over `history + block_bytes`, compute
    cross-entropy loss against the block targets, backward + optimizer step.

    Both compress and decompress call this with the same arguments at the
    same step, so weights stay in lockstep.
    """
    if isinstance(model, FixedModelState):
        model.optimizer_state = fixed_train_block(
            model.weights,
            model.optimizer_state,
            history,
            block_bytes,
            n_heads=model.n_heads,
            n_layers=model.n_layers,
            context=CONTEXT,
            tied_params=model.tied_params,
            use_rope=model.use_rope,
        )
        return

    if optimizer is None:
        raise ValueError("PyTorch training requires an optimizer")

    full = (history + block_bytes)[-CONTEXT:]
    m = len(block_bytes)
    n_hist = len(full) - m

    device = next(model.parameters()).device
    x = torch.tensor([full], dtype=torch.long, device=device)
    logits, _ = model(x, kv_caches=None, pos_offset=0)
    # Predictions for block bytes come from logits at positions [n_hist-1 .. n_hist+m-2]
    block_logits = logits[0, n_hist - 1 : n_hist + m - 1]

    targets = torch.tensor(block_bytes, dtype=torch.long, device=device)
    loss = F.cross_entropy(block_logits, targets)
    optimizer.zero_grad()
    loss.backward()
    # Linear LR warmup. Stored on the optimizer (no separate counter).
    step_num = getattr(optimizer, "_kolmo_step", 0) + 1
    optimizer._kolmo_step = step_num
    if step_num <= LR_WARMUP_STEPS:
        warm = step_num / LR_WARMUP_STEPS
        for g in optimizer.param_groups:
            g["lr"] = LR * warm
    elif step_num == LR_WARMUP_STEPS + 1:
        # Pin to base LR exactly once after warmup completes.
        for g in optimizer.param_groups:
            g["lr"] = LR
    optimizer.step()
    # Historical note: this used to round gradients to 1/8192 and Adam state
    # to 1/16384 to make cross-machine PyTorch produce identical updates. The
    # rounding rounded `exp_avg_sq` (squared gradients, typically O(1e-7)) all
    # the way to zero, which made `m / (sqrt(v) + eps)` blow up by 1e8 and
    # weights exploded after step 2. We don't need it any more: cross-machine
    # determinism now lives in the Q15 fixed-point engine (KOLMO_FIXED=1).
    # Within-machine CPU PyTorch is deterministic without intervention.


def update_history(history: list[int], new_bytes: list[int]) -> list[int]:
    """Append new bytes to the sliding-window history."""
    history = history + new_bytes
    if len(history) > CONTEXT:
        history = history[-CONTEXT:]
    return history


def append_copy_history(copy_history: bytearray, byte: int) -> None:
    """Append one byte to copy history while bounding long-file memory.

    Copy offsets are capped at COPY_WINDOW, so older bytes are never addressable
    by the compressed stream. Trim in chunks rather than every byte to avoid
    repeatedly shifting the bytearray front on long files.
    """
    copy_history.append(byte)
    if len(copy_history) > 2 * COPY_WINDOW:
        del copy_history[:-COPY_WINDOW]


def find_copy(data: bytes, pos: int, known: bytes | bytearray) -> tuple[int, int] | None:
    """Find a simple non-overlapping LZ-style match in recent known bytes.

    Returns (offset, length), where offset=1 means "copy from the previous byte".
    """
    remaining = len(data) - pos
    if remaining < COPY_MIN:
        return None

    window = known[-COPY_WINDOW:]
    key = data[pos : pos + COPY_MIN]
    best_offset = 0
    best_len = 0

    idx = window.rfind(key)
    while idx != -1:
        offset = len(window) - idx
        max_len = min(COPY_MAX, remaining, offset)
        length = COPY_MIN
        while length < max_len and window[idx + length] == data[pos + length]:
            length += 1
        if length > best_len:
            best_offset = offset
            best_len = length
        idx = window.rfind(key, 0, idx)

    if best_len < COPY_MIN:
        return None
    return best_offset, best_len


class RollingCopyMatcher:
    """Fast LZ-style matcher for the compressor.

    The old compressor called `find_copy(data, pos, copy_history)` at every
    position, and `find_copy` searched the current window with repeated
    `rfind` calls. That is fine for tiny files but too expensive for big ones.

    This matcher indexes COPY_MIN-byte keys by absolute position as soon as
    those bytes are known. At probe time it only checks positions with the same
    8-byte key, newest first, and caps the candidate chain. This is the same
    broad shape as practical LZ compressors: hash lookup first, byte compare
    only for plausible candidates.
    """

    def __init__(
        self,
        data: bytes,
        *,
        window: int = COPY_WINDOW,
        max_candidates: int = COPY_CANDIDATES,
    ) -> None:
        self.data = data
        self.window = window
        self.max_candidates = max_candidates
        self._index: defaultdict[bytes, deque[int]] = defaultdict(deque)
        self._indexed_positions: deque[tuple[int, bytes]] = deque()
        self._next_index_pos = 0

    def _index_known_prefix(self, pos: int) -> None:
        """Index every COPY_MIN-byte key fully known before `pos`."""
        limit = min(pos - COPY_MIN + 1, len(self.data) - COPY_MIN + 1)
        while self._next_index_pos < limit:
            start = self._next_index_pos
            key = self.data[start : start + COPY_MIN]
            self._index[key].append(start)
            self._indexed_positions.append((start, key))
            self._next_index_pos += 1

    def _prune_old(self, pos: int) -> None:
        min_start = pos - self.window
        while self._indexed_positions and self._indexed_positions[0][0] < min_start:
            old_start, key = self._indexed_positions.popleft()
            candidates = self._index.get(key)
            if not candidates:
                continue
            if candidates[0] == old_start:
                candidates.popleft()
            if not candidates:
                del self._index[key]

    def candidates(self, pos: int) -> list[tuple[int, int]]:
        """Return plausible (offset, length) copy candidates at `pos`.

        Candidates are newest-first (same order the matcher inspects them),
        capped by `max_candidates`, and already filtered to length >= COPY_MIN.
        The compressor can use this to choose by estimated coding cost instead
        of blindly taking the longest match.
        """
        remaining = len(self.data) - pos
        if remaining < COPY_MIN:
            return []

        self._index_known_prefix(pos)
        self._prune_old(pos)
        key = self.data[pos : pos + COPY_MIN]
        candidates = self._index.get(key)
        if not candidates:
            return []

        min_start = pos - self.window
        while candidates and candidates[0] < min_start:
            candidates.popleft()
        if not candidates:
            return []

        out: list[tuple[int, int]] = []
        checked = 0
        for start in reversed(candidates):
            offset = pos - start
            if offset <= 0 or offset > self.window:
                continue
            # Non-overlapping copy: the copied span can't read bytes that are
            # being produced by this same copy event.
            max_len = min(COPY_MAX, remaining, offset)
            length = COPY_MIN
            while (
                length < max_len
                and self.data[start + length] == self.data[pos + length]
            ):
                length += 1
            if length >= COPY_MIN:
                out.append((offset, length))
                if length == COPY_MAX:
                    break
            checked += 1
            if checked >= self.max_candidates:
                break

        return out

    def find(self, pos: int) -> tuple[int, int] | None:
        candidates = self.candidates(pos)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])

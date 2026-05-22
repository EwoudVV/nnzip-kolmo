"""Shared logic for compress and decompress.

Both directions need to walk the model through the *same* training trajectory:
build identical weights, predict identical probabilities, take identical
optimizer steps. The KV cache lets each direction do most of the predictions
incrementally (O(T) per byte instead of O(T²)); the training step still needs
a full forward over the recent history with gradient tracking, but that's
only done once per block.
"""

import os
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
CONTEXT = 256  # sliding-window cap (max tokens kept in KV cache)
BLOCK_SIZE = 16  # bytes between optimizer steps
BOS = 0  # implicit start-of-stream byte, never written to disk
COPY_PROB = 0.005
COPY_WINDOW = 8192
COPY_MIN = 8
COPY_MAX = 256
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
    b"bytes within an eight-kilobyte window.\n\n"

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
    """Adaptive probability model over match lengths (encoded as offsets from
    COPY_MIN, so the symbol range is [0, COPY_MAX - COPY_MIN]).

    Length distribution is steep — most matches are at or near COPY_MIN. The
    static 1/k prior is decent but not perfect for any particular corpus.
    """

    def __init__(self, n: int, prior_strength: float = 16.0):
        prior = length_probs(n) * prior_strength
        self.counts = prior.astype(np.float64)

    def probs_for(self, max_n: int) -> np.ndarray:
        p = self.counts[:max_n].copy()
        return p / p.sum()

    def observe(self, length_offset: int) -> None:
        self.counts[length_offset] += 1.0


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


class OffsetModel:
    """Adaptive probability model for copy offsets.

    Both compress and decompress hold an instance and call `observe` after
    every copy event, in the same order with the same offsets — so the
    distribution evolves bit-identically on both sides.

    The model maintains Laplace-smoothed counts over the offset range 1..N.
    Counts start at the static 1/sqrt(k) prior (scaled so its total mass is
    `prior_strength`), so very early events have a sensible distribution
    before any are observed. As events accumulate, the empirical distribution
    increasingly dominates.
    """

    def __init__(self, window: int, prior_strength: float = 128.0):
        self.window = window
        # Initialize with 1/sqrt(k) prior scaled to total mass = prior_strength.
        prior = offset_probs(window) * prior_strength
        self.counts = prior.astype(np.float64)

    def probs_for(self, max_offset: int) -> np.ndarray:
        """Return normalized probabilities over offsets 1..max_offset
        (returned as array of length max_offset)."""
        p = self.counts[:max_offset].copy()
        return p / p.sum()

    def observe(self, offset: int) -> None:
        """Record a 1:1 offset observation. offset is 1-indexed (offset=1 is
        the immediately previous byte)."""
        self.counts[offset - 1] += 1.0


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
    model = KolmoTransformer()
    stable_init_model(model, SEED)
    if _use_fixed():
        fixed_model = FixedModelState(
            weights=extract_fixed_weights(model),
            tied_params=tied_param_pairs(model),
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
    with torch.no_grad():
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
        )
        if caches and caches[0]["k"].shape[1] > CONTEXT:
            caches = trim_caches(caches, CONTEXT)
        probs = _probs_from_q15_logits(last_logits_q)
        return probs, caches, pos_offset + 1

    device = next(model.parameters()).device
    x = torch.tensor([[byte]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)
    freqs = logits_to_int_freqs(last_logits)
    probs = freqs.astype(np.float64) / float(TOTAL_FREQ)
    return probs, caches, pos_offset + 1


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

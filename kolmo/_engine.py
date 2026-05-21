"""Shared logic for compress and decompress.

Both directions need to walk the model through the *same* training trajectory:
build identical weights, predict identical probabilities, take identical
optimizer steps. The KV cache lets each direction do most of the predictions
incrementally (O(T) per byte instead of O(T²)); the training step still needs
a full forward over the recent history with gradient tracking, but that's
only done once per block.
"""

import numpy as np
import torch
import torch.nn.functional as F

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
_SEED_BASE = (
    b"English text is full of small regularities. Letters form words, words "
    b"form phrases, and phrases repeat with punctuation, spacing, and rhythm. "
    b"A compressor that begins from a blank model wastes bits learning that "
    b"spaces are common, vowels follow consonants, and sentences often return "
    b"to familiar patterns. This short seed paragraph gives the online model a "
    b"deterministic prior without storing learned weights in the compressed file."
)
_SEED_EXTRA = (
    b"\n\nThe morning train crossed the river while people read notes, checked maps, "
    b"and talked quietly about school, weather, work, and travel. A useful "
    b"English prior should know that spaces are common, commas separate clauses, "
    b"and a period is often followed by a space and a capital letter. "
    b"\n\nReference entry: compression is the process of representing information with "
    b"fewer symbols. A dictionary method stores repeated phrases as pointers, "
    b"while a statistical method assigns shorter codes to likely events. Both "
    b"approaches rely on patterns that appear again after they have been seen. "
    b"\n\nDialogue:\nAlice: Did the model remember the phrase from earlier?\nBen: It remembered "
    b"letters and spaces, but not the exact sentence.\nAlice: Then we need a "
    b"better prior or a small memory for repeated text.\n"
    b"\n# Notes\n\n- Train deterministically.\n- Keep encoder and decoder symmetric.\n- "
    b"Measure gzip, kolmo, ratio, and time.\n- Revert changes that only help tiny cases.\n"
    b"\nThe second paragraph repeats the lesson in different words: text contains "
    b"local spelling rules, medium-range grammar, and long-range reuse. A model "
    b"that handles only local spelling will plateau, but a model with useful "
    b"memory can keep improving as the document becomes longer. "
    b"\n\nNumbers and punctuation also matter: 2026-05-20, 1,024 bytes, 2,048 bytes, "
    b"and 4,096 bytes should be parsed as ordinary text rather than surprises. "
    b"Lists, headings, and quoted speech are common in mixed corpora. "
    b"\n\nA final neutral passage describes a city library with shelves, tables, lamps, "
    b"catalog records, quiet readers, printed forms, and old magazines. The same "
    b"words return in nearby sentences, and the compressor should spend fewer "
    b"bits each time the pattern becomes familiar. "
)
SEED_CORPUS = _SEED_BASE + _SEED_EXTRA * 3
EVENT_PROBS = np.array([1.0 - COPY_PROB, COPY_PROB], dtype=np.float64)


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
    import os
    forced = os.environ.get("KOLMO_DEVICE", "").lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def new_model_and_optimizer() -> tuple[KolmoTransformer, torch.optim.Optimizer]:
    """Build a model with deterministic init. Both compress and decompress
    must call this and get bit-identical starting weights."""
    torch.manual_seed(SEED)
    model = KolmoTransformer()
    stable_init_model(model, SEED)
    model.to(_select_device())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    _prime_model(model, optimizer)
    return model, optimizer


def _prime_model(
    model: KolmoTransformer,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train on a tiny built-in corpus before real data starts."""
    history = [BOS]
    for pos in range(0, len(SEED_CORPUS), BLOCK_SIZE):
        block = list(SEED_CORPUS[pos : pos + BLOCK_SIZE])
        train_block(model, optimizer, history, block)
        history = update_history(history, block)


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
    device = next(model.parameters()).device
    x = torch.tensor([history], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, caches = model(x, kv_caches=None, pos_offset=0)
    probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy().astype(np.float64)
    return probs, caches, len(history)


def step_cache(
    model: KolmoTransformer,
    byte: int,
    caches: list,
    pos_offset: int,
) -> tuple[np.ndarray, list, int]:
    """Feed one new byte using the cache. Returns (probs over next byte,
    updated caches, new pos_offset)."""
    device = next(model.parameters()).device
    x = torch.tensor([[byte]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, caches = model(x, kv_caches=caches, pos_offset=pos_offset)
    caches = _trim_caches(caches, CONTEXT)
    probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy().astype(np.float64)
    return probs, caches, pos_offset + 1


def train_block(
    model: KolmoTransformer,
    optimizer: torch.optim.Optimizer,
    history: list[int],
    block_bytes: list[int],
) -> None:
    """Run a full forward with gradient over `history + block_bytes`, compute
    cross-entropy loss against the block targets, backward + optimizer step.

    Both compress and decompress call this with the same arguments at the
    same step, so weights stay in lockstep.
    """
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


def update_history(history: list[int], new_bytes: list[int]) -> list[int]:
    """Append new bytes to the sliding-window history."""
    history = history + new_bytes
    if len(history) > CONTEXT:
        history = history[-CONTEXT:]
    return history


def find_copy(data: bytes, pos: int, known: bytes) -> tuple[int, int] | None:
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

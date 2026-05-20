"""Measure the actual distribution of copy offsets and lengths produced by
find_copy on real text. Useful for tuning the encoded probability distribution
without guessing.

Usage:
    python3 benchmarks/copy_stats.py

Iterates over the benchmark corpus prefixes and reports a histogram of offset
log-bins and length values that find_copy would actually emit. Compares the
empirical distribution to the current 1/k prior the encoder uses.
"""

import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from kolmo._engine import (  # noqa: E402
    COPY_MAX,
    COPY_MIN,
    COPY_WINDOW,
    find_copy,
    length_probs,
    offset_probs,
)


CORPUS_DIR = Path(
    "/Users/kids/compression-experiment/compression-experiments/benchmarks/corpus"
)
SOURCE_FILES = [
    "prose_modern.txt",
    "wiki_factual.txt",
    "dialogue.txt",
    "markdown_docs.md",
    "code_python.py",
    "json_data.json",
    "repetitive.txt",
]


def gather_events(data: bytes) -> list[tuple[int, int]]:
    """Replay find_copy over data and return the list of (offset, length)
    events the encoder would emit."""
    history = bytearray()
    events = []
    pos = 0
    while pos < len(data):
        copy = find_copy(data, pos, bytes(history))
        if copy is None:
            history.append(data[pos])
            pos += 1
            continue
        offset, length = copy
        events.append((offset, length))
        chunk = data[pos : pos + length]
        history.extend(chunk)
        pos += length
    return events


def offset_bin(offset: int) -> str:
    """Coarse log-spaced bin for visualizing offset distribution."""
    if offset <= 0:
        return "0"
    e = int(math.log2(offset))
    lo = 1 << e
    hi = 1 << (e + 1)
    return f"{lo}-{hi - 1}"


def main() -> None:
    parts = []
    for f in SOURCE_FILES:
        parts.append((CORPUS_DIR / f).read_bytes().rstrip(b"\n"))
    raw = b"\n\n".join(parts) + b"\n"

    for n in [4096, 8192, 16384]:
        if n > len(raw):
            break
        data = raw[:n]
        events = gather_events(data)
        if not events:
            print(f"{n}B: no copy events")
            continue

        offsets = [o for o, _ in events]
        lengths = [length for _, length in events]
        copied = sum(lengths)

        bins = Counter(offset_bin(o) for o in offsets)
        len_hist = Counter(lengths)

        print(f"\n{n}B  copy_events={len(events)}  bytes_copied={copied}  "
              f"({100 * copied / n:.1f}%)")
        print("  offset log-bins:")
        for bin_label, cnt in sorted(
            bins.items(),
            key=lambda kv: int(kv[0].split("-")[0]),
        ):
            pct = 100 * cnt / len(events)
            bar = "#" * int(pct / 2)
            print(f"    {bin_label:>9}: {cnt:>4}  {pct:5.1f}%  {bar}")
        print("  length histogram:")
        for length, cnt in sorted(len_hist.items()):
            pct = 100 * cnt / len(events)
            bar = "#" * int(pct / 2)
            print(f"    {length:>3}: {cnt:>4}  {pct:5.1f}%  {bar}")

        # Compare empirical vs current encoder prior
        max_offset = min(COPY_WINDOW, n)
        prior_offset = offset_probs(max_offset)
        emp_offset_bits = 0.0
        prior_offset_bits = 0.0
        for off in offsets:
            empirical_p = offsets.count(off) / len(offsets)
            emp_offset_bits += -math.log2(empirical_p) if empirical_p > 0 else 0
            prior_offset_bits += -math.log2(prior_offset[off - 1])
        print(f"  offset coding cost (current 1/k prior): {prior_offset_bits:.0f} bits"
              f"  | empirical optimum: {emp_offset_bits:.0f} bits"
              f"  | uniform: {len(events) * math.log2(max_offset):.0f} bits")


if __name__ == "__main__":
    main()

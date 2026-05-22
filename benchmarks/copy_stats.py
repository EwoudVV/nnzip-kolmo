"""Measure the actual distribution of copy offsets and lengths produced by
find_copy on real text. Useful for tuning the encoded probability distribution
without guessing.

Usage:
    python3 benchmarks/copy_stats.py

Iterates over the benchmark corpus prefixes and reports a histogram of offset
log-bins and length values that find_copy would actually emit. Reports both
the local (short) corpus and the procedurally-extended long corpus.
"""

import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from kolmo._engine import (  # noqa: E402
    COPY_WINDOW,
    RollingCopyMatcher,
    length_probs,
)


CORPUS_DIR = HERE / "corpus"
SHORT_FILES = [
    "prose_modern.txt",
    "wiki_factual.txt",
    "dialogue.txt",
    "markdown_docs.md",
]


def build_short_corpus() -> bytes:
    parts = [
        (CORPUS_DIR / f).read_bytes().rstrip(b"\n") for f in SHORT_FILES
    ]
    return b"\n\n".join(parts) + b"\n"


def build_long_corpus() -> bytes:
    """Same as benchmarks/long_crossover.py — extends the short corpus with
    deterministic procedurally-generated paragraphs."""
    parts = [
        (CORPUS_DIR / f).read_bytes().rstrip(b"\n")
        for f in [
            "prose_modern.txt",
            "wiki_factual.txt",
            "dialogue.txt",
            "markdown_docs.md",
            "code_python.py",
            "json_data.json",
            "repetitive.txt",
        ]
    ]
    subjects = [
        "the archivist", "a small research team", "the compression model",
        "the city council", "an old river map", "the school robotics club",
        "the public library", "a weather station",
    ]
    actions = [
        "recorded", "compared", "revised", "summarized",
        "questioned", "indexed", "tested", "explained",
    ]
    objects = [
        "a table of measurements", "the repeated phrase in the report",
        "a sequence of byte patterns", "the notes from the meeting",
        "a paragraph with several clauses", "the local history archive",
        "a list of names and dates", "the result of the benchmark",
    ]
    for i in range(240):
        s = subjects[i % len(subjects)]
        a = actions[(i * 3 + 1) % len(actions)]
        o = objects[(i * 5 + 2) % len(objects)]
        paragraph = (
            f"\n\nSection {i:03d}. {s.title()} {a} {o}. The entry "
            f"included numbers {128 + i}, {256 + 2 * i}, and {4096 + 3 * i}, "
            "plus a note about context, memory, prediction, and exact reuse. "
            "In the dialogue version, one person asked whether the earlier "
            "sentence could be copied, and another answered that literal "
            "prediction and copy references solve different problems. The "
            "wording changes each time, but ordinary English punctuation and "
            "spacing remain stable."
        ).encode("ascii")
        parts.append(paragraph)
    return b"\n\n".join(parts) + b"\n"


def gather_events(data: bytes, window: int) -> list[tuple[int, int]]:
    """Replay the compressor's rolling matcher, returning (offset, length)
    events."""
    matcher = RollingCopyMatcher(data, window=window)
    events = []
    pos = 0
    while pos < len(data):
        copy = matcher.find(pos)
        if copy is None:
            pos += 1
            continue
        offset, length = copy
        events.append((offset, length))
        pos += length
    return events


def offset_bin(offset: int) -> str:
    if offset <= 0:
        return "0"
    e = int(math.log2(offset))
    lo = 1 << e
    hi = 1 << (e + 1)
    return f"{lo}-{hi - 1}"


def report(name: str, data: bytes, window: int) -> None:
    events = gather_events(data, window)
    if not events:
        print(f"{name}: no copy events at window={window}")
        return

    offsets = [o for o, _ in events]
    lengths = [length for _, length in events]
    copied = sum(lengths)
    bins = Counter(offset_bin(o) for o in offsets)

    print(f"\n{name}  window={window}B  events={len(events)}  "
          f"copied={copied}B ({100 * copied / len(data):.1f}%)")
    print("  offset log-bins:")
    for bin_label, cnt in sorted(
        bins.items(), key=lambda kv: int(kv[0].split("-")[0])
    ):
        pct = 100 * cnt / len(events)
        bar = "#" * int(pct / 2)
        print(f"    {bin_label:>10}: {cnt:>4}  {pct:5.1f}%  {bar}")


def main() -> None:
    short = build_short_corpus()
    long_corp = build_long_corpus()

    print(f"short corpus: {len(short):,}B   long corpus: {len(long_corp):,}B")

    for window in [8192, 16384, 65536]:
        print(f"\n=== window={window}B ===")
        for n in [4096, 8192]:
            if n <= len(short):
                report(f"short[{n}]", short[:n], window)
        for n in [16384, 32768]:
            if n <= len(long_corp):
                report(f"long[{n}]", long_corp[:n], window)


if __name__ == "__main__":
    main()

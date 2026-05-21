"""Longer crossover benchmark for the 8KB+ slope.

The checked-in corpus only has ~8.9KB of clean prose/wiki/dialogue/markdown
text, so this benchmark appends deterministic varied English paragraphs. This
is not a publication-grade corpus; it is a slope test for whether gzip pulls
away as file size grows.
"""

import gzip
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

sys.stdout.reconfigure(line_buffering=True)

from kolmo import compress, decompress  # noqa: E402
from kolmo._engine import COPY_WINDOW  # noqa: E402

NNZIP_CORPUS = HERE / "corpus"
SOURCE_FILES = [
    "prose_modern.txt",
    "wiki_factual.txt",
    "dialogue.txt",
    "markdown_docs.md",
    "code_python.py",
    "json_data.json",
    "repetitive.txt",
]
SIZES = [8192, 16384, 32768]


def build_corpus() -> bytes:
    parts = [(NNZIP_CORPUS / fname).read_bytes().rstrip(b"\n") for fname in SOURCE_FILES]
    subjects = [
        "the archivist",
        "a small research team",
        "the compression model",
        "the city council",
        "an old river map",
        "the school robotics club",
        "the public library",
        "a weather station",
    ]
    actions = [
        "recorded",
        "compared",
        "revised",
        "summarized",
        "questioned",
        "indexed",
        "tested",
        "explained",
    ]
    objects = [
        "a table of measurements",
        "the repeated phrase in the report",
        "a sequence of byte patterns",
        "the notes from the meeting",
        "a paragraph with several clauses",
        "the local history archive",
        "a list of names and dates",
        "the result of the benchmark",
    ]
    for i in range(240):
        s = subjects[i % len(subjects)]
        a = actions[(i * 3 + 1) % len(actions)]
        o = objects[(i * 5 + 2) % len(objects)]
        paragraph = (
            f"\n\nSection {i:03d}. {s.title()} {a} {o}. "
            f"The entry included numbers {128 + i}, {256 + 2 * i}, "
            f"and {4096 + 3 * i}, plus a note about context, memory, "
            "prediction, and exact reuse. In the dialogue version, one person "
            "asked whether the earlier sentence could be copied, and another "
            "answered that literal prediction and copy references solve "
            "different problems. The wording changes each time, but ordinary "
            "English punctuation and spacing remain stable."
        ).encode("ascii")
        parts.append(paragraph)
    return b"\n\n".join(parts) + b"\n"


def main() -> None:
    raw = build_corpus()
    print(
        f"long corpus size: {len(raw):,} B "
        f"(mixed local corpus + deterministic generated text)"
    )
    print(f"copy window: {COPY_WINDOW} B")
    print()
    print(f"{'size':>6} | {'gzip':>15} | {'kolmo':>16} | {'enc + dec time':>16}  verdict")
    print("-" * 84)

    for n in SIZES:
        data = raw[:n]
        gz = gzip.compress(data, compresslevel=9)
        gz_pct = 100.0 * len(gz) / n

        t = time.monotonic()
        blob = compress(data)
        enc_t = time.monotonic() - t

        t = time.monotonic()
        recovered = decompress(blob)
        dec_t = time.monotonic() - t

        assert recovered == data, f"round-trip FAILED at {n}B"
        ko_pct = 100.0 * len(blob) / n

        delta = ko_pct - gz_pct
        if delta < -0.5:
            verdict = f"kolmo wins by {-delta:.1f}pp"
        elif delta > 0.5:
            verdict = f"gzip wins by {delta:.1f}pp"
        else:
            verdict = "tie"

        print(
            f"{n:>6} | {len(gz):>5}B ({gz_pct:>5.1f}%) | "
            f"{len(blob):>5}B ({ko_pct:>5.1f}%) | "
            f"{enc_t:>5.1f}s + {dec_t:>4.1f}s   {verdict}"
        )


if __name__ == "__main__":
    main()

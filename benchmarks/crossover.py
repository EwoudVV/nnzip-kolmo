"""Find the crossover point with gzip — at what file size does kolmo's online
learning start winning?

Builds a clean ~8.8KB corpus by concatenating the four prose-style nnzip
benchmark files (prose / wiki / dialogue / markdown — all clean ASCII, no
duplicates). Compresses prefixes of this corpus with both kolmo and gzip and
reports a ratio table.
"""

import gzip
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

# Line-buffered output so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)

from kolmo import compress, decompress  # noqa: E402

NNZIP_CORPUS = Path(
    "/Users/kids/compression-experiment/compression-experiments/benchmarks/corpus"
)
SOURCE_FILES = [
    "prose_modern.txt",
    "wiki_factual.txt",
    "dialogue.txt",
    "markdown_docs.md",
]
SIZES = [1024, 2048, 4096, 8192]


def build_corpus() -> bytes:
    parts = []
    for fname in SOURCE_FILES:
        parts.append((NNZIP_CORPUS / fname).read_bytes().rstrip(b"\n"))
    return b"\n\n".join(parts) + b"\n"


def main() -> None:
    raw = build_corpus()
    print(f"corpus size: {len(raw):,} B (no duplication, varied genres)")
    print()
    print(f"{'size':>6} | {'gzip':>15} | {'kolmo':>16} | {'enc + dec time':>16}  verdict")
    print("-" * 80)

    for n in SIZES:
        if n > len(raw):
            print(f"{n:>6} | skipped — exceeds corpus size {len(raw)}")
            continue
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

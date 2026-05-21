"""Real-corpus benchmark — kolmo vs gzip on a cleaned chunk of Pride and
Prejudice (Project Gutenberg).

This is the honest test: text we did NOT tune any constants against, no
procedural padding, no structural repetition synthetically inserted. If
kolmo wins here, the architecture works on real English prose.

Run from the repo root:
    python3 benchmarks/real_corpus.py

Requires a 50 KB Pride and Prejudice file at the path below; we clean it
in-script (strip BOM, CRLF, smart quotes, Gutenberg header) before testing.
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

RAW_BOOK = Path("/Users/kids/compression-experiment/bigfile.txt")
SIZES = [1024, 2048, 4096, 8192, 16384]


def clean_text(raw: bytes) -> bytes:
    """Strip Gutenberg encoding noise so we measure language compression,
    not encoding artifacts."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    raw = raw.replace(b"\r\n", b"\n")
    smart_to_ascii = {
        b"\xe2\x80\x9c": b'"',
        b"\xe2\x80\x9d": b'"',
        b"\xe2\x80\x98": b"'",
        b"\xe2\x80\x99": b"'",
        b"\xe2\x80\x94": b"--",
        b"\xe2\x80\x93": b"-",
        b"\xe2\x80\xa6": b"...",
    }
    for src, dst in smart_to_ascii.items():
        raw = raw.replace(src, dst)
    idx = raw.find(b"Chapter I")
    if idx > 0:
        raw = raw[idx:]
    return bytes(b for b in raw if b < 128)


def main() -> None:
    if not RAW_BOOK.exists():
        print(f"ERROR: raw book not found at {RAW_BOOK}", file=sys.stderr)
        sys.exit(1)
    raw = clean_text(RAW_BOOK.read_bytes())
    print(f"clean corpus size: {len(raw):,}B (Pride and Prejudice, cleaned)")
    print()
    print(f"{'size':>6} | {'gzip':>15} | {'kolmo':>16} | "
          f"{'enc + dec':>14}  verdict")
    print("-" * 80)
    for n in SIZES:
        if n > len(raw):
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

        print(f"{n:>6} | {len(gz):>5}B ({gz_pct:>5.1f}%) | "
              f"{len(blob):>5}B ({ko_pct:>5.1f}%) | "
              f"{enc_t:>5.1f}s + {dec_t:>4.1f}s  {verdict}")


if __name__ == "__main__":
    main()

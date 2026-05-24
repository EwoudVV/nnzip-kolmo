"""Inspect the copy matcher on real enwik prefixes.

This is a cheap ratio diagnostic: before changing neural architecture, check
whether the LZ-style copy layer is leaving obvious bits on the table via low
coverage, short matches, or COPY_MAX saturation.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from enwik_prefix import default_enwik_path, parse_sizes
from kolmo._engine import COPY_MAX, RollingCopyMatcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=default_enwik_path())
    parser.add_argument("--sizes", default="32kb,64kb,128kb,256kb")
    args = parser.parse_args()
    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    raw_all = args.path.read_bytes()[: max(sizes)]

    print(f"file: {args.path}")
    print(
        "     size | events | copied bytes | copied% | mean len | sat@COPY_MAX | top lengths"
    )
    print("-" * 104)
    for n in sizes:
        data = raw_all[:n]
        matcher = RollingCopyMatcher(data)
        pos = 0
        lengths: Counter[int] = Counter()
        total = 0
        events = 0
        saturated = 0
        while pos < len(data):
            copy = matcher.find(pos)
            if copy is None:
                pos += 1
                continue
            _offset, length = copy
            lengths[length] += 1
            total += length
            events += 1
            saturated += int(length == COPY_MAX)
            pos += length
        mean = total / max(events, 1)
        top = " ".join(f"{length}:{count}" for length, count in lengths.most_common(8))
        print(
            f"{n:>9,d} | {events:>6,d} | {total:>12,d} | "
            f"{100*total/n:>6.1f}% | {mean:>8.1f} | "
            f"{saturated:>12,d} | {top}"
        )


if __name__ == "__main__":
    main()

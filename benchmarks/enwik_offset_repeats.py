"""Diagnose whether copy offsets repeat on enwik prefixes.

Offset coding dominates current copy-header cost (~18 of ~25 bits/event).
If exact offsets often repeat or fall in a small recent-offset cache, we can
encode "use recent offset #k" much cheaper than bucket+residual.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from enwik_prefix import default_enwik_path, parse_sizes
import kolmo._engine as engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=default_enwik_path())
    parser.add_argument("--sizes", default="128kb,256kb,512kb")
    parser.add_argument("--window", type=int, default=engine.COPY_WINDOW)
    parser.add_argument("--candidates", type=int, default=engine.COPY_CANDIDATES)
    parser.add_argument("--recent", type=int, default=8)
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    raw_all = args.path.read_bytes()[: max(sizes)]

    print(f"file: {args.path}")
    print(
        f"window: {args.window:,} candidates: {args.candidates} "
        f"recent cache: {args.recent}"
    )
    print(
        "     size | events | same prev | in last4 | in last8 | in last16 | "
        "top exact offsets"
    )
    print("-" * 116)

    for n in sizes:
        data = raw_all[:n]
        matcher = engine.RollingCopyMatcher(
            data,
            window=args.window,
            max_candidates=args.candidates,
        )
        pos = 0
        events = 0
        same_prev = 0
        in4 = 0
        in8 = 0
        in16 = 0
        offset_counts: Counter[int] = Counter()
        recent: deque[int] = deque(maxlen=max(args.recent, 16))

        while pos < n:
            copy = matcher.find(pos)
            if copy is None:
                pos += 1
                continue
            offset, length = copy
            events += 1
            offset_counts[offset] += 1
            recent_list = list(recent)
            same_prev += int(bool(recent_list) and recent_list[-1] == offset)
            in4 += int(offset in recent_list[-4:])
            in8 += int(offset in recent_list[-8:])
            in16 += int(offset in recent_list[-16:])
            if offset in recent:
                recent.remove(offset)
            recent.append(offset)
            pos += length

        top = " ".join(
            f"{offset}:{count}" for offset, count in offset_counts.most_common(10)
        )
        print(
            f"{n:>9,d} | {events:>6,d} | "
            f"{100*same_prev/max(events, 1):>8.2f}% | "
            f"{100*in4/max(events, 1):>7.2f}% | "
            f"{100*in8/max(events, 1):>7.2f}% | "
            f"{100*in16/max(events, 1):>8.2f}% | {top}"
        )


if __name__ == "__main__":
    main()

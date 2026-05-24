"""Estimate the copy layer's bit economics on enwik prefixes.

This intentionally ignores neural literal probabilities. It answers the
front-end question: for the LZ-style copy stream we currently choose, how many
bits do copy headers cost, how many literal bytes remain, and how long does a
match need to be before it plausibly beats spelling the bytes literally?

The models here are exactly the adaptive event/offset/length models used by
compress.py, so the reported header bits track the real arithmetic-coded copy
headers. Literal bytes are charged a configurable flat bpb (default 8) as a
cheap lower/upper proxy; neural bits will vary, but an 8-byte far copy that
costs 18 header bits is suspicious regardless.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from enwik_prefix import default_enwik_path, parse_sizes
import kolmo._engine as engine


@dataclass
class BucketStats:
    events: int = 0
    bytes: int = 0
    bits: float = 0.0
    profitable: int = 0


def bits_for_prob(p: float) -> float:
    return -math.log2(max(p, 1e-300))


def copy_header_bits(
    *,
    offset: int,
    length: int,
    pos: int,
    n: int,
    window: int,
    event_model: engine.EventModel,
    offset_model: engine.OffsetModel,
    length_model: engine.LengthModel,
) -> tuple[float, float, float]:
    """Return (event_bits, offset_bits, length_bits) for one copy event
    under the current adaptive model states, without mutating them."""
    probs = event_model.probs()
    ev_bits = bits_for_prob(float(probs[1]))
    max_offset = min(window, pos)
    max_len_symbols = min(engine.COPY_MAX, n - pos) - engine.COPY_MIN + 1

    offset_bucket = offset_model.bucket_for(offset)
    bucket_probs = offset_model.probs_for(max_offset)
    off_bits = bits_for_prob(float(bucket_probs[offset_bucket]))
    offset_lo, offset_hi = offset_model.bucket_bounds(offset_bucket, max_offset)
    offset_width = offset_hi - offset_lo + 1
    if offset_width > 1:
        residual_probs = offset_model.residual_probs_for(offset_bucket, max_offset)
        off_bits += bits_for_prob(float(residual_probs[offset - offset_lo]))

    len_bits = 0.0
    if max_len_symbols > 1:
        length_offset = length - engine.COPY_MIN
        length_bucket = length_model.bucket_for(length_offset)
        len_probs = length_model.probs_for(max_len_symbols)
        len_bits = bits_for_prob(float(len_probs[length_bucket]))
        len_lo, len_hi = length_model.bucket_bounds(length_bucket, max_len_symbols)
        if len_hi > len_lo:
            residual_probs = length_model.residual_probs_for(
                length_bucket,
                max_len_symbols,
            )
            len_bits += bits_for_prob(float(residual_probs[length_offset - len_lo]))
    return ev_bits, off_bits, len_bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=default_enwik_path())
    parser.add_argument("--sizes", default="32kb,64kb,128kb,256kb")
    parser.add_argument("--window", type=int, default=engine.COPY_WINDOW)
    parser.add_argument("--candidates", type=int, default=engine.COPY_CANDIDATES)
    parser.add_argument(
        "--literal-bpb",
        type=float,
        default=8.0,
        help="proxy literal cost used to judge whether a copy was profitable",
    )
    parser.add_argument(
        "--show-buckets",
        action="store_true",
        help="print stats by copy length bucket",
    )
    parser.add_argument(
        "--choose-cost-aware",
        action="store_true",
        help=(
            "choose the candidate with the best literal-bpb savings instead "
            "of the longest candidate"
        ),
    )
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    raw_all = args.path.read_bytes()[: max(sizes)]

    print(f"file: {args.path}")
    print(
        f"window: {args.window:,} candidates: {args.candidates} "
        f"literal proxy: {args.literal_bpb:.2f} bpb"
    )
    print(
        "     size | events | copied% | lit bytes | copy hdr bpb | "
        "avg hdr bits | profitable | est stream bpb"
    )
    print("-" * 98)

    for n in sizes:
        data = raw_all[:n]
        matcher = engine.RollingCopyMatcher(
            data,
            window=args.window,
            max_candidates=args.candidates,
        )
        offset_model = engine.OffsetModel(args.window)
        event_model = engine.EventModel()
        length_model = engine.LengthModel(engine.COPY_MAX - engine.COPY_MIN + 1)

        pos = 0
        events = 0
        copied = 0
        literals = 0
        copy_bits = 0.0
        event_bits = 0.0
        copy_event_bits = 0.0
        offset_bits = 0.0
        length_bits = 0.0
        profitable = 0
        lengths: Counter[int] = Counter()
        length_buckets: dict[str, BucketStats] = {
            "8-9": BucketStats(),
            "10-11": BucketStats(),
            "12-15": BucketStats(),
            "16-23": BucketStats(),
            "24-31": BucketStats(),
            "32+": BucketStats(),
        }

        while pos < n:
            if args.choose_cost_aware:
                all_candidates = matcher.candidates(pos)
                best_savings = 0.0
                copy = None
                for cand_offset, cand_length in all_candidates:
                    ev_b, off_b, len_b = copy_header_bits(
                        offset=cand_offset,
                        length=cand_length,
                        pos=pos,
                        n=n,
                        window=args.window,
                        event_model=event_model,
                        offset_model=offset_model,
                        length_model=length_model,
                    )
                    bits = ev_b + off_b + len_b
                    savings = args.literal_bpb * cand_length - bits
                    if savings > best_savings:
                        best_savings = savings
                        copy = (cand_offset, cand_length)
            else:
                copy = matcher.find(pos)
            probs = event_model.probs()
            if copy is None:
                event_bits += bits_for_prob(float(probs[0]))
                event_model.observe(0)
                literals += 1
                pos += 1
                continue

            offset, length = copy
            events += 1
            copied += length
            lengths[length] += 1

            ev_bits, off_bits, len_bits = copy_header_bits(
                offset=offset,
                length=length,
                pos=pos,
                n=n,
                window=args.window,
                event_model=event_model,
                offset_model=offset_model,
                length_model=length_model,
            )
            bits = ev_bits
            bits += off_bits
            bits += len_bits

            event_bits += ev_bits
            copy_event_bits += ev_bits
            offset_bits += off_bits
            length_bits += len_bits
            copy_bits += bits
            if bits < args.literal_bpb * length:
                profitable += 1

            if length <= 9:
                label = "8-9"
            elif length <= 11:
                label = "10-11"
            elif length <= 15:
                label = "12-15"
            elif length <= 23:
                label = "16-23"
            elif length <= 31:
                label = "24-31"
            else:
                label = "32+"
            st = length_buckets[label]
            st.events += 1
            st.bytes += length
            st.bits += bits
            st.profitable += int(bits < args.literal_bpb * length)

            event_model.observe(1)
            offset_model.observe(offset)
            length_model.observe(length - engine.COPY_MIN)
            pos += length

        literal_bits = args.literal_bpb * literals
        est_bits = literal_bits + copy_bits
        print(
            f"{n:>9,d} | {events:>6,d} | {100*copied/n:>6.1f}% | "
            f"{literals:>9,d} | {copy_bits/max(copied, 1):>12.3f} | "
            f"{copy_bits/max(events, 1):>12.2f} | "
            f"{100*profitable/max(events, 1):>9.1f}% | "
            f"{est_bits/n:>14.3f}"
        )
        print(
            "          hdr/event components: "
            f"event={copy_event_bits/max(events, 1):.2f} "
            f"offset={offset_bits/max(events, 1):.2f} "
            f"length={length_bits/max(events, 1):.2f}"
        )

        if args.show_buckets:
            for label, st in length_buckets.items():
                if st.events == 0:
                    continue
                print(
                    f"          len {label:<5} events={st.events:<6,d} "
                    f"bytes={st.bytes:<7,d} hdr/event={st.bits/st.events:>6.2f} "
                    f"hdr/b={st.bits/st.bytes:>5.2f} "
                    f"profitable={100*st.profitable/st.events:>5.1f}%"
                )


if __name__ == "__main__":
    main()

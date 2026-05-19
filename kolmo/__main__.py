"""CLI entry point. Usage:

    python -m kolmo compress INPUT OUTPUT
    python -m kolmo decompress INPUT OUTPUT
"""

import sys
import time

from kolmo import compress, decompress


def _usage_and_exit():
    print(
        "usage:\n"
        "  python -m kolmo compress INPUT OUTPUT\n"
        "  python -m kolmo decompress INPUT OUTPUT",
        file=sys.stderr,
    )
    sys.exit(2)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) != 3 or argv[0] not in ("compress", "decompress"):
        _usage_and_exit()
    op, inpath, outpath = argv

    with open(inpath, "rb") as f:
        data = f.read()

    fn = compress if op == "compress" else decompress
    t = time.monotonic()
    out = fn(data)
    elapsed = time.monotonic() - t

    with open(outpath, "wb") as f:
        f.write(out)

    if op == "compress":
        pct = 100.0 * len(out) / len(data) if data else 0.0
        print(
            f"{len(data):,}B → {len(out):,}B  ({pct:.1f}%)  "
            f"in {elapsed:.1f}s  ({len(data) / elapsed:.0f} B/s)"
        )
    else:
        print(
            f"{len(data):,}B → {len(out):,}B  "
            f"in {elapsed:.1f}s  ({len(out) / elapsed:.0f} B/s)"
        )


if __name__ == "__main__":
    main()

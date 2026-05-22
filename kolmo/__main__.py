"""CLI entry point.

Usage:
    kolmo c INPUT OUTPUT        # compress
    kolmo d INPUT OUTPUT        # decompress
    kolmo compress INPUT OUTPUT   # long forms also work
    kolmo decompress INPUT OUTPUT

Environment variables that affect behavior:
    KOLMO_FIXED=1           use the bit-deterministic Q15 integer engine
                            (cross-machine identical, ~20x slower than PyTorch)
    KOLMO_SKIP_PRIME=1      skip the seed-corpus warmup (mostly for tests)
    KOLMO_NO_SEED_CACHE=1   force fixed-mode prime to re-run, bypassing cache
    KOLMO_CACHE_DIR=path    override the primed-state cache location
    KOLMO_DEVICE=cpu|cuda   force PyTorch path to a specific device
"""

import sys
import time

from kolmo import compress, decompress

_COMPRESS_NAMES = {"c", "compress"}
_DECOMPRESS_NAMES = {"d", "decompress"}


def _usage_and_exit(stream=sys.stderr, code: int = 2) -> None:
    print(
        "usage:\n"
        "  kolmo c|compress    INPUT OUTPUT\n"
        "  kolmo d|decompress  INPUT OUTPUT\n"
        "\n"
        "Run `kolmo --help` for environment-variable options.",
        file=stream,
    )
    sys.exit(code)


def _help_and_exit() -> None:
    print(__doc__.strip())
    sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] in ("-h", "--help", "help"):
        _help_and_exit()
    if len(argv) != 3:
        _usage_and_exit()
    op, inpath, outpath = argv
    if op in _COMPRESS_NAMES:
        fn = compress
        kind = "compress"
    elif op in _DECOMPRESS_NAMES:
        fn = decompress
        kind = "decompress"
    else:
        _usage_and_exit()

    with open(inpath, "rb") as f:
        data = f.read()

    t = time.monotonic()
    out = fn(data)
    elapsed = time.monotonic() - t

    with open(outpath, "wb") as f:
        f.write(out)

    if kind == "compress":
        pct = 100.0 * len(out) / len(data) if data else 0.0
        print(
            f"{len(data):,}B -> {len(out):,}B  ({pct:.1f}%)  "
            f"in {elapsed:.1f}s  ({len(data) / max(elapsed, 1e-9):.0f} B/s)"
        )
    else:
        print(
            f"{len(data):,}B -> {len(out):,}B  "
            f"in {elapsed:.1f}s  ({len(out) / max(elapsed, 1e-9):.0f} B/s)"
        )


if __name__ == "__main__":
    main()

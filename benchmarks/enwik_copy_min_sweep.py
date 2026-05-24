"""Sweep COPY_MIN values on enwik prefixes.

The copy-cost diagnostic can suggest that very short matches are marginal, but
the real answer depends on neural literal probabilities and the exact
arithmetic stream. This benchmark changes COPY_MIN in-process, reloads the
compress/decompress modules so both sides agree on the format, and measures
actual compressed sizes.

This is an experiment harness, not a stable file-format knob. If a COPY_MIN
change wins and becomes the default, old blobs are format-incompatible unless
the magic/version is bumped.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from enwik_prefix import default_enwik_path, parse_sizes


def run_one(raw: bytes, copy_min: int, window: int, decode: bool) -> tuple[int, float, float, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("KOLMO_FIXED", None)
    env["KOLMO_USE_ROPE"] = "1"
    script = """
import hashlib
import importlib
import sys
import time

import kolmo._engine as engine

engine.COPY_MIN = __COPY_MIN__
engine.COPY_WINDOW = __WINDOW__

import kolmo.compress as compress_mod
import kolmo.decompress as decompress_mod

compress_mod = importlib.reload(compress_mod)
decompress_mod = importlib.reload(decompress_mod)

data = sys.stdin.buffer.read()
t = time.perf_counter()
blob = compress_mod.compress(data)
enc = time.perf_counter() - t
dec = -1.0
if __DECODE__:
    t = time.perf_counter()
    out = decompress_mod.decompress(blob)
    dec = time.perf_counter() - t
    if out != data:
        raise SystemExit("round-trip mismatch")
print(len(blob), f"{enc:.3f}", f"{dec:.3f}", hashlib.sha256(blob).hexdigest()[:16])
""".replace("__COPY_MIN__", repr(copy_min)).replace(
        "__WINDOW__", repr(window)
    ).replace(
        "__DECODE__", repr(decode)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout.decode("utf-8", errors="replace"))
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(proc.returncode)
    out = proc.stdout.decode("utf-8").strip().split()
    return int(out[0]), float(out[1]), float(out[2]), out[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=default_enwik_path())
    parser.add_argument("--sizes", default="16kb,32kb")
    parser.add_argument("--copy-mins", default="8,10,12,16")
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    copy_mins = [int(x.strip()) for x in args.copy_mins.split(",") if x.strip()]
    raw_all = args.path.read_bytes()[: max(sizes)]
    decode = not args.no_decode

    print(f"file: {args.path}")
    print(f"window: {args.window:,} decode: {decode}")
    print("     size | copy_min |     bytes |   ratio |    bpb |      enc |      dec | sha")
    print("-" * 86)
    for n in sizes:
        raw = raw_all[:n]
        gz = gzip.compress(raw, compresslevel=9)
        print(
            f"{n:>9,d} | {'gzip-9':>8} | {len(gz):>9,d} | "
            f"{len(gz)/n:>7.4f} | {8*len(gz)/n:>6.3f} | {'-':>8} | {'-':>8} | -",
            flush=True,
        )
        for copy_min in copy_mins:
            t = time.perf_counter()
            blob_len, enc, dec, sha = run_one(raw, copy_min, args.window, decode)
            _wall = time.perf_counter() - t
            dec_s = "skip" if dec < 0 else f"{dec:8.1f}s"
            print(
                f"{n:>9,d} | {copy_min:>8,d} | {blob_len:>9,d} | "
                f"{blob_len/n:>7.4f} | {8*blob_len/n:>6.3f} | "
                f"{enc:>8.1f}s | {dec_s} | {sha}",
                flush=True,
            )


if __name__ == "__main__":
    main()

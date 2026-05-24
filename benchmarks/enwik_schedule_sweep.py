"""Sweep online-training schedules on enwik prefixes.

The default sublinear schedule is a speed win, but Hutter cares about ratio on
long files. This harness patches the training schedule constants in-process,
reloads compressor/decompressor modules, and measures the actual blob size.

Schedules are encoded as max multipliers:
  max_mult=1   -> train every 16 bytes forever (best ratio candidate, slow)
  max_mult=32  -> current default: double every 4KB, cap near CONTEXT

Like other architecture/schedule sweeps, this is an experiment harness. If the
default schedule changes, old blobs are format-incompatible unless the magic
or metadata changes too.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from enwik_prefix import default_enwik_path, parse_sizes


def run_one(raw: bytes, max_mult: int, doubling_bytes: int, decode: bool) -> tuple[int, float, float, str]:
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

engine._TRAIN_SCHEDULE_MAX_MULT = __MAX_MULT__
engine._TRAIN_SCHEDULE_DOUBLING_BYTES = __DOUBLING_BYTES__

compress_mod = importlib.import_module("kolmo.compress")
decompress_mod = importlib.import_module("kolmo.decompress")
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
""".replace("__MAX_MULT__", repr(max_mult)).replace(
        "__DOUBLING_BYTES__", repr(doubling_bytes)
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
    parser.add_argument("--max-mults", default="1,2,4,8,32")
    parser.add_argument("--doubling-bytes", type=int, default=4096)
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    max_mults = [int(x.strip()) for x in args.max_mults.split(",") if x.strip()]
    raw_all = args.path.read_bytes()[: max(sizes)]
    decode = not args.no_decode

    print(f"file: {args.path}")
    print(f"doubling_bytes: {args.doubling_bytes:,} decode: {decode}")
    print("     size | max_mult |     bytes |   ratio |    bpb |      enc |      dec | sha")
    print("-" * 86)
    for n in sizes:
        raw = raw_all[:n]
        gz = gzip.compress(raw, compresslevel=9)
        print(
            f"{n:>9,d} | {'gzip-9':>8} | {len(gz):>9,d} | "
            f"{len(gz)/n:>7.4f} | {8*len(gz)/n:>6.3f} | {'-':>8} | {'-':>8} | -",
            flush=True,
        )
        for max_mult in max_mults:
            blob_len, enc, dec, sha = run_one(
                raw,
                max_mult=max_mult,
                doubling_bytes=args.doubling_bytes,
                decode=decode,
            )
            dec_s = "skip" if dec < 0 else f"{dec:8.1f}s"
            print(
                f"{n:>9,d} | {max_mult:>8,d} | {blob_len:>9,d} | "
                f"{blob_len/n:>7.4f} | {8*blob_len/n:>6.3f} | "
                f"{enc:>8.1f}s | {dec_s} | {sha}",
                flush=True,
            )


if __name__ == "__main__":
    main()

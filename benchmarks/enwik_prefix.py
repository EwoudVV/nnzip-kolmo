"""Benchmark kolmo on prefixes of enwik9.

This is the practical Hutter-work loop: use real enwik bytes, report bpb,
compare to gzip, and print progress after every prefix so long runs do not
look hung.

Examples:
    python benchmarks/enwik_prefix.py --path ~/Downloads/enwik9/enwik9
    python benchmarks/enwik_prefix.py --sizes 8192,16384,32768 --variant abs
    python benchmarks/enwik_prefix.py --variant rope --no-decode
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def default_enwik_path() -> Path | None:
    candidates = [
        Path(os.environ.get("ENWIK_PATH", "")),
        Path.home() / "Downloads" / "enwik9" / "enwik9",
        Path.home() / "Downloads" / "enwik9",
        Path.home() / "Downloads" / "enwik8",
        REPO / "data" / "enwik9",
        REPO / "data" / "enwik8",
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def parse_sizes(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip().lower().replace("_", "")
        if not part:
            continue
        mult = 1
        if part.endswith("kb"):
            mult = 1024
            part = part[:-2]
        elif part.endswith("mb"):
            mult = 1024 * 1024
            part = part[:-2]
        elif part.endswith("k"):
            mult = 1024
            part = part[:-1]
        elif part.endswith("m"):
            mult = 1024 * 1024
            part = part[:-1]
        out.append(int(float(part) * mult))
    return out


def run_one(raw: bytes, variant: str, decode: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("KOLMO_FIXED", None)
    env.pop("KOLMO_USE_ROPE", None)
    if variant == "abs":
        env["KOLMO_USE_ROPE"] = "0"
    elif variant == "rope":
        env["KOLMO_USE_ROPE"] = "1"
    elif variant == "fixed":
        env["KOLMO_FIXED"] = "1"
        env["KOLMO_USE_ROPE"] = "0"
    elif variant == "fixed-rope":
        env["KOLMO_FIXED"] = "1"
        env["KOLMO_USE_ROPE"] = "1"
    else:
        raise ValueError(f"unknown variant {variant!r}")

    script = """
import hashlib
import os
import sys
import time
from kolmo import compress, decompress

data = sys.stdin.buffer.read()
t = time.perf_counter()
blob = compress(data)
enc = time.perf_counter() - t
dec = -1.0
if __DECODE__:
    t = time.perf_counter()
    out = decompress(blob)
    dec = time.perf_counter() - t
    if out != data:
        raise SystemExit("round-trip mismatch")
print(len(blob), f"{enc:.3f}", f"{dec:.3f}", hashlib.sha256(blob).hexdigest()[:16])
""".replace("__DECODE__", repr(decode))
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
    blob_len = int(out[0])
    enc = float(out[1])
    dec = float(out[2])
    sha = out[3]
    bpb = 8.0 * blob_len / len(raw)
    ratio = blob_len / len(raw)
    dec_s = "skip" if dec < 0 else f"{dec:8.1f}s"
    print(
        f"{len(raw):>9,d} | {variant:<10} | {blob_len:>9,d} | "
        f"{ratio:>7.4f} | {bpb:>6.3f} | {enc:>8.1f}s | {dec_s} | {sha}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=default_enwik_path())
    parser.add_argument(
        "--sizes",
        default="4kb,8kb,16kb,32kb",
        help="comma-separated sizes, supports k/kb/m/mb suffixes",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=["abs", "rope", "fixed", "fixed-rope"],
        help="variant to run; repeatable. Defaults to rope (current model default).",
    )
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit(
            "enwik file not found; pass --path or set ENWIK_PATH "
            "(expected e.g. ~/Downloads/enwik9/enwik9)"
        )

    sizes = parse_sizes(args.sizes)
    variants = args.variant or ["rope"]
    max_size = max(sizes)
    raw_all = args.path.read_bytes()[:max_size]
    if len(raw_all) < max_size:
        raise SystemExit(f"{args.path} only has {len(raw_all)} bytes")

    print(f"file: {args.path}")
    print(f"python: {sys.executable}")
    print("     size | variant    |     bytes |   ratio |    bpb |      enc |      dec | sha")
    print("-" * 88)
    for n in sizes:
        raw = raw_all[:n]
        gz = gzip.compress(raw, compresslevel=9)
        print(
            f"{n:>9,d} | {'gzip-9':<10} | {len(gz):>9,d} | "
            f"{len(gz)/n:>7.4f} | {8*len(gz)/n:>6.3f} | {'-':>8} | {'-':>8} | -",
            flush=True,
        )
        for variant in variants:
            run_one(raw, variant, decode=not args.no_decode)


if __name__ == "__main__":
    main()

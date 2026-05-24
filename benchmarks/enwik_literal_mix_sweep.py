"""Sweep adaptive literal-model mix weights on enwik prefixes."""

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


def parse_weights(text: str) -> list[tuple[float, float, float, float]]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = [float(x) for x in part.split(":")]
        if len(pieces) == 2:
            w2 = 0.0
            w1, w0 = pieces
            confidence = 0.0
        elif len(pieces) == 3:
            w2, w1, w0 = pieces
            confidence = 0.0
        elif len(pieces) == 4:
            w2, w1, w0, confidence = pieces
        else:
            raise ValueError(
                "weights must be order1:order0, order2:order1:order0, "
                "or order2:order1:order0:confidence"
            )
        out.append((w2, w1, w0, confidence))
    return out


def run_one(
    raw: bytes,
    w2: float,
    w1: float,
    w0: float,
    confidence: float,
    decode: bool,
) -> tuple[int, float, float, str]:
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

engine.LITERAL_ORDER1_WEIGHT = __W1__
engine.LITERAL_ORDER0_WEIGHT = __W0__
engine.LITERAL_ORDER2_WEIGHT = __W2__
engine.LITERAL_ORDER2_CONFIDENCE = __CONFIDENCE__

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
""".replace("__W2__", repr(w2)).replace("__W1__", repr(w1)).replace("__W0__", repr(w0)).replace("__CONFIDENCE__", repr(confidence)).replace(
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
    parser.add_argument("--sizes", default="16kb")
    parser.add_argument(
        "--weights",
        default="0:0:0,0:0.05:0.01,0.05:0.03:0.005,0.10:0.05:0.01",
        help="comma-separated order2:order1:order0[:confidence] weights",
    )
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    weights = parse_weights(args.weights)
    raw_all = args.path.read_bytes()[: max(sizes)]
    decode = not args.no_decode

    print(f"file: {args.path}")
    print(f"decode: {decode}")
    print("     size | order2 | order1 | order0 |   conf |     bytes |   ratio |    bpb |      enc | sha")
    print("-" * 103)
    for n in sizes:
        raw = raw_all[:n]
        gz = gzip.compress(raw, compresslevel=9)
        print(
            f"{n:>9,d} | {'gzip':>6} | {'-':>6} | {'-':>6} | {'-':>6} | {len(gz):>9,d} | "
            f"{len(gz)/n:>7.4f} | {8*len(gz)/n:>6.3f} | {'-':>8} | -",
            flush=True,
        )
        for w2, w1, w0, confidence in weights:
            blob_len, enc, _dec, sha = run_one(raw, w2, w1, w0, confidence, decode)
            print(
                f"{n:>9,d} | {w2:>6.3f} | {w1:>6.3f} | {w0:>6.3f} | {confidence:>6.1f} | {blob_len:>9,d} | "
                f"{blob_len/n:>7.4f} | {8*blob_len/n:>6.3f} | {enc:>8.1f}s | {sha}",
                flush=True,
            )


if __name__ == "__main__":
    main()

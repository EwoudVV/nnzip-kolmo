"""Sweep the copy selector's literal-bpb proxy on enwik prefixes.

COPY_LITERAL_BPB is encoder-only: it decides whether a candidate copy's
event+offset+length header is worth paying compared with spelling those bytes
as literals. The decoder just follows the explicit event stream, so changing
this threshold changes ratio but not the blob format.

After adding the adaptive order-2 literal model, literals got much cheaper.
This harness finds the new threshold that keeps useful copies and rejects
copy headers the literal model can beat.
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


def run_one(
    raw: bytes,
    copy_bpb: float,
    use_literal_proxy: bool,
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

engine.COPY_LITERAL_BPB = __COPY_BPB__
engine.COPY_USE_LITERAL_MODEL_PROXY = __USE_LITERAL_PROXY__

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
""".replace("__COPY_BPB__", repr(copy_bpb)).replace(
        "__USE_LITERAL_PROXY__", repr(use_literal_proxy)
    ).replace("__DECODE__", repr(decode))
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
    parser.add_argument("--copy-bpbs", default="2.0,2.25,2.5,2.75,3.0,3.25")
    parser.add_argument("--literal-proxy", action="store_true")
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    copy_bpbs = [float(x.strip()) for x in args.copy_bpbs.split(",") if x.strip()]
    raw_all = args.path.read_bytes()[: max(sizes)]
    decode = not args.no_decode

    print(f"file: {args.path}")
    print(f"decode: {decode}")
    print(f"literal proxy: {args.literal_proxy}")
    print("     size | copy_bpb |     bytes |   ratio |    bpb |      enc | sha")
    print("-" * 76)
    for n in sizes:
        raw = raw_all[:n]
        gz = gzip.compress(raw, compresslevel=9)
        print(
            f"{n:>9,d} | {'gzip':>8} | {len(gz):>9,d} | "
            f"{len(gz)/n:>7.4f} | {8*len(gz)/n:>6.3f} | {'-':>8} | -",
            flush=True,
        )
        for copy_bpb in copy_bpbs:
            blob_len, enc, _dec, sha = run_one(
                raw,
                copy_bpb,
                use_literal_proxy=args.literal_proxy,
                decode=decode,
            )
            print(
                f"{n:>9,d} | {copy_bpb:>8.3f} | {blob_len:>9,d} | "
                f"{blob_len/n:>7.4f} | {8*blob_len/n:>6.3f} | "
                f"{enc:>8.1f}s | {sha}",
                flush=True,
            )


if __name__ == "__main__":
    main()

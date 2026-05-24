"""Sweep model context length on enwik prefixes.

CONTEXT controls how much byte history the transformer sees during cached
inference and training. Hutter-scale files have structure beyond 256 bytes
(wiki links, templates, table rows, references), so this is a real ratio lever
even though it is expensive.
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


def run_one(raw: bytes, context: int, decode: bool) -> tuple[int, float, float, str]:
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

engine.CONTEXT = __CONTEXT__

# RoPE buffers must cover the largest cached position. The class default is
# 512, enough for context <= 511. For larger contexts, patch the constructor
# default in this subprocess without changing the repo default.
from kolmo import model as model_mod
orig_init = model_mod.KolmoTransformer.__init__
def patched_init(self, *args, **kwargs):
    # Cached inference can warm `context` positions and then batch-feed a copy
    # chunk before the next training boundary, so max_context needs headroom
    # beyond CONTEXT itself. Production default is 512 for CONTEXT=256.
    kwargs.setdefault("max_context", max(512, 2 * __CONTEXT__ + 16))
    orig_init(self, *args, **kwargs)
model_mod.KolmoTransformer.__init__ = patched_init

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
""".replace("__CONTEXT__", repr(context)).replace("__DECODE__", repr(decode))
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
    parser.add_argument("--contexts", default="128,256,384,512")
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    if args.path is None or not args.path.is_file():
        raise SystemExit("enwik file not found; pass --path or set ENWIK_PATH")

    sizes = parse_sizes(args.sizes)
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    raw_all = args.path.read_bytes()[: max(sizes)]
    decode = not args.no_decode

    print(f"file: {args.path}")
    print(f"decode: {decode}")
    print("     size |  context |     bytes |   ratio |    bpb |      enc |      dec | sha")
    print("-" * 86)
    for n in sizes:
        raw = raw_all[:n]
        gz = gzip.compress(raw, compresslevel=9)
        print(
            f"{n:>9,d} | {'gzip-9':>8} | {len(gz):>9,d} | "
            f"{len(gz)/n:>7.4f} | {8*len(gz)/n:>6.3f} | {'-':>8} | {'-':>8} | -",
            flush=True,
        )
        for context in contexts:
            blob_len, enc, dec, sha = run_one(raw, context, decode)
            dec_s = "skip" if dec < 0 else f"{dec:8.1f}s"
            print(
                f"{n:>9,d} | {context:>8,d} | {blob_len:>9,d} | "
                f"{blob_len/n:>7.4f} | {8*blob_len/n:>6.3f} | "
                f"{enc:>8.1f}s | {dec_s} | {sha}",
                flush=True,
            )


if __name__ == "__main__":
    main()

"""Bench torch.compile vs eager on the PyTorch path.

cProfile on a 2 KB compress (PyTorch path, full preset, skip-prime) showed
~3.7 s out of 14.8 s in `torch._C._nn.linear`, almost all of it small-matmul
dispatch overhead rather than actual matmul math. `torch.compile` fuses
the model graph and removes most of that dispatch tax. This script
quantifies the win at a real scale.

Requirements: torch >= 2.4, Python >= 3.12 (older venvs raise
"Dynamo is not supported on Python 3.12+"). The kolmo Mac dev venv has
torch 2.2.2 which can't run this bench; the Windows venv (torch 2.6+cu124)
can.

Usage:
    python benchmarks/torch_compile_speedup.py [enwik9_path]

Runs 16 KB of enwik9 through draft + full presets × {compile off, on},
skip-prime, and prints a table of bpb + wall time.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    enwik = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if enwik is None:
        # Try a few common locations.
        candidates = [
            Path.home() / "Downloads" / "enwik9" / "enwik9",
            Path.home() / "Downloads" / "enwik9",
            Path.home() / "kolmo-test" / "enwik9",
        ]
        for c in candidates:
            if c.is_file():
                enwik = c
                break
    if enwik is None or not enwik.is_file():
        print(
            "enwik9 not found; pass path as the first argument or place it "
            "at ~/Downloads/enwik9/enwik9",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"enwik9 = {enwik}")
    print(f"repo   = {repo}")
    print(f"python = {sys.executable}")
    print()

    child = '''
import os, sys, time
sys.path.insert(0, r"{repo}")
os.environ["KOLMO_MODEL"] = "{preset}"
os.environ["KOLMO_SKIP_PRIME"] = "1"
os.environ["KOLMO_TORCH_COMPILE"] = "{compile}"
import warnings; warnings.simplefilter("default")
from kolmo import compress, decompress
data = open(r"{enwik}", "rb").read(16384)
t = time.perf_counter(); blob = compress(data); enc = time.perf_counter() - t
assert decompress(blob) == data
bpb = len(blob) * 8 / len(data)
print(f"{{label!r}}: {{len(data)}} -> {{len(blob)}}  bpb={{bpb:.4f}}  enc={{enc:.1f}}s")
'''
    py = sys.executable
    for preset in ("draft", "full"):
        for compile_flag in ("0", "1"):
            label = f"{preset}_compile={compile_flag}"
            code = "label = " + repr(label) + "\n" + child.format(
                repo=str(repo),
                preset=preset,
                compile=compile_flag,
                enwik=str(enwik),
            )
            r = subprocess.run(
                [py, "-c", code], capture_output=True, text=True
            )
            print(
                r.stdout.strip()
                if r.returncode == 0
                else f"FAIL ({label}):\n{r.stderr[-600:]}"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()

"""Kaggle bench kernel for kolmo.

Pushed and executed remotely via `kaggle kernels push` (see run_bench.py
in this directory). Each run clones current main — so it always benches
the latest pushed commit — downloads enwik8 (byte-identical to the first
100 MB of enwik9), runs the CONFIGS matrix, and writes RESULT lines both
to the kernel log and to /kaggle/working/results.txt, which the local
driver fetches when the run completes.

Edit CONFIGS below, then `python benchmarks/kaggle/run_bench.py`.
"""

import os
import subprocess
import sys
import zipfile

REPO = "https://github.com/EwoudVV/nnzip-kolmo.git"
# Everything in /kaggle/working is preserved (and later downloaded) as
# run output — keep only results.txt there. The clone and the corpus go
# to /kaggle/temp, which is discarded with the session.
WORK = "/kaggle/working"
TMP = "/kaggle/temp"
SRC = f"{TMP}/nnzip-kolmo"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"mkdir -p {TMP}")
sh(f"git clone --depth 1 {REPO} {SRC}")
sh(f"pip install -q -e {SRC}")
sh(
    f"wget -q http://mattmahoney.net/dc/enwik8.zip -O {TMP}/enwik8.zip"
    f" || wget -q https://mattmahoney.net/dc/enwik8.zip -O {TMP}/enwik8.zip"
)
zipfile.ZipFile(f"{TMP}/enwik8.zip").extract("enwik8", TMP)

# The kernels API doesn't let us pick the GPU model, and Kaggle's torch
# build has dropped Pascal (P100 = sm_60): cuda.is_available() lies, the
# first real kernel launch fails. Probe with an actual op and fall back
# to CPU-only so a campaign never dies on pool roulette.
import torch  # noqa: E402  (after pip install)

gpu_ok = False
if torch.cuda.is_available():
    try:
        (torch.ones(8, device="cuda") * 2).sum().item()
        gpu_ok = True
        print(f"GPU usable: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:  # noqa: BLE001 — any launch failure means CPU
        print(f"GPU present but unusable ({exc}); CPU only", flush=True)

# ---------------------------------------------------------------------------
# The campaign matrix. Mac reference numbers (draft, KOLMO_SKIP_PRIME=1):
#   8K default 2.8633 | 8K best 2.6797 | 16K best 2.7949 | 64K best 2.6475
# Never compare raw bpb across machines (PyTorch floats differ) — compare
# deltas within one machine.
# ---------------------------------------------------------------------------
P4 = "balanced_delimiter,after_number,in_text,position_modulo"
BEST = {
    "KOLMO_MIXER": "logistic",
    "KOLMO_LOGISTIC_BUCKETS": "1",
    "KOLMO_PREDICTORS": P4,
}
BASE = {"KOLMO_MODEL": "draft", "KOLMO_SKIP_PRIME": "1", "KOLMO_PROGRESS": "1"}

def _buckets(n: int) -> dict:
    return {**BEST, "KOLMO_LOGISTIC_BUCKETS": str(n)}


CONFIGS = [
    # Bucket-crossover + default-flip campaign (~6-7 h on T4 hardware).
    # Question 1: at what scale do bucketed mixer weights (more
    # specialization, less data per weight) overtake b=1? At <=64 KB,
    # b=1 wins decisively. Question 2: does logistic keep beating the
    # cost_aware default as size grows? (default-flip evidence)
    ("128K | logistic b=1 ", 131072, {"KOLMO_DEVICE": "cuda", **_buckets(1)}),
    ("128K | logistic b=16", 131072, {"KOLMO_DEVICE": "cuda", **_buckets(16)}),
    ("256K | cost_aware   ", 262144, {"KOLMO_DEVICE": "cuda"}),
    ("256K | logistic b=1 ", 262144, {"KOLMO_DEVICE": "cuda", **_buckets(1)}),
    ("256K | logistic b=4 ", 262144, {"KOLMO_DEVICE": "cuda", **_buckets(4)}),
    ("256K | logistic b=16", 262144, {"KOLMO_DEVICE": "cuda", **_buckets(16)}),
]

CHILD = """
import time, hashlib
payload = open('/kaggle/temp/enwik8', 'rb').read(SIZE)
from kolmo import compress, decompress
t0 = time.time()
blob = compress(payload)
print('--- compressed, decompressing (round-trip check) ---', flush=True)
assert decompress(blob) == payload, 'ROUND TRIP FAILED'
t = time.time() - t0
line = (f"LABEL  {len(blob)*8/len(payload):.4f} bpb  {len(blob)} B"
        f"  {t:.0f}s  {hashlib.sha256(blob).hexdigest()[:8]}")
print('RESULT', line, flush=True)
open('/kaggle/working/results.txt', 'a').write(line + '\\n')
"""

for label, size, extra in CONFIGS:
    if extra.get("KOLMO_DEVICE") == "cuda" and not gpu_ok:
        print(f"SKIP {label} (no usable GPU)", flush=True)
        open(f"{WORK}/results.txt", "a").write(f"{label}  SKIPPED no GPU\n")
        continue
    print(f"\n========== {label} ==========", flush=True)
    env = dict(os.environ, **BASE, **extra)
    code = CHILD.replace("SIZE", str(size)).replace("LABEL", label)
    # 256 KB at T4-host speed extrapolates to ~70 min/config; give each
    # config 2.5 h before declaring it hung. Kernel ceiling is 12 h.
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=SRC, env=env, timeout=9000
    )
    if r.returncode != 0:
        open(f"{WORK}/results.txt", "a").write(f"{label}  FAILED rc={r.returncode}\n")

print("\n=== RESULTS ===", flush=True)
try:
    print(open(f"{WORK}/results.txt").read(), flush=True)
except FileNotFoundError:
    print("(no results written)", flush=True)

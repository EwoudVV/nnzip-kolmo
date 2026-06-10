#!/usr/bin/env python3
"""Push the bench kernel to Kaggle, wait for it, fetch the results.

One-time setup:
  1. kaggle.com -> avatar -> Settings -> API -> "Create New Token"
     (downloads kaggle.json)
  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
     && chmod 600 ~/.kaggle/kaggle.json
  3. pip install kaggle

Then a full remote bench is one command:
    python benchmarks/kaggle/run_bench.py

Edit the CONFIGS list in bench_kernel.py to change the campaign; the
push uploads whatever is in this directory. Each run clones current
main on the Kaggle side, so push your kolmo commits first.
"""

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
KERNEL = "ewoudvanvooren/kolmo-bench"
RUNS = HERE / "runs"
POLL_SECONDS = 45


def kaggle(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kaggle", *args], text=True, capture_output=True, check=False
    )


def main() -> int:
    # `--watch` attaches to the run already in flight (poll + fetch only).
    # Without it, the script pushes a NEW version first — never use the
    # bare form to "check on" a running campaign or you will restart it.
    if "--watch" not in sys.argv:
        push = kaggle("kernels", "push", "-p", str(HERE))
        print((push.stdout + push.stderr).strip(), flush=True)
        if push.returncode != 0 or "error" in (push.stdout + push.stderr).lower():
            return 1

    print(f"polling every {POLL_SECONDS}s ...", flush=True)
    # Only a real KernelWorkerStatus verdict ends the watch. Anything else
    # (DNS blips, timeouts, API hiccups — their tracebacks contain "Error"
    # and used to be mistaken for a failed run) is transient: keep polling
    # unless it fails many times in a row.
    consecutive_failures = 0
    while True:
        time.sleep(POLL_SECONDS)
        st = kaggle("kernels", "status", KERNEL)
        line = (st.stdout + st.stderr).strip()
        if "KernelWorkerStatus" not in line:
            consecutive_failures += 1
            print(
                time.strftime("%H:%M:%S"),
                f"poll failed ({consecutive_failures}/20): "
                + line.splitlines()[-1][:120],
                flush=True,
            )
            if consecutive_failures >= 20:
                print("giving up after ~15 min without connectivity; "
                      "the Kaggle-side run continues — rerun this script "
                      "or `kaggle kernels output` later to fetch results")
                return 1
            continue
        consecutive_failures = 0
        print(time.strftime("%H:%M:%S"), line, flush=True)
        if "KernelWorkerStatus.RUNNING" in line or "QUEUED" in line:
            continue
        if "KernelWorkerStatus.COMPLETE" not in line:
            print("run did not complete cleanly; fetching log anyway")
        break

    out = RUNS / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        fetch = kaggle("kernels", "output", KERNEL, "-p", str(out))
        if fetch.returncode == 0:
            break
        print(f"fetch attempt {attempt + 1} failed; retrying in 60s")
        time.sleep(60)
    print((fetch.stdout + fetch.stderr).strip(), flush=True)

    results = out / "results.txt"
    if results.exists():
        print("\n=== RESULTS ===\n" + results.read_text())
    else:
        log = next(out.glob("*.log"), None)
        if log is not None:
            print("\n=== LOG TAIL ===\n" + log.read_text()[-4000:])
    print(f"(full output in {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

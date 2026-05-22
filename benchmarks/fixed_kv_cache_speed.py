"""Measure the inference speedup from the fixed-point KV cache.

Before the cache, every new byte triggered a full `fixed_forward` over the
entire history — quadratic per block. After: warm once, then stream tokens
one at a time, each a single-row attention against the cached K/V.

Numbers are from the prod-sized model (d_model=256, n_heads=8, n_layers=4)
so they reflect actual compress-path cost, not toy sizes.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from kolmo.fixed_kv_cache import fixed_step, fixed_warm  # noqa: E402
from kolmo.fixed_model import extract_fixed_weights, fixed_forward  # noqa: E402
from kolmo.model import KolmoTransformer  # noqa: E402
from kolmo.stable_init import stable_init_model  # noqa: E402


def main() -> None:
    model = KolmoTransformer()  # production dims
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)

    history_len = 32
    n_steps = 16  # one BLOCK_SIZE worth of bytes

    history = np.arange(history_len, dtype=np.int64) % 256
    new_bytes = (np.arange(n_steps, dtype=np.int64) * 7 + 13) % 256

    # Pre-cache: each step calls fixed_forward over the growing history.
    t0 = time.perf_counter()
    seq = history.copy()
    for byte in new_bytes:
        seq = np.append(seq, byte)
        _ = fixed_forward(seq, weights)[-1]
    pre_cache_s = time.perf_counter() - t0

    # Post-cache: warm once, then step.
    t0 = time.perf_counter()
    _, caches = fixed_warm(history, weights)
    pos = history_len
    for byte in new_bytes:
        _, caches = fixed_step(int(byte), caches, weights, pos_offset=pos)
        pos += 1
    post_cache_s = time.perf_counter() - t0

    print(f"history_len={history_len}, n_steps={n_steps}")
    print(f"  pre-cache  (recompute full forward each step): {pre_cache_s:6.3f}s")
    print(f"  post-cache (warm + N step calls):              {post_cache_s:6.3f}s")
    print(f"  speedup:                                       {pre_cache_s / post_cache_s:6.2f}x")


if __name__ == "__main__":
    main()

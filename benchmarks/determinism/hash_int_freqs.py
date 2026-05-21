"""Hash deterministic integer frequencies from stable-initialized weights.

This is the critical Rung 2 probe: do the quantized integer frequencies that
the arithmetic coder will see match across machines?

If yes, the determinism plan works for inference: float ops on the way to
logits can vary by ~1 ULP, but the quantization grid collapses those errors
and we get identical integer counts to feed the arithmetic coder.

If no, we need to replace np.exp (the likely culprit — different libm
implementations) with a polynomial approximation or a precomputed table.

Run on Mac and Windows, compare hashes.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo.det_probs import logits_to_int_freqs  # noqa: E402
from kolmo.model import KolmoTransformer  # noqa: E402
from kolmo.np_model import extract_weights, kolmo_forward  # noqa: E402
from kolmo.stable_init import stable_init_model  # noqa: E402


def main() -> None:
    # Test on BOTH the tiny model (matches earlier probes) AND the production
    # config — small errors might survive a 2-layer model but not a 4-layer one.
    for label, kwargs, n_heads, n_layers in [
        ("tiny", dict(d_model=64, n_heads=4, n_layers=2, max_context=128), 4, 2),
        ("prod", dict(d_model=256, n_heads=8, n_layers=4, max_context=512), 8, 4),
    ]:
        model = KolmoTransformer(**kwargs)
        stable_init_model(model, seed=42)

        # Use a probe context that's nontrivial — random byte values plus the
        # BOS marker. Same on every machine because no RNG is used here.
        input_ids = np.array([0, 84, 104, 101, 32, 113, 117, 105, 99, 107], dtype=np.int64)

        logits = kolmo_forward(
            input_ids,
            extract_weights(model),
            n_heads=n_heads,
            n_layers=n_layers,
        )
        last_logits = logits[-1].astype(np.float64)

        freqs = logits_to_int_freqs(last_logits)
        digest = hashlib.sha256(freqs.tobytes()).hexdigest()
        print(f"{label:5}  vocab={len(freqs):4}  "
              f"sum={int(freqs.sum())}  "
              f"max={int(freqs.max())}  min={int(freqs.min())}  "
              f"sha256={digest}")
        # Show top-5 most likely symbols for sanity
        top = np.argsort(-freqs)[:5].tolist()
        print(f"        top5: {[(int(i), int(freqs[i])) for i in top]}")


if __name__ == "__main__":
    main()

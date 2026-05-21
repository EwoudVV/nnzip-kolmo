"""Hash int_freqs produced by the PYTORCH forward path used in compress.

The int-freqs probe (hash_int_freqs.py) uses the NumPy forward, which seems
to round-trip-deterministic across Mac and Windows. The full compress path
uses the PyTorch forward, and hash_no_training showed cross-machine bytes
DIFFER. This probe isolates: does PyTorch + det_probs match across machines
or not?

If the int_freqs here differ, PyTorch float errors are bigger than the 1/16384
grid. We then either coarsen the grid or replace PyTorch forward with NumPy
forward in the compress pipeline.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo._engine import warm_cache  # noqa: E402
from kolmo.det_probs import TOTAL_FREQ, logits_to_int_freqs  # noqa: E402
from kolmo.model import KolmoTransformer  # noqa: E402
from kolmo.stable_init import stable_init_model  # noqa: E402


def main() -> None:
    # Use the production config so this matches what compress actually does.
    model = KolmoTransformer()
    stable_init_model(model, seed=42)

    # Same input prefix as hash_int_freqs.py so we can compare with the
    # NumPy-forward probe directly.
    history = [0, 84, 104, 101, 32, 113, 117, 105, 99, 107]

    # Call the real PyTorch forward used inside compress.
    device = next(model.parameters()).device
    x = torch.tensor([history], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x, kv_caches=None, pos_offset=0)
    last_logits = logits[0, -1].cpu().numpy().astype(np.float64)

    freqs = logits_to_int_freqs(last_logits)
    digest = hashlib.sha256(freqs.tobytes()).hexdigest()
    print(f"pytorch logits[:5]: {last_logits[:5]}")
    print(f"freqs sum={int(freqs.sum())}  max={int(freqs.max())}  min={int(freqs.min())}")
    print(f"sha256: {digest}")
    print(f"top5: {[(int(i), int(freqs[i])) for i in np.argsort(-freqs)[:5]]}")


if __name__ == "__main__":
    main()

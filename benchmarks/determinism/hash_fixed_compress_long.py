"""Cross-machine determinism probe — long payload, PPM literal default.

Why this exists in addition to `hash_fixed_compress.py`:

The 45-byte payload in the original probe is too short to expose subtle
training-time divergences. With KOLMO_SKIP_PRIME=1 and a 45-byte input the
fixed-point pipeline only fires ~2 training steps, which wasn't enough to
amplify a single-bit difference in the Q15 RoPE cos table into observable
blob bytes. That's how a real platform-dependence bug shipped under green CI:

  - `torch.cos` on float32 disagreed in the last bit between Mac's Accelerate
    libm and Windows' UCRT libm for 2 entries of the (512, 16) cos table
  - the existing probe didn't run enough training to surface the cascade,
    so CI passed
  - PPM-mode payloads as small as 100 bytes diverged silently on real machines

This probe uses a longer, more structurally varied payload that triggers
several training steps and exercises the PPM literal-model path through
realistic byte transitions. If anything in the fixed-point training
pipeline (forward, backward, Adam, RoPE quantization, literal-model float
reductions, ...) becomes platform-dependent again, the resulting hash will
differ across the determinism matrix runners and the workflow will fail
loudly instead of silently.
"""

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

os.environ["KOLMO_FIXED"] = "1"
os.environ["KOLMO_SKIP_PRIME"] = "1"
# Explicit defaults — the probe should fail if a future commit changes them.
os.environ.setdefault("KOLMO_MODEL", "full")
os.environ.setdefault("KOLMO_LITERAL", "ppm")
os.environ.setdefault("KOLMO_NEURAL_WEIGHT", "0.50")

from kolmo import compress, decompress  # noqa: E402


def main() -> None:
    # ~280 bytes of mixed structured text. The mix of letters, spaces,
    # punctuation, repeats, and a numeric span makes the literal-model
    # walk hit several context orders and exercises copy detection on
    # the repeated phrase. Empirically this triggers ~5-6 training
    # steps under the default block-size schedule — enough that any
    # cumulative ULP drift in the trained weights becomes visible in
    # the encoded blob bytes.
    data = (
        b"The quick brown fox jumps over the lazy dog. "
        b"The quick brown fox jumps over the lazy dog again. "
        b"== Section ==\n"
        b"[[link|text]] {{template}}\n"
        b"Numbers: 1, 2, 3, 1024, 65536, 3.14159.\n"
        b"Determinism across machines, byte for byte, is the goal.\n"
    )
    blob = compress(data)
    out = decompress(blob)
    assert out == data, (
        "fixed-mode compress/decompress mismatch on this machine — "
        "cross-machine probe is meaningless until round-trip works locally"
    )
    print(f"input bytes: {len(data)}")
    print(f"output bytes: {len(blob)}")
    print(f"output sha256: {hashlib.sha256(blob).hexdigest()}")
    print(f"first 32 bytes: {blob[:32].hex()}")


if __name__ == "__main__":
    main()

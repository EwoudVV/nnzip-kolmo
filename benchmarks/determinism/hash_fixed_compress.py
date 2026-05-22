"""Hash a kolmo blob produced in fixed-point mode.

The earlier probes (hash_q15_matmul, hash_fixed_forward, hash_fixed_training)
test individual fixed-point ops. This one is the end-to-end claim: do
KOLMO_FIXED=1 compress and decompress agree across machines on the actual
bytes that get written to disk?

KOLMO_SKIP_PRIME=1 keeps the run short enough for CI — the seed corpus would
add hundreds of training steps. The compress pipeline still exercises the
full fixed-point engine on the actual payload: warm cache, step, train block,
optimizer step. If this hash matches across runners, the bulletproof claim
holds end to end.
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

from kolmo import compress, decompress  # noqa: E402


def main() -> None:
    data = b"deterministic across machines, byte for byte."
    blob = compress(data)
    out = decompress(blob)
    assert out == data, "fixed-mode compress/decompress mismatch on this machine"
    print(f"input bytes: {len(data)}")
    print(f"output bytes: {len(blob)}")
    print(f"output sha256: {hashlib.sha256(blob).hexdigest()}")
    print(f"first 32 bytes: {blob[:32].hex()}")


if __name__ == "__main__":
    main()

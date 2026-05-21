"""Compress with the seed warmup DISABLED and the per-block training DISABLED.

This isolates the inference path (transformer forward + det_probs +
arithmetic coder) from the training path. If Mac and Windows agree here,
deterministic inference works and the only remaining nondeterminism source
is the training step (PyTorch backward + Adam), which is task #4.

If Mac and Windows still disagree here, there's some nondeterminism on the
inference path we haven't isolated yet.
"""

import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import kolmo._engine as engine  # noqa: E402
from kolmo import compress  # noqa: E402


def main() -> None:
    # Disable seed warmup AND end-of-block training. Both are no-ops.
    original_prime = engine._prime_model
    original_train = engine.train_block

    def no_prime(model, optimizer):
        pass  # do nothing

    def no_train(model, optimizer, history, block_bytes):
        pass  # do nothing

    engine._prime_model = no_prime
    engine.train_block = no_train

    try:
        data = b"The quick brown fox jumps over the lazy dog. " * 50
        blob = compress(data)
        print(f"input bytes:  {len(data)}")
        print(f"output bytes: {len(blob)}")
        print(f"sha256:       {hashlib.sha256(blob).hexdigest()}")
    finally:
        engine._prime_model = original_prime
        engine.train_block = original_train


if __name__ == "__main__":
    main()

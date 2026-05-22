"""Hash fixed-point training after a few deterministic blocks.

This probe is for cross-machine verification. It starts from stable
PyTorch-initialized weights, quantizes them to Q15, runs fixed_train_block a
few times, then hashes both weights and integer Adam state.

Mac and Windows should print identical hashes. If they do, the full
fixed-point training path is bit-identical for this configuration.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from kolmo.fixed_model import extract_fixed_weights  # noqa: E402
from kolmo.fixed_train import fixed_train_block  # noqa: E402
from kolmo.model import KolmoTransformer  # noqa: E402
from kolmo.stable_init import stable_init_model  # noqa: E402


def _hash_weights_and_state(weights, state) -> str:
    h = hashlib.sha256()
    for name in sorted(weights):
        h.update(name.encode("utf-8"))
        h.update(weights[name].tobytes())
    h.update(f"step:{state.step}".encode("utf-8"))
    h.update(f"b1:{state.beta1_pow_q30}".encode("utf-8"))
    h.update(f"b2:{state.beta2_pow_q30}".encode("utf-8"))
    for name in sorted(state.m):
        h.update(f"m:{name}".encode("utf-8"))
        h.update(state.m[name].tobytes())
        h.update(f"v:{name}".encode("utf-8"))
        h.update(state.v[name].tobytes())
    return h.hexdigest()


def run_case(label: str, kwargs: dict, n_heads: int, n_layers: int, context: int) -> None:
    model = KolmoTransformer(**kwargs)
    stable_init_model(model, seed=42)
    weights = extract_fixed_weights(model)

    history = [0] + list(b"the quick brown ")
    blocks = [list(b"fox jumps over "), list(b"the lazy dog.")]
    state = None
    for block in blocks:
        state = fixed_train_block(
            weights,
            state,
            history,
            block,
            n_heads=n_heads,
            n_layers=n_layers,
            context=context,
        )
        history = (history + block)[-context:]

    digest = _hash_weights_and_state(weights, state)
    print(f"{label:8} steps={state.step} sha256={digest}")


def main() -> None:
    run_case(
        "tiny",
        dict(d_model=16, n_heads=4, n_layers=1, max_context=64),
        n_heads=4,
        n_layers=1,
        context=64,
    )
    run_case(
        "small",
        dict(d_model=64, n_heads=4, n_layers=2, max_context=128),
        n_heads=4,
        n_layers=2,
        context=128,
    )


if __name__ == "__main__":
    main()

"""Cache the primed fixed-point state to disk.

Priming the seed corpus takes ~3 minutes per `compress` / `decompress` call
in fixed mode (one training step per BLOCK_SIZE bytes of the seed, each step
~2s of int64 numpy work). Since the result is deterministic by Rung 2 design
— same inputs produce byte-identical state on every machine, every time —
the prime only needs to happen once per (seed_corpus + architecture + init).
Save it; load it on subsequent runs.

Cache invalidation: the file name embeds a hash of every input that affects
the primed state. Change the seed corpus, model architecture, init seed, or
block size, and the hash flips → a different file path → automatic re-prime.
Stale caches just sit on disk until manually deleted.

Disable with `KOLMO_NO_SEED_CACHE=1`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from kolmo.fixed_optim import FixedAdamState, Q30

# Bump if the cache file format changes incompatibly. Old caches just stop
# matching and get regenerated.
CACHE_FORMAT_VERSION = 1


def cache_dir() -> Path:
    """Where seed state caches live. Override with `KOLMO_CACHE_DIR`."""
    override = os.environ.get("KOLMO_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "kolmo"


def cache_disabled() -> bool:
    return os.environ.get("KOLMO_NO_SEED_CACHE", "").lower() in {"1", "true", "yes"}


def compute_config_hash(
    *,
    seed_corpus: bytes,
    model_config: dict,
    init_seed: int,
    block_size: int,
) -> str:
    """Hash everything that determines the primed state.

    Stable across machines: same bytes in → same bytes out. We can theoretically
    even share cache files between users with matching configs.
    """
    h = hashlib.sha256()
    h.update(f"v{CACHE_FORMAT_VERSION}\n".encode("utf-8"))
    h.update(b"seed_corpus:")
    h.update(seed_corpus)
    h.update(b"\nmodel_config:")
    for key in sorted(model_config):
        h.update(f"  {key}={model_config[key]}\n".encode("utf-8"))
    h.update(f"init_seed:{init_seed}\n".encode("utf-8"))
    h.update(f"block_size:{block_size}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def cache_path_for(config_hash: str) -> Path:
    return cache_dir() / f"seed_state_{config_hash}.npz"


def save_state(
    path: Path,
    weights: dict[str, np.ndarray],
    state: FixedAdamState,
    tied_params: list[tuple[str, str]],
) -> None:
    """Write the primed state to `path` as a compressed .npz.

    Tied aliases are NOT stored — only the canonical weight is saved, and
    aliases are reconstructed on load. This both saves disk space and
    prevents the file from carrying a stale alias (mismatched data after
    deserialization would be a silent bug).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    alias_set = {alias for _, alias in tied_params}

    payload: dict[str, np.ndarray] = {}
    for name, value in weights.items():
        if name in alias_set:
            continue
        payload[f"w/{name}"] = value
    payload["adam/step"] = np.array(state.step, dtype=np.int64)
    payload["adam/beta1_pow_q30"] = np.array(state.beta1_pow_q30, dtype=np.int64)
    payload["adam/beta2_pow_q30"] = np.array(state.beta2_pow_q30, dtype=np.int64)
    for name, value in state.m.items():
        payload[f"m/{name}"] = value
    for name, value in state.v.items():
        payload[f"v/{name}"] = value
    if tied_params:
        payload["tied/canonical"] = np.array(
            [c for c, _ in tied_params], dtype=object
        )
        payload["tied/alias"] = np.array(
            [a for _, a in tied_params], dtype=object
        )

    # Write to a temp file and rename, so a partial write (e.g. machine
    # crash mid-save) never leaves a half-written cache that subsequent
    # runs would happily load and produce wrong output from.
    # numpy's savez auto-appends ".npz" to string paths that don't already
    # end in .npz, which would defeat this rename — pass an open handle.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, path)


def load_state(
    path: Path,
) -> tuple[dict[str, np.ndarray], FixedAdamState, list[tuple[str, str]]]:
    """Load primed state from a previously-saved cache file."""
    data = np.load(path, allow_pickle=True)
    weights: dict[str, np.ndarray] = {}
    m: dict[str, np.ndarray] = {}
    v: dict[str, np.ndarray] = {}
    step = 0
    b1 = Q30
    b2 = Q30
    for key in data.files:
        if key.startswith("w/"):
            weights[key[2:]] = data[key]
        elif key.startswith("m/"):
            m[key[2:]] = data[key]
        elif key.startswith("v/"):
            v[key[2:]] = data[key]
        elif key == "adam/step":
            step = int(data[key])
        elif key == "adam/beta1_pow_q30":
            b1 = int(data[key])
        elif key == "adam/beta2_pow_q30":
            b2 = int(data[key])

    tied_params: list[tuple[str, str]] = []
    if "tied/canonical" in data.files:
        for canonical, alias in zip(
            data["tied/canonical"], data["tied/alias"], strict=True
        ):
            canonical_s = str(canonical)
            alias_s = str(alias)
            tied_params.append((canonical_s, alias_s))
            weights[alias_s] = weights[canonical_s]

    state = FixedAdamState(
        step=step,
        m=m,
        v=v,
        beta1_pow_q30=b1,
        beta2_pow_q30=b2,
    )
    return weights, state, tied_params

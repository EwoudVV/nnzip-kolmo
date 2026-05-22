"""Platform-stable parameter initialization for kolmo.

PyTorch's `manual_seed` is reproducible on one machine, but the exact initial
weights differ across platforms. Rung 2 needs the compressor and decompressor
to begin from identical bytes everywhere, so this module fills a PyTorch model
from a tiny deterministic integer PRNG instead of PyTorch's RNG.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


def _seed_for_name(base_seed: int, name: str) -> np.uint64:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], "little"))


def _splitmix64(seed: np.uint64, size: int) -> np.ndarray:
    """Vectorized SplitMix64 stream."""
    x = np.arange(size, dtype=np.uint64) + seed + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _uniform_float32(name: str, shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Uniform values in [0, 1) using exactly 24 random mantissa bits."""
    bits = _splitmix64(_seed_for_name(seed, name), int(np.prod(shape)))
    mantissa = (bits >> np.uint64(40)).astype(np.uint32)
    values = mantissa.astype(np.float32) * np.float32(1.0 / (1 << 24))
    return values.reshape(shape)


def _uniform_symmetric(
    name: str,
    shape: tuple[int, ...],
    bound: float,
    seed: int,
) -> np.ndarray:
    values = _uniform_float32(name, shape, seed)
    return (values * np.float32(2.0 * bound) - np.float32(bound)).astype(
        np.float32
    )


def stable_init_model(model: torch.nn.Module, seed: int) -> None:
    """Fill model parameters with platform-stable float32 values.

    The distributions mirror PyTorch defaults closely enough for compression:
    embeddings get variance-1 uniform values, Linear weights use
    +/-1/sqrt(fan_in), Linear biases use the same bound, LayerNorm weights are
    ones, and LayerNorm biases are zeros.
    """
    params = dict(model.named_parameters())

    with torch.no_grad():
        for name, param in params.items():
            shape = tuple(param.shape)

            if name.endswith("emb.weight"):
                # Use the Linear init scale (1/sqrt(d_model)) rather than
                # nn.Embedding's default variance-1. The model ties token_emb
                # with the output head, so the same matrix has to serve both
                # as residual-stream embedding AND as final logit projection;
                # a variance-1 init would make logits ~28x larger than a
                # Linear init, breaking quantization error budgets and
                # blowing up early-train softmax confidence. Matches GPT-2.
                d_model = shape[1]
                bound = 1.0 / math.sqrt(d_model)
                arr = _uniform_symmetric(name, shape, bound, seed)
            elif name.endswith(".weight") and param.ndim == 2:
                bound = 1.0 / math.sqrt(shape[1])
                arr = _uniform_symmetric(name, shape, bound, seed)
            elif name.endswith(".weight"):
                arr = np.ones(shape, dtype=np.float32)
            elif name.endswith(".bias") and (
                ".ln" in name or name.startswith("ln_f.")
            ):
                arr = np.zeros(shape, dtype=np.float32)
            elif name.endswith(".bias"):
                weight_name = f"{name[:-5]}.weight"
                fan_in = params[weight_name].shape[1]
                bound = 1.0 / math.sqrt(fan_in)
                arr = _uniform_symmetric(name, shape, bound, seed)
            else:
                raise ValueError(f"unknown parameter kind: {name}")

            param.copy_(torch.from_numpy(arr).to(param.device, dtype=param.dtype))

"""Regression test: after prime, the model should actually predict better.

This caught (in retrospect) a silent bug in the PyTorch-path Adam grad/state
rounding code: it rounded `exp_avg_sq` to zero on a 1/16384 grid, which made
Adam's `m / (sqrt(v) + eps)` blow up by 1e8 per step. Weights exploded,
model never learned, compression dropped from 50% to 181%.

The fast unit tests didn't catch it because they all used KOLMO_SKIP_PRIME=1
to keep test time down. End-to-end round-trip tests passed too — both
compress and decompress walked the same (broken) trajectory, so the bytes
still matched on the other end; the blob was just much bigger than the input.

This test pins the "actually learning" property: after one block of
training on a tiny synthetic corpus, the model's cross-entropy on
held-out text should drop materially below the uniform-byte baseline of
8 bits/byte. If a future change silently breaks training (gradient
explosion, vanishing learning rate, broken backward, etc.) this fails.
"""

import numpy as np
import torch
import torch.nn.functional as F

from kolmo._engine import BLOCK_SIZE, BOS, LR, train_block, update_history
from kolmo.model import KolmoTransformer
from kolmo.stable_init import stable_init_model


def _cross_entropy_bpb(model: KolmoTransformer, history: list[int], target_bytes: bytes) -> float:
    """Average bits-per-byte cross-entropy of `target_bytes` given `history`."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor([history + list(target_bytes)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x, kv_caches=None, pos_offset=0)
    log_probs = F.log_softmax(logits[0], dim=-1)
    targets = torch.tensor(list(target_bytes), dtype=torch.long, device=device)
    # logits[i] predicts byte at position i+1
    target_log_probs = log_probs[
        len(history) - 1 : len(history) - 1 + len(target_bytes)
    ]
    nll = -target_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.mean().item() / float(np.log(2))


def test_short_prime_actually_drops_cross_entropy():
    """Training on a tiny English corpus should drop bpb on held-out text.

    Quantitative threshold: >= 1.0 bpb improvement after 8 training steps.
    Random uniform = 8.0; random untrained model is around 8.1; a working
    primed model on this small workload should hit 6.0-7.0.
    """
    torch.manual_seed(42)
    model = KolmoTransformer()
    stable_init_model(model, 42)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    held_out = b"The library held quiet readers and old books.\n"
    history = [BOS]
    bpb_before = _cross_entropy_bpb(model, history, held_out)

    # Tiny synthetic prime: 8 blocks of BLOCK_SIZE bytes (~128 bytes total).
    # Enough to exercise the training loop without burning test time.
    prime = (
        b"the quick brown fox jumps over the lazy dog. "
        b"the rain in spain falls mainly on the plain. "
        b"a stitch in time saves nine. all that glitters is not gold.\n"
    )
    for pos in range(0, min(len(prime), 8 * BLOCK_SIZE), BLOCK_SIZE):
        block = list(prime[pos : pos + BLOCK_SIZE])
        train_block(model, optimizer, history, block)
        history = update_history(history, block)

    bpb_after = _cross_entropy_bpb(model, history, held_out)

    assert bpb_after < bpb_before - 1.0, (
        f"prime did not train the model: "
        f"bpb {bpb_before:.2f} -> {bpb_after:.2f} "
        f"(expected at least 1.0 bpb drop)"
    )
    assert not any(
        torch.isnan(p).any() or torch.isinf(p).any()
        for p in model.parameters()
    ), "weights went NaN/Inf during training"
    # Catch the specific gradient-explosion failure mode: weights stay
    # within a sane magnitude. Initial weights are bounded by ~1.0; after
    # 8 training steps they shouldn't explode by 3 orders of magnitude.
    max_w = max(p.detach().abs().max().item() for p in model.parameters())
    assert max_w < 100.0, (
        f"weights exploded during training: max |w| = {max_w:.1f} "
        f"(expected < 100; >10000 indicates Adam denominator collapse)"
    )

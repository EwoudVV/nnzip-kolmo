"""Shared logic for compress and decompress.

Both directions need to walk the model through the *same* training trajectory:
build identical weights, predict identical probabilities, take identical
optimizer steps. Anything that affects model state has to live here so the two
sides can't drift.
"""

import numpy as np
import torch
import torch.nn as nn

from kolmo.model import KolmoTransformer

SEED = 42
LR = 1e-3
CONTEXT = 512
BOS = 0  # implicit start-of-stream byte, never written to disk


def new_model_and_optimizer() -> tuple[KolmoTransformer, torch.optim.Optimizer]:
    """Build a model with deterministic init. Both compress and decompress
    must call this and get bit-identical starting weights."""
    torch.manual_seed(SEED)
    model = KolmoTransformer()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, optimizer


def predict(model: KolmoTransformer, context: list[int]) -> tuple[np.ndarray, torch.Tensor]:
    """Forward pass with gradient tracking. Returns (probs as numpy float64,
    logits tensor for backprop).

    Gradient tracking stays on because the same forward pass feeds both the
    encoding (probs) and the training step (logits → loss). One forward per
    byte is the whole compute budget.
    """
    x = torch.tensor([context], dtype=torch.long)
    logits = model(x)  # (1, T, 256)
    last_logits = logits[0, -1]  # (256,)
    probs = torch.softmax(last_logits, dim=-1).detach().numpy().astype(np.float64)
    return probs, last_logits


def train_step(optimizer: torch.optim.Optimizer, last_logits: torch.Tensor, byte: int) -> None:
    """One backward + optimizer step on the actual byte. Both compress and
    decompress call this with the same byte at the same step, so weights stay
    in lockstep."""
    target = torch.tensor([byte], dtype=torch.long)
    loss = nn.functional.cross_entropy(last_logits.unsqueeze(0), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def slide(context: list[int], byte: int) -> list[int]:
    """Append byte and trim to max context length."""
    context = context + [byte]
    if len(context) > CONTEXT:
        context = context[-CONTEXT:]
    return context

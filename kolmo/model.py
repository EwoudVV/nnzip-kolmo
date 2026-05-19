"""Tiny decoder-only transformer used as the probability model for compression.

Default config produces a ~3M-parameter model with byte-level vocabulary. This
is intentionally small: at Rung 1 we just want to prove the online-training
architecture round-trips correctly. Bigger models come at Rung 3 once the
plumbing is solid.
"""

import torch
import torch.nn as nn


class KolmoTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_context: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_context = max_context

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_context, d_model)

        # Pre-norm (norm_first=True): more stable than post-norm when training
        # from scratch with online updates. The Hutter contenders all use
        # variants of pre-norm for the same reason.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) of byte ints in [0, 256). Returns logits (B, T, vocab)."""
        B, T = x.shape
        if T > self.max_context:
            raise ValueError(
                f"context length {T} exceeds max_context {self.max_context}"
            )

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(positions)

        # Causal mask: position i can only attend to positions 0..i.
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

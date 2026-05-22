"""Tiny decoder-only transformer with KV cache support.

The previous implementation recomputed attention over the full context for
every byte, even though the model weights are frozen between training steps.
This version supports incremental forward — feed one new token, attend against
cached K/V from previous tokens, output one prediction. Between training
steps, that drops the per-byte cost from O(T²) down to O(T) where T is the
context length.

The cache is invalidated at every training step because the model weights
change, so cached K/V (which were computed from the old weights) become
stale. Both compressor and decompressor rebuild the cache from scratch after
each gradient step, in lockstep, so they stay in sync.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(t_new: int, t_total: int, device: torch.device) -> torch.Tensor:
    """Mask of shape (t_new, t_total). True = positions to MASK OUT.

    The last t_new positions in the input are the new queries; the first
    (t_total - t_new) positions are already-cached keys/values. A new query
    at index i (within the new range) can attend to all cached positions plus
    positions 0..i within the new range.
    """
    t_past = t_total - t_new
    rows = torch.arange(t_new, device=device).unsqueeze(1)
    cols = torch.arange(t_total, device=device).unsqueeze(0)
    return cols > (t_past + rows)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        B, T_new, D = x.shape
        qkv = self.qkv(x).view(B, T_new, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T_new, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        # Detach when storing to break the graph — backward must not flow
        # through old forwards, only through the current one.
        new_cache = {"k": k.detach(), "v": v.detach()}

        T_total = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T_new, T_total)
        mask = _causal_mask(T_new, T_total, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T_new, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T_new, D)
        return self.proj(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        h, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class KolmoTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_context: int = 16384,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_context = max_context
        self.tie_weights = tie_weights

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_context, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            # token_emb maps id -> d_model row; head maps d_model -> logit over
            # ids. Both use a (vocab_size, d_model) matrix. Sharing one matrix
            # means a gradient flowing back from the head also improves the
            # embedding (and vice versa), and cuts ~65K parameters from a
            # ~2M-param model. Standard trick from modern LMs.
            self.head.weight = self.token_emb.weight

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: list | None = None,
        pos_offset: int = 0,
    ):
        """x: (B, T_new) of byte ints. Returns (logits, new_kv_caches).

        kv_caches: optional list of per-layer KV caches.
        pos_offset: absolute starting position for the new tokens' position
            embeddings. When using KV cache, this should equal the number of
            tokens already cached (so positions stay monotonic).
        """
        B, T_new = x.shape
        if pos_offset + T_new > self.max_context:
            raise ValueError(
                f"position {pos_offset + T_new} exceeds max_context {self.max_context}"
            )
        positions = torch.arange(pos_offset, pos_offset + T_new, device=x.device)
        h = self.token_emb(x) + self.pos_emb(positions).unsqueeze(0)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(h, cache)
            new_caches.append(new_cache)

        h = self.ln_f(h)
        return self.head(h), new_caches

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

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


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE, Su et al. 2021).

    Encodes position by rotating each (q[2i], q[2i+1]) pair by an angle
    proportional to the token's absolute position. Two tokens at relative
    distance r see each other through cos/sin of (r * theta_i) — the
    geometry encodes relative position, not absolute.

    Two wins over absolute pos_emb:
    1. Zero parameters (precomputed cos/sin buffers). Frees the 131K of
       the (max_context, d_model) embedding table for actual learning.
    2. Generalizes to context lengths longer than trained on — useful
       when we eventually run on enwik-scale files with longer history.

    Standard in modern LLMs (LLaMA, Mistral, Qwen, GPT-NeoX).
    """

    def __init__(self, d_head: int, max_context: int, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires even d_head"
        # theta_i = base^(-2i/d_head) for i in 0..d_head/2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        positions = torch.arange(max_context, dtype=torch.float32)
        # (max_context, d_head/2)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def apply(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Rotate `x` by the angles for the given absolute positions.

        x: (..., T, d_head). position_ids: (T,) absolute positions.
        Returns: same shape as x.
        """
        cos = self.cos[position_ids].to(x.dtype)  # (T, d_head/2)
        sin = self.sin[position_ids].to(x.dtype)
        # Broadcast over batch and head dims (everything left of T).
        for _ in range(x.dim() - 2):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x1 = x[..., 0::2]  # (..., T, d_head/2)
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.empty_like(x)
        y[..., 0::2] = y1
        y[..., 1::2] = y2
        return y


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Shared across layers (one rope module owns the cos/sin tables).
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        B, T_new, D = x.shape
        qkv = self.qkv(x).view(B, T_new, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T_new, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            assert position_ids is not None, "RoPE requires position_ids"
            q = self.rope.apply(q, position_ids)
            k = self.rope.apply(k, position_ids)
            # Cached K is already rotated for its original positions, so
            # concatenating works naturally — each position's K carries the
            # rotation for its own absolute position.

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        new_cache = {"k": k.detach(), "v": v.detach()}

        T_total = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T_new, T_total)
        mask = _causal_mask(T_new, T_total, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T_new, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T_new, D)
        return self.proj(out), new_cache


class GeGLUFFN(nn.Module):
    """GeGLU feed-forward block.

    Replaces the standard `Linear -> GELU -> Linear` FFN with a gated
    variant: `Linear -> (GELU(gate) * up) -> Linear`. Modern best practice
    (LLaMA, PaLM, etc.) — typically gives 1-3% better ratio on text at
    small model sizes for roughly the same parameter count and compute.

    The intermediate dim is `8 * d_model / 3` (rounded to a multiple of
    32) so that the three linear projections together have ~the same
    param count as the original two projections at 4 * d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        d_ff = ((d_model * 8 + 32 * 3 - 1) // (3 * 32)) * 32
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_type: str = "gelu",
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope=rope)
        self.ln2 = nn.LayerNorm(d_model)
        if ffn_type == "geglu":
            self.ffn = GeGLUFFN(d_model)
        elif ffn_type == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        else:
            raise ValueError(f"unknown ffn_type: {ffn_type!r}")
        self.ffn_type = ffn_type

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        h, new_cache = self.attn(self.ln1(x), kv_cache, position_ids=position_ids)
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
        max_context: int = 512,
        # CONTEXT=256 (sliding-window cap) + BLOCK_SIZE=16 (max new tokens
        # per step before training resets pos_offset to 0) means the highest
        # absolute position ever indexed is ~272. 512 is 2x headroom. The
        # old default of 16384 made pos_emb a 4.2M-param tensor where 99%
        # of rows were dead weight — Adam still spent 30% of its time
        # updating them every step.
        tie_weights: bool = True,
        ffn_type: str = "gelu",
        use_rope: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_context = max_context
        self.tie_weights = tie_weights
        self.ffn_type = ffn_type
        self.use_rope = use_rope

        self.token_emb = nn.Embedding(vocab_size, d_model)
        if use_rope:
            # No learned position embedding — RoPE encodes position via
            # rotation inside attention. Saves a (max_context, d_model)
            # parameter table.
            self.pos_emb = None
            d_head = d_model // n_heads
            self.rope = RotaryPositionalEmbedding(d_head, max_context)
        else:
            self.pos_emb = nn.Embedding(max_context, d_model)
            self.rope = None
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_type=ffn_type, rope=self.rope)
             for _ in range(n_layers)]
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
        if self.use_rope:
            # No additive position embedding — RoPE rotates Q/K inside attention.
            h = self.token_emb(x)
        else:
            h = self.token_emb(x) + self.pos_emb(positions).unsqueeze(0)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(h, cache, position_ids=positions if self.use_rope else None)
            new_caches.append(new_cache)

        h = self.ln_f(h)
        return self.head(h), new_caches

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

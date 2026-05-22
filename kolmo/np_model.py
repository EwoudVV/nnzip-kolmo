"""Pure-NumPy forward pass for the kolmo transformer.

This is the Rung 2 stepping stone. PyTorch's float operations diverge between
CPU architectures (M1 vs x86), which breaks cross-machine round-trip. NumPy
with controlled threading is at least *closer* to deterministic, and gives us
a path to fixed-point math later.

Same architecture as `kolmo.model.KolmoTransformer`:
  - byte-level token embedding
  - learned position embedding
  - N pre-norm transformer blocks (LayerNorm → attention → LayerNorm → FFN)
  - final LayerNorm
  - linear output head

The forward function takes a dict of weights extracted from the PyTorch model
and the input bytes; returns logits with the same shape as PyTorch's output.
Validate by comparing against the PyTorch model on the same weights.
"""

import math

import numpy as np


def _layernorm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """LayerNorm over the last axis. Matches torch.nn.LayerNorm semantics:
    mean and biased variance, then scale + shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU (matches torch.nn.GELU default, not the tanh approximation)."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """erf via np.vectorize on math.erf — slow but reproducible across machines."""
    return np.vectorize(math.erf)(x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax: subtract max before exp."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _causal_mask(t_new: int, t_total: int) -> np.ndarray:
    """True = mask out. New query at row i can attend to columns 0..(t_past + i)."""
    t_past = t_total - t_new
    rows = np.arange(t_new)[:, None]
    cols = np.arange(t_total)[None, :]
    return cols > (t_past + rows)


def _attention(
    x: np.ndarray,
    qkv_w: np.ndarray,
    proj_w: np.ndarray,
    n_heads: int,
) -> np.ndarray:
    """Causal self-attention without bias.

    Matches our model's `bias=False`. x is (T, d_model), return is
    (T, d_model).
    """
    T, D = x.shape
    d_head = D // n_heads

    qkv = x @ qkv_w.T  # (T, 3D)
    qkv = qkv.reshape(T, 3, n_heads, d_head)
    q = qkv[:, 0].transpose(1, 0, 2)  # (H, T, d_head)
    k = qkv[:, 1].transpose(1, 0, 2)
    v = qkv[:, 2].transpose(1, 0, 2)

    scale = 1.0 / math.sqrt(d_head)
    scores = q @ k.transpose(0, 2, 1) * scale  # (H, T, T)
    mask = _causal_mask(T, T)
    scores = np.where(mask, -np.inf, scores)
    attn = _softmax(scores, axis=-1)

    out = attn @ v  # (H, T, d_head)
    out = out.transpose(1, 0, 2).reshape(T, D)
    return out @ proj_w.T


def _block(
    x: np.ndarray,
    weights: dict,
    n_heads: int,
) -> np.ndarray:
    """One pre-norm transformer block."""
    # Attention sublayer
    h = _layernorm(x, weights["ln1.weight"], weights["ln1.bias"])
    h = _attention(h, weights["attn.qkv.weight"], weights["attn.proj.weight"], n_heads)
    x = x + h

    # FFN sublayer
    h = _layernorm(x, weights["ln2.weight"], weights["ln2.bias"])
    h = h @ weights["ffn.0.weight"].T + weights["ffn.0.bias"]
    h = _gelu(h)
    h = h @ weights["ffn.2.weight"].T + weights["ffn.2.bias"]
    return x + h


def kolmo_forward(
    input_ids: np.ndarray,
    weights: dict,
    n_heads: int = 8,
    n_layers: int = 4,
    pos_offset: int = 0,
) -> np.ndarray:
    """Pure-NumPy forward pass.

    input_ids is (T,) of byte ints. Returns (T, vocab_size).
    """
    T = len(input_ids)
    positions = np.arange(pos_offset, pos_offset + T)
    h = weights["token_emb.weight"][input_ids] + weights["pos_emb.weight"][positions]

    for i in range(n_layers):
        prefix = f"blocks.{i}."
        block_w = {
            k[len(prefix):]: v
            for k, v in weights.items()
            if k.startswith(prefix)
        }
        h = _block(h, block_w, n_heads)

    h = _layernorm(h, weights["ln_f.weight"], weights["ln_f.bias"])
    return h @ weights["head.weight"].T


def extract_weights(model) -> dict:
    """Pull weights from a PyTorch KolmoTransformer into a name->ndarray dict.

    Handles weight-tied parameters: `named_parameters()` deduplicates by
    Parameter id, so a tied pair (e.g. token_emb.weight === head.weight)
    only yields one name. We then re-add every aliased name pointing at the
    same underlying array.
    """
    seen_ids: dict[int, str] = {}
    weights: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        seen_ids[id(param)] = name
        weights[name] = param.detach().cpu().numpy().astype(np.float32)
    for name, param in model.named_parameters(remove_duplicate=False):
        canonical = seen_ids.get(id(param))
        if canonical is not None and name not in weights:
            weights[name] = weights[canonical]
    return weights

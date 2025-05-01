"""A tiny decoder-only transformer LM, and the sequence helpers built on it.

Everything here is deliberately small: a couple of hundred thousand parameters,
learned positional embeddings, pre-norm blocks, and manual attention with an
additive causal mask.  No KV cache, because the toy experiment regenerates each
sequence from scratch every step and clarity is worth more than speed at this
size.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=8)
def causal_mask(length: int) -> mx.array:
    """Additive mask, 0 on and below the diagonal and -inf above it."""
    keep = mx.tril(mx.ones((length, length), dtype=mx.float32))
    return (1.0 - keep) * mx.array(-1e9, dtype=mx.float32)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, ff_mult * dim, bias=False)
        self.down = nn.Linear(ff_mult * dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(0, 3, 1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim) + causal_mask(T)
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(out)
        h = self.norm2(x)
        return x + self.down(nn.gelu(self.up(h)))


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 15,
        dim: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        ff_mult: int = 4,
        max_len: int = 24,
    ):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = [Block(dim, n_heads, ff_mult) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        """(B, T) int token ids -> (B, T, vocab) logits."""
        B, T = tokens.shape
        if T > self.max_len:
            raise ValueError(f"sequence of {T} exceeds max_len={self.max_len}")
        x = self.token_emb(tokens) + self.pos_emb(mx.arange(T))[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def token_logprobs(model: TinyLM, tokens: mx.array) -> mx.array:
    """log p(tokens[:, t+1] | tokens[:, :t+1]) for t = 0 .. T-2, shape (B, T-1)."""
    logits = model(tokens)[:, :-1]
    targets = tokens[:, 1:]
    logp = nn.log_softmax(logits, axis=-1)
    return mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]


def completion_logprobs(
    model: TinyLM,
    prompt_ids: mx.array,
    prompt_mask: mx.array,
    completion_ids: mx.array,
) -> mx.array:
    """Log-probability of each completion token under ``model``.

    ``prompt_ids`` is (B, P), ``completion_ids`` is (B, C).  The first
    completion token is predicted from the last prompt token, so the returned
    array is (B, C) and aligned with ``completion_ids``.
    """
    tokens = mx.concatenate([prompt_ids, completion_ids], axis=1)
    logp = token_logprobs(model, tokens)
    return logp[:, prompt_ids.shape[1] - 1 :]


def completion_cross_entropy(
    model: TinyLM,
    prompt_ids: mx.array,
    prompt_mask: mx.array,
    completion_ids: mx.array,
    completion_mask: mx.array,
) -> mx.array:
    """Masked mean negative log-likelihood over completion tokens."""
    logp = completion_logprobs(model, prompt_ids, prompt_mask, completion_ids)
    total = mx.sum(logp * completion_mask)
    return -total / mx.maximum(mx.sum(completion_mask), 1.0)


def param_count(model: TinyLM) -> int:
    from mlx.utils import tree_flatten

    return int(sum(v.size for _, v in tree_flatten(model.parameters())))

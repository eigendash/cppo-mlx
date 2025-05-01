"""Sampling completions from the tiny policy, and greedy evaluation."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .data import EOS_ID, Example, pad_prompts
from .model import TinyLM
from .reward import accuracy, reward


def mask_after_eos(completion_ids: mx.array, eos_id: int = EOS_ID) -> mx.array:
    """(B, C) mask that is 1 up to and including the first <eos> in each row.

    Tokens a policy emits after <eos> are not part of the completion, so they
    neither earn reward nor contribute to the loss.
    """
    ids = np.asarray(completion_ids)
    is_eos = ids == eos_id
    seen = np.cumsum(is_eos, axis=1)
    return mx.array(((seen - is_eos) == 0).astype(np.float32))


def repeat_prompts(prompt_ids: mx.array, prompt_mask: mx.array, n_samples: int):
    """(B, P) -> (B*S, P) with each prompt repeated ``n_samples`` times."""
    ids = mx.repeat(prompt_ids, n_samples, axis=0)
    mask = mx.repeat(prompt_mask, n_samples, axis=0)
    return ids, mask


def sample_completions(
    model: TinyLM,
    prompt_ids: mx.array,
    prompt_mask: mx.array,
    n_samples: int = 1,
    max_new_tokens: int = 4,
    temperature: float = 1.0,
    key: mx.array | None = None,
):
    """Autoregressively sample ``n_samples`` completions per prompt.

    Returns ``(completion_ids, completion_mask)``, both of shape
    ``(B, n_samples, max_new_tokens)``.  Prompts are repeated along a flat batch
    so the whole group is generated with one forward pass per token.
    """
    if key is None:
        key = mx.random.key(0)
    B = prompt_ids.shape[0]
    ids, _ = repeat_prompts(prompt_ids, prompt_mask, n_samples)
    generated: list[mx.array] = []
    for _ in range(max_new_tokens):
        key, subkey = mx.random.split(key)
        cur = mx.concatenate([ids] + [g[:, None] for g in generated], axis=1)
        logits = model(cur)[:, -1] / temperature
        nxt = mx.random.categorical(logits, axis=-1, key=subkey)
        mx.eval(nxt)
        generated.append(nxt)
    completion = mx.stack(generated, axis=1).reshape(B, n_samples, max_new_tokens)
    return completion, mask_after_eos(completion)


def greedy_completions(
    model: TinyLM,
    prompt_ids: mx.array,
    prompt_mask: mx.array,
    max_new_tokens: int = 5,
    eos_id: int = EOS_ID,
) -> list[list[int]]:
    """Greedy decode one completion per prompt, stopping at <eos>."""
    ids = prompt_ids
    finished = [False] * ids.shape[0]
    out: list[list[int]] = [[] for _ in range(ids.shape[0])]
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1]
        nxt = np.asarray(mx.argmax(logits, axis=-1))
        for i, t in enumerate(nxt):
            if finished[i]:
                continue
            out[i].append(int(t))
            if int(t) == eos_id:
                finished[i] = True
        ids = mx.concatenate([ids, mx.array(nxt)[:, None]], axis=1)
    return out


def greedy_accuracy(model: TinyLM, examples: list[Example], max_new_tokens: int = 5) -> float:
    prompt, mask = pad_prompts(examples)
    completions = greedy_completions(model, prompt, mask, max_new_tokens=max_new_tokens)
    hits = sum(accuracy(c, ex) for c, ex in zip(completions, examples))
    return hits / max(len(examples), 1)


def score_completions(completion_ids, examples) -> np.ndarray:
    """Reward matrix of shape (B, G) for a (B, G, C) batch of completions."""
    arr = np.asarray(completion_ids)
    B, G, _ = arr.shape
    rewards = np.zeros((B, G), dtype=np.float32)
    for b, ex in enumerate(examples):
        for g in range(G):
            rewards[b, g] = reward(arr[b, g], ex)
    return rewards

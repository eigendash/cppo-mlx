"""The GRPO objective and the CPPO pruning layer.

GRPO, for a question ``q`` with a group of ``G`` completions sampled from the
old policy, is

    A_i = (r_i - mean(r)) / std(r)                                    (paper eq. 3)
    J   = E[ 1/G sum_i 1/|o_i| sum_t min(rho_t A_i, clip(rho_t) A_i)
             - beta * D_KL ]                                          (paper eq. 1)

with ``rho_t = pi_theta(o_t) / pi_theta_old(o_t)`` and the k3 KL estimator

    D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1                  (paper eq. 2)

which is non-negative and exactly zero when the policy equals the reference.

CPPO adds a retention rule on top: only completions with |A_i| >= gamma
(paper eq. 7) or the top-k by |A_i| with k = floor(G (1 - P)) (paper eq. 8-10)
take part in the loss.  The sum in the paper's eq. 9 is divided by k rather than
by G, which is what keeps the retained update at the scale of a full group; we
call that ``normalisation="retained"`` and it is the default.  With
``normalisation="group"`` the divisor stays G, which is the literal reading of
eq. 7 and makes the update shrink with the pruning rate.

The point that matters for tests: with ``prune_threshold=0`` and
``prune_rate=0`` every completion is retained, k equals G, and ``cppo_loss``
computes exactly the same number as ``grpo_loss``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

DEFAULT_CLIP = 0.2
DEFAULT_BETA = 0.04


def group_advantages(rewards: mx.array, eps: float = 1e-4) -> mx.array:
    """Normalise rewards within each group, shape (B, G) -> (B, G).

    Uses the population standard deviation, and a constant group gives exactly
    zero advantage (the eps only guards the division).
    """
    rewards = mx.array(rewards, dtype=mx.float32)
    if rewards.ndim == 1:
        rewards = rewards[None]
    mean = mx.mean(rewards, axis=-1, keepdims=True)
    std = mx.std(rewards, axis=-1, keepdims=True)
    return (rewards - mean) / (std + eps)


def kl_k3(logp: mx.array, logp_ref: mx.array) -> mx.array:
    """Schulman's k3 estimator, elementwise, >= 0 and 0 when the two agree."""
    log_ratio = logp_ref - logp
    return mx.exp(log_ratio) - log_ratio - 1.0


def prune_mask(
    advantages: mx.array,
    prune_threshold: float = 0.0,
    prune_rate: float = 0.0,
) -> mx.array:
    """Float mask (B, G), 1 for retained completions.

    ``prune_threshold`` drops completions with |A| < threshold (paper eq. 7);
    it may empty a group, which then contributes nothing.

    ``prune_rate`` P keeps the k = max(1, floor(G (1 - P))) largest |A| values
    per group (paper eq. 8-10).  Ties are broken by the lower index, so the
    result is deterministic.  When both are given the threshold is applied first
    and the rate then caps the survivors.
    """
    advantages = mx.array(advantages, dtype=mx.float32)
    if advantages.ndim != 2:
        raise ValueError("advantages must be (B, G)")
    if not 0.0 <= prune_rate < 1.0:
        raise ValueError("prune_rate must be in [0, 1)")
    if prune_threshold < 0:
        raise ValueError("prune_threshold must be >= 0")
    abs_adv = np.abs(np.asarray(advantages))
    keep = np.ones(abs_adv.shape, dtype=bool)
    if prune_threshold > 0.0:
        keep &= abs_adv >= prune_threshold
    if prune_rate > 0.0:
        G = abs_adv.shape[1]
        k = max(1, int(math.floor(G * (1.0 - prune_rate))))
        order = np.argsort(-abs_adv, axis=1, kind="stable")[:, :k]
        top = np.zeros_like(keep)
        np.put_along_axis(top, order, True, axis=1)
        keep &= top
    return mx.array(keep.astype(np.float32))


def _core_loss(
    logp: mx.array,
    logp_old: mx.array,
    logp_ref: mx.array,
    advantages: mx.array,
    token_mask: mx.array,
    completion_weight: mx.array,
    beta: float,
    clip_eps: float,
    denominator: mx.array | None = None,
):
    ratio = mx.exp(logp - logp_old)
    adv = advantages[..., None]
    surrogate = mx.minimum(ratio * adv, mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)
    kl = kl_k3(logp, logp_ref)
    per_token = (surrogate - beta * kl) * token_mask
    lengths = mx.sum(token_mask, axis=-1)
    per_completion = mx.sum(per_token, axis=-1) / mx.maximum(lengths, 1.0)
    weight = completion_weight * (lengths > 0)
    total_weight = mx.sum(weight)
    if denominator is None:
        denominator = total_weight
    objective = mx.sum(per_completion * weight) / mx.maximum(denominator, 1.0)
    n_sampled = float(advantages.shape[0] * advantages.shape[1])
    info = {
        "loss": -objective,
        "objective": objective,
        "kl": mx.sum(kl * token_mask) / mx.maximum(mx.sum(token_mask), 1.0),
        "ratio_mean": mx.sum(ratio * token_mask) / mx.maximum(mx.sum(token_mask), 1.0),
        "n_retained": total_weight,
        "retained_fraction": mx.sum(completion_weight) / n_sampled,
    }
    return -objective, info


def grpo_loss(
    logp: mx.array,
    logp_old: mx.array,
    logp_ref: mx.array,
    advantages: mx.array,
    token_mask: mx.array,
    beta: float = DEFAULT_BETA,
    clip_eps: float = DEFAULT_CLIP,
):
    """Full-group GRPO loss (lower is better) and a small info dict."""
    advantages = mx.array(advantages, dtype=mx.float32)
    ones = mx.ones(advantages.shape, dtype=mx.float32)
    return _core_loss(logp, logp_old, logp_ref, advantages, token_mask, ones, beta, clip_eps)


def cppo_loss(
    logp: mx.array,
    logp_old: mx.array,
    logp_ref: mx.array,
    advantages: mx.array,
    token_mask: mx.array,
    prune_threshold: float = 0.0,
    prune_rate: float = 0.0,
    beta: float = DEFAULT_BETA,
    clip_eps: float = DEFAULT_CLIP,
    normalisation: str = "retained",
):
    """CPPO loss: the GRPO surrogate restricted to the retained completions.

    ``normalisation="retained"`` divides by the number of retained completions
    (paper eq. 9, where the divisor is k).  ``normalisation="group"`` divides by
    the size of the whole group (paper eq. 7, divisor G).
    """
    if normalisation not in ("retained", "group"):
        raise ValueError("normalisation must be 'retained' or 'group'")
    advantages = mx.array(advantages, dtype=mx.float32)
    mask = prune_mask(advantages, prune_threshold, prune_rate)
    denominator = None
    if normalisation == "group":
        # The literal eq. 7: the divisor stays the group size even after pruning.
        denominator = mx.array(float(advantages.shape[0] * advantages.shape[1]))
    return _core_loss(
        logp, logp_old, logp_ref, advantages, token_mask, mask, beta, clip_eps, denominator
    )


def per_group_losses(
    logp: mx.array,
    logp_old: mx.array,
    logp_ref: mx.array,
    advantages: mx.array,
    token_mask: mx.array,
    completion_weight: mx.array,
    beta: float = DEFAULT_BETA,
    clip_eps: float = DEFAULT_CLIP,
) -> mx.array:
    """Per-completion loss contributions (B, G), for gradient-noise statistics."""
    ratio = mx.exp(logp - logp_old)
    adv = advantages[..., None]
    surrogate = mx.minimum(ratio * adv, mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)
    kl = kl_k3(logp, logp_ref)
    per_token = (surrogate - beta * kl) * token_mask
    lengths = mx.sum(token_mask, axis=-1)
    per_completion = mx.sum(per_token, axis=-1) / mx.maximum(lengths, 1.0)
    return -completion_weight * per_completion

"""The GRPO objective: group-normalised advantages, clipped surrogate, KL.

GRPO, for a question ``q`` with a group of ``G`` completions sampled from the
old policy, is

    A_i = (r_i - mean(r)) / std(r)                                    (paper eq. 3)
    J   = E[ 1/G sum_i 1/|o_i| sum_t min(rho_t A_i, clip(rho_t) A_i)
             - beta * D_KL ]                                          (paper eq. 1)

with ``rho_t = pi_theta(o_t) / pi_theta_old(o_t)`` and the k3 KL estimator

    D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1                  (paper eq. 2)

which is non-negative and exactly zero when the policy equals the reference.
"""

from __future__ import annotations

import mlx.core as mx

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

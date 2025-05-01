import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from cppo.data import EOS_ID, digit_id, make_example, pad_prompts
from cppo.model import TinyLM, completion_logprobs
from cppo.objectives import grpo_loss, group_advantages, kl_k3
from cppo.sample import sample_completions, score_completions


def completion(text: str) -> list[int]:
    return [digit_id(int(c)) for c in text] + [EOS_ID]


def random_inputs(B=2, G=4, T=3, seed=0):
    mx.random.seed(seed)
    logp = mx.random.normal((B, G, T)) * 0.1
    logp_old = logp - mx.random.normal((B, G, T)) * 0.05
    logp_ref = logp + mx.random.normal((B, G, T)) * 0.05
    advantages = mx.random.normal((B, G))
    token_mask = mx.ones((B, G, T), dtype=mx.float32)
    return logp, logp_old, logp_ref, advantages, token_mask


def budget_rewards():
    """Three completions for 47+58 with hand-computed rewards."""
    ex = make_example(47, 58)
    comps = [completion("501"), completion("105"), completion("999")]
    ids = mx.array([comps])  # (1, 3, C)
    rewards = score_completions(ids, [ex]).reshape(-1)
    return rewards


def test_rewards_and_advantages_of_a_three_completion_group():
    rewards = np.asarray(budget_rewards())
    np.testing.assert_allclose(rewards, [3.0, 2.5, 1.0])  # direct, parsed, wrong
    adv = np.asarray(group_advantages(mx.array(rewards)))
    mean, std = rewards.mean(), rewards.std()
    np.testing.assert_allclose(adv, ((rewards - mean) / (std + 1e-4))[None], rtol=1e-6)
    np.testing.assert_allclose(adv.sum(axis=-1), 0.0, atol=1e-6)
    assert adv[0, 0] > adv[0, 1] > 0 > adv[0, 2]


def test_advantages_match_hand_computed_values():
    rewards = mx.array([[3.0, 1.0, 1.0]])
    adv = np.asarray(group_advantages(rewards))
    mean = 5.0 / 3.0
    std = np.sqrt(((3.0 - mean) ** 2 + 2 * (1.0 - mean) ** 2) / 3.0)
    np.testing.assert_allclose(adv, [[(3.0 - mean) / (std + 1e-4), (1.0 - mean) / (std + 1e-4), (1.0 - mean) / (std + 1e-4)]], rtol=1e-5)


def test_constant_group_has_zero_advantage():
    adv = np.asarray(group_advantages(mx.array([[2.0, 2.0, 2.0, 2.0]])))
    np.testing.assert_allclose(adv, 0.0, atol=1e-6)


def test_kl_estimator_is_zero_on_identical_models_and_positive_otherwise():
    logp = mx.array([[0.1, -0.2, 0.3]])
    assert float(mx.max(mx.abs(kl_k3(logp, logp)))) == pytest.approx(0.0, abs=1e-6)
    other = logp + 0.5
    kl = np.asarray(kl_k3(logp, other))
    assert np.all(kl > 0)
    np.testing.assert_allclose(kl, np.exp(0.5) - 0.5 - 1.0, rtol=1e-5)


def test_kl_penalty_scales_with_beta_and_vanishes_when_policy_is_reference():
    logp, logp_old, logp_ref, advantages, token_mask = random_inputs(seed=5)
    zero, info0 = grpo_loss(logp, logp, logp, advantages, token_mask)
    penalised, info1 = grpo_loss(logp, logp_old, logp_ref, advantages, token_mask, beta=0.04)
    unpenalised, _ = grpo_loss(logp, logp_old, logp_ref, advantages, token_mask, beta=0.0)
    assert float(info0["kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(info1["kl"]) > 0
    np.testing.assert_allclose(
        float(penalised) - float(unpenalised), 0.04 * float(info1["kl"]), rtol=1e-3
    )
    assert float(zero) != float(penalised)


def test_clip_bounds_the_positive_direction_only():
    # ratio = 5, A = 2, eps = 0.2 -> per-token surrogate = min(10, 2.4) = 2.4
    logp = mx.array([[[np.log(5.0)]]])
    logp_old = mx.zeros_like(logp)
    adv = mx.array([[2.0]])
    mask = mx.ones_like(logp)
    loss, _ = grpo_loss(logp, logp_old, logp, adv, mask, beta=0.0)
    np.testing.assert_allclose(float(loss), -2.4, rtol=1e-5)
    # A = -2: min(-10, -2.4) = -10, the pessimistic direction is not clipped.
    loss_neg, _ = grpo_loss(logp, logp_old, logp, mx.array([[-2.0]]), mask, beta=0.0)
    np.testing.assert_allclose(float(loss_neg), 10.0, rtol=1e-5)


def test_gradients_are_finite():
    mx.random.seed(11)
    model = TinyLM(vocab_size=15, dim=32, n_layers=1, n_heads=4, max_len=16)
    examples = [make_example(47, 58), make_example(12, 34)]
    prompt, pmask = pad_prompts(examples)
    comp, cmask = sample_completions(model, prompt, pmask, n_samples=3, max_new_tokens=4, key=mx.random.key(1))
    B, G, C = comp.shape
    logp = completion_logprobs(
        model,
        mx.repeat(prompt, G, axis=0),
        mx.repeat(pmask, G, axis=0),
        comp.reshape(B * G, C),
    ).reshape(B, G, C)
    advantages = group_advantages(mx.array(score_completions(comp, examples)))

    def loss_of(m):
        lp = completion_logprobs(
            m, mx.repeat(prompt, G, axis=0), mx.repeat(pmask, G, axis=0), comp.reshape(B * G, C)
        ).reshape(B, G, C)
        return grpo_loss(lp, logp, logp, advantages, cmask)[0]

    grads = mx.grad(loss_of)(model)
    leaves = [np.asarray(v) for _, v in tree_flatten(grads)]
    assert leaves
    assert all(np.all(np.isfinite(v)) for v in leaves)
    assert any(np.any(v != 0) for v in leaves)


def test_loss_is_negative_mean_advantage_when_policy_equals_old_policy():
    logp, logp_old, logp_ref, advantages, token_mask = random_inputs()
    loss, info = grpo_loss(logp, logp, logp, advantages, token_mask, beta=0.0)
    np.testing.assert_allclose(float(loss), -float(mx.mean(advantages)), rtol=1e-5)
    assert float(info["kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(info["ratio_mean"]) == pytest.approx(1.0, abs=1e-6)


def test_token_mask_excludes_padding_from_the_per_completion_mean():
    logp = mx.array([[[1.0, 1.0], [1.0, 1.0]]])
    token_mask = mx.array([[[1.0, 0.0], [1.0, 1.0]]])
    adv = mx.array([[2.0, -2.0]])
    loss, _ = grpo_loss(logp, logp, logp, adv, token_mask, beta=0.0)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)

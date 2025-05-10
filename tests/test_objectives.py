import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppo.data import EOS_ID, PAD_ID, digit_id, make_example, pad_prompts
from cppo.model import TinyLM, completion_logprobs
from cppo.objectives import (
    cppo_loss,
    grpo_loss,
    group_advantages,
    kl_k3,
    per_group_losses,
    prune_mask,
)
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


def test_threshold_zero_reduces_to_grpo_exactly():
    logp, logp_old, logp_ref, advantages, token_mask = random_inputs()
    ref, _ = grpo_loss(logp, logp_old, logp_ref, advantages, token_mask)
    for kwargs in (
        dict(prune_threshold=0.0),
        dict(prune_rate=0.0),
        dict(prune_threshold=0.0, prune_rate=0.0),
        dict(prune_threshold=0.0, prune_rate=0.0, normalisation="group"),
    ):
        got, _ = cppo_loss(logp, logp_old, logp_ref, advantages, token_mask, **kwargs)
        assert np.array_equal(np.asarray(ref), np.asarray(got)), kwargs


def test_threshold_zero_reduces_to_grpo_on_a_real_rollout():
    mx.random.seed(3)
    model = TinyLM(vocab_size=15, dim=32, n_layers=1, n_heads=4, max_len=16)
    examples = [make_example(47, 58), make_example(12, 34)]
    prompt, mask = pad_prompts(examples)
    comp, cmask = sample_completions(model, prompt, mask, n_samples=3, max_new_tokens=4, key=mx.random.key(0))
    B, G, C = comp.shape
    flat_prompt = mx.repeat(prompt, G, axis=0)
    flat_mask = mx.repeat(mask, G, axis=0)
    logp = completion_logprobs(model, flat_prompt, flat_mask, comp.reshape(B * G, C)).reshape(B, G, C)
    rewards = score_completions(comp, examples)
    advantages = group_advantages(mx.array(rewards))
    ref, info_ref = grpo_loss(logp, logp, logp, advantages, cmask)
    got, info = cppo_loss(logp, logp, logp, advantages, cmask, prune_threshold=0.0)
    assert np.array_equal(np.asarray(ref), np.asarray(got))
    assert float(info["retained_fraction"]) == pytest.approx(1.0)
    # Ratio is exactly 1 and the policy is the reference, so the loss is the
    # negative mean advantage.
    assert abs(float(ref) - (-float(mx.mean(advantages)))) < 1e-6
    assert float(info_ref["kl"]) == pytest.approx(0.0, abs=1e-6)


def test_prune_mask_threshold_uses_absolute_advantage():
    adv = mx.array([[3.0, -2.0, 0.5, -0.1]])
    mask = np.asarray(prune_mask(adv, prune_threshold=1.0))
    np.testing.assert_array_equal(mask, [[1, 1, 0, 0]])
    mask = np.asarray(prune_mask(adv, prune_threshold=2.5))
    np.testing.assert_array_equal(mask, [[1, 0, 0, 0]])
    mask = np.asarray(prune_mask(adv, prune_threshold=0.0))
    np.testing.assert_array_equal(mask, [[1, 1, 1, 1]])


def test_prune_mask_threshold_can_empty_a_group():
    adv = mx.array([[0.2, -0.1]])
    mask = np.asarray(prune_mask(adv, prune_threshold=1.0))
    np.testing.assert_array_equal(mask, [[0, 0]])


def test_prune_rate_keeps_floor_g_times_one_minus_p():
    adv = mx.array([[4.0, 2.0, -1.0, -3.0, 0.5, 0.4, 0.3, 0.2]])
    for rate, k in ((0.5, 4), (0.75, 2), (0.125, 7)):
        assert int(np.asarray(prune_mask(adv, prune_rate=rate)).sum()) == k
    g16 = mx.array([list(np.linspace(-1, 1, 16))])
    assert int(np.asarray(prune_mask(g16, prune_rate=0.75)).sum()) == 4
    assert int(np.asarray(prune_mask(g16, prune_rate=0.875)).sum()) == 2


def test_prune_rate_never_empties_a_group_and_breaks_ties_by_index():
    adv = mx.array([[1.0, -1.0, 1.0, -1.0]])
    mask = np.asarray(prune_mask(adv, prune_rate=0.9))
    k = max(1, int(np.floor(4 * 0.1)))
    np.testing.assert_array_equal(mask, [[1, 0, 0, 0]])
    assert k == 1
    mask2 = np.asarray(prune_mask(adv, prune_rate=0.5))
    np.testing.assert_array_equal(mask2, [[1, 1, 0, 0]])


def test_prune_rate_validation():
    adv = mx.array([[1.0, -1.0]])
    with pytest.raises(ValueError):
        prune_mask(adv, prune_rate=1.0)
    with pytest.raises(ValueError):
        prune_mask(adv, prune_threshold=-0.5)
    with pytest.raises(ValueError):
        prune_mask(mx.array([1.0, -1.0]))


def test_renormalisation_matches_hand_computed_tiny_case():
    # One group, four completions, ratio 1, no KL, so the loss is the negative
    # mean of the retained advantages.
    adv = mx.array([[4.0, 2.0, -1.0, -3.0]])
    logp = mx.zeros((1, 4, 2))
    mask = mx.ones((1, 4, 2), dtype=mx.float32)
    loss, info = cppo_loss(logp, logp, logp, adv, mask, prune_rate=0.5, beta=0.0)
    np.testing.assert_allclose(float(loss), -0.5, rtol=1e-6)  # mean(4, -3)
    assert int(info["n_retained"]) == 2
    loss3, _ = cppo_loss(logp, logp, logp, adv, mask, prune_rate=0.25, beta=0.0)
    np.testing.assert_allclose(float(loss3), -1.0, rtol=1e-6)  # mean(4, -3, 2)
    # eq. 7's divisor instead: (4 - 3) / 4
    lossg, _ = cppo_loss(logp, logp, logp, adv, mask, prune_rate=0.5, beta=0.0, normalisation="group")
    np.testing.assert_allclose(float(lossg), -0.25, rtol=1e-6)
    # A threshold keeps only the single largest |A|.
    loss1, _ = cppo_loss(logp, logp, logp, adv, mask, prune_threshold=3.5, beta=0.0)
    np.testing.assert_allclose(float(loss1), -4.0, rtol=1e-6)


def test_update_scale_stays_bounded_after_pruning():
    mx.random.seed(7)
    for rate in (0.0, 0.25, 0.5, 0.75):
        adv = mx.random.normal((4, 8)) * 3.0
        logp = mx.zeros((4, 8, 2))
        mask = mx.ones((4, 8, 2), dtype=mx.float32)
        loss, info = cppo_loss(logp, logp, logp, adv, mask, prune_rate=rate, beta=0.0)
        assert abs(float(loss)) <= float(mx.max(mx.abs(adv))) + 1e-6
        if rate > 0:
            assert float(info["retained_fraction"]) <= 1.0 - rate + 1e-6


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


def test_all_pruned_batch_gives_zero_loss_and_zero_gradient():
    adv = mx.array([[0.1, -0.1]])
    logp = mx.array([[[0.5], [-0.5]]])
    mask = mx.ones((1, 2, 1), dtype=mx.float32)
    loss, info = cppo_loss(logp, logp, logp, adv, mask, prune_threshold=1.0, beta=0.04)
    assert float(info["n_retained"]) == 0
    np.testing.assert_allclose(float(loss), 0.0, atol=0.0)
    grad = mx.grad(lambda x: cppo_loss(x, logp, logp, adv, mask, prune_threshold=1.0)[0])(logp)
    assert np.all(np.isfinite(np.asarray(grad)))
    np.testing.assert_allclose(np.asarray(grad), 0.0, atol=1e-8)


def test_gradients_are_finite_for_grpo_and_cppo():
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

    for fn in (
        lambda m: grpo_loss(
            completion_logprobs(m, mx.repeat(prompt, G, axis=0), mx.repeat(pmask, G, axis=0), comp.reshape(B * G, C)).reshape(B, G, C),
            logp, logp, advantages, cmask)[0],
        lambda m: cppo_loss(
            completion_logprobs(m, mx.repeat(prompt, G, axis=0), mx.repeat(pmask, G, axis=0), comp.reshape(B * G, C)).reshape(B, G, C),
            logp, logp, advantages, cmask, prune_rate=0.5)[0],
    ):
        grads = mx.grad(fn)(model)
        leaves = [np.asarray(v) for _, v in tree_flatten(grads)]
        assert leaves
        assert all(np.all(np.isfinite(v)) for v in leaves)
        assert any(np.any(v != 0) for v in leaves)


def test_per_group_losses_sum_to_the_pruned_objective():
    logp, logp_old, logp_ref, advantages, token_mask = random_inputs(seed=2)
    mask = prune_mask(advantages, prune_rate=0.5)
    per_group = per_group_losses(logp, logp_old, logp_ref, advantages, token_mask, mask)
    loss, info = cppo_loss(logp, logp_old, logp_ref, advantages, token_mask, prune_rate=0.5)
    np.testing.assert_allclose(float(mx.sum(per_group)), float(loss) * int(info["n_retained"]), rtol=1e-5)

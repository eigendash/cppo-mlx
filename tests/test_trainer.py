import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest

from cppo.data import AdditionTask, make_example, pad_prompts
from cppo.model import TinyLM, completion_logprobs
from cppo.objectives import cppo_loss, grpo_loss
from cppo.sample import score_completions
from cppo.trainer import (
    TrainConfig,
    clone_model,
    collect_rollout,
    dynamic_question_count,
    evaluate,
    flat_params,
    gradient_stats,
    retained_per_question,
    rl_step,
    rollout_loss,
    train_rl,
    train_sft,
    train_sft_to_accuracy,
)


def small_model(seed=0) -> TinyLM:
    mx.random.seed(seed)
    return TinyLM(vocab_size=15, dim=32, n_layers=1, n_heads=4, max_len=16)


@pytest.fixture(scope="module")
def task():
    return AdditionTask(digits=2, n_train=64, n_test=16, seed=0)


def test_retained_per_question_follows_floor_g_one_minus_p():
    assert retained_per_question(16, 0.0) == 16
    assert retained_per_question(16, 0.5) == 8
    assert retained_per_question(16, 0.75) == 4
    assert retained_per_question(16, 0.875) == 2
    assert retained_per_question(8, 0.5) == 4
    assert retained_per_question(3, 0.5) == 1
    assert retained_per_question(4, 0.9) == 1


def test_dynamic_question_count_scales_the_batch():
    assert dynamic_question_count(8, 0.0) == 8
    assert dynamic_question_count(8, 0.5) == 16
    assert dynamic_question_count(8, 0.75) == 32
    assert dynamic_question_count(5, 0.5) == 10
    assert dynamic_question_count(8, 0.875) == 64
    for rate in (0.0, 0.5, 0.75):
        # Retained completions per step are what the gradient budget counts.
        assert dynamic_question_count(8, rate) * retained_per_question(8, rate) >= 8 * 8 * 0.999
    with pytest.raises(ValueError):
        dynamic_question_count(8, 1.0)


def test_grpo_rollout_keeps_every_completion(task):
    model = small_model()
    cfg = TrainConfig(group_size=4, questions=2, seed=0)
    rollout = collect_rollout(model, model, task.train[:2], cfg, mx.random.key(0))
    assert rollout.sampled == 8
    assert rollout.n_retained == 8
    assert rollout.retained_fraction == 1.0
    assert rollout.r_completion.shape == (8, cfg.max_new_tokens)
    assert rollout.r_logp_old.shape == rollout.r_logp_ref.shape == (8, cfg.max_new_tokens)


def test_cppo_rollout_prunes_before_the_forward_passes(task):
    model = small_model()
    cfg = TrainConfig(group_size=8, questions=2, prune_rate=0.5, seed=0)
    rollout = collect_rollout(model, model, task.train[:2], cfg, mx.random.key(1))
    assert rollout.sampled == 16
    assert rollout.n_retained == 8  # k = 4 per question, two questions
    assert rollout.retained_fraction == pytest.approx(0.5)
    # The retained views are exactly the gathered rows, in row-major order.
    np.testing.assert_array_equal(np.asarray(rollout.r_completion), np.asarray(rollout.completion)[rollout.retained])
    np.testing.assert_array_equal(
        np.asarray(rollout.r_completion_mask), np.asarray(rollout.completion_mask)[rollout.retained]
    )
    # k per question, and the kept ones are the largest |A| in each group.
    np.testing.assert_array_equal(rollout.retained.sum(axis=1), [4, 4])
    for b in range(2):
        kept = np.abs(np.asarray(rollout.advantages)[b])[rollout.retained[b]]
        dropped = np.abs(np.asarray(rollout.advantages)[b])[~rollout.retained[b]]
        assert kept.min() >= dropped.max() - 1e-6


def full_batch_logps(model, rollout):
    B, G, C = rollout.completion.shape
    logp = completion_logprobs(
        model,
        mx.repeat(rollout.prompt, G, axis=0),
        mx.repeat(rollout.prompt_mask, G, axis=0),
        rollout.completion.reshape(B * G, C),
    ).reshape(B, G, C)
    mx.eval(logp)
    return logp


def test_gathered_rollout_loss_matches_the_dense_objective(task):
    model = small_model()
    reference = small_model(seed=7)
    for rate in (0.0, 0.5, 0.75):
        cfg = TrainConfig(group_size=4, questions=3, prune_rate=rate, seed=0)
        rollout = collect_rollout(model, reference, task.train[:3], cfg, mx.random.key(2))
        gathered, _ = rollout_loss(model, rollout, cfg)
        logp = full_batch_logps(model, rollout)
        dense, _ = cppo_loss(
            logp,
            logp,
            full_batch_logps(reference, rollout),
            rollout.advantages,
            rollout.completion_mask,
            prune_rate=rate,
            beta=cfg.beta,
            clip_eps=cfg.clip_eps,
        )
        np.testing.assert_allclose(float(gathered), float(dense), rtol=1e-4, atol=1e-6)
        if rate == 0.0:
            plain, _ = grpo_loss(
                logp,
                logp,
                full_batch_logps(reference, rollout),
                rollout.advantages,
                rollout.completion_mask,
                beta=cfg.beta,
                clip_eps=cfg.clip_eps,
            )
            np.testing.assert_allclose(float(gathered), float(plain), rtol=1e-4, atol=1e-6)


def test_rollout_loss_uses_group_normalisation_when_asked(task):
    model = small_model()
    reference = small_model(seed=5)
    cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.5, seed=0, normalisation="group")
    rollout = collect_rollout(model, reference, task.train[:2], cfg, mx.random.key(3))
    grouped, info = rollout_loss(model, rollout, cfg)
    retained_cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.5, seed=0)
    retained, _ = rollout_loss(model, rollout, retained_cfg)
    # Group normalisation divides by the sampled count, retained by k.
    np.testing.assert_allclose(
        float(grouped), float(retained) * rollout.n_retained / rollout.sampled, rtol=1e-4
    )


def test_rl_step_updates_the_policy_and_not_the_reference(task):
    model = small_model()
    reference = clone_model(model)
    before = flat_params(reference)
    cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.5, seed=0)
    rollout = collect_rollout(model, reference, task.train[:2], cfg, mx.random.key(4))
    optimizer = optim.AdamW(learning_rate=cfg.lr)
    loss, info, norm = rl_step(model, rollout, optimizer, cfg)
    assert np.isfinite(loss) and np.isfinite(norm)
    np.testing.assert_array_equal(flat_params(reference), before)
    assert not np.array_equal(flat_params(model), before)
    assert int(info["retained"]) == rollout.n_retained


def test_clone_model_copies_weights(task):
    model = small_model()
    clone = clone_model(model)
    np.testing.assert_array_equal(flat_params(clone), flat_params(model))
    cfg = TrainConfig(group_size=4, questions=2, seed=0)
    rollout = collect_rollout(model, clone, task.train[:2], cfg, mx.random.key(6))
    rl_step(clone, rollout, optim.AdamW(learning_rate=1e-2), cfg)
    assert not np.array_equal(flat_params(clone), flat_params(model))


def test_gradient_stats_are_bounded_and_degenerate_when_nothing_is_pruned(task):
    model = small_model()
    reference = small_model(seed=9)
    cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.0, seed=0)
    rollout = collect_rollout(model, reference, task.train[:2], cfg, mx.random.key(5))
    stats = gradient_stats(model, reference, rollout, cfg)
    assert 0.0 <= stats["full_snr"] <= 1.0 + 1e-6
    assert stats["cosine"] == pytest.approx(1.0, abs=1e-6)  # nothing pruned
    assert stats["norm_ratio"] == pytest.approx(1.0, abs=1e-6)

    pruned_cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.5, seed=0)
    pruned_rollout = collect_rollout(model, reference, task.train[:2], pruned_cfg, mx.random.key(5))
    stats = gradient_stats(model, reference, pruned_rollout, pruned_cfg)
    assert 0.0 <= stats["pruned_snr"] <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= stats["cosine"] <= 1.0 + 1e-6
    assert np.isfinite(stats["pruned_grad_norm"]) and np.isfinite(stats["full_grad_norm"])


def test_budget_accounting_matches_the_allocation_rule(task):
    model = small_model()
    reference = clone_model(model)
    cfg = TrainConfig(group_size=8, questions=2, prune_rate=0.5, seed=0, lr=1e-4)
    budget = train_rl(model, reference, task, cfg, steps=3, key_offset=0)
    n_questions = dynamic_question_count(cfg.questions, cfg.prune_rate)
    k = retained_per_question(cfg.group_size, cfg.prune_rate)
    assert budget.steps == 3
    assert budget.sampled_completions == 3 * n_questions * cfg.group_size
    assert budget.retained_completions == 3 * n_questions * k
    assert budget.sampled_completions == 2 * budget.retained_completions
    assert budget.gradient_tokens == budget.retained_completions * cfg.max_new_tokens
    assert len(budget.losses) == 3 and all(np.isfinite(v) for v in budget.losses)
    assert budget.wall_time > 0


def test_grpo_budget_saves_nothing(task):
    model = small_model()
    reference = clone_model(model)
    cfg = TrainConfig(group_size=4, questions=2, seed=0, lr=1e-4)
    budget = train_rl(model, reference, task, cfg, steps=2, key_offset=0)
    assert budget.sampled_completions == budget.retained_completions == 2 * 2 * 4


def test_sft_warmup_reduces_the_loss_and_improves_accuracy():
    task = AdditionTask(digits=1, n_train=40, n_test=40, seed=0)
    model = small_model(seed=1)
    before = evaluate(model, task.test)
    cfg = TrainConfig(sft_steps=200, sft_batch=32, seed=0)
    losses = train_sft(model, task, cfg)
    assert np.mean(losses[-20:]) < np.mean(losses[:20])
    assert evaluate(model, task.test) > before


def test_train_rl_records_an_eval_curve(task):
    model = small_model()
    reference = clone_model(model)
    cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.5, seed=0, lr=1e-4)
    budget = train_rl(model, reference, task, cfg, steps=4, eval_examples=task.test, eval_every=2)
    assert [p["step"] for p in budget.curve] == [2, 4]
    assert all(0.0 <= p["accuracy"] <= 1.0 for p in budget.curve)
    assert budget.curve[-1]["sampled_completions"] == budget.sampled_completions


def test_dynamic_allocation_can_be_switched_off_for_a_matched_sample_budget(task):
    on = TrainConfig(group_size=8, questions=2, prune_rate=0.5)
    off = TrainConfig(group_size=8, questions=2, prune_rate=0.5, dynamic_allocation=False)
    assert on.questions_per_step() == 4 and on.samples_per_step() == 32
    assert off.questions_per_step() == 2 and off.samples_per_step() == 16
    model = small_model()
    reference = clone_model(model)
    off.lr = 1e-4
    budget = train_rl(model, reference, task, off, steps=3, key_offset=0)
    assert budget.sampled_completions == 3 * 2 * 8
    assert budget.retained_completions == 3 * 2 * 4
    assert all(np.isfinite(v) for v in budget.reward_means)


def test_budget_records_reward_and_degeneracy_statistics(task):
    model = small_model()
    reference = clone_model(model)
    cfg = TrainConfig(group_size=4, questions=2, prune_rate=0.5)
    cfg.lr = 1e-4
    budget = train_rl(model, reference, task, cfg, steps=6, eval_every=3, eval_examples=task.test)
    assert len(budget.reward_means) == 6
    assert all(0.0 <= v <= 3.0 for v in budget.reward_means)
    assert all(0.0 <= d <= 1.0 for d in budget.degenerate)
    assert all(0.0 <= point["mean_reward"] <= 3.0 for point in budget.curve)
    assert all(0.0 <= point["degenerate_fraction"] <= 1.0 for point in budget.curve)


def test_sft_band_stops_at_the_first_check_that_reaches_the_target():
    task = AdditionTask(digits=1, n_train=50, n_test=31, seed=0)
    model = small_model(seed=3)
    cfg = TrainConfig(sft_steps=0, sft_batch=32, seed=0)
    steps, accuracy = train_sft_to_accuracy(
        model, task, cfg, task.test, target=0.0, max_steps=600, check_every=50
    )
    assert steps == 50  # already at or above the target at the first check
    assert accuracy >= 0.0


def test_sft_band_gives_up_at_max_steps_when_the_target_is_out_of_reach():
    task = AdditionTask(digits=1, n_train=50, n_test=31, seed=0)
    model = small_model(seed=4)
    cfg = TrainConfig(sft_steps=0, sft_batch=32, seed=0)
    steps, accuracy = train_sft_to_accuracy(
        model, task, cfg, task.test, target=1.01, max_steps=100, check_every=25
    )
    assert steps == 100
    assert accuracy < 1.01

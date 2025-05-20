"""Rollouts, the supervised warm-up, and an RL step for GRPO / CPPO.

The paper's Algorithm 1 differs from GRPO in two places, and both live here:

* *Completion pruning*: keep the top k = max(1, floor(G (1 - P))) completions by
  |A| per question.  Pruning happens right after the rewards, and only the
  retained completions are forwarded through the policy, reference and old
  policy models -- that is the compute CPPO removes.
* *Dynamic completion allocation*: because pruning leaves the gradient budget
  underfilled, sample ``b / (1 - P)`` questions per step instead of ``b`` and
  keep k completions from each, so the step still fills the same gradient batch
  (``dynamic_question_count``, paper Appendix B step 7).

The supervised warm-up is not in the paper.  It is here because GRPO needs a
policy that already succeeds sometimes: a randomly initialised tiny LM never
gets a correct completion, every group has zero advantage, and there is no
gradient.  Both arms in the experiment start from the same warmed-up model, and
that model is also the frozen reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .data import AdditionTask, EOS_ID, Example, PAD_ID, pad_prompts
from .model import TinyLM, completion_cross_entropy, completion_logprobs
from .objectives import (
    cppo_loss,
    grpo_loss,
    group_advantages,
    per_group_losses,
    prune_mask,
    retained_loss,
)
from .sample import greedy_accuracy, mask_after_eos, sample_completions, score_completions


@dataclass
class TrainConfig:
    group_size: int = 8
    questions: int = 8  # questions whose retained completions fill the batch
    prune_rate: float = 0.0
    prune_threshold: float = 0.0
    dynamic_allocation: bool = True
    normalisation: str = "retained"
    beta: float = 0.04
    clip_eps: float = 0.2
    lr: float = 3e-4
    temperature: float = 1.0
    max_new_tokens: int = 5
    sft_steps: int = 500
    sft_batch: int = 64
    sft_lr: float = 3e-3
    max_len: int = 24
    seed: int = 0

    @property
    def prunes(self) -> bool:
        return self.prune_rate > 0.0 or self.prune_threshold > 0.0

    def questions_per_step(self) -> int:
        """Questions sampled per step, after dynamic allocation if enabled."""
        if self.prune_rate > 0.0 and self.dynamic_allocation:
            return dynamic_question_count(self.questions, self.prune_rate)
        return self.questions

    def samples_per_step(self) -> int:
        return self.questions_per_step() * self.group_size


def retained_per_question(group_size: int, prune_rate: float) -> int:
    """k = max(1, floor(G (1 - P))), the count CPPO keeps per question."""
    return max(1, int(math.floor(group_size * (1.0 - prune_rate))))


def dynamic_question_count(questions: int, prune_rate: float) -> int:
    """b / (1 - P) questions per step, rounded up (paper Appendix B)."""
    if not 0.0 <= prune_rate < 1.0:
        raise ValueError("prune_rate must be in [0, 1)")
    return int(math.ceil(questions / (1.0 - prune_rate)))


@dataclass
class Rollout:
    """Sampled completions, and the retained subset that reaches the forwards."""

    examples: list[Example]
    prompt: mx.array
    prompt_mask: mx.array
    completion: mx.array
    completion_mask: mx.array
    rewards: np.ndarray
    advantages: mx.array
    retained: np.ndarray  # (B, G) bool
    r_prompt: mx.array
    r_prompt_mask: mx.array
    r_completion: mx.array
    r_completion_mask: mx.array
    r_advantages: mx.array
    r_logp_old: mx.array
    r_logp_ref: mx.array

    @property
    def batch(self) -> int:
        return self.prompt.shape[0]

    @property
    def group_size(self) -> int:
        return self.completion.shape[1]

    @property
    def sampled(self) -> int:
        return self.batch * self.group_size

    @property
    def n_retained(self) -> int:
        return int(self.retained.sum())

    @property
    def retained_fraction(self) -> float:
        return self.n_retained / max(self.sampled, 1)

    @property
    def degenerate_groups(self) -> int:
        """Groups whose rewards are all identical (zero advantage, no signal)."""
        return int(np.sum(np.asarray(self.rewards).std(axis=-1) == 0))


@dataclass
class Budget:
    steps: int = 0
    sampled_completions: int = 0
    retained_completions: int = 0
    gradient_tokens: int = 0
    wall_time: float = 0.0
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    reward_means: list[float] = field(default_factory=list)
    degenerate: list[float] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "sampled_completions": self.sampled_completions,
            "retained_completions": self.retained_completions,
            "gradient_tokens": self.gradient_tokens,
            "wall_time": round(self.wall_time, 2),
            "mean_loss": float(np.mean(self.losses)) if self.losses else None,
            "mean_grad_norm": float(np.mean(self.grad_norms)) if self.grad_norms else None,
            "grad_norm_std": float(np.std(self.grad_norms)) if self.grad_norms else None,
            "mean_reward": float(np.mean(self.reward_means)) if self.reward_means else None,
            "degenerate_fraction": float(np.mean(self.degenerate)) if self.degenerate else None,
            "curve": self.curve,
        }


def flat_grads(grads) -> np.ndarray:
    return np.concatenate([np.asarray(v).ravel() for _, v in tree_flatten(grads)])


def flat_params(model) -> np.ndarray:
    return np.concatenate([np.asarray(v).ravel() for _, v in tree_flatten(model.parameters())])


def clone_model(model: TinyLM) -> TinyLM:
    """A frozen copy with the same weights (the reference model)."""
    clone = TinyLM(
        vocab_size=model.vocab_size,
        dim=model.token_emb.weight.shape[1],
        n_layers=len(model.blocks),
        n_heads=model.blocks[0].n_heads,
        ff_mult=model.blocks[0].up.weight.shape[0] // model.token_emb.weight.shape[1],
        max_len=model.max_len,
    )
    clone.update(model.parameters())
    mx.eval(clone.parameters())
    return clone


def collate_supervised(examples: list[Example], pad_id: int = PAD_ID):
    prompt, prompt_mask = pad_prompts(examples, pad_id)
    width = max(len(e.target_tokens) for e in examples)
    ids = [[pad_id] * width for _ in examples]
    mask = [[0.0] * width for _ in examples]
    for i, e in enumerate(examples):
        target = e.target_tokens
        ids[i][: len(target)] = target
        mask[i][: len(target)] = [1.0] * len(target)
    return prompt, prompt_mask, mx.array(ids, dtype=mx.int32), mx.array(mask, dtype=mx.float32)


def sft_step(model: TinyLM, batch: list[Example], optimizer, lr: float) -> float:
    prompt, prompt_mask, target, target_mask = collate_supervised(batch)

    def loss_fn(m):
        return completion_cross_entropy(m, prompt, prompt_mask, target, target_mask)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    optimizer.learning_rate = lr
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    return float(loss)


def train_sft(model: TinyLM, task: AdditionTask, cfg: TrainConfig, verbose: bool = False) -> list[float]:
    """Teacher-forced warm-up on the task's own targets."""
    optimizer = optim.AdamW(learning_rate=cfg.sft_lr)
    rng = np.random.default_rng(cfg.seed)
    losses: list[float] = []
    for step in range(cfg.sft_steps):
        idx = rng.integers(0, len(task.train), size=cfg.sft_batch)
        losses.append(sft_step(model, [task.train[i] for i in idx], optimizer, cfg.sft_lr))
        if verbose and (step + 1) % 100 == 0:
            print(f"  sft step {step + 1:4d} loss {np.mean(losses[-100:]):.4f}", flush=True)
    return losses


def collect_rollout(
    model: TinyLM,
    reference: TinyLM,
    examples: list[Example],
    cfg: TrainConfig,
    key: mx.array,
) -> Rollout:
    """Sample G completions per question, score them, prune, then forward.

    The policy / reference forwards are run only on the retained completions,
    which is exactly the compute CPPO saves (paper Fig. 2).
    """
    prompt, prompt_mask = pad_prompts(examples)
    completion, completion_mask = sample_completions(
        model,
        prompt,
        prompt_mask,
        n_samples=cfg.group_size,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        key=key,
    )
    mx.eval(completion, completion_mask)
    rewards = score_completions(completion, examples)
    advantages = group_advantages(mx.array(rewards))
    mx.eval(advantages)
    keep = np.asarray(prune_mask(advantages, cfg.prune_threshold, cfg.prune_rate)) > 0
    index = np.argwhere(keep)  # (K, 2), row-major so the gather order is fixed
    rows = mx.array(index[:, 0], dtype=mx.int32)
    cols = mx.array(index[:, 1], dtype=mx.int32)
    r_prompt = prompt[rows]
    r_prompt_mask = prompt_mask[rows]
    r_completion = completion[rows, cols]
    r_completion_mask = completion_mask[rows, cols]
    r_advantages = advantages[rows, cols]
    if index.shape[0]:
        r_logp_old = completion_logprobs(model, r_prompt, r_prompt_mask, r_completion)
        r_logp_ref = completion_logprobs(reference, r_prompt, r_prompt_mask, r_completion)
    else:
        width = completion.shape[-1]
        r_logp_old = mx.zeros((0, width))
        r_logp_ref = mx.zeros((0, width))
    mx.eval(r_logp_old, r_logp_ref, r_advantages)
    return Rollout(
        examples=examples,
        prompt=prompt,
        prompt_mask=prompt_mask,
        completion=completion,
        completion_mask=completion_mask,
        rewards=rewards,
        advantages=advantages,
        retained=keep,
        r_prompt=r_prompt,
        r_prompt_mask=r_prompt_mask,
        r_completion=r_completion,
        r_completion_mask=r_completion_mask,
        r_advantages=r_advantages,
        r_logp_old=r_logp_old,
        r_logp_ref=r_logp_ref,
    )


def rollout_denominator(rollout: Rollout, cfg: TrainConfig):
    """1/k after pruning (paper eq. 9) or 1/G (paper eq. 7)."""
    if cfg.normalisation == "group":
        return mx.array(float(rollout.sampled))
    return None


def rollout_loss(model: TinyLM, rollout: Rollout, cfg: TrainConfig):
    """CPPO (or GRPO, when nothing is pruned) loss over the retained set.

    A threshold can prune every completion of every question.  Those steps have
    no gradient, so the loss is the constant zero.
    """
    if rollout.n_retained == 0:
        zero = mx.zeros(1).sum()
        return zero, {
            "loss": zero,
            "objective": -zero,
            "kl": zero,
            "ratio_mean": zero,
            "n_retained": 0.0,
            "retained_fraction": 0.0,
            "pruned": True,
            "retained": 0,
            "empty": True,
        }
    logp = completion_logprobs(model, rollout.r_prompt, rollout.r_prompt_mask, rollout.r_completion)
    loss, info = retained_loss(
        logp,
        rollout.r_logp_old,
        rollout.r_logp_ref,
        rollout.r_advantages,
        rollout.r_completion_mask,
        denominator=rollout_denominator(rollout, cfg),
        beta=cfg.beta,
        clip_eps=cfg.clip_eps,
        n_sampled=rollout.sampled,
    )
    info["pruned"] = cfg.prunes
    info["retained"] = rollout.n_retained
    return loss, info


def rl_step(model: TinyLM, rollout: Rollout, optimizer, cfg: TrainConfig):
    """One gradient step; returns (loss, info, grad_norm)."""

    def loss_fn(m):
        return rollout_loss(m, rollout, cfg)[0]

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    norm = float(np.linalg.norm(flat_grads(grads)))
    optimizer.learning_rate = cfg.lr
    optimizer.update(model, grads)
    _, info = rollout_loss(model, rollout, cfg)
    mx.eval(model.parameters(), optimizer.state, loss, *info.values())
    return float(loss), info, norm


def gradient_stats(model: TinyLM, reference: TinyLM, rollout: Rollout, cfg: TrainConfig) -> dict:
    """Per-question gradient spread for the full group and for the pruned group.

    ``snr = ||mean_b g_b|| / sqrt(mean_b ||g_b||^2)`` estimates how much of a
    step is signal rather than sampling noise; ``cosine`` is the similarity
    between the pruned step direction and the full-group one on the same
    rollout, which is the direct check on CPPO's "unbiased-ish" claim.  Both
    gradients are taken at the current policy with the ratio set to 1 (the
    start-of-step situation), so this is instrumentation and not the training
    path: it forwards every sampled completion on purpose.
    """
    B, G, C = rollout.completion.shape
    flat_prompt = mx.repeat(rollout.prompt, G, axis=0)
    flat_mask = mx.repeat(rollout.prompt_mask, G, axis=0)
    flat_completion = rollout.completion.reshape(B * G, C)
    logp_all = completion_logprobs(model, flat_prompt, flat_mask, flat_completion).reshape(B, G, C)
    logp_ref_all = completion_logprobs(reference, flat_prompt, flat_mask, flat_completion).reshape(B, G, C)
    mx.eval(logp_all, logp_ref_all)
    full_weight = mx.ones((B, G), dtype=mx.float32)
    pruned_weight = mx.array(rollout.retained.astype(np.float32))

    def question_loss(m, b: int, weight):
        logp = completion_logprobs(m, flat_prompt, flat_mask, flat_completion).reshape(B, G, C)
        per_group = per_group_losses(
            logp[b : b + 1],
            logp_all[b : b + 1],
            logp_ref_all[b : b + 1],
            rollout.advantages[b : b + 1],
            rollout.completion_mask[b : b + 1],
            weight[b : b + 1],
            beta=cfg.beta,
            clip_eps=cfg.clip_eps,
        )
        return mx.sum(per_group) / mx.maximum(mx.sum(weight[b : b + 1]), 1.0)

    out: dict[str, float] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, weight in (("full", full_weight), ("pruned", pruned_weight)):
        grads = [flat_grads(mx.grad(lambda m, b=b, w=weight: question_loss(m, b, w))(model)) for b in range(B)]
        g = np.stack(grads)
        mean = g.mean(axis=0)
        vectors[name] = mean
        out[f"{name}_grad_norm"] = float(np.linalg.norm(mean))
        out[f"{name}_snr"] = float(np.linalg.norm(mean) / max(np.sqrt((g**2).sum(axis=1).mean()), 1e-12))
    denom = float(np.linalg.norm(vectors["full"]) * np.linalg.norm(vectors["pruned"]))
    out["cosine"] = float(np.dot(vectors["full"], vectors["pruned"]) / denom) if denom > 0 else 0.0
    out["norm_ratio"] = (
        out["pruned_grad_norm"] / out["full_grad_norm"] if out["full_grad_norm"] > 0 else 0.0
    )
    return out


def shuffle_examples(examples: list[Example], rng: np.random.Generator) -> list[Example]:
    order = rng.permutation(len(examples))
    return [examples[i] for i in order]


def evaluate(model: TinyLM, examples: list[Example]) -> float:
    return greedy_accuracy(model, examples)


def train_rl(
    model: TinyLM,
    reference: TinyLM,
    task: AdditionTask,
    cfg: TrainConfig,
    steps: int,
    eval_examples: list[Example] | None = None,
    eval_every: int = 0,
    key_offset: int = 0,
    verbose: bool = False,
) -> Budget:
    """Run ``steps`` RL steps on the task's training split, tracking costs."""
    import time

    optimizer = optim.AdamW(learning_rate=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    data = shuffle_examples(task.train, rng)
    n_questions = cfg.questions_per_step()
    budget = Budget()
    started = time.perf_counter()
    for step in range(steps):
        start = (step * n_questions) % max(len(data) - n_questions, 1)
        batch = data[start : start + n_questions]
        rollout = collect_rollout(model, reference, batch, cfg, mx.random.key(key_offset + step))
        loss, _, norm = rl_step(model, rollout, optimizer, cfg)
        budget.steps += 1
        budget.sampled_completions += rollout.sampled
        budget.retained_completions += rollout.n_retained
        budget.gradient_tokens += rollout.n_retained * rollout.completion.shape[-1]
        budget.losses.append(loss)
        budget.grad_norms.append(norm)
        budget.reward_means.append(float(rollout.rewards.mean()))
        budget.degenerate.append(rollout.degenerate_groups / max(rollout.batch, 1))
        if eval_every and eval_examples is not None and (step + 1) % eval_every == 0:
            window = budget.reward_means[-eval_every:]
            budget.curve.append(
                {
                    "step": step + 1,
                    "sampled_completions": budget.sampled_completions,
                    "retained_completions": budget.retained_completions,
                    "accuracy": evaluate(model, eval_examples),
                    "mean_reward": float(np.mean(window)),
                    "degenerate_fraction": float(np.mean(budget.degenerate[-eval_every:])),
                    "wall_time": round(time.perf_counter() - started, 3),
                }
            )
            if verbose:
                point = budget.curve[-1]
                print(
                    f"  step {point['step']:4d} sampled {point['sampled_completions']:6d} "
                    f"acc {point['accuracy']:.4f}",
                    flush=True,
                )
    budget.wall_time = time.perf_counter() - started
    return budget


__all__ = [
    "Budget",
    "Rollout",
    "TrainConfig",
    "clone_model",
    "collate_supervised",
    "collect_rollout",
    "dynamic_question_count",
    "evaluate",
    "flat_grads",
    "flat_params",
    "gradient_stats",
    "retained_per_question",
    "rl_step",
    "rollout_loss",
    "shuffle_examples",
    "train_rl",
    "train_sft",
]

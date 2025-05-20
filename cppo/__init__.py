"""A small MLX implementation of GRPO with CPPO-style completion pruning.

The paper is *CPPO: Accelerating the Training of Group Relative Policy
Optimization-Based Reasoning Models* (arXiv:2503.22342).  This package holds a
tiny causal LM, a programmatic-reward synthetic task, the GRPO objective, and
the CPPO pruning / dynamic-allocation layer on top of it.
"""

from .data import AdditionTask, Example, VOCAB, VOCAB_SIZE, make_example, pad_prompts
from .model import TinyLM, completion_cross_entropy, completion_logprobs, param_count
from .objectives import (
    cppo_loss,
    grpo_loss,
    group_advantages,
    kl_k3,
    per_group_losses,
    prune_mask,
    retained_loss,
)
from .reward import accuracy, reward
from .sample import greedy_accuracy, sample_completions, score_completions
from .trainer import (
    Budget,
    Rollout,
    TrainConfig,
    clone_model,
    collect_rollout,
    dynamic_question_count,
    evaluate,
    gradient_stats,
    retained_per_question,
    rl_step,
    rollout_loss,
    train_rl,
    train_sft,
)

__version__ = "0.1.0"

__all__ = [
    "AdditionTask",
    "Budget",
    "Example",
    "Rollout",
    "TinyLM",
    "TrainConfig",
    "VOCAB",
    "VOCAB_SIZE",
    "accuracy",
    "clone_model",
    "collect_rollout",
    "completion_cross_entropy",
    "completion_logprobs",
    "cppo_loss",
    "dynamic_question_count",
    "evaluate",
    "gradient_stats",
    "greedy_accuracy",
    "grpo_loss",
    "group_advantages",
    "kl_k3",
    "make_example",
    "pad_prompts",
    "param_count",
    "per_group_losses",
    "prune_mask",
    "retained_loss",
    "retained_per_question",
    "reward",
    "rl_step",
    "rollout_loss",
    "sample_completions",
    "score_completions",
    "train_rl",
    "train_sft",
]

# cppo-mlx

A small MLX implementation of Completion Pruning Policy Optimization, the method
in *CPPO: Accelerating the Training of Group Relative Policy Optimization-Based
Reasoning Models* (arXiv:2503.22342, submitted 28 March 2025). The paper starts
from GRPO: sample a group of completions per question, score them with a
rule-based reward, normalise the rewards within the group into advantages, and
take a clipped policy-gradient step with a KL penalty against a frozen
reference. It observes that a completion's contribution to that gradient scales
with its advantage, so completions with a near-zero advantage can be dropped
before the policy / reference / old-policy forward passes, and the compute they
free up can be spent on completions from *additional* questions instead.

This repo is a small independent implementation of that mechanism on a synthetic
task, written from the paper's method section and appendix. It is **not** a
reproduction of the paper's results: the paper trains Qwen2.5-1.5B/7B on GSM8K
and MATH with vLLM and reports up to 7.98x/3.48x speedups on two or four 80GB
GPUs. Nothing here is comparable to that. The model is a 338,304-parameter
character-level transformer trained on CPU-only 2-digit addition, and the
accuracy differences between the arms below are mostly inside seed noise.

## What is in here

### The task and the reward — `cppo/data.py`, `cppo/reward.py`

Two-digit addition where the answer is written **least significant digit first**,
so carries propagate left to right and a decoder-only model can produce the
answer in one pass of local computation:

```
prompt       <bos> 4 7 + 5 8 =
completion   5 0 1 <eos>          (47 + 58 = 105)
```

The vocabulary is ten digits plus `+`, `=`, `<pad>`, `<eos>`, `<bos>`. Pairs are
generated with a seeded `random.Random`, train and test are disjoint, and asking
for more distinct pairs than the digit range holds raises instead of looping.

The reward is the paper's rule-based reward (its Appendix A), adapted to a task
with no free-form text: `R = R_format + R_accuracy`, where `R_format = 1` if the
completion is a non-empty run of digits ended by `<eos>`, `R_accuracy = 2` for
the canonical least-significant-first string, `1.5` if the completion is not
that string but its digits read in the usual most-significant-first order spell
the right number, and `0` otherwise. Tokens after `<eos>` are ignored, and the
completion's length for the per-token mean is up to and including `<eos>`.

### The policy — `cppo/model.py`, `cppo/sample.py`

A 3-layer, 96-wide, 4-head decoder-only transformer with learned positional
embeddings, pre-norm blocks and an additive causal mask (`TinyLM`). Prompts are
expanded to a flat batch of `B * G` rows, so all `G` completions per question
are generated with one forward pass per token (`sample_completions`). Tokens
after the first `<eos>` are masked out of both the reward and the loss.

### GRPO — `cppo/objectives.py`

Implemented as the paper writes it: `A_i = (r_i - mean(r)) / std(r)` with the
population standard deviation and an `eps` in the denominator so a group of
identical rewards gives exactly zero advantage; the clipped surrogate
`min(rho_t A_i, clip(rho_t, 1-eps, 1+eps) A_i)`; the k3 KL estimator
`pi_ref/pi_theta - log(pi_ref/pi_theta) - 1`, which is non-negative and exactly
zero when the policy is the reference; and a per-completion mean over the
completion's own tokens before the mean over the group.

### CPPO — `cppo/objectives.py`, `cppo/trainer.py`

Two retention rules from the paper:

* an absolute threshold, `|A_i| >= gamma` (paper eq. 7), which may empty a group;
* the top-k rule the paper actually uses in its experiments,
  `k = max(1, floor(G (1 - P)))` (eq. 8-10), which never empties a group and
  breaks ties by index so it is deterministic.

`normalisation="retained"` divides by the number of retained completions, which
is the 1/k of eq. 9 and keeps the retained update at the scale of a full group;
`normalisation="group"` keeps eq. 7's divisor `G` and shrinks the update with
the pruning rate. The default is `retained`, and the invariant that matters is
that with `prune_threshold=0` and `prune_rate=0` every completion is retained,
`k = G`, and `cppo_loss` returns bit-identical numbers to `grpo_loss`.

Pruning happens after the rewards and before any model forward: `collect_rollout`
computes rewards and advantages, builds the retention mask, gathers the retained
completions, and only then computes the old-policy and reference log-probabilities
for those rows. That is where the paper's saving comes from (its Figure 2).

Dynamic completion allocation follows the paper's Appendix B, which says the
allocation is done *before* pruning: sample `b / (1 - P)` questions per step
instead of `b`, keep `k` completions from each, and the gradient batch is filled
with the same number of completions as a GRPO step while covering more distinct
questions (`dynamic_question_count`). Setting `dynamic_allocation=False` gives
the matched-sample-budget variant, where the same completions are drawn but only
`k` per question reach the backward pass.

### Training loop and cost accounting — `cppo/trainer.py`

`train_rl` cycles a shuffled training split, calls `collect_rollout` and
`rl_step`, and records per-run costs: sampled completions, retained completions,
gradient tokens (retained completions x completion length, i.e. the tokens that
actually pass through the policy, reference and old-policy forwards), wall time,
per-step loss and gradient norm, and the mean reward and fraction of degenerate
groups. `gradient_stats` measures, on a fixed rollout, the per-question gradient
spread `snr = ||mean_b g_b|| / sqrt(mean_b ||g_b||^2)` for the full group and
for the pruned group, and the cosine similarity between the two step directions.

A supervised warm-up (`train_sft_to_accuracy`) is not in the paper. GRPO needs a
policy that already succeeds sometimes: a randomly initialised model never emits
a correct completion, every group has zero advantage, and there is no gradient.
The warm-up trains on the task's own targets until greedy accuracy first reaches
0.5, which took 550, 650 and 1100 steps for the three seeds. Both arms of every
seed start from that one model, which is also the frozen reference.

## Checks

`pytest` runs 56 tests. The ones that carry the weight:

* `test_threshold_zero_reduces_to_grpo_exactly` and
  `test_threshold_zero_reduces_to_grpo_on_a_real_rollout` — with the threshold
  and the rate at zero, `cppo_loss` is **bitwise equal** (`np.array_equal`) to
  `grpo_loss`, on random inputs and on a real rollout through a real model.
* `test_rewards_and_advantages_of_a_three_completion_group` — a hand-built group
  of three completions for `47 + 58` (canonical `501`, reversed `105`, wrong
  `999`) gives rewards `[3.0, 2.5, 1.0]` and advantages matching the arithmetic
  computed independently; `test_advantages_match_hand_computed_values` does the
  same for `[3.0, 1.0, 1.0]`.
* `test_renormalisation_matches_hand_computed_tiny_case` — with ratio 1 and no
  KL, `P=0.5` on advantages `[4, 2, -1, -3]` gives loss `-0.5` (the mean of
  `4, -3`), `P=0.25` gives `-1.0`, the `group` normalisation gives `-0.25`, and a
  threshold of 3.5 gives `-4.0`. `test_update_scale_stays_bounded_after_pruning`
  checks the loss stays inside `max |A|` at every prune rate.
* `test_gathered_rollout_loss_matches_the_dense_objective` — the trainer's
  gathered retained-completion path equals `cppo_loss` on the dense `(B, G)`
  batch, which is the same maths written twice, and equals `grpo_loss` at rate 0.
* `test_kl_estimator_is_zero_on_identical_models_and_positive_otherwise`,
  `test_kl_penalty_scales_with_beta_and_vanishes_when_policy_is_reference` — the
  KL term is exactly 0 when the policy is the reference and enters the loss as
  `beta * KL`; `test_clip_bounds_the_positive_direction_only` pins the clip
  semantics with `ratio = 5` and `A = +-2`.
* `test_gradients_are_finite_for_grpo_and_cppo` — every parameter's gradient is
  finite and at least one is non-zero, through `mx.grad` on the real model.
* `test_all_pruned_batch_gives_zero_loss_and_zero_gradient` — a threshold that
  prunes everything gives loss 0 and a finite zero gradient rather than a NaN.
* `test_cppo_rollout_prunes_before_the_forward_passes` — the retained rows are
  exactly the gathered top-k, and the kept `|A|` in each group dominates the
  dropped ones.
* `test_budget_accounting_matches_the_allocation_rule` — sampled, retained and
  gradient-token counts equal what `b / (1-P)`, `k = floor(G(1-P))` predict.
* Plus task/tokenizer/reward contracts, sampling determinism, mask-after-`<eos>`,
  the reference model staying frozen while the policy moves, and the task
  generator refusing to ask for more pairs than exist.

## Experiment

`scripts/experiment.py`, three seeds, one shared frozen reference per seed, 400
optimizer steps per arm. Policy: the 338,304-parameter `TinyLM`; group size
`G = 8`; AdamW `lr = 3e-4`; `beta = 0.04`; clip `0.2`; sampling temperature 1.0;
greedy evaluation on 128 held-out pairs after every 50 steps. The four arms:

| arm | pruning | questions/step | sampled/step | retained/step |
|---|---|---:|---:|---:|
| `grpo` | none | 8 | 64 | 64 |
| `cppo_rate50_samples` | top-k, P = 0.5, no dynamic allocation | 8 | 64 | 32 |
| `cppo_threshold1_samples` | `\|A\| >= 1.0`, no dynamic allocation | 8 | 64 | at most 64 (mean 11.5% of sampled) |
| `cppo_rate50_dynamic` | top-k, P = 0.5, dynamic allocation | 16 | 128 | 64 |

The run takes about 5 minutes on CPU (313 s total, including the three
warm-ups). The script sets MLX's default device to the CPU because the GPU
embedding backward pass accumulates with atomics: two identical forward/backward
passes on the GPU differ by ~1.5e-8, which over hundreds of steps changes the
sampled completions and the final accuracy, while the CPU path is bit-identical
across runs. The committed log is one such run; the numbers below are copied
from it verbatim.

### Table 1 — matched sampled-completion budget (mean +- std over 3 seeds)

Same number of completions generated per step, same 400 steps, so both arms see
the same 25,600 completions.

| arm | steps | sampled | retained | gradient tokens | wall (s) | test accuracy | accuracy gain / 1000 sampled |
|---|---:|---:|---:|---:|---:|---:|---:|
| `grpo` | 400 | 25600 | 25600 | 128000 | 24.4 | 0.5755 +- 0.0670 | 0.0017 |
| `cppo_rate50_samples` | 400 | 25600 | 12800 | 64000 | 19.2 | 0.5521 +- 0.0940 | 0.0008 |
| `cppo_threshold1_samples` | 400 | 25600 | 2945 | 14727 | 15.3 | 0.5807 +- 0.0550 | 0.0019 |

### Table 2 — matched step budget with dynamic completion allocation

Same 400 steps and the same 25,600 retained completions (the same gradient
batch, so the same backward cost), but CPPO samples 16 questions per step
instead of 8, i.e. twice the completions.

| arm | questions/step | sampled | retained | wall (s) | test accuracy |
|---|---:|---:|---:|---:|---:|
| `grpo` | 8 | 25600 | 25600 | 24.4 | 0.5755 +- 0.0670 |
| `cppo_rate50_dynamic` | 16 | 51200 | 25600 | 32.9 | 0.6380 +- 0.0706 |

### Table 3 — accuracy at a matched wall-clock of 30 s

| arm | accuracy | steps reached | wall (s) |
|---|---:|---:|---:|
| `grpo` | 0.5755 | 400 | 24.4 |
| `cppo_rate50_samples` | 0.5521 | 400 | 19.2 |
| `cppo_threshold1_samples` | 0.5807 | 400 | 15.3 |
| `cppo_rate50_dynamic` | 0.6172 | 350 | 28.8 |

### Table 4 — accuracy against the sampled-completion budget (mean over seeds)

| sampled | `grpo` | `cppo_rate50_samples` | `cppo_threshold1_samples` | `cppo_rate50_dynamic` |
|---:|---|---|---|---|
| 3200 | 0.4948 +- 0.0894 | 0.4401 +- 0.0161 | 0.3203 +- 0.0776 | - |
| 6400 | 0.5469 +- 0.0338 | 0.5078 +- 0.0510 | 0.4635 +- 0.0516 | 0.5365 +- 0.0792 |
| 9600 | 0.5807 +- 0.0301 | 0.5104 +- 0.0579 | 0.4714 +- 0.0327 | - |
| 12800 | 0.5911 +- 0.0772 | 0.5026 +- 0.0761 | 0.5286 +- 0.0242 | 0.5443 +- 0.0224 |
| 16000 | 0.5859 +- 0.0628 | 0.5417 +- 0.1237 | 0.5052 +- 0.0405 | - |
| 19200 | 0.5547 +- 0.0522 | 0.5391 +- 0.1073 | 0.5677 +- 0.0224 | 0.5651 +- 0.0772 |
| 22400 | 0.5990 +- 0.0512 | 0.5651 +- 0.0579 | 0.5938 +- 0.0609 | - |
| 25600 | 0.5755 +- 0.0670 | 0.5521 +- 0.0940 | 0.5807 +- 0.0550 | 0.5964 +- 0.0479 |

The starting points were 0.5391, 0.5547 and 0.5000 for seeds 0, 1, 2, so the
overall gain over 400 RL steps is a few points at most. Per seed the final
accuracies were (`grpo`, `rate50_samples`, `threshold1`, `rate50_dynamic`):
seed 0 `0.6562, 0.6406, 0.6406, 0.7188`; seed 1 `0.5781, 0.5938, 0.5938,
0.6484`; seed 2 `0.4922, 0.4219, 0.5078, 0.5469`. Seed 2's `grpo` run actually
lost 0.8 points. The dynamic-allocation arm is the only one that is ahead of
GRPO on all three seeds, by 0.055 to 0.070.

### Table 5 — gradient-noise statistics on a fixed rollout (mean over 3 seeds)

`snr` is `||mean_b g_b|| / sqrt(mean_b ||g_b||^2)` over the questions of the
rollout, computed at the start-of-step policy with the ratio set to 1; `cos` is
the cosine similarity between the pruned step direction and the full-group one
on the same rollout, and the last column is the ratio of their norms.

| prune rate | when | snr full | snr pruned | cos(pruned, full) | \|\|g_pruned\|\| / \|\|g_full\|\| |
|---:|---|---:|---:|---:|---:|
| 0.5 | initial policy | 0.477 | 0.445 | 0.793 | 1.660 |
| 0.75 | initial policy | 0.477 | 0.449 | 0.897 | 3.156 |
| 0.5 | after 400 steps | 0.484 | 0.483 | 0.992 | 1.718 |
| 0.75 | after 400 steps | 0.484 | 0.482 | 0.959 | 3.073 |

Other measured per-run statistics (mean over seeds): mean reward 2.506 (`grpo`),
2.378 (`rate50_samples`), 2.388 (`threshold1`), 2.627 (`rate50_dynamic`) out of
a maximum of 3; fraction of groups whose eight rewards are all identical, i.e.
zero advantage, 0.454, 0.459, 0.438 and 0.542; mean gradient norm ~0.95-0.98
with a per-step standard deviation of 0.08-0.13; sampled completions per
retained completion 1.00, 2.00, 9.76 and 2.00.

### What these numbers mean, and what they do not

* Dropping half of the completions before the model forwards (`rate50_samples`)
  changed the final accuracy by -0.023 against GRPO while halving the gradient
  tokens (128,000 -> 64,000) and cutting wall time by 21%. The threshold arm cut
  the gradient tokens by 8.7x (down to 11.5% of sampled completions) for +0.005
  accuracy. Both differences are far inside the +-0.055 to +-0.094 seed spread,
  so the honest statement is "no measurable accuracy loss at this scale", not
  "CPPO is as accurate as GRPO".
* At a matched step and gradient batch (table 2), spending the freed budget on
  twice as many questions is the only change that helps on every seed
  (+0.063 +- 0.071 mean, at twice the sampled completions and 1.35x the wall
  clock). It is not a free win: per generated completion it is slightly behind
  GRPO (table 4).
* Gradient noise barely moves: with `P = 0.5` the pruned direction is 0.79
  cosine-similar to the full-group direction at the warmed-up policy and 0.99
  after training, with the same signal-to-noise ratio, while the norm ratio
  grows with the prune rate because the 1/k normalisation amplifies the
  survivors.
* **This is not the paper's speedup, and it is not a wall-clock speedup at
  all.** The paper counts the three big forward passes per completion on a 1.5B
  model, where pruning removes most of the cost. Here generation dominates: at a
  matched gradient batch the dynamic arm generates twice as many tokens and
  takes 1.35x the wall clock. The analogue of the paper's cost model in this
  repo is the `gradient tokens` column, where `P = 0.5` does exactly half the
  work. The paper's reported 7.98x (GSM8K) and 3.48x (MATH) speedups are not
  reproduced or tested here.
* The reward is close to saturated by the end (mean 2.5/3, and 44-54% of groups
  are already all-correct), because RL is trained on the same 512 pairs the
  warm-up used. Most of the remaining gradient is the KL pull, which is what
  makes the pruned and full directions so similar after training.

## Where this may differ from the paper

* The paper gives two retention rules and uses them in different places: the
  threshold `|A_i| >= gamma` in the objective (eq. 7, divisor `G`) and the top-k
  with `k = floor(G (1-P))` in the experiments (eq. 8-10, divisor `k`). I
  implement both. The top-k rate with the 1/k divisor is the default because
  that is what the paper's ablations and Algorithm 1 use; the threshold keeps
  eq. 7's `G` divisor available as `normalisation="group"`.
* Dynamic allocation is implemented where the paper's Appendix B says it
  happens ("we perform completion allocation before completion pruning ... sample
  a batch of `b / (1-p)` questions and then sample `G` completions for each"),
  and not as the main-text Figure 3 describes it (prune first, then backfill
  pruned slots from new questions). The paper says the two are equivalent and
  that the appendix order is an implementation choice; the appendix order is
  much simpler to account for.
* Algorithm 1 runs `mu` inner gradient epochs per rollout. I take one gradient
  step per rollout (`mu = 1`), which is the common GRPO setting and keeps the
  toy cheap. The importance ratio is therefore always taken against the policy
  that generated the completions, exactly as in the paper, but the off-policy
  behaviour of `mu > 1` is untested here.
* The paper's reward has a "matches after regular parsing" tier for free-form
  text. This task has no text to parse, so I read that tier as "the right number
  in the wrong digit order" (reward 1.5). Something had to fill that slot; a
  different reading would give different groups a different reward spread.
* Advantages use the population standard deviation (not `ddof=1`) with an
  `eps = 1e-4` in the denominator. The paper's eq. 3 does not say which
  standard deviation; the paper's own code is not part of what I read.
* The supervised warm-up and the "warm up until accuracy first reaches 0.5"
  stopping rule are mine, not the paper's, and the paper's models start from an
  instruction-tuned checkpoint instead.
* The toy uses a character vocabulary, a 338k-parameter transformer, no vLLM, no
  distributed pruning, and no `mu` epochs. Long completions, a learned reward
  ambiguity, and the bucket effect that motivates the unified single/multi-GPU
  top-k rule (paper section 3.3) do not exist here: every group has exactly the
  same completion length, so the "keep exactly k per question" rule is what
  bounds the batch.
* Seeding: the experiment is run on MLX's CPU device because the GPU embedding
  backward is not bit-reproducible (measured ~1.5e-8 per-step gradient
  difference). Within that, everything is seeded and the committed numbers
  reproduce exactly.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/experiment.py
```

The experiment writes `results/experiment_log.txt` and
`results/experiment.json` and takes about five minutes. `--device gpu` switches
MLX's default device; `--help` lists the rest (seeds, step budget, group size,
learning rate, warm-up target).

"""GRPO vs CPPO on a synthetic addition task, at matched sample and step budgets.

Run with the repository's interpreter:

    /Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/experiment.py

What it does, per seed:

1. Supervised warm-up of the tiny policy on the task's own targets.  Both arms
   start from this model, and it is also the frozen reference.
2. GRPO and three CPPO configurations are trained for the same number of
   optimizer steps, each logging its own completion budget (sample count,
   retained count, gradient tokens, wall time) and an accuracy curve.
3. Gradient-noise statistics are measured on a fixed rollout before and after
   training: the per-question gradient signal-to-noise ratio for the full group
   and for the pruned group, and the cosine similarity between the two step
   directions.

Results are written to ``results/experiment_log.txt`` and
``results/experiment.json`` and printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cppo import (
    AdditionTask,
    TinyLM,
    TrainConfig,
    clone_model,
    collect_rollout,
    dynamic_question_count,
    evaluate,
    gradient_stats,
    retained_per_question,
    train_rl,
    train_sft_to_accuracy,
)

ARMS = ["grpo", "cppo_rate50_samples", "cppo_threshold1_samples", "cppo_rate50_dynamic"]
MATCHED_SAMPLES_ARMS = ["grpo", "cppo_rate50_samples", "cppo_threshold1_samples"]
MATCHED_STEPS_ARMS = ["grpo", "cppo_rate50_dynamic"]


class Tee:
    """Print to stdout and to the results log at the same time."""

    def __init__(self, path: Path):
        self.file = path.open("w")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def arm_config(name: str, args) -> TrainConfig:
    common = dict(
        group_size=args.group_size,
        questions=args.questions,
        lr=args.lr,
        beta=args.beta,
        clip_eps=args.clip,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        sft_batch=args.sft_batch,
        seed=args.data_seed,
    )
    if name == "grpo":
        return TrainConfig(**common)
    if name == "cppo_rate50_samples":
        return TrainConfig(**common, prune_rate=0.5, dynamic_allocation=False)
    if name == "cppo_threshold1_samples":
        return TrainConfig(**common, prune_threshold=1.0, dynamic_allocation=False)
    if name == "cppo_rate50_dynamic":
        return TrainConfig(**common, prune_rate=0.5, dynamic_allocation=True)
    raise ValueError(name)


def arm_description(name: str, cfg: TrainConfig) -> dict:
    return {
        "prune_rate": cfg.prune_rate,
        "prune_threshold": cfg.prune_threshold,
        "dynamic_allocation": cfg.dynamic_allocation,
        "questions_per_step": cfg.questions_per_step(),
        "sampled_per_step": cfg.samples_per_step(),
        "retained_per_step": (
            cfg.questions_per_step() * retained_per_question(cfg.group_size, cfg.prune_rate)
            if cfg.prune_rate > 0.0
            else "at most " + str(cfg.samples_per_step()) if cfg.prune_threshold > 0.0 else cfg.samples_per_step()
        ),
    }


def accuracy_per_1000_sampled(initial: float, final: float, sampled: int) -> float:
    if sampled == 0:
        return 0.0
    return (final - initial) / (sampled / 1000.0)


def curve_at_or_before(curve, key: str, limit: float):
    points = [p for p in curve if p[key] <= limit]
    return points[-1] if points else None


def summarise_run(initial_accuracy: float, final_accuracy: float, budget) -> dict:
    out = budget.as_dict()
    out["initial_accuracy"] = initial_accuracy
    out["final_accuracy"] = final_accuracy
    out["accuracy_per_1000_sampled"] = accuracy_per_1000_sampled(
        initial_accuracy, final_accuracy, budget.sampled_completions
    )
    out["sampled_per_retained"] = budget.sampled_completions / max(budget.retained_completions, 1)
    out["seconds_per_step"] = budget.wall_time / max(budget.steps, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--wall-clock-match", type=float, default=30.0)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--sft-target", type=float, default=0.5)
    parser.add_argument("--sft-max-steps", type=int, default=1500)
    parser.add_argument("--sft-check-every", type=int, default=50)
    parser.add_argument("--sft-batch", type=int, default=64)
    parser.add_argument("--task-digits", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log = Tee(args.out / "experiment_log.txt")
    sys.stdout = log

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    started = time.time()

    print("CPPO toy experiment: GRPO vs completion pruning on synthetic addition")
    print("paper: CPPO: Accelerating the Training of Group Relative Policy")
    print("       Optimization-Based Reasoning Models (arXiv:2503.22342)")
    print(f"date: {time.strftime('%Y-%m-%d')}")
    print(f"device: {mx.default_device()} (cpu is bit-reproducible; gpu embedding backward is not)")
    print(f"seeds: {args.seeds}  steps per arm: {args.steps}  eval every {args.eval_every}")
    print(
        f"task: {args.task_digits}-digit addition, answer written least significant digit first; "
        f"{args.train_size} train / {args.test_size} test pairs"
    )
    print(
        f"policy: 3 layers, 96 wide, 4 heads ({338304} params); reference = the warmed-up policy, frozen"
    )
    print(
        f"optimiser: AdamW lr={args.lr}, clip={args.clip}, beta={args.beta}, temperature={args.temperature}, "
        f"group size G={args.group_size}"
    )
    print(
        f"warm-up: supervised until greedy test accuracy first reaches {args.sft_target} "
        f"(at most {args.sft_max_steps} steps, checked every {args.sft_check_every})"
    )
    print()

    arm_descriptions = {name: arm_description(name, arm_config(name, args)) for name in ARMS}
    print("step cost per arm:")
    for name in ARMS:
        d = arm_descriptions[name]
        print(
            f"  {name:26s} questions/step {d['questions_per_step']:3d}  "
            f"sampled/step {d['sampled_per_step']:4d}  retained/step {str(d['retained_per_step']):>10s}"
        )
    print()

    results: dict = {
        "paper": {
            "title": "CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models",
            "arxiv": "2503.22342",
        },
        "device": str(mx.default_device()),
        "config": vars(args) | {"out": str(args.out)},
        "arm_descriptions": arm_descriptions,
        "runs": {},
        "gradient_noise": {},
        "init_accuracy": {},
    }

    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        task = AdditionTask(
            digits=args.task_digits,
            n_train=args.train_size,
            n_test=args.test_size,
            seed=args.data_seed,
        )
        mx.random.seed(seed)
        base = TinyLM(max_len=24)
        t0 = time.time()
        sft_steps, init_accuracy = train_sft_to_accuracy(
            base,
            task,
            TrainConfig(sft_steps=0, sft_batch=args.sft_batch, seed=seed),
            task.test,
            target=args.sft_target,
            max_steps=args.sft_max_steps,
            check_every=args.sft_check_every,
        )
        print(
            f"  sft: {sft_steps} steps in {time.time() - t0:.1f}s until greedy test accuracy "
            f"{init_accuracy:.4f} (target {args.sft_target})",
            flush=True,
        )
        reference = clone_model(base)
        results["init_accuracy"][str(seed)] = init_accuracy
        results["sft_steps"] = results.get("sft_steps", {}) | {str(seed): sft_steps}
        results["runs"][str(seed)] = {}

        # A fixed rollout, used for the gradient-noise statistics.
        stat_cfg = TrainConfig(group_size=args.group_size, questions=4, seed=seed)
        stat_questions = task.train[:4]
        stats_init = {}
        for rate in (0.5, 0.75):
            rate_cfg = TrainConfig(
                group_size=args.group_size, questions=4, prune_rate=rate, seed=seed
            )
            rollout = collect_rollout(base, reference, stat_questions, rate_cfg, mx.random.key(999))
            stats_init[str(rate)] = gradient_stats(base, reference, rollout, rate_cfg)
        results["gradient_noise"][str(seed)] = {"init": stats_init}

        trained: dict = {}
        for name in ARMS:
            cfg = arm_config(name, args)
            model = clone_model(base)
            t0 = time.time()
            budget = train_rl(
                model,
                reference,
                task,
                cfg,
                steps=args.steps,
                eval_examples=task.test,
                eval_every=args.eval_every,
                key_offset=0,
            )
            final_accuracy = evaluate(model, task.test)
            summary = summarise_run(init_accuracy, final_accuracy, budget)
            results["runs"][str(seed)][name] = summary
            trained[name] = model
            print(
                f"  {name:26s} steps {budget.steps:4d}  sampled {budget.sampled_completions:7d}  "
                f"retained {budget.retained_completions:7d}  wall {budget.wall_time:6.1f}s  "
                f"acc {final_accuracy:.4f}  gain/1000 sampled {summary['accuracy_per_1000_sampled']:.4f}  "
                f"last-loss {np.mean(budget.losses[-20:]):.4f}",
                flush=True,
            )

        # Gradient statistics after training, on the same questions, using the
        # policy that spent the freed compute on extra questions.
        trained_model = trained["cppo_rate50_dynamic"]
        print("  gradient-noise statistics (init -> final policy):", flush=True)
        stats_final = {}
        kept = {}
        for rate in (0.5, 0.75):
            rate_cfg = TrainConfig(group_size=args.group_size, questions=4, prune_rate=rate, seed=seed)
            rollout = collect_rollout(trained_model, reference, stat_questions, rate_cfg, mx.random.key(999))
            stats_final[str(rate)] = gradient_stats(trained_model, reference, rollout, rate_cfg)
        results["gradient_noise"][str(seed)]["final"] = stats_final
        for rate in ("0.5", "0.75"):
            a, b = stats_init[rate], stats_final[rate]
            print(
                f"    P={rate}: snr full {a['full_snr']:.3f}->{b['full_snr']:.3f}  "
                f"snr pruned {a['pruned_snr']:.3f}->{b['pruned_snr']:.3f}  "
                f"cos(pruned, full) {a['cosine']:.3f}->{b['cosine']:.3f}  "
                f"||g_pruned||/||g_full|| {a['norm_ratio']:.3f}->{b['norm_ratio']:.3f}",
                flush=True,
            )
        del trained, kept, trained_model
        print(flush=True)
    sys.stdout = log.stdout

    # ---------------------------------------------------------------- tables
    seeds = [str(s) for s in args.seeds]

    def mean_of(arm: str, field: str):
        values = [results["runs"][s][arm][field] for s in seeds]
        return float(np.mean(values)), float(np.std(values))

    tables: dict = {"matched_samples": [], "matched_steps": [], "wall_clock": [], "curves": {}}
    print("table 1: matched sampled-completion budget (same completions generated per step)")
    print(f"{'arm':26s} {'steps':>6s} {'sampled':>8s} {'retained':>9s} {'grad-tok':>9s} {'wall s':>7s} "
          f"{'acc':>14s} {'acc/1k sampled':>15s}")
    for arm in MATCHED_SAMPLES_ARMS:
        acc, acc_std = mean_of(arm, "final_accuracy")
        wall, _ = mean_of(arm, "wall_time")
        sampled, _ = mean_of(arm, "sampled_completions")
        retained, _ = mean_of(arm, "retained_completions")
        gtokens, _ = mean_of(arm, "gradient_tokens")
        rate, _ = mean_of(arm, "accuracy_per_1000_sampled")
        row = {
            "arm": arm,
            "steps": int(mean_of(arm, "steps")[0]),
            "sampled_completions": int(sampled),
            "retained_completions": int(retained),
            "gradient_tokens": int(gtokens),
            "wall_time": wall,
            "accuracy_mean": acc,
            "accuracy_std": acc_std,
            "accuracy_per_1000_sampled": rate,
        }
        tables["matched_samples"].append(row)
        print(f"{arm:26s} {row['steps']:6d} {row['sampled_completions']:8d} {row['retained_completions']:9d} "
              f"{row['gradient_tokens']:9d} {wall:7.1f} {acc:8.4f} +- {acc_std:.4f} {rate:15.4f}")

    print()
    print("table 2: matched step budget and matched gradient batch (dynamic completion allocation)")
    print(f"{'arm':26s} {'q/step':>6s} {'sampled':>8s} {'retained':>9s} {'wall s':>7s} {'acc':>14s}")
    for arm in MATCHED_STEPS_ARMS:
        acc, acc_std = mean_of(arm, "final_accuracy")
        wall, _ = mean_of(arm, "wall_time")
        sampled, _ = mean_of(arm, "sampled_completions")
        retained, _ = mean_of(arm, "retained_completions")
        row = {
            "arm": arm,
            "questions_per_step": arm_descriptions[arm]["questions_per_step"],
            "sampled_completions": int(sampled),
            "retained_completions": int(retained),
            "wall_time": wall,
            "accuracy_mean": acc,
            "accuracy_std": acc_std,
        }
        tables["matched_steps"].append(row)
        print(f"{arm:26s} {row['questions_per_step']:6d} {row['sampled_completions']:8d} "
              f"{row['retained_completions']:9d} {wall:7.1f} {acc:8.4f} +- {acc_std:.4f}")

    print()
    limit = args.wall_clock_match
    print(f"table 3: accuracy at a matched wall-clock of {limit:.0f}s per seed")
    print(f"{'arm':26s} {'acc @':>8s} {'steps @':>8s} {'wall @':>7s}")
    for arm in ARMS:
        accs, steps_at, walls = [], [], []
        for s in seeds:
            point = curve_at_or_before(results["runs"][s][arm]["curve"], "wall_time", limit)
            if point:
                accs.append(point["accuracy"])
                steps_at.append(point["step"])
                walls.append(point["wall_time"])
        if accs:
            row = {
                "arm": arm,
                "wall_limit": limit,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "steps_mean": float(np.mean(steps_at)),
                "wall_time_mean": float(np.mean(walls)),
            }
            tables["wall_clock"].append(row)
            print(f"{arm:26s} {row['accuracy_mean']:8.4f} {row['steps_mean']:8.1f} {row['wall_time_mean']:7.1f}")

    print()
    print("table 4: accuracy along the sampled-completion budget (mean over seeds)")
    header = "sampled".ljust(9) + "".join(arm[:20].rjust(22) for arm in ARMS)
    print(header)
    budgets = sorted({p["sampled_completions"] for s in seeds for p in results["runs"][s]["grpo"]["curve"]})
    for budget in budgets:
        cells = ""
        for arm in ARMS:
            values = [
                point["accuracy"]
                for s in seeds
                for point in results["runs"][s][arm]["curve"]
                if point["sampled_completions"] == budget
            ]
            cells += (f"{np.mean(values):.4f} +- {np.std(values):.4f}" if values else "-").rjust(22)
        print(str(budget).ljust(9) + cells)
        tables["curves"][str(budget)] = {
            arm: [
                point["accuracy"]
                for s in seeds
                for point in results["runs"][s][arm]["curve"]
                if point["sampled_completions"] == budget
            ]
            for arm in ARMS
        }

    print()
    print("table 5: gradient-noise statistics on a fixed rollout (mean over seeds)")
    print(f"{'prune rate':>10s} {'when':>6s} {'snr full':>9s} {'snr pruned':>11s} {'cos(pruned,full)':>17s} {'||gp||/||gf||':>14s}")
    tables["gradient_noise"] = []
    for when in ("init", "final"):
        for rate in ("0.5", "0.75"):
            rows = [results["gradient_noise"][s][when][rate] for s in seeds]
            row = {
                "when": when,
                "prune_rate": float(rate),
                "snr_full": float(np.mean([r["full_snr"] for r in rows])),
                "snr_pruned": float(np.mean([r["pruned_snr"] for r in rows])),
                "cosine": float(np.mean([r["cosine"] for r in rows])),
                "norm_ratio": float(np.mean([r["norm_ratio"] for r in rows])),
            }
            tables["gradient_noise"].append(row)
            print(f"{rate:>10s} {when:>6s} {row['snr_full']:9.3f} {row['snr_pruned']:11.3f} "
                  f"{row['cosine']:17.3f} {row['norm_ratio']:14.3f}")

    results.pop("_models", None)
    results["tables"] = tables
    results["total_wall_time"] = time.time() - started
    print()
    print(f"total wall time {results['total_wall_time']:.1f}s")

    with (args.out / "experiment.json").open("w") as handle:
        json.dump(results, handle, indent=2, default=float)
    print(f"wrote {args.out / 'experiment.json'} and {args.out / 'experiment_log.txt'}")
    sys.stdout = log.stdout
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

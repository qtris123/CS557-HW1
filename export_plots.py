"""Export six report figures and a CSV from the actual TensorBoard scalars."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from run_experiments import is_complete, plan, run_dir

METRICS = [
    ("train-dqn/avg_loss", "Mean training Huber loss"),
    ("train-dqn/episode_reward", "Training episode return"),
    ("train-dqn/eval_mean_reward", "Greedy evaluation return"),
]
AGENTS = [("p1_dqn", "DQN + uniform"), ("p1_double_per", "Double DQN + PER"),
          ("p2a_double", "Double DQN + uniform"), ("p2a_dueling_per", "Dueling DQN + PER")]


def moving_mean(y, window):
    y = np.asarray(y, dtype=float)
    sums = np.concatenate([[0.0], np.cumsum(y)])
    starts = np.maximum(0, np.arange(len(y)) + 1 - window)
    return (sums[1:] - sums[starts]) / (np.arange(len(y)) + 1 - starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=10, help="Causal episode window for training curves")
    parser.add_argument("--allow_partial", action="store_true", help="For debugging only")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")
    root = Path(args.output_root)
    destination = root / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    data, rows = {}, []
    for _, env, _, name, _, _ in plan():
        directory = run_dir(root, env, name, args.seed)
        if not directory.exists():
            if args.allow_partial:
                continue
            raise SystemExit(f"Missing run: {directory}")
        if not args.allow_partial and not is_complete(directory):
            raise SystemExit(f"Incomplete run: {directory}")
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        available = events.Tags().get("scalars", [])
        series = {tag: [(p.step, p.value) for p in events.Scalars(tag)]
                  for tag, _ in METRICS if tag in available}
        data[env, name] = series
        config = json.loads((directory / "config.json").read_text())
        train = series.get("train-dqn/episode_reward", [])
        evaluation = series.get("train-dqn/eval_mean_reward", [])
        rows.append(dict(env=env, run=name, seed=args.seed, lr=config["lr"],
                         epsilon_decay=config["epsilon_decay"],
                         completed_episodes=train[-1][0] if train else 0,
                         final_eval_mean_reward=evaluation[-1][1] if evaluation else "",
                         best_eval_mean_reward=max(y for _, y in evaluation) if evaluation else "",
                         last_10_train_mean_reward=float(np.mean([y for _, y in train[-10:]])) if train else ""))
    groups = []
    for env in ("LunarLander-v2", "CarRacing-v2"):
        groups.extend([(f"part1_{env}", env, AGENTS[:2]), (f"part2A_{env}", env, AGENTS)])
    groups.extend([
        ("part2B_lr", "LunarLander-v2", [("p2b_lr_1em5", "lr=1e-5"), ("p1_double_per", "lr=1e-4"), ("p2b_lr_1em3", "lr=1e-3")]),
        ("part2B_epsilon_decay", "LunarLander-v2", [("p2b_decay_5000", "decay=5000"), ("p1_double_per", "decay=15000"), ("p2b_decay_50000", "decay=50000")]),
    ])
    for title, env, selected in groups:
        if not any((env, name) in data for name, _ in selected):
            continue
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        for ax, (tag, ylabel) in zip(axes, METRICS):
            for name, label in selected:
                points = data.get((env, name), {}).get(tag, [])
                if not points:
                    continue
                x, y = map(np.asarray, zip(*points))
                if not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite scalar: {env}/{name}/{tag}")
                if tag != "train-dqn/eval_mean_reward":
                    y = moving_mean(y, args.smooth)
                marker = "o" if len(x) == 1 or tag == "train-dqn/eval_mean_reward" else None
                ax.plot(x, y, label=label, linewidth=1.5, marker=marker, markersize=3)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel("Training episode")
        heading = "Part 1" if title.startswith("part1_") else "Part 2A" if title.startswith("part2A_") else "Part 2B: learning rate" if title == "part2B_lr" else "Part 2B: epsilon decay"
        fig.suptitle(f"{heading} - {env}\nSeed {args.seed}; training: trailing mean of {args.smooth}; evaluation: mean of 3 episodes, unsmoothed", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(destination / f"{title}.png", dpi=180)
        plt.close(fig)
    if rows:
        with (destination / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved figures and summary: {destination.resolve()}")


if __name__ == "__main__":
    main()

"""Print or execute the 12-run HW1 plan. Existing completed runs are reused."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = dict(lr=1e-4, batch_size=64, save_interval=20, hidden_size=128,
            replay_size=10000, epsilon_start=0.9, epsilon_end=0.05,
            epsilon_decay=15000, gamma=0.99, target_update=2, tau=0.005,
            alpha=0.6, beta=0.4)
VARIANTS = [
    ("p1", "p1_dqn", []),
    ("p1", "p1_double_per", ["--use_double_dqn", "--use_per"]),
    ("p2a", "p2a_double", ["--use_double_dqn"]),
    ("p2a", "p2a_dueling_per", ["--use_dueling_dqn", "--use_per"]),
]


def run_dir(output_root, env, name, seed):
    return Path(output_root) / "runs" / env.lower().replace("-", "_") / f"{name}_seed{seed}"


def is_complete(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    config_path = Path(directory) / "config.json"
    if not config_path.is_file() or not (Path(directory) / "completed.json").is_file():
        return False
    config = json.loads(config_path.read_text())
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    tags = events.Tags().get("scalars", [])
    for tag in ("train-dqn/episode_reward", "train-dqn/eval_mean_reward"):
        if tag not in tags or max(p.step for p in events.Scalars(tag)) < config["episodes"]:
            return False
    return True


def plan(lunar_episodes=500, car_episodes=100):
    jobs = []
    for env, episodes in [("LunarLander-v2", lunar_episodes), ("CarRacing-v2", car_episodes)]:
        for part, name, flags in VARIANTS:
            jobs.append((part, env, episodes, name, flags, {}))
    # The p1_double_per Lunar baseline is the central value in both sweeps.
    for name, changes in [
        ("p2b_lr_1em5", {"lr": 1e-5}),
        ("p2b_lr_1em3", {"lr": 1e-3}),
        ("p2b_decay_5000", {"epsilon_decay": 5000}),
        ("p2b_decay_50000", {"epsilon_decay": 50000}),
    ]:
        jobs.append(("p2b", "LunarLander-v2", lunar_episodes, name,
                     ["--use_double_dqn", "--use_per"], changes))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only print commands")
    parser.add_argument("--part", choices=["all", "p1", "p2a", "p2b"], default="all")
    parser.add_argument("--env", choices=["all", "LunarLander-v2", "CarRacing-v2"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default=".")
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--lunar_episodes", type=int, default=500)
    parser.add_argument("--car_episodes", type=int, default=100)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    jobs = [j for j in plan(args.lunar_episodes, args.car_episodes)
            if (args.part == "all" or j[0] == args.part) and
            (args.env == "all" or j[1] == args.env)]
    if not jobs:
        parser.error("No runs match this part/environment combination")
    for part, env, episodes, name, flags, changes in jobs:
        settings = dict(BASE, **changes)
        cmd = [sys.executable, str(ROOT / "main.py"), "--env", env,
               "--num_epochs", str(episodes), "--seed", str(args.seed),
               "--run_name", name, "--output_root", str(output_root)]
        for key, value in settings.items():
            cmd.extend([f"--{key}", str(value)])
        cmd.extend(flags)
        if not args.videos:
            cmd.append("--no_video")
        print(f"[{part}] {shlex.join(cmd)}", flush=True)
        if args.execute:
            directory = run_dir(output_root, env, name, args.seed)
            if directory.exists():
                config_path = directory / "config.json"
                if config_path.is_file():
                    old = json.loads(config_path.read_text())
                    expected = dict(settings, episodes=episodes, env=env, seed=args.seed,
                                    use_double_dqn="--use_double_dqn" in flags,
                                    dueling_dqn="--use_dueling_dqn" in flags,
                                    use_per="--use_per" in flags,
                                    record_video=args.videos)
                    if any(old.get(key) != value for key, value in expected.items()):
                        raise SystemExit(f"Existing run has different settings: {directory}")
                if is_complete(directory):
                    print(f"Reusing completed run: {directory}", flush=True)
                    continue
                raise SystemExit(f"Incomplete run exists: {directory}. Move it aside before retrying.")
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

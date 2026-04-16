import re
import csv
import math
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from train import make_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all Connect4 checkpoints vs uniform-random-legal opponent."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "maskable_ppo"],
        help="Algorithm used in checkpoint filenames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training seed used in checkpoint filenames, e.g. seed0.",
    )
    # parser.add_argument(
    #     "--eval-seed",
    #     type=int,
    #     default=0,
    #     help="Evaluation seed used to initialize env RNG.",
    # )
    # parser.add_argument(
    #     "--n-games",
    #     type=int,
    #     default=1000,
    #     help="Number of evaluation games per checkpoint.",
    # )
    # parser.add_argument(
    #     "--models-dir",
    #     type=str,
    #     default="checkpoints",
    #     help="Directory containing checkpoint .zip files.",
    # )
    return parser.parse_args()


def extract_step(model_path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps\.zip$", model_path.name)
    if match:
        return int(match.group(1))
    return None


def find_model_files(models_dir: str, algo: str, seed: int) -> list[tuple[int, Path]]:
    models_path = Path(models_dir)
    pattern = f"{algo}_connect4_seed{seed}_*_steps.zip"

    matched = []
    for path in models_path.glob(pattern):
        step = extract_step(path)
        if step is not None:
            matched.append((step, path))

    matched.sort(key=lambda x: x[0])
    return matched


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games == 0:
        return 0.0, 0.0

    phat = wins / games
    denom = 1.0 + (z * z) / games
    center = (phat + (z * z) / (2.0 * games)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) / games) + ((z * z) / (4.0 * games * games)))
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def load_model(model_path: str, algo: str, vec_env: DummyVecEnv):
    if algo == "maskable_ppo":
        return MaskablePPO.load(model_path, env=vec_env)
    if algo == "ppo":
        return PPO.load(model_path, env=vec_env)
    raise ValueError(f"Unsupported algo: {algo}")


def evaluate(model, vec_env: DummyVecEnv, algo: str, n_games: int) -> tuple[int, int]:
    wins = 0
    games = 0

    obs = vec_env.reset()

    while games < n_games:
        if algo == "maskable_ppo":
            action_masks = get_action_masks(vec_env)
            actions, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
        else:
            actions, _ = model.predict(
                obs,
                deterministic=True,
            )

        obs, rewards, dones, infos = vec_env.step(actions)

        if dones[0]:
            games += 1
            terminal_reward = float(rewards[0])

            if terminal_reward > 0.0:
                wins += 1

    return wins, games


def save_rows_csv(rows: list[dict], csv_path: str):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "wins",
        "games",
        "win_rate",
        "ci_low",
        "ci_high",
        "algo",
        "seed",
    ]

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    model_files = find_model_files(
        models_dir="checkpoints",
        algo=args.algo,
        seed=args.seed,
    )

    if not model_files:
        raise FileNotFoundError(
            f"No checkpoint files found for pattern: "
            f"{args.algo}_connect4_seed{args.seed}_*_steps.zip in checkpoints"
        )

    rows = []

    print(f"Found {len(model_files)} checkpoint(s).")

    for step, model_path in model_files:
        print(f"Evaluating step={step} | file={model_path.name}")

        vec_env = DummyVecEnv([make_env(args.seed, args.algo)])
        model = load_model(str(model_path), args.algo, vec_env)

        wins, games = evaluate(
            model=model,
            vec_env=vec_env,
            algo=args.algo,
            n_games=1000,
        )

        win_rate = wins / games if games > 0 else 0.0
        ci_low, ci_high = wilson_interval(wins, games)

        rows.append({
            "step": step,
            "wins": wins,
            "games": games,
            "win_rate": f"{win_rate:.6f}",
            "ci_low": f"{ci_low:.6f}",
            "ci_high": f"{ci_high:.6f}",
            "algo": args.algo,
            "seed": args.seed,
        })

        print(
            f"  wins={wins}, games={games}, "
            f"win_rate={win_rate:.6f}, ci_95=[{ci_low:.6f}, {ci_high:.6f}]"
        )

        vec_env.close()

    csv_out = Path("results") / f"eval_{args.algo}_seed{args.seed}.csv"
    save_rows_csv(rows, str(csv_out))

    print("\nDone.")
    print(f"Saved {len(rows)} row(s) to: {csv_out}")

if __name__ == "__main__":
    main()
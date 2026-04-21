import re
import csv
import math
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from __future__ import annotations

from scripts.train import make_env


MODEL_RE = re.compile(
    r"^(?P<algo>.+?)_connect4_seed(?P<seed>\d+)_(?P<step>\d+)_steps\.zip$"
)


def parse_model_file(model_path: Path) -> tuple[str, int, int] | None:
    match = MODEL_RE.match(model_path.name)
    if not match:
        return None

    algo = match.group("algo")
    seed = int(match.group("seed"))
    step = int(match.group("step"))
    return algo, step, seed


def find_model_files(models_dir: str) -> list[tuple[str, int, int, Path]]:
    models_path = Path(models_dir)
    matched: list[tuple[str, int, int, Path]] = []

    for path in models_path.glob("*.zip"):
        parsed = parse_model_file(path)
        if parsed is None:
            continue

        algo, step, seed = parsed
        matched.append((algo, step, seed, path))

    matched.sort(key=lambda x: (x[0], x[1], x[2]))
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
    model_files = find_model_files(models_dir="checkpoints")

    if not model_files:
        raise FileNotFoundError("No models files found")

    print(f"Found {len(model_files)} models.")

    rows = []

    for algo, step, seed, model_path in model_files:
        print(f"Evaluating step={step} | file={model_path.name}")

        vec_env = DummyVecEnv([make_env(seed*1000, algo)])
        model = load_model(str(model_path), algo, vec_env)

        wins, games = evaluate(
            model=model,
            vec_env=vec_env,
            algo=algo,
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
            "algo": algo,
            "seed": seed,
        })

        print(
            f"  wins={wins}, games={games}, "
            f"win_rate={win_rate:.6f}, ci_95=[{ci_low:.6f}, {ci_high:.6f}]"
        )

        vec_env.close()

    csv_out = Path("results") / f"eval_{algo}_seed{seed}.csv"
    save_rows_csv(rows, str(csv_out))

    print("\nDone.")
    print(f"Saved {len(rows)} row(s) to: {csv_out}")

if __name__ == "__main__":
    main()
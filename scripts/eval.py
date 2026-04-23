import re
import csv
import math
from collections import defaultdict
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import argparse
from scripts.train import make_env, read_config

MODEL_RE = re.compile(
    r"^(?P<algo>.+?)_connect4_seed(?P<seed>\d+)_(?P<step>\d+)_steps\.zip$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Ścieżka do pliku YAML z konfiguracją",
    )
    return parser.parse_args()


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


    if not matched:
        raise FileNotFoundError("No models files found")

    print(f"Found {len(matched)} models.")

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


def run_n_games(model, vec_env: DummyVecEnv, algo: str, n_games: int) -> tuple[int, int]:
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


def run_model(model_path: Path, algo: str, run_param) -> tuple[int, int]:
    vec_env = DummyVecEnv([make_env(run_param["eval_seed"], algo)])
    model = load_model(str(model_path), algo, vec_env)

    wins, games = run_n_games(
        model=model,
        vec_env=vec_env,
        algo=algo,
        n_games=run_param["n_games"],
    )

    vec_env.close()

    return wins, games


def run_models(model_paths: list[tuple[str, int, int, Path]], run_param) -> list[dict[str, int]]:
    results: list[dict[str, int]] = []
    for algo, step, seed, model_path in model_paths:
        print(f"Evaluating algo={algo} | step={step} | seed={seed} | file={model_path.name}")

        wins, games = run_model(
            model_path=model_path,
            algo=algo,
            run_param=run_param
        )

        results.append({
            "algo": algo,
            "step": step,
            "seed": seed,
            "wins": wins,
            "games": games,
        })

        print(f"wins={wins}, games={games}")

    return results


# def calculate_results(results: dict[tuple[str, int, int], dict[str, int]]) -> dict[tuple[str, int], dict[str, float]]:
#     grouped = defaultdict(lambda: {"wins": 0, "games": 0})
#
#     for (algo, step, _seed), values in results.items():
#         grouped[(algo, step)]["wins"] += values["wins"]
#         grouped[(algo, step)]["games"] += values["games"]
#
#     calculated_results = {}
#
#     for (algo, step), values in grouped.items():
#         win_rate = values["wins"] / values["games"] if values["games"] > 0 else 0.0
#         ci_low, ci_high = wilson_interval(values["wins"], values["games"])
#         calculated_results[(algo, step)] ={
#             "win_rate": win_rate,
#             "ci_low": ci_low,
#             "ci_high": ci_high,
#         }
#
#     return calculated_results


def save_results(results: list[dict[str, int]], results_config) -> None:
    results_path = Path(results_config["results_dir"]) / results_config["results_file"]

    fieldnames = list(results[0].keys())

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    config = read_config(args.config)

    # wyszukanie wytrenowanych modeli
    model_files = find_model_files(config["models_dir"])

    # ewaluacja modeli
    results = run_models(model_files, config["run_param"])

    # zapis wynikow do csv
    save_results(results, config["results"])


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from scripts.train import read_config
from scripts.eval import Result
from scripts.summarize import load_results, group_results_by_algo_and_seed, GroupedResults


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/plot.yaml"),
        help="Ścieżka do pliku config.",
    )
    return parser.parse_args()


def plot_results(grouped_results: GroupedResults, config) -> None:
    plt.figure(figsize=(12, 7))

    for (algo, seed), rows in sorted(grouped_results.items()):
        if algo in config["show_algo"] and seed in config["show_seeds"]:
            rows = sorted(rows, key=lambda row: int(row["step"]))

            steps = [int(row["step"]) for row in rows]
            win_rates = [float(row["win_rate"]) for row in rows]
            ci_lows = [float(row["ci_low"]) for row in rows]
            ci_highs = [float(row["ci_high"]) for row in rows]

            if bool(config["show_wilson_interval"]):
                y_errors = [
                    [
                        win_rate - ci_low
                        for win_rate, ci_low in zip(win_rates, ci_lows)
                    ],
                    [
                        ci_high - win_rate
                        for win_rate, ci_high in zip(win_rates, ci_highs)
                    ],
                ]

                plt.errorbar(
                    steps,
                    win_rates,
                    yerr=y_errors,
                    fmt=".",
                    linestyle="-",
                    capsize=2,
                    label=f"{algo}, seed={seed}",
                )

            else:
                plt.plot(
                    steps,
                    win_rates,
                    ".",
                    linestyle="-",
                    label=f"{algo}, seed={seed}"
                )

    plt.xlabel("step")
    plt.ylabel("win_rate")
    plt.title(f"Win rate vs step dla przebiegow algo: {config["show_algo"]}, seed={config["show_seeds"]}")
    plt.ylim(0.5, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(config["results_dir"]) / config["plot_file"], dpi=300, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    results: list[Result] = load_results(Path(config["results_dir"]) / config["results_file"])

    grouped_results = group_results_by_algo_and_seed(results)

    plot_results(grouped_results, config)


if __name__ == "__main__":
    main()
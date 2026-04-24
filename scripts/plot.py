import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


Row = dict[str, str | int | float]
GroupKey = tuple[str, int]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=r"results\evals.csv",
        help="Ścieżka do pliku CSV z wynikami ewaluacji.",
    )
    return parser.parse_args()


def parse_value(value: str) -> str | int | float:
    value = value.strip()

    if value == "":
        return value

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_results_csv(csv_path: str | Path) -> list[Row]:
    results: list[Row] = []

    with open(csv_path, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            parsed_row: Row = {
                key: parse_value(value)
                for key, value in row.items()
            }

            results.append(parsed_row)

    return results


def group_results(results: list[Row]) -> dict[GroupKey, list[Row]]:
    grouped: dict[GroupKey, list[Row]] = defaultdict(list)

    for row in results:
        algo = str(row["algo"])
        seed = int(row["seed"])

        grouped[(algo, seed)].append(row)

    return dict(grouped)


def plot_grouped_results(grouped_results: dict[GroupKey, list[Row]]) -> None:
    plt.figure(figsize=(12, 7))

    for (algo, seed), rows in sorted(grouped_results.items()):
        rows = sorted(rows, key=lambda row: int(row["step"]))

        steps = [int(row["step"]) for row in rows]
        win_rates = [float(row["win_rate"]) for row in rows]
        ci_lows = [float(row["ci_low"]) for row in rows]
        ci_highs = [float(row["ci_high"]) for row in rows]

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
            #yerr=y_errors if 1==0 else 0,
            fmt=".",
            linestyle="-",
            capsize=4,
            label=f"{algo}, seed={seed}",
        )

    plt.xlabel("step")
    plt.ylabel("win_rate")
    plt.title("Win rate vs step dla wszystkich przebiegów")
    plt.ylim(0.5, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    results: list[Row] = load_results_csv(args.csv_path)

    grouped_results = group_results(results)

    plot_grouped_results(grouped_results)


if __name__ == "__main__":
    main()
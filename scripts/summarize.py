import csv
from collections import defaultdict
from pathlib import Path
import argparse
from scripts.train import read_config
from scripts.eval import Result

SummaryRow = dict[str, str | int | float | None]
GroupedResults = dict[tuple[str, int], list[Result]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/summarize.yaml"),
        help="Ścieżka do pliku config.",
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


def load_results(csv_path: str | Path) -> list[Result]:
    results: list[Result] = []

    with open(csv_path, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            parsed_row: Result = {
                key: parse_value(value)
                for key, value in row.items()
            }

            results.append(parsed_row)

    return results


def group_results_by_algo_and_seed(results: list[Result]) -> GroupedResults:
    grouped: GroupedResults = defaultdict(list)

    for result in results:
        algo = str(result["algo"])
        seed = int(result["seed"])

        grouped[(algo, seed)].append(result)

    return dict(grouped)


def group_summary(summary: list[SummaryRow]) -> dict[str, list[SummaryRow]]:
    grouped: dict[str, list[SummaryRow]] = defaultdict(list)

    for row in summary:
        algo = str(row["algo"])

        grouped[algo].append(row)

    return dict(grouped)


def first_step_at_or_above(results: list[Result], metric: str, threshold: float,) -> int | None:
    sorted_results = sorted(results, key=lambda result: int(result["step"]))

    for result in sorted_results:
        step = int(result["step"])
        if float(result[metric]) >= threshold:
            return step

    return None


def interpolated_step_at_threshold(results: list[Result],threshold: float,) -> float | None:
    sorted_results = sorted(results, key=lambda result: int(result["step"]))

    if not sorted_results:
        return None

    first_step = int(sorted_results[0]["step"])
    first_value = float(sorted_results[0]["win_rate"])

    if first_value >= threshold:
        return float(first_step)

    for previous_result, current_result in zip(sorted_results, sorted_results[1:]):
        previous_step = int(previous_result["step"])
        current_step = int(current_result["step"])

        previous_value = float(previous_result["win_rate"])
        current_value = float(current_result["win_rate"])

        if previous_value < threshold <= current_value:
            if current_value == previous_value:
                return float(current_step)

            interpolated_step = previous_step + (
                (threshold - previous_value)
                / (current_value - previous_value)
            ) * (current_step - previous_step)

            return interpolated_step

    return None


def calc_algo_avg(summary: list[SummaryRow]) -> list[SummaryRow]:
    grouped_summary: dict[str, list[SummaryRow]] = group_summary(summary)

    for algo, algo_summary in grouped_summary.items():
        sum_win_rate = 0
        sum_ci_low = 0
        sum_interpolated = 0
        count = 0
        for row in algo_summary:
            if row["seed"] != "step_avg":
                sum_win_rate += row["step_win_rate_ge_0_9"]
                sum_ci_low += row["step_ci_low_ge_0_9"]
                sum_interpolated += row["interpolated_step_win_rate_0_9"]
                count += 1

        summary.append(
            {
                "algo": algo,
                "seed": "avg",
                "step_win_rate_ge_0_9": sum_win_rate / count,
                "step_ci_low_ge_0_9": sum_ci_low / count,
                "interpolated_step_win_rate_0_9": sum_interpolated / count,
            }
        )

    return summary


def build_threshold_summary(grouped_results: GroupedResults,threshold: float,) -> list[SummaryRow]:
    summary: list[SummaryRow] = []

    for (algo, seed), results in sorted(grouped_results.items()):
        step_win_rate = first_step_at_or_above(
            results=results,
            metric="win_rate",
            threshold=threshold,
        )

        step_ci_low = first_step_at_or_above(
            results=results,
            metric="ci_low",
            threshold=threshold,
        )

        interpolated_step_win_rate = interpolated_step_at_threshold(
            results=results,
            threshold=threshold,
        )

        summary.append(
            {
                "algo": algo,
                "seed": seed,
                "step_win_rate_ge_0_9": step_win_rate,
                "step_ci_low_ge_0_9": step_ci_low,
                "interpolated_step_win_rate_0_9": interpolated_step_win_rate,
            }
        )

    summary = calc_algo_avg(summary)

    return summary


def print_threshold_summary_table(summary: list[SummaryRow],) -> None:
    headers = list(summary[0].keys())

    rows_as_strings: list[list[str]] = []

    for row in summary:
        row_elem: list[str] = []
        for key in headers:
            row_elem.append(str(row[key]))
        rows_as_strings.append(row_elem)

    column_widths = [
        max(len(headers[column_index]), *(len(row[column_index]) for row in rows_as_strings))
        for column_index in range(len(headers))
    ]

    header_line = " | ".join(
        header.ljust(column_widths[column_index])
        for column_index, header in enumerate(headers)
    )

    separator_line = "-+-".join(
        "-" * column_width
        for column_width in column_widths
    )

    print(header_line)
    print(separator_line)

    for row in rows_as_strings:
        print(
            " | ".join(
                value.ljust(column_widths[column_index])
                for column_index, value in enumerate(row)
            )
        )


def save_summary(summary: list[SummaryRow], summary_config) -> None:
    results_path = Path(summary_config["results_dir"]) / summary_config["summary_file"]

    fieldnames = list(summary[0].keys())

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    results = load_results(Path(config["results_dir"]) / config["results_file"])
    grouped_results = group_results_by_algo_and_seed(results)

    summary = build_threshold_summary(
        grouped_results=grouped_results,
        threshold=float(config["threshold"]),
    )

    print_threshold_summary_table(summary)
    save_summary(summary, config)


if __name__ == "__main__":
    main()
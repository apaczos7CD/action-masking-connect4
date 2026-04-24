from typing import Optional
from scripts.plot import parse_args, load_results_csv, group_results, Row, GroupKey


ThresholdSummaryRow = dict[str, str | int | float | None]


def first_step_at_or_above(
    rows: list[Row],
    metric: str,
    threshold: float = 0.9,
) -> int | None:
    """
    Zwraca najmniejszy step, dla którego dana metryka osiąga co najmniej threshold.
    Np. metric='win_rate' albo metric='ci_low'.
    """
    sorted_rows = sorted(rows, key=lambda row: int(row["step"]))

    for row in sorted_rows:
        step = int(row["step"])
        value = float(row[metric])

        if value >= threshold:
            return step

    return None


def interpolated_step_at_threshold(
    rows: list[Row],
    metric: str = "win_rate",
    threshold: float = 0.9,
) -> float | None:
    """
    Przybliża dokładny step, w którym metryka przekracza threshold,
    używając interpolacji liniowej między dwoma sąsiednimi punktami.

    Domyślnie interpoluje po win_rate.
    """
    sorted_rows = sorted(rows, key=lambda row: int(row["step"]))

    if not sorted_rows:
        return None

    first_step = int(sorted_rows[0]["step"])
    first_value = float(sorted_rows[0][metric])

    if first_value >= threshold:
        return float(first_step)

    for previous_row, current_row in zip(sorted_rows, sorted_rows[1:]):
        previous_step = int(previous_row["step"])
        current_step = int(current_row["step"])

        previous_value = float(previous_row[metric])
        current_value = float(current_row[metric])

        if previous_value < threshold <= current_value:
            if current_value == previous_value:
                return float(current_step)

            interpolated_step = previous_step + (
                (threshold - previous_value)
                / (current_value - previous_value)
            ) * (current_step - previous_step)

            return interpolated_step

    return None


def build_threshold_summary(
    grouped_results: dict[GroupKey, list[Row]],
    threshold: float = 0.9,
) -> list[ThresholdSummaryRow]:
    summary: list[ThresholdSummaryRow] = []

    for (algo, seed), rows in sorted(grouped_results.items()):
        step_win_rate = first_step_at_or_above(
            rows=rows,
            metric="win_rate",
            threshold=threshold,
        )

        step_ci_low = first_step_at_or_above(
            rows=rows,
            metric="ci_low",
            threshold=threshold,
        )

        interpolated_step_win_rate = interpolated_step_at_threshold(
            rows=rows,
            metric="win_rate",
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

    return summary


def format_optional_int(value: int | None) -> str:
    if value is None:
        return "-"

    return str(value)


def format_optional_float(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"

    return f"{value:.{decimals}f}"


def print_threshold_summary_table(
    summary: list[ThresholdSummaryRow],
) -> None:
    headers = [
        "algo",
        "seed",
        "min step win_rate >= 0.9",
        "min step ci_low >= 0.9",
        "interpolated step win_rate = 0.9",
    ]

    rows_as_strings: list[list[str]] = []

    for row in summary:
        rows_as_strings.append(
            [
                str(row["algo"]),
                str(row["seed"]),
                format_optional_int(row["step_win_rate_ge_0_9"]),  # type: ignore[arg-type]
                format_optional_int(row["step_ci_low_ge_0_9"]),  # type: ignore[arg-type]
                format_optional_float(row["interpolated_step_win_rate_0_9"]),  # type: ignore[arg-type]
            ]
        )

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

def main() -> None:
    args = parse_args()

    results: list[Row] = load_results_csv(args.csv_path)

    grouped_results = group_results(results)

    summary = build_threshold_summary(
        grouped_results=grouped_results,
        threshold=0.9,
    )

    print_threshold_summary_table(summary)


if __name__ == "__main__":
    main()
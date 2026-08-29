"""Plot FP8 training curves against the mean of multiple baseline runs.

Outputs, for every configured max-step cutoff:
1. A PNG containing metric curves and FP8-minus-baseline-mean errors.
2. A per-step CSV with baseline statistics and FP8 errors.
3. A summary CSV with MAE, RMSE, bias, relative errors, correlation, and AUC.

All baseline curves are linearly interpolated only inside their observed step
ranges. Values are never extrapolated beyond a run's available data.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


FP8_LABEL = "fp8"
LOG_FILES = {
    "fp8": Path("logs/xx-11111.log"),
    "job-79b28e79": Path(
        "logs/modelarts-job-79b28e79-b747-46a6-b6db-f9e6594c1db0-worker-0.log"
    ),
    "job-a8319e6d": Path(
        "logs/modelarts-job-a8319e6d-efa5-49d7-bf71-a2915a53d844-worker-0.log"
    ),
    "job-d85462ea": Path(
        "logs/modelarts-job-d85462ea-6b7a-4b75-8483-6add6406eac5-worker-0.log"
    ),
}

METRICS = (
    "reward/all/pass_rate/mean",
    "reward/all/hit_reward/mean",
    "reward/all/pass_at_1/mean",
)

MAX_STEPS_LIST = (30, 100)
OUTPUT_DIR = Path("metrics_analysis")
OUTPUT_TAG = "11111"

COLORS = {
    "fp8": "red",
    "job-79b28e79": "#1f77b4",
    "job-a8319e6d": "#ff7f0e",
    "job-d85462ea": "#2ca02c",
}

# Used for the per-step np.isclose rate in the statistical summary.
RTOL = 1.0e-3
ATOL = 1.0e-5
RELATIVE_EPS = 1.0e-8

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep\s*:\s*(\d+)")
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


Series = Dict[int, float]
ParsedData = Dict[str, Dict[str, Series]]


def validate_configuration() -> list[str]:
    if FP8_LABEL not in LOG_FILES:
        raise ValueError(f"FP8_LABEL={FP8_LABEL!r} is not present in LOG_FILES")

    baseline_labels = [label for label in LOG_FILES if label != FP8_LABEL]
    if not baseline_labels:
        raise ValueError("At least one baseline log is required")

    missing = [str(path) for path in LOG_FILES.values() if not path.is_file()]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing log files:\n  {formatted}")
    return baseline_labels


def parse_logs(max_steps: int) -> ParsedData:
    """Parse every configured metric independently from every log line."""
    data: ParsedData = {
        label: {metric: {} for metric in METRICS} for label in LOG_FILES
    }

    metric_patterns = {
        metric: re.compile(
            re.escape(metric) + rf"\s*:\s*({FLOAT_PATTERN})"
        )
        for metric in METRICS
    }

    for label, path in LOG_FILES.items():
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for raw_line in file:
                line = ANSI_RE.sub("", raw_line)
                step_match = STEP_RE.search(line)
                if not step_match:
                    continue

                step = int(step_match.group(1))
                if step > max_steps:
                    continue

                for metric, pattern in metric_patterns.items():
                    value_match = pattern.search(line)
                    if value_match:
                        value = float(value_match.group(1))
                        if math.isfinite(value):
                            # If a step appears multiple times, keep the latest value.
                            data[label][metric][step] = value
    return data


def interpolate_series(series: Mapping[int, float], target_steps: np.ndarray) -> np.ndarray:
    """Interpolate within the observed range; return NaN outside that range."""
    result = np.full(target_steps.shape, np.nan, dtype=np.float64)
    if not series:
        return result

    source_steps = np.asarray(sorted(series), dtype=np.float64)
    source_values = np.asarray([series[int(step)] for step in source_steps], dtype=np.float64)
    valid = (target_steps >= source_steps[0]) & (target_steps <= source_steps[-1])
    result[valid] = np.interp(target_steps[valid], source_steps, source_values)
    return result


def align_metric(
    data: ParsedData,
    metric: str,
    baseline_labels: Iterable[str],
) -> dict[str, np.ndarray]:
    """Align FP8 and all baselines at FP8 steps and compute baseline statistics."""
    fp8_series = data[FP8_LABEL][metric]
    if not fp8_series:
        return {}

    steps = np.asarray(sorted(fp8_series), dtype=np.float64)
    fp8_values = np.asarray([fp8_series[int(step)] for step in steps], dtype=np.float64)

    labels = list(baseline_labels)
    baseline_matrix = np.vstack(
        [interpolate_series(data[label][metric], steps) for label in labels]
    )

    # Require all configured baselines so the comparison always means the
    # average of the same set of runs.
    valid = np.all(np.isfinite(baseline_matrix), axis=0) & np.isfinite(fp8_values)
    steps = steps[valid]
    fp8_values = fp8_values[valid]
    baseline_matrix = baseline_matrix[:, valid]
    if steps.size == 0:
        return {}

    baseline_mean = np.mean(baseline_matrix, axis=0)
    baseline_std = np.std(baseline_matrix, axis=0, ddof=0)
    baseline_min = np.min(baseline_matrix, axis=0)
    baseline_max = np.max(baseline_matrix, axis=0)
    signed_error = fp8_values - baseline_mean
    abs_error = np.abs(signed_error)
    relative_error = abs_error / np.maximum(np.abs(baseline_mean), RELATIVE_EPS)
    smape = (
        abs_error
        / np.maximum((np.abs(fp8_values) + np.abs(baseline_mean)) / 2.0, RELATIVE_EPS)
    )
    z_score = np.divide(
        signed_error,
        baseline_std,
        out=np.full_like(signed_error, np.nan),
        where=baseline_std > RELATIVE_EPS,
    )

    result = {
        "steps": steps,
        "fp8": fp8_values,
        "baseline_matrix": baseline_matrix,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_min": baseline_min,
        "baseline_max": baseline_max,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "relative_error": relative_error,
        "smape": smape,
        "z_score": z_score,
    }
    for index, label in enumerate(labels):
        result[f"baseline:{label}"] = baseline_matrix[index]
    return result


def trapezoid(values: np.ndarray, steps: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, steps))
    return float(np.trapz(values, steps))


def summarize(metric: str, max_steps: int, aligned: Mapping[str, np.ndarray]) -> dict[str, object]:
    steps = aligned["steps"]
    fp8 = aligned["fp8"]
    baseline_mean = aligned["baseline_mean"]
    baseline_std = aligned["baseline_std"]
    signed_error = aligned["signed_error"]
    abs_error = aligned["abs_error"]

    if fp8.size >= 2 and np.std(fp8) > 0 and np.std(baseline_mean) > 0:
        correlation = float(np.corrcoef(fp8, baseline_mean)[0, 1])
    else:
        correlation = float("nan")

    within_range = (fp8 >= aligned["baseline_min"]) & (fp8 <= aligned["baseline_max"])
    within_one_std = abs_error <= baseline_std
    close = np.isclose(fp8, baseline_mean, rtol=RTOL, atol=ATOL)
    fp8_auc = trapezoid(fp8, steps)
    baseline_auc = trapezoid(baseline_mean, steps)

    return {
        "metric": metric,
        "max_steps": max_steps,
        "compared_points": int(fp8.size),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
        "bias": float(np.mean(signed_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_relative_pct": float(np.mean(aligned["relative_error"]) * 100.0),
        "smape_pct": float(np.mean(aligned["smape"]) * 100.0),
        "pearson_correlation": correlation,
        "allclose_rate_pct": float(np.mean(close) * 100.0),
        "within_baseline_range_pct": float(np.mean(within_range) * 100.0),
        "within_1std_pct": float(np.mean(within_one_std) * 100.0),
        "fp8_auc": fp8_auc,
        "baseline_mean_auc": baseline_auc,
        "auc_difference": fp8_auc - baseline_auc,
    }


def write_per_step_csv(
    path: Path,
    max_steps: int,
    aligned_by_metric: Mapping[str, Mapping[str, np.ndarray]],
    baseline_labels: list[str],
) -> None:
    baseline_fields = [f"baseline_{label}" for label in baseline_labels]
    fieldnames = [
        "max_steps",
        "metric",
        "step",
        "fp8",
        *baseline_fields,
        "baseline_mean",
        "baseline_std",
        "baseline_min",
        "baseline_max",
        "signed_error",
        "abs_error",
        "relative_error_pct",
        "smape_pct",
        "z_score",
        "isclose",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric, aligned in aligned_by_metric.items():
            if not aligned:
                continue
            for index, step in enumerate(aligned["steps"]):
                row: dict[str, object] = {
                    "max_steps": max_steps,
                    "metric": metric,
                    "step": int(step),
                    "fp8": aligned["fp8"][index],
                    "baseline_mean": aligned["baseline_mean"][index],
                    "baseline_std": aligned["baseline_std"][index],
                    "baseline_min": aligned["baseline_min"][index],
                    "baseline_max": aligned["baseline_max"][index],
                    "signed_error": aligned["signed_error"][index],
                    "abs_error": aligned["abs_error"][index],
                    "relative_error_pct": aligned["relative_error"][index] * 100.0,
                    "smape_pct": aligned["smape"][index] * 100.0,
                    "z_score": aligned["z_score"][index],
                    "isclose": bool(
                        np.isclose(
                            aligned["fp8"][index],
                            aligned["baseline_mean"][index],
                            rtol=RTOL,
                            atol=ATOL,
                        )
                    ),
                }
                for label in baseline_labels:
                    row[f"baseline_{label}"] = aligned[f"baseline:{label}"][index]
                writer.writerow(row)


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def short_metric_name(metric: str) -> str:
    return metric.removeprefix("reward/all/").removesuffix("/mean")


def plot_analysis(
    path: Path,
    max_steps: int,
    aligned_by_metric: Mapping[str, Mapping[str, np.ndarray]],
    summaries: Mapping[str, Mapping[str, object]],
    baseline_labels: list[str],
) -> None:
    fig, axes = plt.subplots(
        len(METRICS),
        2,
        figsize=(16, 4.2 * len(METRICS)),
        squeeze=False,
    )

    for row, metric in enumerate(METRICS):
        curve_ax, error_ax = axes[row]
        aligned = aligned_by_metric.get(metric, {})
        if not aligned:
            curve_ax.text(0.5, 0.5, "No aligned data", ha="center", va="center")
            error_ax.text(0.5, 0.5, "No aligned data", ha="center", va="center")
            continue

        steps = aligned["steps"]
        for label in baseline_labels:
            curve_ax.plot(
                steps,
                aligned[f"baseline:{label}"],
                label=label,
                color=COLORS.get(label),
                linewidth=1.0,
                alpha=0.45,
            )

        mean = aligned["baseline_mean"]
        std = aligned["baseline_std"]
        curve_ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color="gray",
            alpha=0.20,
            label="baseline mean ± 1 std",
        )
        curve_ax.plot(steps, mean, color="black", linewidth=2.0, label="baseline mean")
        curve_ax.plot(
            steps,
            aligned["fp8"],
            color=COLORS[FP8_LABEL],
            linewidth=2.0,
            label=FP8_LABEL,
        )
        curve_ax.set_title(metric)
        curve_ax.set_ylabel(short_metric_name(metric))
        curve_ax.grid(True, alpha=0.3)
        curve_ax.legend(fontsize=8, ncol=2)

        error_ax.axhline(0.0, color="black", linewidth=1.0)
        error_ax.plot(
            steps,
            aligned["signed_error"],
            color="purple",
            linewidth=1.5,
            label="FP8 - baseline mean",
        )
        error_ax.fill_between(
            steps,
            0.0,
            aligned["signed_error"],
            where=aligned["signed_error"] >= 0,
            color="red",
            alpha=0.18,
        )
        error_ax.fill_between(
            steps,
            0.0,
            aligned["signed_error"],
            where=aligned["signed_error"] < 0,
            color="blue",
            alpha=0.18,
        )
        summary = summaries[metric]
        stats_text = (
            f"MAE={summary['mae']:.4g}\n"
            f"RMSE={summary['rmse']:.4g}\n"
            f"Bias={summary['bias']:.4g}\n"
            f"Corr={summary['pearson_correlation']:.4g}\n"
            f"isclose={summary['allclose_rate_pct']:.1f}%"
        )
        error_ax.text(
            0.02,
            0.98,
            stats_text,
            transform=error_ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        error_ax.set_title(f"{short_metric_name(metric)} error")
        error_ax.set_ylabel("Signed error")
        error_ax.grid(True, alpha=0.3)
        error_ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    fig.suptitle(
        f"FP8 vs baseline mean (first {max_steps} steps, {len(baseline_labels)} baselines)",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_summary(summaries: list[dict[str, object]]) -> None:
    print(
        f"{'metric':18s} {'points':>6s} {'MAE':>11s} {'RMSE':>11s} "
        f"{'bias':>11s} {'rel%':>9s} {'corr':>9s} {'isclose%':>10s}"
    )
    print("-" * 92)
    for summary in summaries:
        print(
            f"{short_metric_name(str(summary['metric'])):18s} "
            f"{int(summary['compared_points']):6d} "
            f"{float(summary['mae']):11.4e} "
            f"{float(summary['rmse']):11.4e} "
            f"{float(summary['bias']):11.4e} "
            f"{float(summary['mean_abs_relative_pct']):9.3f} "
            f"{float(summary['pearson_correlation']):9.4f} "
            f"{float(summary['allclose_rate_pct']):10.2f}"
        )


def analyze(max_steps: int, baseline_labels: list[str]) -> None:
    data = parse_logs(max_steps)
    aligned_by_metric = {
        metric: align_metric(data, metric, baseline_labels) for metric in METRICS
    }
    summaries = [
        summarize(metric, max_steps, aligned_by_metric[metric])
        for metric in METRICS
        if aligned_by_metric[metric]
    ]
    summary_by_metric = {str(row["metric"]): row for row in summaries}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{max_steps}steps-{OUTPUT_TAG}"
    plot_path = OUTPUT_DIR / f"metrics_plot_{stem}.png"
    per_step_path = OUTPUT_DIR / f"metrics_per_step_{stem}.csv"
    summary_path = OUTPUT_DIR / f"metrics_summary_{stem}.csv"

    write_per_step_csv(per_step_path, max_steps, aligned_by_metric, baseline_labels)
    write_summary_csv(summary_path, summaries)
    plot_analysis(
        plot_path,
        max_steps,
        aligned_by_metric,
        summary_by_metric,
        baseline_labels,
    )

    print(f"\nFirst {max_steps} steps")
    print_summary(summaries)
    print(f"Saved plot   : {plot_path}")
    print(f"Saved steps  : {per_step_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    baseline_labels = validate_configuration()
    print(f"FP8 run      : {FP8_LABEL}")
    print(f"Baseline runs: {', '.join(baseline_labels)}")
    print(f"isclose      : rtol={RTOL}, atol={ATOL}")
    for max_steps in MAX_STEPS_LIST:
        analyze(max_steps, baseline_labels)


if __name__ == "__main__":
    main()

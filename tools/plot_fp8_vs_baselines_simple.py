"""Simple FP8-versus-baseline curve plot and error statistics."""

import csv
import math
import re
from pathlib import Path

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
RELATIVE_EPS = 1.0e-8

COLORS = {
    "fp8": "red",
    "job-79b28e79": "#1f77b4",
    "job-a8319e6d": "#ff7f0e",
    "job-d85462ea": "#2ca02c",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep\s*:\s*(\d+)")
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def validate_files():
    missing = [str(path) for path in LOG_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing log files:\n  " + "\n  ".join(missing))


def parse_logs(max_steps):
    data = {
        label: {metric: {} for metric in METRICS}
        for label in LOG_FILES
    }
    patterns = {
        metric: re.compile(re.escape(metric) + rf"\s*:\s*({FLOAT_PATTERN})")
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

                for metric, pattern in patterns.items():
                    value_match = pattern.search(line)
                    if value_match:
                        value = float(value_match.group(1))
                        if math.isfinite(value):
                            data[label][metric][step] = value
    return data


def interpolate(series, target_steps):
    """Interpolate only inside the observed step range; never extrapolate."""
    result = np.full(target_steps.shape, np.nan, dtype=np.float64)
    if not series:
        return result

    source_steps = np.asarray(sorted(series), dtype=np.float64)
    source_values = np.asarray([series[int(step)] for step in source_steps])
    valid = (target_steps >= source_steps[0]) & (target_steps <= source_steps[-1])
    result[valid] = np.interp(target_steps[valid], source_steps, source_values)
    return result


def align_metric(data, metric, baseline_labels):
    fp8_series = data[FP8_LABEL][metric]
    if not fp8_series:
        return None

    steps = np.asarray(sorted(fp8_series), dtype=np.float64)
    fp8_values = np.asarray([fp8_series[int(step)] for step in steps])
    baseline_values = np.vstack(
        [interpolate(data[label][metric], steps) for label in baseline_labels]
    )

    # Always compare against the mean of all three baselines.
    valid = np.all(np.isfinite(baseline_values), axis=0) & np.isfinite(fp8_values)
    steps = steps[valid]
    fp8_values = fp8_values[valid]
    baseline_values = baseline_values[:, valid]
    if steps.size == 0:
        return None

    baseline_mean = np.mean(baseline_values, axis=0)
    baseline_std = np.std(baseline_values, axis=0)
    abs_error = np.abs(fp8_values - baseline_mean)
    relative_mask = np.abs(baseline_mean) > RELATIVE_EPS

    if np.any(relative_mask):
        mean_relative_error_pct = (
            np.mean(abs_error[relative_mask] / np.abs(baseline_mean[relative_mask]))
            * 100.0
        )
    else:
        mean_relative_error_pct = float("nan")

    return {
        "steps": steps,
        "fp8": fp8_values,
        "baselines": baseline_values,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "mean_abs_error": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_relative_error_pct": float(mean_relative_error_pct),
        "compared_points": int(steps.size),
        "relative_error_points": int(np.count_nonzero(relative_mask)),
    }


def short_name(metric):
    return metric.removeprefix("reward/all/").removesuffix("/mean")


def print_statistics(max_steps, rows):
    print(f"\nFirst {max_steps} steps")
    print(
        f"{'metric':18s} {'points':>7s} {'mean abs error':>16s} "
        f"{'max abs error':>16s} {'mean relative':>15s}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{short_name(row['metric']):18s} "
            f"{row['compared_points']:7d} "
            f"{row['mean_abs_error']:16.6e} "
            f"{row['max_abs_error']:16.6e} "
            f"{row['mean_relative_error_pct']:14.4f}%"
        )


def save_statistics(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(max_steps, baseline_labels):
    data = parse_logs(max_steps)
    aligned = {
        metric: align_metric(data, metric, baseline_labels)
        for metric in METRICS
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    rows = []

    for ax, metric in zip(axes, METRICS):
        result = aligned[metric]
        if result is None:
            ax.text(0.5, 0.5, "No aligned data", ha="center", va="center")
            ax.set_title(metric)
            continue

        steps = result["steps"]
        for index, label in enumerate(baseline_labels):
            ax.plot(
                steps,
                result["baselines"][index],
                label=label,
                color=COLORS[label],
                linewidth=1.0,
                alpha=0.55,
            )

        baseline_mean = result["baseline_mean"]
        baseline_std = result["baseline_std"]
        ax.fill_between(
            steps,
            baseline_mean - baseline_std,
            baseline_mean + baseline_std,
            color="gray",
            alpha=0.18,
            label="baseline mean ? std",
        )
        ax.plot(
            steps,
            baseline_mean,
            color="black",
            linewidth=2.0,
            label="baseline mean",
        )
        ax.plot(
            steps,
            result["fp8"],
            color=COLORS[FP8_LABEL],
            linewidth=2.0,
            label=FP8_LABEL,
        )
        ax.set_title(metric)
        ax.set_ylabel(short_name(metric))
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        rows.append(
            {
                "metric": metric,
                "max_steps": max_steps,
                "compared_points": result["compared_points"],
                "relative_error_points": result["relative_error_points"],
                "mean_abs_error": result["mean_abs_error"],
                "max_abs_error": result["max_abs_error"],
                "mean_relative_error_pct": result["mean_relative_error_pct"],
            }
        )

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"FP8 vs baseline mean (first {max_steps} steps)", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{max_steps}steps-{OUTPUT_TAG}"
    plot_path = OUTPUT_DIR / f"metrics_plot_{stem}.png"
    summary_path = OUTPUT_DIR / f"metrics_error_summary_{stem}.csv"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print_statistics(max_steps, rows)
    save_statistics(summary_path, rows)
    print(f"Saved plot   : {plot_path}")
    print(f"Saved summary: {summary_path}")


def main():
    validate_files()
    baseline_labels = [label for label in LOG_FILES if label != FP8_LABEL]
    print(f"FP8 run      : {FP8_LABEL}")
    print(f"Baseline runs: {', '.join(baseline_labels)}")
    for max_steps in MAX_STEPS_LIST:
        analyze(max_steps, baseline_labels)


if __name__ == "__main__":
    main()

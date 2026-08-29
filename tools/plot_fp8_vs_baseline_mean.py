"""Plot one FP8 metric against the mean of three baseline runs."""

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC = "reward/all/pass_rate/mean"
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
STEP_RE = re.compile(r"\bstep\b[\"']?\s*[:=]\s*(\d+)", re.IGNORECASE)
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(
    re.escape(METRIC)
    + rf"[\"']?\s*[:=]\s*(?:tensor\s*\(\s*)?({FLOAT_PATTERN})"
)


def validate_files():
    missing = [str(path) for path in LOG_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing log files:\n  " + "\n  ".join(missing))


def parse_logs(max_steps):
    data = {label: {} for label in LOG_FILES}
    for label, path in LOG_FILES.items():
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for raw_line in file:
                line = ANSI_RE.sub("", raw_line)
                step_match = STEP_RE.search(line)
                value_match = METRIC_RE.search(line)
                if not step_match or not value_match:
                    continue

                step = int(step_match.group(1))
                value = float(value_match.group(1))
                if step <= max_steps and math.isfinite(value):
                    # If the same step appears more than once, keep the latest value.
                    data[label][step] = value
    return data


def interpolate(series, target_steps):
    """Fill missing internal steps without extrapolating outside the log range."""
    result = np.full(target_steps.shape, np.nan, dtype=np.float64)
    if not series:
        return result

    source_steps = np.asarray(sorted(series), dtype=np.float64)
    source_values = np.asarray([series[int(step)] for step in source_steps])
    valid = (target_steps >= source_steps[0]) & (target_steps <= source_steps[-1])
    result[valid] = np.interp(target_steps[valid], source_steps, source_values)
    return result


def align_curves(data, baseline_labels):
    fp8_series = data[FP8_LABEL]
    if not fp8_series:
        raise RuntimeError(f"No values found for {METRIC} in the FP8 log")

    steps = np.asarray(sorted(fp8_series), dtype=np.float64)
    fp8_values = np.asarray([fp8_series[int(step)] for step in steps])
    baseline_values = np.vstack(
        [interpolate(data[label], steps) for label in baseline_labels]
    )

    # The baseline mean must always contain the same three baseline runs.
    valid = np.all(np.isfinite(baseline_values), axis=0) & np.isfinite(fp8_values)
    steps = steps[valid]
    fp8_values = fp8_values[valid]
    baseline_values = baseline_values[:, valid]
    if steps.size == 0:
        raise RuntimeError("No common valid steps between FP8 and all baselines")

    baseline_mean = np.mean(baseline_values, axis=0)
    baseline_std = np.std(baseline_values, axis=0)
    abs_error = np.abs(fp8_values - baseline_mean)

    relative_mask = np.abs(baseline_mean) > RELATIVE_EPS
    if np.any(relative_mask):
        mean_relative_error_pct = float(
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
        "compared_points": int(steps.size),
        "relative_error_points": int(np.count_nonzero(relative_mask)),
        "mean_abs_error": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_relative_error_pct": mean_relative_error_pct,
    }


def save_summary(path, max_steps, result):
    row = {
        "metric": METRIC,
        "max_steps": max_steps,
        "compared_points": result["compared_points"],
        "relative_error_points": result["relative_error_points"],
        "mean_abs_error": result["mean_abs_error"],
        "max_abs_error": result["max_abs_error"],
        "mean_relative_error_pct": result["mean_relative_error_pct"],
    }

def format_report(max_steps, result):
    return [
        f"First {max_steps} steps",
        f"Compared points            : {result['compared_points']}",
        f"Mean absolute error        : {result['mean_abs_error']:.6e}",
        f"Maximum absolute error     : {result['max_abs_error']:.6e}",
        f"Mean relative error        : {result['mean_relative_error_pct']:.4f}%",
        f"Relative-error valid points: {result['relative_error_points']}",
    ]


    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def plot_and_report(max_steps, baseline_labels):
    data = parse_logs(max_steps)
    result = align_curves(data, baseline_labels)
    steps = result["steps"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, label in enumerate(baseline_labels):
        ax.plot(
            steps,
            result["baselines"][index],
            label=label,
            color=COLORS[label],
            linewidth=1.0,
            alpha=0.55,
        )

    """
    baseline_mean = result["baseline_mean"]
    baseline_std = result["baseline_std"]
    ax.fill_between(
        steps,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color="gray",
        alpha=0.18,
        label="baseline mean ± std",
    )
    ax.plot(
        steps,
        baseline_mean,
        color="black",
        linewidth=2.0,
        label="baseline mean",
    )
    """
    ax.plot(
        steps,
        result["fp8"],
        color=COLORS[FP8_LABEL],
        linewidth=2.0,
        label=FP8_LABEL,
    )

    ax.set_title(f"FP8 vs baselines (first {max_steps} steps)")
    ax.set_xlabel("Step")
    ax.set_ylabel("pass_rate/mean")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    experiment_dir = OUTPUT_DIR / OUTPUT_TAG
    experiment_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{max_steps}steps"
    plot_path = experiment_dir / f"fp8_vs_baseline_{stem}.png"
    summary_path = experiment_dir / f"fp8_error_summary_{stem}.csv"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    save_summary(summary_path, max_steps, result)

    report_lines = format_report(max_steps, result)
    print("\n" + "\n".join(report_lines))
    return report_lines
    print(f"Saved plot                 : {plot_path}")
    print(f"Saved summary              : {summary_path}")


def main():
    validate_files()
    baseline_labels = [label for label in LOG_FILES if label != FP8_LABEL]
    print(f"Metric        : {METRIC}")
    print(f"FP8 run       : {FP8_LABEL}")
    print(f"Baseline runs : {', '.join(baseline_labels)}")
    all_report_lines = [
        f"Metric        : {METRIC}",
        f"FP8 run       : {FP8_LABEL}",
        f"Baseline runs : {', '.join(baseline_labels)}",
    ]
    for max_steps in MAX_STEPS_LIST:
        all_report_lines.append("")
        all_report_lines.extend(plot_and_report(max_steps, baseline_labels))

    experiment_dir = OUTPUT_DIR / OUTPUT_TAG
    experiment_dir.mkdir(parents=True, exist_ok=True)
    txt_path = experiment_dir / "fp8_error_summary.txt"
    txt_path.write_text("\n".join(all_report_lines) + "\n", encoding="utf-8")
    print(f"\nSaved text summary         : {txt_path}")


if __name__ == "__main__":
    main()

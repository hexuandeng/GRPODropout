import argparse
import csv
import hashlib
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


# Edit these defaults when you want to run the script directly from an IDE.
INPUT_PATH = Path(__file__).resolve().parent / "logs" / "qwen3-1.7b"
OUTPUT_DIR = Path(__file__).resolve().parent / "logs" / "curve" / "qwen3-1.7b"
LOG_PATTERN = "*.log"
METRIC_NAME = "critic/rewards/mean"
SMOOTH_WINDOW = 5
MIN_POINTS = 1
RECURSIVE = True


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.+)")
METRIC_RE = re.compile(
    r"([A-Za-z0-9_./@-]+):([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_metric_points(log_path: Path, metric_name: str):
    points = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            step_match = STEP_RE.search(line)
            if not step_match:
                continue

            step = int(step_match.group(1))
            metrics_text = step_match.group(2)
            for key, value in METRIC_RE.findall(metrics_text):
                if key != metric_name:
                    continue
                value = float(value)
                if math.isfinite(value):
                    points.append((step, value))
                break

    return points


def moving_average(values, window: int):
    if window <= 1:
        return values

    smoothed = []
    running_sum = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / len(queue))
    return smoothed


def safe_output_stem(root: Path, log_path: Path):
    try:
        rel = log_path.relative_to(root)
    except ValueError:
        rel = log_path.name

    rel_text = str(rel).replace("\\", "__").replace("/", "__")
    stem = Path(rel_text).with_suffix("").as_posix()
    digest = hashlib.sha1(str(log_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def plot_single_log(label: str, points, output_path: Path, metric_name: str, smooth_window: int):
    steps = [step for step, _ in points]
    values = [value for _, value in points]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, values, linewidth=1.1, alpha=0.45, label=metric_name)
    if smooth_window > 1 and len(values) > 1:
        plt.plot(
            steps,
            moving_average(values, smooth_window),
            linewidth=2.0,
            label=f"moving average ({smooth_window})",
        )
    plt.title(label)
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_all_logs(log_points, output_path: Path, metric_name: str, smooth_window: int):
    fig, ax = plt.subplots(figsize=(16, 8))

    for label, points in log_points:
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        if smooth_window > 1 and len(values) > 1:
            values = moving_average(values, smooth_window)
        ax.plot(steps, values, linewidth=1.5, alpha=0.85, label=label)

    suffix = f" smoothed ({smooth_window})" if smooth_window > 1 else ""
    ax.set_title(f"{metric_name}{suffix}")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.25)
    legend_cols = 1 if len(log_points) <= 20 else 2
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=legend_cols)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_points_csv(log_points, csv_path: Path, metric_name: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["log", "step", metric_name])
        for label, points in log_points:
            for step, value in points:
                writer.writerow([label, step, value])


def write_summary_csv(summary_rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "log",
                "points",
                "first_step",
                "last_step",
                "first_value",
                "last_value",
                "min_value",
                "max_value",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def find_logs(root: Path, pattern: str, recursive: bool):
    if root.is_file():
        return [root]
    globber = root.rglob if recursive else root.glob
    return sorted(path for path in globber(pattern) if path.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Plot critic/rewards/mean curves from all training logs under a directory."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=INPUT_PATH,
        help=f"Log file or directory to scan. Defaults to INPUT_PATH={INPUT_PATH}.",
    )
    parser.add_argument("--pattern", default=LOG_PATTERN, help=f"Log filename pattern. Defaults to {LOG_PATTERN}.")
    parser.add_argument(
        "--metric",
        default=METRIC_NAME,
        help=f"Metric key to plot. Defaults to METRIC_NAME={METRIC_NAME}.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help=f"Output directory. Defaults to {OUTPUT_DIR}.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=SMOOTH_WINDOW,
        help=f"Moving-average window. Defaults to SMOOTH_WINDOW={SMOOTH_WINDOW}.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_POINTS,
        help=f"Skip logs with fewer points than this. Defaults to MIN_POINTS={MIN_POINTS}.",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top level of the directory.")
    args = parser.parse_args()

    root = args.root.resolve()
    logs = find_logs(root, args.pattern, recursive=RECURSIVE and not args.no_recursive)
    if not logs:
        raise RuntimeError(f"No logs matched {args.pattern!r} under {root}")

    log_points = []
    summary_rows = []
    skipped = []
    per_log_dir = args.out_dir / "per_log"

    for log_path in logs:
        points = parse_metric_points(log_path, args.metric)
        if len(points) < args.min_points:
            skipped.append((log_path, len(points)))
            continue

        label = str(log_path.relative_to(root)) if root.is_dir() else log_path.name
        log_points.append((label, points))

        values = [value for _, value in points]
        summary_rows.append(
            {
                "log": label,
                "points": len(points),
                "first_step": points[0][0],
                "last_step": points[-1][0],
                "first_value": values[0],
                "last_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
            }
        )

        png_name = f"{safe_output_stem(root if root.is_dir() else log_path.parent, log_path)}.png"
        plot_single_log(label, points, per_log_dir / png_name, args.metric, args.smooth_window)

    if not log_points:
        raise RuntimeError(f"No logs contained at least {args.min_points} points for {args.metric!r}.")

    combined_png = args.out_dir / f"all_{args.metric.replace('/', '_')}.png"
    points_csv = args.out_dir / f"all_{args.metric.replace('/', '_')}_points.csv"
    summary_csv = args.out_dir / "summary.csv"

    plot_all_logs(log_points, combined_png, args.metric, args.smooth_window)
    write_points_csv(log_points, points_csv, args.metric)
    write_summary_csv(summary_rows, summary_csv)

    print(f"scanned_logs={len(logs)}")
    print(f"plotted_logs={len(log_points)}")
    print(f"skipped_logs={len(skipped)}")
    print(f"combined_png={combined_png}")
    print(f"per_log_dir={per_log_dir}")
    print(f"points_csv={points_csv}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()

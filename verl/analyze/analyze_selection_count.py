from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLLOUT_ROOT = Path(__file__).resolve().parents[1] / "rollout_logs"
DEFAULT_LOG_ROOT = Path(__file__).resolve().parent / "logs"
UNKNOWN_METHOD = "unknown"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ROLLOUT_SUFFIX_RE = re.compile(r"rollout_logs/([^\s'\"),]+)")
SELECT_METHOD_RE = re.compile(
    r"(?:custom_training_module\.kwargs\.)?select_method(?:=|['\"]?\s*:\s*['\"])([A-Za-z0-9_\-]+)"
)
UPDATE_COUNT_RE = re.compile(
    r"(?:custom_training_module\.kwargs\.)?n_update_samples_per_prompt(?:=|['\"]?\s*:\s*)(\d+)"
)
MODEL_RE = re.compile(r"(qwen\d+(?:[-.]\d+(?:\.\d+)?)?[bB])")


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {file_path} line {line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_csv(file_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _step_from_path(file_path: Path) -> int | None:
    return _safe_int(file_path.stem)


def _step_sort_value(value: Any) -> int:
    parsed = _safe_int(value)
    if parsed is None:
        return 10**18
    return parsed


def _file_sort_key(file_path: Path) -> tuple[str, str, int, str]:
    return (str(file_path.parent), file_path.parent.name, _step_sort_value(file_path.stem), file_path.name)


def _clean_method(value: Any) -> str:
    if value is None:
        return UNKNOWN_METHOD
    method = str(value).strip()
    return method or UNKNOWN_METHOD


def _most_common(values: Iterable[Any], default: Any = None) -> Any:
    cleaned = [value for value in values if value not in (None, "")]
    if not cleaned:
        return default
    return Counter(cleaned).most_common(1)[0][0]


def _derive_model(experiment: str) -> str:
    match = MODEL_RE.search(experiment)
    if match:
        return match.group(1).upper().replace("QWEN", "qwen")
    return "unknown_model"


def _derive_experiment_dataset(file_path: Path, rollout_root: Path) -> tuple[str, str, str, str]:
    try:
        rel = file_path.relative_to(rollout_root)
    except ValueError:
        rel = file_path
    experiment = rel.parts[0] if len(rel.parts) >= 1 else ""
    dataset = rel.parts[1] if len(rel.parts) >= 2 else ""
    model = _derive_model(experiment)
    suffix = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else experiment
    return model, experiment, dataset, suffix


def _question_key(row: dict[str, Any], fallback_index: int) -> str:
    group_key = row.get("group_key")
    if group_key not in (None, ""):
        return str(group_key)
    question_id = row.get("question_id")
    if question_id not in (None, ""):
        return str(question_id)
    return f"__missing_group_key_{fallback_index}"


def _first_question_id(rows: list[dict[str, Any]]) -> Any:
    for row in rows:
        question_id = row.get("question_id")
        if question_id not in (None, ""):
            return question_id
    return ""


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_avg_selected_rollouts_per_question": 0.0,
            "median_avg_selected_rollouts_per_question": 0.0,
            "min_avg_selected_rollouts_per_question": 0.0,
            "max_avg_selected_rollouts_per_question": 0.0,
        }
    return {
        "mean_avg_selected_rollouts_per_question": float(statistics.mean(values)),
        "median_avg_selected_rollouts_per_question": float(statistics.median(values)),
        "min_avg_selected_rollouts_per_question": float(min(values)),
        "max_avg_selected_rollouts_per_question": float(max(values)),
    }


def _rollout_suffix_from_text(text: str) -> str | None:
    match = ROLLOUT_SUFFIX_RE.search(text)
    if not match:
        return None
    suffix = match.group(1).strip().strip("'\"),]")
    parts = [part for part in suffix.split("/") if part and part != "${ROLLOUT_LOG_DIR}"]
    if not parts:
        return None
    return "/".join(parts)


def _parse_logs(log_root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if not log_root.exists():
        return metadata

    for log_file in sorted(log_root.rglob("*.log")):
        current_suffix: str | None = None
        current_method: str | None = None
        current_update_count: int | None = None

        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = ANSI_RE.sub("", raw_line)
                suffix = _rollout_suffix_from_text(line)
                if suffix is not None:
                    current_suffix = suffix

                method_match = SELECT_METHOD_RE.search(line)
                if method_match:
                    current_method = method_match.group(1)

                update_count_match = UPDATE_COUNT_RE.search(line)
                if update_count_match:
                    current_update_count = int(update_count_match.group(1))

        if current_suffix is None:
            continue

        metadata[current_suffix] = {
            "configured_select_method_from_log": current_method or "",
            "configured_n_update_samples_per_prompt": current_update_count if current_update_count is not None else "",
            "source_log": str(log_file),
        }
    return metadata


def _find_log_metadata(suffix: str, metadata_by_suffix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if suffix in metadata_by_suffix:
        return metadata_by_suffix[suffix]

    best_key = ""
    for key in metadata_by_suffix:
        if suffix.endswith(key) or key.endswith(suffix):
            if len(key) > len(best_key):
                best_key = key
    if best_key:
        return metadata_by_suffix[best_key]
    return {
        "configured_select_method_from_log": "",
        "configured_n_update_samples_per_prompt": "",
        "source_log": "",
    }


def _build_rows_for_file(
    file_path: Path,
    rollout_root: Path,
    drop_unknown: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_jsonl(file_path)
    if not rows:
        return [], []

    model, experiment, dataset, rollout_suffix = _derive_experiment_dataset(file_path, rollout_root)
    fallback_step = _step_from_path(file_path)
    rows_by_method: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        method = _clean_method(row.get("select_method"))
        if drop_unknown and method == UNKNOWN_METHOD:
            continue
        rows_by_method[method].append((index, row))

    step_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []

    for method, indexed_rows in rows_by_method.items():
        method_rows = [row for _, row in indexed_rows]
        step = _most_common((_safe_int(row.get("step")) for row in method_rows), default=fallback_step)
        step = fallback_step if step is None else step

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fallback_index, row in indexed_rows:
            groups[_question_key(row, fallback_index)].append(row)

        selected_rollouts = sum(_safe_bool(row.get("selected_for_update")) for row in method_rows)
        total_rollouts = len(method_rows)
        question_count = len(groups)
        base_selected_rollouts = sum(_safe_bool(row.get("selector_base_selected_for_update")) for row in method_rows)
        ablation_added_rollouts = sum(_safe_bool(row.get("selector_ablation_added_for_update")) for row in method_rows)

        for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
            group_selected = sum(_safe_bool(row.get("selected_for_update")) for row in group_rows)
            group_total = len(group_rows)
            question_rows.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "dataset": dataset,
                    "step": step,
                    "select_method": method,
                    "group_key": group_key,
                    "question_id": _first_question_id(group_rows),
                    "rollout_count": group_total,
                    "selected_rollouts": group_selected,
                    "unselected_rollouts": group_total - group_selected,
                    "selected_ratio": _ratio(group_selected, group_total),
                    "path": str(file_path),
                }
            )

        step_rows.append(
            {
                "model": model,
                "experiment": experiment,
                "dataset": dataset,
                "step": step,
                "select_method": method,
                "question_count": question_count,
                "total_rollouts": total_rollouts,
                "selected_rollouts": selected_rollouts,
                "unselected_rollouts": total_rollouts - selected_rollouts,
                "avg_selected_rollouts_per_question": _ratio(selected_rollouts, question_count),
                "avg_rollouts_per_question": _ratio(total_rollouts, question_count),
                "selected_ratio": _ratio(selected_rollouts, total_rollouts),
                "base_selected_rollouts": base_selected_rollouts,
                "ablation_added_rollouts": ablation_added_rollouts,
                "rollout_suffix": rollout_suffix,
                "path": str(file_path),
            }
        )

    return step_rows, question_rows


def _aggregate_experiments(
    step_rows: list[dict[str, Any]], metadata_by_suffix: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    per_step_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    steps: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    suffixes: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)

    for row in step_rows:
        key = (str(row["model"]), str(row["experiment"]), str(row["dataset"]), str(row["select_method"]))
        if key not in grouped:
            grouped[key] = {
                "model": row["model"],
                "experiment": row["experiment"],
                "dataset": row["dataset"],
                "select_method": row["select_method"],
                "file_count": 0,
                "step_count": 0,
                "first_step": "",
                "last_step": "",
                "total_questions": 0,
                "total_rollouts": 0,
                "selected_rollouts": 0,
                "unselected_rollouts": 0,
                "selected_ratio": 0.0,
                "base_selected_rollouts": 0,
                "ablation_added_rollouts": 0,
            }

        agg = grouped[key]
        agg["file_count"] += 1
        agg["total_questions"] += int(row["question_count"])
        agg["total_rollouts"] += int(row["total_rollouts"])
        agg["selected_rollouts"] += int(row["selected_rollouts"])
        agg["unselected_rollouts"] += int(row["unselected_rollouts"])
        agg["base_selected_rollouts"] += int(row["base_selected_rollouts"])
        agg["ablation_added_rollouts"] += int(row["ablation_added_rollouts"])
        per_step_values[key].append(float(row["avg_selected_rollouts_per_question"]))
        step = _safe_int(row.get("step"))
        if step is not None:
            steps[key].add(step)
        suffixes[key].add(str(row.get("rollout_suffix", "")))

    output_rows: list[dict[str, Any]] = []
    for key, agg in grouped.items():
        step_list = sorted(steps[key])
        agg["step_count"] = len(step_list)
        agg["first_step"] = step_list[0] if step_list else ""
        agg["last_step"] = step_list[-1] if step_list else ""
        agg["selected_ratio"] = _ratio(agg["selected_rollouts"], agg["total_rollouts"])
        agg.update(_numeric_summary(per_step_values[key]))

        metadata = {
            "configured_select_method_from_log": "",
            "configured_n_update_samples_per_prompt": "",
            "source_log": "",
        }
        for suffix in sorted(suffixes[key], key=len, reverse=True):
            metadata = _find_log_metadata(suffix, metadata_by_suffix)
            if metadata.get("source_log") or metadata.get("configured_select_method_from_log"):
                break
        agg.update(metadata)
        output_rows.append(agg)

    return sorted(output_rows, key=lambda row: (row["model"], row["experiment"], row["dataset"], row["select_method"]))


def _aggregate_methods(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    per_step_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    experiments: dict[tuple[str, str], set[str]] = defaultdict(set)
    steps: dict[tuple[str, str], set[tuple[str, str, int]]] = defaultdict(set)

    for row in step_rows:
        model = str(row["model"])
        method = str(row["select_method"])
        key = (model, method)
        if key not in grouped:
            grouped[key] = {
                "model": model,
                "select_method": method,
                "experiment_count": 0,
                "step_count": 0,
                "file_count": 0,
                "total_questions": 0,
                "total_rollouts": 0,
                "selected_rollouts": 0,
                "unselected_rollouts": 0,
                "selected_ratio": 0.0,
                "base_selected_rollouts": 0,
                "ablation_added_rollouts": 0,
            }

        agg = grouped[key]
        agg["file_count"] += 1
        agg["total_questions"] += int(row["question_count"])
        agg["total_rollouts"] += int(row["total_rollouts"])
        agg["selected_rollouts"] += int(row["selected_rollouts"])
        agg["unselected_rollouts"] += int(row["unselected_rollouts"])
        agg["base_selected_rollouts"] += int(row["base_selected_rollouts"])
        agg["ablation_added_rollouts"] += int(row["ablation_added_rollouts"])
        per_step_values[key].append(float(row["avg_selected_rollouts_per_question"]))
        experiments[key].add(f"{row['experiment']}/{row['dataset']}")
        step = _safe_int(row.get("step"))
        if step is not None:
            steps[key].add((str(row["experiment"]), str(row["dataset"]), step))

    output_rows: list[dict[str, Any]] = []
    for key, agg in grouped.items():
        agg["experiment_count"] = len(experiments[key])
        agg["step_count"] = len(steps[key])
        agg["selected_ratio"] = _ratio(agg["selected_rollouts"], agg["total_rollouts"])
        agg.update(_numeric_summary(per_step_values[key]))
        output_rows.append(agg)

    return sorted(output_rows, key=lambda row: (row["model"], row["select_method"]))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start : start + stride])
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _blank_canvas(width: int, height: int) -> bytearray:
    return bytearray([255, 255, 255] * width * height)


def _set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    x, y = x0, y0
    while True:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                _set_pixel(pixels, width, height, x + ox, y + oy, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * error
        if e2 >= dy:
            error += dy
            x += sx
        if e2 <= dx:
            error += dx
            y += sy


def _draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    left, right = sorted((max(0, x0), min(width - 1, x1)))
    top, bottom = sorted((max(0, y0), min(height - 1, y1)))
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            _set_pixel(pixels, width, height, x, y, color)


def _fallback_line_plot(path: Path, series: dict[str, list[tuple[int, float]]]) -> None:
    width, height = 1400, 800
    left, right, top, bottom = 80, 40, 40, 80
    pixels = _blank_canvas(width, height)
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    if not all_x or not all_y:
        return
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(0.0, min(all_y)), max(all_y)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1.0

    axis_color = (30, 30, 30)
    grid_color = (225, 225, 225)
    _draw_line(pixels, width, height, left, top, left, height - bottom, axis_color)
    _draw_line(pixels, width, height, left, height - bottom, width - right, height - bottom, axis_color)
    for i in range(1, 5):
        y = top + int(plot_height * i / 5)
        _draw_line(pixels, width, height, left, y, width - right, y, grid_color)

    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    for series_index, (_, points) in enumerate(sorted(series.items())):
        points = sorted(points)
        color = palette[series_index % len(palette)]
        scaled: list[tuple[int, int]] = []
        for x_value, y_value in points:
            x = left + int((x_value - min_x) / (max_x - min_x) * plot_width)
            y = height - bottom - int((y_value - min_y) / (max_y - min_y) * plot_height)
            scaled.append((x, y))
        for (x0, y0), (x1, y1) in zip(scaled, scaled[1:]):
            _draw_line(pixels, width, height, x0, y0, x1, y1, color)
        for x, y in scaled:
            _draw_rect(pixels, width, height, x - 3, y - 3, x + 3, y + 3, color)

    _write_png(path, width, height, pixels)


def _fallback_bar_plot(path: Path, values: list[float]) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 80, 40, 40, 80
    pixels = _blank_canvas(width, height)
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0
    _draw_line(pixels, width, height, left, top, left, height - bottom, (30, 30, 30))
    _draw_line(pixels, width, height, left, height - bottom, width - right, height - bottom, (30, 30, 30))
    bar_count = max(len(values), 1)
    slot_width = max(plot_width // bar_count, 1)
    bar_width = max(int(slot_width * 0.65), 1)
    for index, value in enumerate(values):
        x0 = left + index * slot_width + (slot_width - bar_width) // 2
        x1 = x0 + bar_width
        y1 = height - bottom
        y0 = y1 - int(float(value) / max_value * plot_height)
        _draw_rect(pixels, width, height, x0, y0, x1, y1, (31, 119, 180))
    _write_png(path, width, height, pixels)


def _plot_outputs(
    output_dir: Path,
    step_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
) -> list[Path]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in step_rows:
        step = _safe_int(row.get("step"))
        if step is None:
            continue
        label = f"{row['experiment']}/{row['dataset']}/{row['select_method']}"
        series[label].append((step, float(row["avg_selected_rollouts_per_question"])))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        print(f"matplotlib unavailable, writing simple PNG plots instead: {exc}")
        saved: list[Path] = []
        if series:
            path = output_dir / "avg_selected_rollouts_per_question_by_step.png"
            _fallback_line_plot(path, series)
            saved.append(path)
        if method_rows:
            path = output_dir / "mean_avg_selected_rollouts_per_question_by_method.png"
            _fallback_bar_plot(path, [float(row["mean_avg_selected_rollouts_per_question"]) for row in method_rows])
            saved.append(path)
            path = output_dir / "selected_rollouts_by_method.png"
            _fallback_bar_plot(path, [float(row["selected_rollouts"]) for row in method_rows])
            saved.append(path)
        if experiment_rows:
            path = output_dir / "mean_avg_selected_rollouts_per_question_by_model_experiment_method.png"
            _fallback_bar_plot(path, [float(row["mean_avg_selected_rollouts_per_question"]) for row in experiment_rows])
            saved.append(path)
            path = output_dir / "selected_rollouts_by_model_experiment_method.png"
            _fallback_bar_plot(path, [float(row["selected_rollouts"]) for row in experiment_rows])
            saved.append(path)
        return saved

    saved: list[Path] = []

    if series:
        plt.figure(figsize=(14, 8))
        for label, points in sorted(series.items()):
            points = sorted(points)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            plt.plot(xs, ys, marker="o", markersize=2, linewidth=1, label=label)
        plt.xlabel("Step")
        plt.ylabel("Average selected rollouts per question")
        plt.title("Average selected rollouts per question by step")
        plt.grid(True, alpha=0.3)
        if len(series) <= 20:
            plt.legend(fontsize=7)
        else:
            plt.legend([], [], frameon=False)
            plt.figtext(0.01, 0.01, f"Legend omitted: {len(series)} series", fontsize=8)
        plt.tight_layout()
        path = output_dir / "avg_selected_rollouts_per_question_by_step.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(path)

    if method_rows:
        labels = [f"{row['model']} / {row['select_method']}" for row in method_rows]
        mean_values = [float(row["mean_avg_selected_rollouts_per_question"]) for row in method_rows]
        selected_values = [int(row["selected_rollouts"]) for row in method_rows]

        plt.figure(figsize=(max(10, len(labels) * 0.8), 6))
        plt.bar(labels, mean_values)
        plt.xlabel("Model / selector method")
        plt.ylabel("Mean per-step avg selected rollouts per question")
        plt.title("Mean selected rollouts per question by model and selector method")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = output_dir / "mean_avg_selected_rollouts_per_question_by_method.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(path)

        plt.figure(figsize=(max(10, len(labels) * 0.8), 6))
        plt.bar(labels, selected_values)
        plt.xlabel("Model / selector method")
        plt.ylabel("Total selected rollouts")
        plt.title("Total selected rollouts by model and selector method")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = output_dir / "selected_rollouts_by_method.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(path)

    if experiment_rows:
        labels = [f"{row['model']} / {row['experiment']} / {row['select_method']}" for row in experiment_rows]
        mean_values = [float(row["mean_avg_selected_rollouts_per_question"]) for row in experiment_rows]
        selected_values = [int(row["selected_rollouts"]) for row in experiment_rows]

        plt.figure(figsize=(max(12, len(labels) * 0.7), 7))
        plt.bar(labels, mean_values)
        plt.xlabel("Model / experiment / selector method")
        plt.ylabel("Mean per-step avg selected rollouts per question")
        plt.title("Mean selected rollouts per question by model, experiment, and selector method")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        path = output_dir / "mean_avg_selected_rollouts_per_question_by_model_experiment_method.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(path)

        plt.figure(figsize=(max(12, len(labels) * 0.7), 7))
        plt.bar(labels, selected_values)
        plt.xlabel("Model / experiment / selector method")
        plt.ylabel("Total selected rollouts")
        plt.title("Total selected rollouts by model, experiment, and selector method")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        path = output_dir / "selected_rollouts_by_model_experiment_method.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(path)

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many rollout samples each GRPO-dropout selector method selects for gradient updates."
    )
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT, help="Root directory of rollout JSONL logs.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root directory of training logs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV and PNG outputs.")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for rollout files.")
    parser.add_argument("--experiment-filter", default=None, help="Only analyze rollout paths containing this substring.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of JSONL files to analyze.")
    parser.add_argument("--drop-unknown", action="store_true", help="Skip rows without a select_method field.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rollout_root: Path = args.rollout_root
    output_dir: Path = args.output_dir or rollout_root / "selection_count_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(rollout_root.rglob(args.pattern), key=_file_sort_key)
    if args.experiment_filter:
        files = [file_path for file_path in files if args.experiment_filter in str(file_path)]
    if args.max_files is not None:
        files = files[: args.max_files]

    metadata_by_suffix = _parse_logs(args.log_root)
    step_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    skipped_empty = 0

    for file_path in files:
        file_step_rows, file_question_rows = _build_rows_for_file(
            file_path=file_path,
            rollout_root=rollout_root,
            drop_unknown=args.drop_unknown,
        )
        if not file_step_rows and not file_question_rows:
            skipped_empty += 1
            continue
        step_rows.extend(file_step_rows)
        question_rows.extend(file_question_rows)

    step_rows.sort(key=lambda row: (row["model"], row["experiment"], row["dataset"], _step_sort_value(row["step"]), row["select_method"]))
    question_rows.sort(
        key=lambda row: (
            row["model"],
            row["experiment"],
            row["dataset"],
            _step_sort_value(row["step"]),
            row["select_method"],
            row["group_key"],
        )
    )
    experiment_rows = _aggregate_experiments(step_rows, metadata_by_suffix)
    method_rows = _aggregate_methods(step_rows)

    output_paths = {
        "selection_counts_by_step.csv": step_rows,
        "selection_counts_by_question.csv": question_rows,
        "selection_counts_by_model_experiment_method.csv": experiment_rows,
        "selection_counts_by_experiment.csv": experiment_rows,
        "selection_counts_by_method.csv": method_rows,
    }
    for file_name, rows in output_paths.items():
        _write_csv(output_dir / file_name, rows)

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = _plot_outputs(output_dir, step_rows, method_rows, experiment_rows)

    print(f"Analyzed {len(files)} rollout files from {rollout_root}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} empty or filtered files")
    for file_name, rows in output_paths.items():
        if rows:
            print(f"Saved: {output_dir / file_name}")
    for path in plot_paths:
        print(f"Saved: {path}")

    if experiment_rows:
        print("\nSelection counts by model / experiment / method:")
        for row in experiment_rows:
            print(
                f"{row['model']} / {row['experiment']} / {row['select_method']}: "
                f"selected={row['selected_rollouts']} / total={row['total_rollouts']} "
                f"ratio={row['selected_ratio']:.4f} "
                f"mean_avg_selected_per_question={row['mean_avg_selected_rollouts_per_question']:.4f}"
            )


if __name__ == "__main__":
    main()

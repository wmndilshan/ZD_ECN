#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PAPER = {
    "device": "TECNO KL4, ARM Mali-G57, Android 14, OpenGL ES 3.2",
    "gpu_mean_ms": 1.324,
    "gpu_p95_ms": 1.369,
    "gpu_p99_ms": 1.398,
    "fps_from_mean": 755.3,
    "fps_from_p99": 715.3,
}


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values):
    if not values:
        return {
            "n": 0,
            "mean_ms": None,
            "stddev_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "coefficient_of_variation": None,
            "confidence_interval_95_ms": None,
            "fps_from_mean": None,
            "fps_from_p99": None,
        }

    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    sem = stddev / math.sqrt(len(values)) if values else None
    ci_half_width = 1.96 * sem if sem is not None else None
    p99 = percentile(values, 99)
    return {
        "n": len(values),
        "mean_ms": mean,
        "stddev_ms": stddev,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": p99,
        "coefficient_of_variation": stddev / mean if mean else None,
        "confidence_interval_95_ms": [
            mean - ci_half_width,
            mean + ci_half_width,
        ] if ci_half_width is not None else None,
        "fps_from_mean": 1000.0 / mean if mean else None,
        "fps_from_p99": 1000.0 / p99 if p99 else None,
    }


def fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, list):
        return f"[{fmt(value[0], digits)}, {fmt(value[1], digits)}]"
    return f"{value:.{digits}f}"


def read_sessions(paths):
    rows = []
    for session_index, path in enumerate(paths, start=1):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iteration = int(row.get("run") or len(rows) + 1)
                cpu_value = parse_float(row.get("mlp_cpu_ms"))
                if cpu_value is None:
                    cpu_value = parse_float(row.get("cpu_ms"))
                rows.append({
                    "session": session_index,
                    "iteration": iteration,
                    "cpu_ms": cpu_value,
                    "gpu_ms": parse_float(row.get("gpu_ms")),
                    "source_file": str(path),
                })
    return rows


def write_raw_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["session", "iteration", "cpu_ms", "gpu_ms", "source_file"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_text(path):
    if path.exists():
        return path.read_text(errors="replace")
    return ""


def write_markdown(path, summary):
    gpu = summary["gpu"]
    cpu = summary["cpu"]
    lines = [
        "# ZD-ECN On-Device Benchmark Summary",
        "",
        "## Protocol",
        "",
        "- Compute path: OpenGL ES / GLSL ES compute shader MLP deformer.",
        "- Architecture: `144 → 64 → 64 → 20,670`.",
        "- Input: 24 joints × 6D rotation = 144 floats.",
        "- Output: 6,890 SMPL vertices × 3 displacement values = 20,670 floats.",
        "- Sessions requested: 5 independent sessions.",
        "- Warmup per session: 100 iterations discarded by the app.",
        "- Measured per session: 1000 iterations.",
        "- Weight mode: fp16 if supported by repo/app build.",
        "",
        "## Results",
        "",
        "| Metric | GPU | CPU | Paper GPU Reference |",
        "|---|---:|---:|---:|",
        f"| Samples | {gpu['n']} | {cpu['n']} | 5000 |",
        f"| Mean latency (ms) | {fmt(gpu['mean_ms'])} | {fmt(cpu['mean_ms'])} | {PAPER['gpu_mean_ms']:.3f} |",
        f"| Std dev (ms) | {fmt(gpu['stddev_ms'])} | {fmt(cpu['stddev_ms'])} | NA |",
        f"| p50 (ms) | {fmt(gpu['p50_ms'])} | {fmt(cpu['p50_ms'])} | NA |",
        f"| p95 (ms) | {fmt(gpu['p95_ms'])} | {fmt(cpu['p95_ms'])} | {PAPER['gpu_p95_ms']:.3f} |",
        f"| p99 (ms) | {fmt(gpu['p99_ms'])} | {fmt(cpu['p99_ms'])} | {PAPER['gpu_p99_ms']:.3f} |",
        f"| Coefficient of variation | {fmt(gpu['coefficient_of_variation'], 6)} | {fmt(cpu['coefficient_of_variation'], 6)} | NA |",
        f"| 95% CI (ms) | {fmt(gpu['confidence_interval_95_ms'])} | {fmt(cpu['confidence_interval_95_ms'])} | NA |",
        f"| FPS from mean | {fmt(gpu['fps_from_mean'], 1)} | {fmt(cpu['fps_from_mean'], 1)} | {PAPER['fps_from_mean']:.1f} |",
        f"| FPS from p99 | {fmt(gpu['fps_from_p99'], 1)} | {fmt(cpu['fps_from_p99'], 1)} | {PAPER['fps_from_p99']:.1f} |",
        "",
        "## Paper Reference",
        "",
        f"- Device: {PAPER['device']}.",
        f"- Mean GPU latency: {PAPER['gpu_mean_ms']:.3f} ms.",
        f"- p95 / p99: {PAPER['gpu_p95_ms']:.3f} ms / {PAPER['gpu_p99_ms']:.3f} ms.",
        f"- Throughput: {PAPER['fps_from_mean']:.1f} FPS mean, {PAPER['fps_from_p99']:.1f} FPS p99.",
        "",
        "## Mismatch Notes",
        "",
    ]
    if gpu["n"] == 0:
        lines.append("- GPU timer samples are unavailable; check `GL_EXT_disjoint_timer_query` support and app logs.")
    else:
        mean_delta = gpu["mean_ms"] - PAPER["gpu_mean_ms"]
        p99_delta = gpu["p99_ms"] - PAPER["gpu_p99_ms"]
        lines.append(f"- Mean GPU latency delta vs paper: {mean_delta:+.3f} ms.")
        lines.append(f"- p99 GPU latency delta vs paper: {p99_delta:+.3f} ms.")
        lines.append("- Likely causes for mismatch include different GPU, thermal state, debug build overhead, timer-query availability, and driver scheduling differences.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_notes(path, output_root, session_paths, device_info_text, zero_dependency_text, failures):
    lines = [
        "# Reproduction Notes",
        "",
        "## Repository Findings",
        "",
        "- Android app: `android_compute_benchmark`.",
        "- Native benchmark core: `android_compute_benchmark/app/src/main/cpp/main.cpp`.",
        "- Compute shaders: `android_compute_benchmark/app/src/main/assets/shaders/*.comp`.",
        "- On-device inference uses OpenGL ES compute shaders and SSBOs.",
        "- No Python, ONNX Runtime, TensorRT, TFLite, or libtorch path is used for on-device inference.",
        "",
        "## Input Files",
        "",
    ]
    for path_item in session_paths:
        lines.append(f"- `{path_item}`")
    lines.extend([
        "",
        "## Device Info",
        "",
        "```text",
        device_info_text.strip() or "Unavailable",
        "```",
        "",
        "## Zero-Dependency APK Check",
        "",
        "```text",
        zero_dependency_text.strip() or "Unavailable",
        "```",
        "",
        "## Failures Or Limitations",
        "",
    ])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None recorded by parser.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def maybe_write_plots(rows, plots_dir):
    gpu_rows = [row for row in rows if row["gpu_ms"] is not None]
    if not gpu_rows:
        return ["Skipped plots: no valid GPU samples."]
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Skipped plots: matplotlib unavailable ({exc})."]

    plots_dir.mkdir(parents=True, exist_ok=True)
    gpu_values = [row["gpu_ms"] for row in gpu_rows]
    ordered = sorted(gpu_values)
    cdf_y = [(index + 1) / len(ordered) for index in range(len(ordered))]

    plt.figure(figsize=(8, 5))
    plt.plot(ordered, cdf_y)
    plt.xlabel("GPU latency (ms)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "latency_cdf.png", dpi=160)
    plt.close()

    by_session = {}
    for row in gpu_rows:
        by_session.setdefault(row["session"], []).append(row["gpu_ms"])
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [by_session[key] for key in sorted(by_session)],
        tick_labels=[str(key) for key in sorted(by_session)],
    )
    plt.xlabel("Session")
    plt.ylabel("GPU latency (ms)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "per_session_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(gpu_values, bins=50)
    plt.xlabel("GPU latency (ms)")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "latency_histogram.png", dpi=160)
    plt.close()
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="benchmark_results")
    parser.add_argument("--session-csv", nargs="+", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    session_paths = [Path(item) for item in args.session_csv]
    rows = read_sessions(session_paths)
    gpu_values = [row["gpu_ms"] for row in rows if row["gpu_ms"] is not None]
    cpu_values = [row["cpu_ms"] for row in rows if row["cpu_ms"] is not None]

    failures = []
    expected_samples = len(session_paths) * 1000
    if len(rows) != expected_samples:
        failures.append(f"Expected {expected_samples} measured rows from {len(session_paths)} sessions, found {len(rows)}.")
    if not gpu_values:
        failures.append("No valid GPU samples found; hardware timer results are unavailable.")

    raw_csv = output_root / "raw" / "benchmark_raw.csv"
    write_raw_csv(raw_csv, rows)

    device_info_text = load_text(output_root / "raw" / "device_info.txt")
    zero_dependency_text = load_text(output_root / "raw" / "zero_dependency_check.txt")
    plot_failures = maybe_write_plots(rows, output_root / "plots")
    failures.extend(plot_failures)

    summary = {
        "paper_reference": PAPER,
        "protocol": {
            "sessions": len(session_paths),
            "warmup_iterations_per_session": 100,
            "measured_iterations_per_session": 1000,
            "expected_samples": expected_samples,
            "architecture": "144 -> 64 -> 64 -> 20670",
            "input_representation": "6D rotation, 24 joints x 6 = 144",
            "output_representation": "6890 SMPL vertices x 3 displacement values = 20670",
        },
        "gpu": summarize(gpu_values),
        "cpu": summarize(cpu_values),
        "files": {
            "raw_csv": str(raw_csv),
            "device_info": str(output_root / "raw" / "device_info.txt"),
            "markdown_summary": str(output_root / "summary" / "benchmark_summary.md"),
            "reproduction_notes": str(output_root / "summary" / "reproduction_notes.md"),
        },
        "failures_or_limitations": failures,
    }

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary_dir / "benchmark_summary.md", summary)
    write_notes(
        summary_dir / "reproduction_notes.md",
        output_root,
        session_paths,
        device_info_text,
        zero_dependency_text,
        failures,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

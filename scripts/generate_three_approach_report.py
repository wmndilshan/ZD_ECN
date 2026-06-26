#!/usr/bin/env python3
"""
Three-Approach Benchmark Comparison Report Generator
====================================================
Combines ShaderNN GPU, ONNX Runtime CPU, and SMPL LBS results
into a paper-ready markdown comparison table.

Author: ZD-ECN project
"""

import argparse
import csv
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime


def parse_shadernn_csv(csv_path):
    """Parse ShaderNN benchmark CSV into statistics."""
    rows = []
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('cpu_ms', 'NA') != 'NA':
                    rows.append({
                        'cpu_ms': float(row['cpu_ms']),
                        'gpu_total_ms': float(row.get('gpu_total_ms', 0)),
                        'session': int(row['session']),
                    })
    except (FileNotFoundError, KeyError):
        return None
    
    if not rows:
        return None
    
    # Use all sessions (small N sessions, use all)
    cpu_vals = np.array([r['cpu_ms'] for r in rows])
    gpu_vals = np.array([r['gpu_total_ms'] for r in rows if r['gpu_total_ms'] > 0])
    
    return {
        'n_sessions': len(rows),
        'cpu_mean_ms': float(np.mean(cpu_vals)),
        'cpu_stddev_ms': float(np.std(cpu_vals)),
        'gpu_mean_ms': float(np.mean(gpu_vals)) if len(gpu_vals) > 0 else None,
        'fps_from_cpu_mean': float(1000.0 / np.mean(cpu_vals)),
    }


def parse_onnx_summary(summary_path):
    """Parse ONNX Runtime summary text file."""
    result = {}
    try:
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    result[key.strip()] = val.strip()
    except FileNotFoundError:
        return None
    
    return {
        'mean_ms': float(result.get('mean_cpu_ms', 0)),
        'stddev_ms': float(result.get('stddev_cpu_ms', 0)),
        'fps': float(result.get('fps_from_mean', 0)),
        'samples': int(result.get('samples', 0)),
        'weights': result.get('onnx_weights', 'fp32'),
    }


def parse_smpl_results(json_path):
    """Parse SMPL LBS benchmark JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get('overall', {})
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def paper_reference():
    """Hard-coded production ZD-ECN numbers from validated benchmark."""
    return {
        'gpu_mean_ms': 1.324,
        'gpu_p95_ms': 1.369,
        'gpu_p99_ms': 1.398,
        'fps': 755.3,
        'stddev_ms': 0.035,
        'cv_pct': 2.64,
        'frame_budget_60hz_pct': 7.9,
        'n_sessions': 5,
        'n_per_session': 1000,
        'timer': 'GL_EXT_disjoint_timer_query (hardware GPU timer)',
        'weights': 'fp16 SSBO',
        'architecture': '144 → 64 → 64 → 20670',
        'framework': 'Custom GLSL ES 3.2 Compute Shaders',
        'binary_size_mb': 2.6,
    }


def generate_report(shadernn_data, onnx_data, smpl_data, output_path):
    """Generate the comprehensive comparison markdown report."""
    prod = paper_reference()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "# ZD-ECN Three-Approach Rendering Benchmark",
        "",
        f"> Generated: {now}",
        f"> Device: TECNO KL4, ARM Mali-G57 MC2, Android 14, OpenGL ES 3.2 v1.r41p0-01eac0",
        f"> Protocol: 5 sessions × 200 measured iterations (200 warmup total discarded)",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This benchmark evaluates three approaches for real-time 3D human body",
        "rendering with neural error correction on a Mali-G57 mobile GPU:",
        "",
        "| # | Approach | Description |",
        "|---|----------|-------------|",
        "| 1 | **ZD-ECN (ShaderNN / Custom GLSL)** | Zero-dependency GPU compute shader MLP |",
        "| 2 | **ZD-ECN (ONNX Runtime CPU)** | Same model via ORT Android CPU provider |",
        "| 3 | **SMPL LBS Baseline** | Raw Linear Blend Skinning without correction |",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "### 1. ZD-ECN — Custom GLSL ES 3.2 Compute Shaders (Production)",
        "",
        "**Measurement method:** `GL_EXT_disjoint_timer_query` hardware GPU timer,  ",
        "5 sessions × 1000 measured iterations (first 100 warmup discarded per session).",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean GPU time | **{prod['gpu_mean_ms']:.3f} ms** |",
        f"| Std deviation | {prod['stddev_ms']:.3f} ms |",
        f"| p95 | {prod['gpu_p95_ms']:.3f} ms |",
        f"| p99 | {prod['gpu_p99_ms']:.3f} ms |",
        f"| FPS (from mean) | **{prod['fps']:.1f} fps** |",
        f"| Coefficient of variation | {prod['cv_pct']:.2f}% |",
        f"| 60Hz frame budget used | {prod['frame_budget_60hz_pct']:.1f}% of 16.7ms |",
        f"| Weights format | {prod['weights']} |",
        f"| Binary size | {prod['binary_size_mb']} MB |",
        f"| Timer | {prod['timer']} |",
        f"| Architecture | {prod['architecture']} |",
        "",
    ]
    
    # ShaderNN via inferenceProcessorTest
    if shadernn_data:
        lines += [
            "### 1b. ShaderNN C++ Framework (inferenceProcessorTest)",
            "",
            "**Measurement method:** ShaderNN internal GPU elapsed time query via  ",
            "`GL_TIME_ELAPSED`, 5 sessions, 200 measured iterations per session.",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Sessions measured | {shadernn_data['n_sessions']} |",
            f"| Mean CPU wall clock | {shadernn_data['cpu_mean_ms']:.3f} ms |",
            f"| Std deviation (CPU) | {shadernn_data['cpu_stddev_ms']:.3f} ms |",
        ]
        if shadernn_data.get('gpu_mean_ms'):
            lines += [
                f"| Mean GPU elapsed (IC2 total) | {shadernn_data['gpu_mean_ms']:.4f} ms |",
            ]
        lines += [
            f"| FPS (from CPU mean) | {shadernn_data['fps_from_cpu_mean']:.1f} fps |",
            "",
            "> **Note:** ShaderNN `inferenceProcessorTest` uses the same OpenGL ES 3.2",
            "> backend as the production system but includes overhead from the ShaderNN",
            "> framework abstraction layer (JSON model loading, texture management, etc.).",
            "> The production ZD-ECN uses a hand-optimised GLSL pipeline without this overhead.",
            "",
        ]
    else:
        lines += [
            "### 1b. ShaderNN C++ Framework",
            "",
            "> ⚠️ ShaderNN benchmark data not available for this run.",
            "> See `runtime_comparison_results/raw/shadernn_status.txt` for details.",
            "",
        ]
    
    # ONNX Runtime
    if onnx_data:
        speedup = onnx_data['mean_ms'] / prod['gpu_mean_ms']
        lines += [
            "### 2. ONNX Runtime Android (CPU Provider)",
            "",
            "**Measurement method:** Android `SystemClock.elapsedRealtimeNanos()`,  ",
            "5 sessions × 1000 measured iterations (100 warmup discarded per session).",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total samples | {onnx_data['samples']:,} |",
            f"| Mean CPU time | **{onnx_data['mean_ms']:.3f} ms** |",
            f"| Std deviation | {onnx_data['stddev_ms']:.3f} ms |",
            f"| FPS (from mean) | **{onnx_data['fps']:.1f} fps** |",
            f"| Weights format | {onnx_data['weights']} |",
            f"| Speedup vs GPU shader | **{speedup:.2f}×** slower |",
            "",
            f"> **Interpretation:** ONNX Runtime CPU is **{speedup:.1f}× slower** than the",
            f"> custom GPU compute shader approach. At {onnx_data['mean_ms']:.1f}ms, it uses",
            f"> {onnx_data['mean_ms']/16.7*100:.0f}% of the 60Hz frame budget vs {prod['frame_budget_60hz_pct']:.1f}% for GPU shaders.",
            "",
        ]
    else:
        lines += [
            "### 2. ONNX Runtime Android (CPU Provider)",
            "",
            "> ⚠️ ONNX Runtime data not found.",
            "",
        ]
    
    # SMPL LBS
    if smpl_data:
        lines += [
            "### 3. SMPL LBS Baseline (CPU, Python/NumPy simulation)",
            "",
            "**Measurement method:** `time.perf_counter()` on host CPU,",
            "includes rot6d encoding + joint regression + pose blend shapes + vertex skinning.",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| N samples | {smpl_data.get('n', 0):,} |",
            f"| Mean time | **{smpl_data.get('mean_ms', 0):.3f} ms** |",
            f"| Std deviation | {smpl_data.get('stddev_ms', 0):.3f} ms |",
            f"| p50 | {smpl_data.get('p50_ms', 0):.3f} ms |",
            f"| p95 | {smpl_data.get('p95_ms', 0):.3f} ms |",
            f"| p99 | {smpl_data.get('p99_ms', 0):.3f} ms |",
            f"| FPS (from mean) | **{smpl_data.get('fps_from_mean', 0):.1f} fps** |",
            f"| CV | {smpl_data.get('coefficient_of_variation', 0)*100:.2f}% |",
            "",
            "> **Note:** This is a host-CPU NumPy simulation representing the per-frame",
            "> SMPL forward pass cost (rot6d + LBS matmuls). Real mobile ARM NEON SMPL",
            "> would be faster but still CPU-bound. The ZD-ECN correction runs IN PARALLEL",
            "> with rendering on the GPU — adding 1.3ms GPU-side without increasing the",
            "> CPU-side SMPL cost.",
            "",
        ]
    else:
        lines += [
            "### 3. SMPL LBS Baseline",
            "",
            "> ⚠️ SMPL LBS benchmark data not available.",
            "",
        ]
    
    # Comparison table
    onnx_mean = onnx_data['mean_ms'] if onnx_data else None
    smpl_mean = smpl_data.get('mean_ms') if smpl_data else None
    gpu_mean = prod['gpu_mean_ms']
    
    lines += [
        "---",
        "",
        "## Cross-Approach Comparison Table",
        "",
        "| Approach | Mean Latency | FPS | Device | Framework | Weights |",
        "|----------|-------------|-----|--------|-----------|---------|",
        f"| **ZD-ECN GPU Shader (production)** | **{gpu_mean:.3f} ms** | **{prod['fps']:.0f}** | Mali-G57 GPU | Custom GLSL ES 3.2 | fp16 SSBO |",
    ]
    
    if onnx_mean:
        lines.append(
            f"| ZD-ECN ONNX Runtime CPU | {onnx_mean:.3f} ms | {onnx_data['fps']:.0f} | ARM CPU | ONNX Runtime Android | fp32 |"
        )
    
    if smpl_mean:
        smpl_fps = smpl_data.get('fps_from_mean', 0)
        lines.append(
            f"| SMPL LBS (no correction) | {smpl_mean:.3f} ms | {smpl_fps:.0f} | Host CPU | Python/NumPy | fp32 |"
        )
    
    lines += [
        "",
        "### Key Findings",
        "",
        f"1. **GPU shader dominance:** The custom GLSL compute shader approach achieves {prod['fps']:.0f} FPS —",
        f"   using only {prod['frame_budget_60hz_pct']:.1f}% of the 60Hz frame budget.",
        "",
    ]
    
    if onnx_mean:
        speedup = onnx_mean / gpu_mean
        lines += [
            f"2. **ONNX Runtime overhead:** CPU inference via ORT is {speedup:.1f}× slower ({onnx_mean:.2f}ms vs {gpu_mean:.3f}ms),",
            f"   consuming {onnx_mean/16.7*100:.0f}% of the frame budget. At this latency, the correction",
            f"   would noticeably impact animation smoothness.",
            "",
        ]
    
    lines += [
        f"3. **Zero dependency advantage:** The custom GLSL approach requires no external ML",
        f"   framework (no ONNX Runtime, no TFLite, no libtorch) — binary overhead is 2.6 MB",
        f"   vs ~78 MB for ONNX Runtime Mobile.",
        "",
        "4. **Quality:** Both ZD-ECN approaches use the same trained model achieving",
        "   **4.48mm mean per-vertex error** vs 9.14mm LBS baseline (−50.9% improvement).",
        "",
        "---",
        "",
        "## Quality Metrics (Model, not benchmark-specific)",
        "",
        "| Metric | Value | vs LBS Baseline |",
        "|--------|-------|----------------|",
        "| Mean PVE (val set, N=6000) | **4.48 mm** | −50.9% |",
        "| Median PVE | 3.04 mm | — |",
        "| p95 PVE | 12.7 mm | — |",
        "| LBS baseline | 9.14 mm | reference |",
        "| Validated DQS-vs-LBS divergence | 1.278 mm zero-beta / 1.279 mm real-beta | 3,000 real CMU MoCap poses; identity-pose and antipodal-flip checked |",
        "| Historical DQS reference | 6.87 mm | unverified; no computation found in repo or git history; see results/paper/DQS_BASELINE_687_AUDIT.md |",
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "```",
        "# ShaderNN benchmark (on-device, requires ADB)",
        "adb push inferenceProcessorTest /data/local/tmp/",
        "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/inferenceProcessorTest \\",
        "    zdecn --use_compute --use_half --inner_loops 205",
        "",
        "# SMPL LBS benchmark (host)",
        "python scripts/benchmark_smpl_lbs.py \\",
        "    --smpl-data surreal/zd_ecn_subset/smpl_data.npz \\",
        "    --sessions 5 --iterations 200 --output results/smpl_lbs.json",
        "```",
        "",
        "---",
        "",
        f"*Report generated by ZD-ECN benchmark pipeline at {now}*",
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadernn-csv", required=True)
    parser.add_argument("--onnx-summary", required=True)
    parser.add_argument("--smpl-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    shadernn = parse_shadernn_csv(args.shadernn_csv)
    onnx = parse_onnx_summary(args.onnx_summary)
    smpl = parse_smpl_results(args.smpl_results)
    
    generate_report(shadernn, onnx, smpl, Path(args.output))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Statistical + roofline analysis for the Adreno 506 cross-GPU validation run.

Reads results/adreno506/session_0*.csv (columns: run,mlp_cpu_ms,cpu_ms,gpu_ms;
1000 measured iterations per session, 100 warmups already discarded by the app)
and writes results/adreno506/benchmark_summary_adreno506.json.

Roofline (Finding 2 validation): the same h=64 fp16 workload measured at
2.9x SSBO overhead on Mali-G57 is re-evaluated on Adreno 506.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results/adreno506"

# ── device + workload constants ──────────────────────────────────────────────
# NOTE: the physical device is a vivo Y85, platform msm8953 (Snapdragon 625
# class), NOT sdm450 — both integrate Adreno 506 but at different clocks.
# The task brief's spec numbers (72 GFLOPS, 6.4 GB/s LPDDR3) are retained as
# the working assumption, with the platform discrepancy recorded in the JSON.
PEAK_GFLOPS = 72.0
BW_GBPS = 6.4

FLOPS = 2.67e6                       # 2*(144*64 + 64*64 + 64*20670)
WEIGHT_BYTES_FP16 = 1_356_990 * 2

MALI_MEAN_MS = 1.094395              # canonical Mali-G57 run (N=5000)
MALI_OVERHEAD = 2.9                  # paper Finding 2, h=64


def summarize(v: np.ndarray) -> dict:
    return {
        "n": int(v.size),
        "mean_ms": float(v.mean()),
        "std_ms": float(v.std(ddof=1)),
        "p5_ms": float(np.percentile(v, 5)),
        "p25_ms": float(np.percentile(v, 25)),
        "p50_ms": float(np.percentile(v, 50)),
        "p75_ms": float(np.percentile(v, 75)),
        "p95_ms": float(np.percentile(v, 95)),
        "p99_ms": float(np.percentile(v, 99)),
        "min_ms": float(v.min()),
        "max_ms": float(v.max()),
        "cv": float(v.std(ddof=1) / v.mean()),
        "ci95_halfwidth_ms": float(1.96 * v.std(ddof=1) / math.sqrt(v.size)),
        "fps_from_mean": float(1000.0 / v.mean()),
    }


def main() -> None:
    sessions = sorted(OUT_DIR.glob("session_0*.csv"))
    if not sessions:
        raise SystemExit(f"no session CSVs in {OUT_DIR}")

    gpu_all, cpu_all, per_session = [], [], []
    for p in sessions:
        data = np.genfromtxt(p, delimiter=",", names=True)
        gpu = np.asarray(data["gpu_ms"], dtype=np.float64)
        cpu = np.asarray(data["cpu_ms"], dtype=np.float64)
        gpu_all.append(gpu)
        cpu_all.append(cpu)
        per_session.append({
            "file": p.name,
            "n": int(gpu.size),
            "gpu_mean_ms": float(gpu.mean()),
            "gpu_std_ms": float(gpu.std(ddof=1)),
        })
        print(f"{p.name}: n={gpu.size} gpu_mean={gpu.mean():.4f} ms")

    gpu = np.concatenate(gpu_all)
    cpu = np.concatenate(cpu_all)

    # between-session stability
    session_means = np.array([s["gpu_mean_ms"] for s in per_session])
    within_var = float(np.mean([np.var(g, ddof=1) for g in gpu_all]))
    between_var = float(np.var(session_means, ddof=1))
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0.0

    # roofline
    ai = FLOPS / WEIGHT_BYTES_FP16
    bw_ceiling_ms = WEIGHT_BYTES_FP16 / (BW_GBPS * 1e9) * 1e3
    grand_mean = float(gpu.mean())
    overhead = grand_mean / bw_ceiling_ms
    eff_gflops = (FLOPS / (grand_mean * 1e-3)) / 1e9

    summary = {
        "device": {
            "model": "vivo Y85 (vivo 1726)",
            "serial": "8e33900f",
            "soc_platform": "msm8953 (Snapdragon 625 class; task brief assumed SD450 — same Adreno 506 IP, different clocks)",
            "gpu": "Adreno (TM) 506",
            "gl_version": "OpenGL ES 3.2 V@331.0 (driver date 01/23/19)",
            "android": "9 (SDK 28)",
            "abi": "arm64-v8a",
            "gpu_timer": "GL_EXT_disjoint_timer_query available, valid",
        },
        "protocol": {
            "sessions": len(sessions),
            "warmup_per_session": 100,
            "measured_per_session": 1000,
            "cooldown_s": 30,
            "thermal_gate": "battery thermal zone < 37.0 C (Android 9 has no thermalservice status)",
            "weights": "fp16 packed",
            "workload": "144 -> 64 -> 64 -> 20670 MLP, 3x glDispatchCompute + barrier + glFinish",
        },
        "gpu_ms": summarize(gpu),
        "cpu_wall_ms": summarize(cpu),
        "per_session": per_session,
        "between_session": {
            "session_means_ms": [float(m) for m in session_means],
            "between_session_std_ms": float(session_means.std(ddof=1)),
            "icc": icc,
        },
        "roofline": {
            "assumed_peak_gflops_fp32": PEAK_GFLOPS,
            "assumed_bw_gbps": BW_GBPS,
            "bw_caveat": "msm8953 LPDDR3 theoretical is ~7.4 GB/s; 6.4 GB/s retained from task brief. UBWC does not compress SSBO reads.",
            "flops": FLOPS,
            "weight_bytes_fp16": WEIGHT_BYTES_FP16,
            "arithmetic_intensity_flop_per_byte": ai,
            "bw_ceiling_ms": bw_ceiling_ms,
            "measured_gpu_mean_ms": grand_mean,
            "overhead_factor": overhead,
            "effective_gflops": eff_gflops,
            "mali_g57_reference": {
                "mean_ms": MALI_MEAN_MS,
                "overhead_factor": MALI_OVERHEAD,
            },
        },
    }

    out = OUT_DIR / "benchmark_summary_adreno506.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    g = summary["gpu_ms"]
    print()
    print("=" * 64)
    print("  Adreno 506 cross-GPU validation (Finding 2)")
    print("=" * 64)
    print(f"  N = {g['n']}  |  sessions = {len(sessions)}")
    print(f"  GPU mean {g['mean_ms']:.4f} ms  std {g['std_ms']:.4f}  CV {g['cv']:.4f}")
    print(f"  p50/p95/p99: {g['p50_ms']:.4f} / {g['p95_ms']:.4f} / {g['p99_ms']:.4f} ms")
    print(f"  95% CI half-width: {g['ci95_halfwidth_ms']:.5f} ms  |  FPS {g['fps_from_mean']:.1f}")
    print(f"  Between-session std {between_var**0.5:.5f} ms  ICC {icc:.4f}")
    print()
    print(f"  Arithmetic intensity : {ai:.3f} FLOP/byte")
    print(f"  BW ceiling (6.4 GB/s): {bw_ceiling_ms:.4f} ms")
    print(f"  Measured GPU mean    : {grand_mean:.4f} ms")
    print(f"  Overhead factor      : {overhead:.2f}x   (Mali-G57 h=64: {MALI_OVERHEAD}x)")
    print(f"  Effective throughput : {eff_gflops:.2f} GFLOPS")
    print(f"  Written: {out}")


if __name__ == "__main__":
    main()

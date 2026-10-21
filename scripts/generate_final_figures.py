#!/usr/bin/env python3
"""
Generate the final paper-quality figure set for ZD-ECN (IEEE style).

Every value is read from a file on disk — nothing is fabricated. If a source
file is missing the figure is SKIPPED and reported in the manifest. The only
spec-provided constants are the Mali-G57 roofline device parameters and the
h=256 fp32 measurement (17.202 ms), which predates the current results tree.

Output: results/paper/figures/final/  (PDF + PNG per figure)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/paper/figures/final"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         9,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'legend.frameon':    False,
    'legend.fontsize':   8,
    'figure.dpi':        300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid':         True,
    'grid.linewidth':    0.4,
    'grid.alpha':        0.4,
    'grid.color':        '#cccccc',
    'lines.linewidth':   1.2,
})

SINGLE_COL = (3.45, 2.6)
DOUBLE_COL = (7.0,  2.6)

COLOR = {
    'zdecn':   '#2166ac',
    'onnx':    '#d6604d',
    'lbs':     '#4d4d4d',
    'shadernn':'#6a3d9a',
    'green':   '#1a9641',
    'ref':     '#b2182b',
    'neutral': '#878787',
}

# ── shared data paths ─────────────────────────────────────────────────────────
BENCH_RAW        = ROOT / "benchmark_results/raw/benchmark_raw.csv"
BENCH_SUMMARY    = ROOT / "benchmark_results/summary/benchmark_summary.json"
TRAIN_LOG        = ROOT / "results/paper/full_retrain_sameframe/train.log"
PER_POSE_CSV     = ROOT / "results/paper/final_validation/full_30k_val_pve_per_pose.csv"
DATASET_NPZ      = ROOT / "data/dataset_sameframe_pose_30k.npz"
GLSL_VALIDATION  = ROOT / "results/paper/final_validation/glsl_validation_summary.json"
TARGET_DIST_SUM  = ROOT / "results/paper/full_retrain_sameframe/target_distribution_summary.json"
CKPT_SELECTION   = ROOT / "results/paper/final_validation/checkpoint_selection.json"
RUNTIME_CMP      = ROOT / "runtime_comparison_results/summary/runtime_comparison_summary.json"
PIPELINE_JSON    = ROOT / "skin_rendering/results/full_pipeline_benchmark.json"
PER_VERTEX_CSV   = ROOT / "results/paper/real_beta_eval/real_beta_per_vertex_errors_mm.csv"
LBS_SPARSE       = ROOT / "models/smpl_reference_data/lbs_weights_sparse.npz"

# Spec-provided constants (see module docstring)
MALI_PEAK_GFLOPS = 460.0
MALI_BW_GBS      = 12.0
H64_FLOPS,  H64_PARAMS,  H64_BYTES_PER  = 2.67e6,  1_356_990, 2   # fp16
H256_FLOPS, H256_PARAMS, H256_BYTES_PER = 10.75e6, 5_396_670, 4   # fp32
H256_MEASURED_MS = 17.202  # spec-provided; predates current results tree

manifest: list[tuple[str, str, str, str]] = []  # (fig, file, source, status)


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def require(*paths: Path) -> list[Path]:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(m) for m in missing))
    return list(paths)


def load_benchmark_raw():
    require(BENCH_RAW)
    sessions, gpu = [], []
    with open(BENCH_RAW) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(int(row["session"]))
            gpu.append(float(row["gpu_ms"]))
    return np.asarray(sessions), np.asarray(gpu)


# ══════════════════════════════════════════════════════════════════════════════
def fig01_latency_cdf():
    _, gpu = load_benchmark_raw()
    x = np.sort(gpu)
    y = np.arange(1, len(x) + 1) / len(x)
    mean, p95, p99 = x.mean(), np.percentile(x, 95), np.percentile(x, 99)

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.plot(x, y, color=COLOR['zdecn'], label="ZD-ECN")
    # stagger the three labels — the lines sit within 0.06 ms of each other
    for v, name, ty in ((mean, "mean", 1.02), (p95, "p95", 1.10),
                        (p99, "p99", 1.18)):
        ax.axvline(v, color=COLOR['ref'], ls='--', lw=0.6, alpha=0.7)
        ax.text(v, ty, f"{name} {v:.3f}", ha='left', va='bottom', fontsize=7,
                color=COLOR['ref'], transform=ax.get_xaxis_transform())
    ax.set_xlim(1.0, 1.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("GPU frame time (ms)")
    ax.set_ylabel("Cumulative probability")
    save(fig, "fig01_latency_cdf")
    return f"N={len(x)}"


def fig02_session_stability():
    sessions, gpu = load_benchmark_raw()
    data = [gpu[sessions == s] for s in range(1, 6)]
    grand_mean = gpu.mean()

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    bp = ax.boxplot(data, tick_labels=[str(s) for s in range(1, 6)],
                    patch_artist=True, whis=1.5,
                    flierprops=dict(markersize=1, alpha=0.3,
                                    markerfacecolor=COLOR['zdecn'],
                                    markeredgecolor='none'),
                    medianprops=dict(color='white', lw=1.0),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8))
    for box in bp['boxes']:
        box.set_facecolor(COLOR['zdecn'])
        box.set_alpha(0.7)
    ax.axhline(grand_mean, color=COLOR['ref'], ls='--', lw=0.8)
    ax.text(5.45, grand_mean, f"{grand_mean:.3f}", va='center', fontsize=7,
            color=COLOR['ref'])
    ax.set_xlabel("Session")
    ax.set_ylabel("GPU frame time (ms)")
    save(fig, "fig02_session_stability")
    return f"grand mean {grand_mean:.4f} ms"


def fig03_roofline():
    require(BENCH_SUMMARY)
    h64_ms = json.loads(BENCH_SUMMARY.read_text())["gpu"]["mean_ms"]

    ai64  = H64_FLOPS  / (H64_PARAMS  * H64_BYTES_PER)
    ai256 = H256_FLOPS / (H256_PARAMS * H256_BYTES_PER)
    pf64  = H64_FLOPS  / (h64_ms * 1e-3) / 1e9
    pf256 = H256_FLOPS / (H256_MEASURED_MS * 1e-3) / 1e9

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ai = np.logspace(-1, 2, 300)
    roof = np.minimum(ai * MALI_BW_GBS, MALI_PEAK_GFLOPS)
    ax.plot(ai, roof, color=COLOR['neutral'], lw=1.0)
    knee = MALI_PEAK_GFLOPS / MALI_BW_GBS
    ax.text(0.5, 0.5 * MALI_BW_GBS * 1.5, f"{MALI_BW_GBS:.0f} GB/s",
            fontsize=7, color=COLOR['neutral'], rotation=38,
            rotation_mode='anchor')
    ax.text(85, MALI_PEAK_GFLOPS * 1.15, f"{MALI_PEAK_GFLOPS:.0f} GFLOPS",
            fontsize=7, color=COLOR['neutral'], ha='right')

    ax.scatter([ai64], [pf64], marker='s', s=30, color=COLOR['zdecn'], zorder=5)
    ax.annotate(f"h=64  {h64_ms:.3f} ms", (ai64, pf64),
                textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.scatter([ai256], [pf256], marker='o', s=30, color=COLOR['onnx'], zorder=5)
    ax.annotate(f"h=256  {H256_MEASURED_MS:.1f} ms", (ai256, pf256),
                textcoords="offset points", xytext=(6, -12), fontsize=7)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, 100); ax.set_ylim(0.1, 1000)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    save(fig, "fig03_roofline")
    return f"h64 {pf64:.2f} GFLOPS @ AI {ai64:.2f}; h256 {pf256:.2f} @ {ai256:.2f}"


def fig04_ssbo_overhead():
    require(BENCH_SUMMARY)
    h64_ms = json.loads(BENCH_SUMMARY.read_text())["gpu"]["mean_ms"]

    ceil64  = H64_PARAMS  * H64_BYTES_PER  / (MALI_BW_GBS * 1e9) * 1e3  # ms
    ceil256 = H256_PARAMS * H256_BYTES_PER / (MALI_BW_GBS * 1e9) * 1e3
    ov64, ov256 = h64_ms / ceil64, H256_MEASURED_MS / ceil256
    mb64  = H64_PARAMS  * H64_BYTES_PER  / 1e6
    mb256 = H256_PARAMS * H256_BYTES_PER / 1e6

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    xs = [0, 1]
    bars = ax.bar(xs, [ov64, ov256], width=0.4, color=COLOR['zdecn'],
                  label="Overhead factor")
    for x, v in zip(xs, [ov64, ov256]):
        ax.text(x, v + 0.25, f"{v:.1f}×", ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["h=64", "h=256"])
    ax.set_ylim(0, 12)
    ax.set_ylabel("SSBO overhead factor")
    ax.grid(axis='x', visible=False)

    ax2 = ax.twinx()
    ax2.plot(xs, [mb64, mb256], color=COLOR['onnx'], marker='o', ms=4,
             label="Weight size")
    ax2.set_ylim(0, 22)
    ax2.set_ylabel("Weight size (MB)")
    ax2.spines['top'].set_visible(False)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7)
    save(fig, "fig04_ssbo_overhead")
    return f"computed overheads {ov64:.1f}x / {ov256:.1f}x"


def fig05_convergence():
    require(TRAIN_LOG)
    rows = []
    for line in TRAIN_LOG.read_text().splitlines():
        line = line.strip()
        if line.startswith('{') and '"epoch"' in line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        raise FileNotFoundError(f"no JSON epoch rows in {TRAIN_LOG}")
    ep   = np.array([r["epoch"] for r in rows])
    pve  = np.array([r["val_pve_mm"] for r in rows])
    tmse = np.array([r["train_mse"] for r in rows])
    vmse = np.array([r["val_mse"] for r in rows])
    lbs  = rows[0]["baseline_val_pve_mm"]

    # The paper checkpoint is the best *saved* checkpoint (epoch 375, 1.947 mm),
    # not the raw val-PVE argmin (epoch 387 — no checkpoint saved there).
    if CKPT_SELECTION.exists():
        ckpt_epoch = json.loads(CKPT_SELECTION.read_text())["used_checkpoint_epoch"]
        best_i = int(np.argmax(ep == ckpt_epoch))
    else:
        best_i = int(np.argmin(pve))
    cross_i = int(np.argmax(pve < lbs))  # first epoch below baseline

    fig, (a, b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    a.plot(ep, pve, color=COLOR['zdecn'])
    a.axhline(lbs, color=COLOR['ref'], ls='--', lw=0.8)
    a.text(ep[-1], lbs + 0.12, f"LBS  {lbs:.2f} mm", ha='right', fontsize=7,
           color=COLOR['ref'])
    a.scatter([ep[cross_i]], [pve[cross_i]], marker='^', s=40,
              color=COLOR['green'], zorder=5)
    a.annotate(f"ep. {ep[cross_i]}", (ep[cross_i], pve[cross_i]),
               textcoords="offset points", xytext=(6, 8), fontsize=7)
    a.scatter([ep[best_i]], [pve[best_i]], marker='o', s=30,
              color=COLOR['ref'], zorder=5)
    a.annotate(f"{pve[best_i]:.3f} mm", (ep[best_i], pve[best_i]),
               textcoords="offset points", xytext=(-6, 10), fontsize=7,
               ha='right')
    a.set_xlabel("Epoch")
    a.set_ylabel("Mean per-vertex error (mm)")

    b.plot(ep, tmse, color=COLOR['neutral'], ls='--', label="Train")
    b.plot(ep, vmse, color=COLOR['zdecn'], label="Val")
    b.set_yscale('log')
    b.set_xlabel("Epoch")
    b.set_ylabel("MSE loss")
    b.legend()
    save(fig, "fig05_convergence")
    return f"{len(rows)} epochs, best {pve[best_i]:.4f} mm @ ep {ep[best_i]}"


def load_per_pose():
    require(PER_POSE_CSV)
    return np.loadtxt(PER_POSE_CSV, skiprows=1, dtype=np.float64)


def fig06_error_distribution():
    v = load_per_pose()
    mean, med, p95 = v.mean(), np.median(v), np.percentile(v, 95)

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.hist(v, bins=40, range=(0, 20), density=True, color=COLOR['zdecn'],
            alpha=0.75, edgecolor='none')
    for val, col, name in ((mean, COLOR['zdecn'], 'mean'),
                           (med, COLOR['neutral'], 'median'),
                           (p95, COLOR['ref'], 'p95')):
        ax.axvline(val, color=col, ls='--', lw=0.8)
        ax.text(val, 1.02, f"{val:.2f}", ha='center', va='bottom', fontsize=7,
                color=col, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 20)
    ax.set_xlabel("Per-pose mean PVE (mm)")
    ax.set_ylabel("Density")
    save(fig, "fig06_error_distribution")
    return f"N={len(v)}, mean {mean:.3f}"


def fig07_baseline_comparison():
    zdecn = load_per_pose()
    require(DATASET_NPZ)
    z = np.load(DATASET_NPZ)
    lbs = z["target_magnitude_mm"][z["val_indices"]]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    bp = ax.boxplot([lbs, zdecn], tick_labels=["LBS", "ZD-ECN"],
                    patch_artist=True, whis=1.5,
                    flierprops=dict(markersize=1, alpha=0.3,
                                    markeredgecolor='none',
                                    markerfacecolor=COLOR['neutral']),
                    medianprops=dict(color='white', lw=1.0),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8))
    for box, col in zip(bp['boxes'], (COLOR['neutral'], COLOR['zdecn'])):
        box.set_facecolor(col)
        box.set_alpha(0.85)
    ax.set_ylabel("Mean PVE per pose (mm)")
    save(fig, "fig07_baseline_comparison")
    return f"LBS {lbs.mean():.3f} vs ZD-ECN {zdecn.mean():.3f} (val N={len(lbs)})"


SMPL_JOINT_NAMES = [
    "pelvis", "L hip", "R hip", "spine1", "L knee", "R knee", "spine2",
    "L ankle", "R ankle", "spine3", "L foot", "R foot", "neck",
    "L collar", "R collar", "head", "L shoulder", "R shoulder",
    "L elbow", "R elbow", "L wrist", "R wrist", "L hand", "R hand",
]


def fig08_perjoint_pve():
    require(PER_VERTEX_CSV, LBS_SPARSE)
    # per-vertex mean error over 3000 real-beta poses; CSV is pose-major ravel
    vals = np.loadtxt(PER_VERTEX_CSV, skiprows=1, dtype=np.float32)
    n_vertex = 6890
    if vals.size % n_vertex:
        raise ValueError(f"per-vertex CSV size {vals.size} not divisible by {n_vertex}")
    per_vertex = vals.reshape(-1, n_vertex).mean(axis=0)

    s = np.load(LBS_SPARSE)
    offsets, indices, weights = s["offsets"], s["indices"], s["weights"]
    dominant = np.empty(n_vertex, dtype=np.int64)
    for v in range(n_vertex):
        lo, hi = offsets[v], offsets[v + 1]
        dominant[v] = indices[lo:hi][np.argmax(weights[lo:hi])]

    joint_pve = np.array([per_vertex[dominant == j].mean() for j in range(24)])
    order = np.argsort(joint_pve)
    overall = per_vertex.mean()

    fig, ax = plt.subplots(figsize=(3.45, 3.8))
    norm = mpl.colors.Normalize(joint_pve.min(), joint_pve.max())
    colors = mpl.cm.RdYlBu_r(norm(joint_pve[order]))
    ax.barh(np.arange(24), joint_pve[order], color=colors, height=0.7)
    ax.set_yticks(np.arange(24))
    ax.set_yticklabels([SMPL_JOINT_NAMES[j] for j in order], fontsize=7)
    ax.axvline(overall, color=COLOR['ref'], ls='--', lw=0.8)
    ax.text(overall, 23.8, "mean", ha='center', va='bottom', fontsize=7,
            color=COLOR['ref'])
    ax.set_xlabel("Mean PVE (mm)")
    ax.grid(axis='y', visible=False)
    save(fig, "fig08_perjoint_pve")
    return f"range {joint_pve.min():.2f}-{joint_pve.max():.2f} mm, overall {overall:.3f}"


def fig09_pipeline_breakdown():
    require(PIPELINE_JSON, BENCH_SUMMARY)
    total = json.loads(PIPELINE_JSON.read_text())["gpu_ms"]["mean"]
    ecn = json.loads(BENCH_SUMMARY.read_text())["gpu"]["mean_ms"]
    other = total - ecn
    fps = 1000.0 / total

    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    ax.barh([0], [ecn], color=COLOR['zdecn'], height=0.45, label="ZD-ECN")
    ax.barh([0], [other], left=[ecn], color=COLOR['neutral'], height=0.45,
            label="SMPL + render")
    ax.text(ecn / 2, 0, f"{ecn:.2f}", ha='center', va='center', color='white',
            fontsize=8)
    ax.text(ecn + other / 2, 0, f"{other:.2f}", ha='center', va='center',
            color='white', fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, total * 1.15)
    ax.set_xlabel("GPU time (ms)")
    ax.set_title(f"Full pipeline  {total:.2f} ms  ({fps:.0f} FPS)", fontsize=9)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(axis='y', visible=False)
    save(fig, "fig09_pipeline_breakdown")
    return f"total {total:.3f} ms = ECN {ecn:.3f} + other {other:.3f}"


def fig10_backend_comparison():
    require(BENCH_SUMMARY, RUNTIME_CMP)
    zdecn = json.loads(BENCH_SUMMARY.read_text())["gpu"]["mean_ms"]
    rc = json.loads(RUNTIME_CMP.read_text())["approaches"]
    shadernn = rc["shadernn_pure"]["cpu_wall_clock"]["mean_ms"]
    onnx_cpu = rc["onnx_runtime_fp16_cpu_ep"]["cpu_wall_clock"]["mean_ms"]
    onnx_nnapi = rc["onnx_runtime_fp16_nnapi_ep"]["cpu_wall_clock"]["mean_ms"]

    entries = sorted([
        ("ZD-ECN", zdecn, COLOR['zdecn'], None),
        ("ShaderNN†", shadernn, COLOR['shadernn'], '///'),
        ("ONNX\nCPU", onnx_cpu, COLOR['onnx'], None),
        ("ONNX\nNNAPI", onnx_nnapi, '#e08a7a', None),
    ], key=lambda e: e[1])

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    xs = np.arange(len(entries))
    for x, (name, val, col, hatch) in zip(xs, entries):
        ax.bar(x, val, width=0.55, color=col, hatch=hatch,
               edgecolor='white' if hatch else 'none', linewidth=0.4)
        ax.text(x, val + 0.12, f"{val:.2f}", ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(e[1] for e in entries) * 1.2)
    ax.grid(axis='x', visible=False)
    fig.text(0.13, -0.14,
             "† GPU timer not available for ShaderNN; CPU wall-clock shown.",
             fontsize=7, style='italic')
    save(fig, "fig10_backend_comparison")
    return (f"zdecn {zdecn:.3f}, shadernn {shadernn:.2f} (CPU), "
            f"onnx {onnx_cpu:.2f}/{onnx_nnapi:.2f}")


def fig11_fp16_error():
    require(GLSL_VALIDATION)
    d = json.loads(GLSL_VALIDATION.read_text())
    mean_abs = d["Validation mean_abs"]
    max_abs = d["Validation max_abs"]
    tail_max = d["Validation tail_max_abs"]
    tail_mean = d["Validation tail_mean_abs"]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.axis('off')
    txt = (
        "GLSL fp16 vs Python fp32 reference\n"
        "(per-vertex absolute difference)\n\n"
        f"mean |err|        {mean_abs * 1e3:8.5f} µm\n"
        f"max  |err|        {max_abs * 1e3:8.5f} µm\n"
        f"tail mean |err|   {tail_mean * 1e3:8.5f} µm\n"
        f"tail max  |err|   {tail_max * 1e3:8.5f} µm\n\n"
        f"max = {max_abs:.4g} mm ≈ {max_abs / 1.947495 * 100:.4f}%\n"
        "of model PVE (1.947 mm)"
    )
    ax.text(0.05, 0.5, txt, fontsize=9, family='monospace',
            va='center', ha='left',
            bbox=dict(boxstyle='square,pad=0.6', fc='none', ec='#666666',
                      lw=0.8))
    save(fig, "fig11_fp16_error")
    return f"summary box (max {max_abs:.3g} mm); raw distribution not archived"


def fig12_dataset_distribution():
    require(DATASET_NPZ, TARGET_DIST_SUM)
    z = np.load(DATASET_NPZ)
    mag = z["target_magnitude_mm"]                       # (30000,)
    aa = z["pose_axis_angle"].reshape(-1, 24, 3)         # (30000, 24, 3)
    ref = json.loads(TARGET_DIST_SUM.read_text())
    lbs_mean = ref["mean_mm"]
    # pose difficulty: max joint rotation magnitude excluding global root
    difficulty = np.linalg.norm(aa[:, 1:, :], axis=2).max(axis=1)

    fig, (a, b) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    a.hist(mag, bins=40, density=True, color=COLOR['zdecn'], alpha=0.75,
           edgecolor='none')
    a.axvline(lbs_mean, color=COLOR['ref'], ls='--', lw=0.8)
    a.text(lbs_mean, 1.02, "LBS baseline", ha='center', va='bottom',
           fontsize=7, color=COLOR['ref'], transform=a.get_xaxis_transform())
    a.set_xlabel("Mean |δv| per pose (mm)")
    a.set_ylabel("Density")

    b.scatter(difficulty, mag, s=3, alpha=0.15, color=COLOR['zdecn'],
              edgecolors='none', rasterized=True)
    bins = np.linspace(difficulty.min(), difficulty.max(), 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    running = np.array([
        mag[(difficulty >= lo) & (difficulty < hi)].mean()
        if ((difficulty >= lo) & (difficulty < hi)).any() else np.nan
        for lo, hi in zip(bins[:-1], bins[1:])
    ])
    b.plot(centers, running, color=COLOR['ref'], lw=1.2)
    b.set_xlabel("Max joint rotation (rad)")
    b.set_ylabel("Mean |δv| per pose (mm)")
    save(fig, "fig12_dataset_distribution")
    return f"N={len(mag)}, mean {mag.mean():.3f} mm"


# ══════════════════════════════════════════════════════════════════════════════
FIGS = [
    ("01", "fig01_latency_cdf",       "benchmark_raw.csv",                fig01_latency_cdf),
    ("02", "fig02_session_stability", "benchmark_raw.csv",                fig02_session_stability),
    ("03", "fig03_roofline",          "benchmark_summary.json + spec",    fig03_roofline),
    ("04", "fig04_ssbo_overhead",     "benchmark_summary.json + spec",    fig04_ssbo_overhead),
    ("05", "fig05_convergence",       "train.log",                        fig05_convergence),
    ("06", "fig06_error_distribution","full_30k_val_pve_per_pose.csv",    fig06_error_distribution),
    ("07", "fig07_baseline_comparison","per_pose csv + dataset npz",      fig07_baseline_comparison),
    ("08", "fig08_perjoint_pve",      "real_beta per-vertex + lbs sparse",fig08_perjoint_pve),
    ("09", "fig09_pipeline_breakdown","full_pipeline_benchmark.json",     fig09_pipeline_breakdown),
    ("10", "fig10_backend_comparison","runtime_comparison_summary.json",  fig10_backend_comparison),
    ("11", "fig11_fp16_error",        "glsl_validation_summary.json",     fig11_fp16_error),
    ("12", "fig12_dataset_distribution","dataset npz + dist summary",     fig12_dataset_distribution),
]


def main():
    failed = []
    for num, stem, source, fn in FIGS:
        try:
            note = fn()
            manifest.append((num, f"{stem}.pdf", source, "OK"))
            print(f"[{num}] {stem}: OK — {note}")
        except FileNotFoundError as e:
            manifest.append((num, f"{stem}.pdf", source, "SKIP"))
            failed.append((stem, f"missing file: {e}"))
            print(f"[{num}] {stem}: SKIP — missing {e}")
        except Exception as e:
            manifest.append((num, f"{stem}.pdf", source, "FAIL"))
            failed.append((stem, str(e)))
            print(f"[{num}] {stem}: FAIL — {e}")

    print()
    print(f"{'Figure':<7}| {'File':<31}| {'Data source':<36}| Status")
    print(f"{'-'*7}|{'-'*32}|{'-'*37}|-------")
    for num, f, src, st in manifest:
        print(f"{num:<7}| {f:<31}| {src:<36}| {st}")
    print()
    ok = sum(1 for m in manifest if m[3] == "OK")
    print(f"Figures generated: {ok} / {len(FIGS)}")
    print(f"Output: {OUT.relative_to(ROOT)}/")
    if failed:
        print("Failed:")
        for stem, why in failed:
            print(f"  {stem}: {why}")
    else:
        print("Failed: none")


if __name__ == "__main__":
    main()

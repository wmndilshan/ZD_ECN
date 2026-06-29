#!/usr/bin/env python3
"""
Evaluate the zero-beta-trained ZD-ECN model on poses with real (non-zero) body shapes.

The production model takes only rot6d_144 (pose) as input — body shape (betas) is not
observed by the model. This script measures how well the pose-correction generalises to
real CMU MoCap body shapes by:

  1. Loading real-beta poses from data/dataset_eval_3k_real_betas.npz.
  2. Computing the same-frame pose-correction target WITH each pose's actual betas using
     the SMPL numpy forward pass (same formula as build_training_dataset.py, beta-aware).
  3. Running the model (pose-only input, zero-beta training) to predict corrections.
  4. Reporting PVE(model, real-beta-target) and LBS baseline PVE(zero, real-beta-target).

A small generalisation gap means body shape does not materially affect the pose-correction
magnitude. A large gap quantifies the shape-generalisation cost and becomes a paper limitation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluate_zdecn_dqs import VERTEX_COUNT               # 6890
from export_zdecn_checkpoint import load_checkpoint_model
from generate_smpl_lbs_validation_fixture import (
    JOINT_COUNT,
    batch_rigid_transform,
    batch_rodrigues,
    load_model,
)

INPUT_SIZE  = 144
OUTPUT_SIZE = VERTEX_COUNT * 3   # 20670


def _skin(v: np.ndarray, transforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """LBS skinning: v (B,V,3), transforms (B,J,4,4), weights (V,J) → (B,V,3)."""
    b = v.shape[0]
    t_flat = transforms.reshape(b, JOINT_COUNT, 16)
    blended = np.matmul(
        np.broadcast_to(weights[None], (b, VERTEX_COUNT, JOINT_COUNT)),
        t_flat,
    ).reshape(b, VERTEX_COUNT, 4, 4)
    v_h = np.concatenate([v, np.ones((b, VERTEX_COUNT, 1), dtype=np.float32)], axis=2)
    return np.matmul(blended, v_h[:, :, :, None])[:, :, :3, 0].astype(np.float32)


def compute_sameframe_targets(
    model_data: dict[str, np.ndarray],
    poses_aa: np.ndarray,   # (N, 72) axis-angle
    betas: np.ndarray,      # (N, 10)  — pass zeros for zero-beta reference
    batch_size: int = 64,
) -> np.ndarray:
    """
    target_mm = (skin_A(theta)(v_shaped + pose_bs(theta)) - skin_A(theta)(v_shaped)) * 1000

    Both terms use identical joint transforms derived from theta and the beta-shaped joints,
    so the result is the pure pose-blendshape contribution in the posed frame.
    """
    n = poses_aa.shape[0]
    identity = np.eye(3, dtype=np.float32)
    targets = np.empty((n, VERTEX_COUNT * 3), dtype=np.float32)

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        pose  = poses_aa[start:end].astype(np.float32)
        beta  = betas[start:end].astype(np.float32)
        b     = pose.shape[0]

        v_shaped = (
            model_data["v_template"][None]
            + np.einsum("bl,mkl->bmk", beta, model_data["shapedirs"])
        )                                                          # (b, V, 3)
        joints = np.einsum("bik,ji->bjk", v_shaped, model_data["J_regressor"])

        rot_mats    = batch_rodrigues(pose.reshape(-1, 3)).reshape(b, JOINT_COUNT, 3, 3)
        pose_feat   = (rot_mats[:, 1:] - identity).reshape(b, -1)
        pose_offsets = np.matmul(pose_feat, model_data["posedirs"]).reshape(b, VERTEX_COUNT, 3)

        _, transforms = batch_rigid_transform(rot_mats, joints, model_data["parents"])

        v_ref  = _skin(v_shaped + pose_offsets, transforms, model_data["lbs_weights"])
        v_base = _skin(v_shaped,                transforms, model_data["lbs_weights"])

        targets[start:end] = ((v_ref - v_base) * 1000.0).reshape(b, VERTEX_COUNT * 3)
        if end % 500 == 0 or end == n:
            print(f"  [target] {end}/{n}", flush=True)

    return targets


def summarize(arr: np.ndarray, label: str) -> dict[str, object]:
    return {
        "label": label,
        "n":    int(arr.size),
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "max":  float(np.max(arr)),
    }


def pct_below(arr: np.ndarray, thr: float) -> float:
    return float((arr < thr).mean() * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("results/paper/full_retrain_sameframe/checkpoint_epoch_0375.pt"))
    ap.add_argument("--dataset",    type=Path,
                    default=Path("data/dataset_eval_3k_real_betas.npz"))
    ap.add_argument("--smpl-model", type=Path,
                    default=Path("models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/paper/real_beta_eval"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────
    print("Loading dataset …")
    ds          = np.load(args.dataset, allow_pickle=False)
    theta_rot6d = ds["theta_rot6d"].astype(np.float32)     # (3000, 144)
    poses_aa    = ds["pose_axis_angle"].astype(np.float32)  # (3000, 72)
    betas       = ds["betas"].astype(np.float32)            # (3000, 10)
    n           = theta_rot6d.shape[0]
    print(f"  {n} poses | beta std = {betas.std():.3f}")

    print("Loading SMPL model …")
    model_data = load_model(args.smpl_model)

    # ── compute targets ───────────────────────────────────────────────────
    print("Computing same-frame targets with real betas …")
    targets_real_mm = compute_sameframe_targets(
        model_data, poses_aa, betas, args.batch_size)

    print("Computing same-frame targets with zero betas (reference) …")
    targets_zero_mm = compute_sameframe_targets(
        model_data, poses_aa, np.zeros_like(betas), args.batch_size)

    # ── run model ─────────────────────────────────────────────────────────
    print("Running ZD-ECN model …")
    model = load_checkpoint_model(args.checkpoint)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            xb = torch.from_numpy(theta_rot6d[start:start + args.batch_size])
            chunks.append(model(xb).cpu().numpy().astype(np.float32))
    pred_mm = np.concatenate(chunks)   # (n, 20670)

    # ── PVE helpers ───────────────────────────────────────────────────────
    def pve(p: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.linalg.norm((p - t).reshape(-1, VERTEX_COUNT, 3), axis=2).mean(axis=1)

    zeros = np.zeros_like(pred_mm)

    pve_model_real = pve(pred_mm, targets_real_mm)
    pve_lbs_real   = pve(zeros,   targets_real_mm)
    pve_model_zero = pve(pred_mm, targets_zero_mm)
    pve_lbs_zero   = pve(zeros,   targets_zero_mm)

    # ── target shape delta: how much do real betas change the target ──────
    shape_delta_pv = np.linalg.norm(
        (targets_real_mm - targets_zero_mm).reshape(-1, VERTEX_COUNT, 3), axis=2
    ).mean(axis=1)

    # ── per-vertex errors for perceptual threshold breakdown ──────────────
    pv_zdecn = np.linalg.norm(
        (pred_mm - targets_real_mm).reshape(-1, VERTEX_COUNT, 3), axis=2
    ).ravel()
    pv_lbs   = np.linalg.norm(
        targets_real_mm.reshape(-1, VERTEX_COUNT, 3), axis=2
    ).ravel()

    gap = float(pve_model_real.mean() - pve_model_zero.mean())

    summary = {
        "checkpoint":  str(args.checkpoint),
        "dataset":     str(args.dataset),
        "n_poses":     n,
        "beta_stats": {
            "std":  float(betas.std()),
            "mean": float(betas.mean()),
            "min":  float(betas.min()),
            "max":  float(betas.max()),
        },
        "target_shape_delta_mm":        summarize(shape_delta_pv, "target_diff_real_vs_zero_beta"),
        "zdecn_pve_real_betas_mm":      summarize(pve_model_real, "zdecn_vs_real_beta_target"),
        "lbs_baseline_real_betas_mm":   summarize(pve_lbs_real,   "lbs_vs_real_beta_target"),
        "zdecn_pve_zero_betas_mm":      summarize(pve_model_zero, "zdecn_vs_zero_beta_target"),
        "lbs_baseline_zero_betas_mm":   summarize(pve_lbs_zero,   "lbs_vs_zero_beta_target"),
        "generalisation_gap_mm":        gap,
        "improvement_real_betas_pct":   float(
            (1.0 - pve_model_real.mean() / pve_lbs_real.mean()) * 100.0),
        "perceptual_thresholds": {
            "zdecn_below_1mm_pct": pct_below(pv_zdecn, 1.0),
            "zdecn_below_2mm_pct": pct_below(pv_zdecn, 2.0),
            "zdecn_below_5mm_pct": pct_below(pv_zdecn, 5.0),
            "lbs_below_1mm_pct":   pct_below(pv_lbs,   1.0),
            "lbs_below_2mm_pct":   pct_below(pv_lbs,   2.0),
            "lbs_below_5mm_pct":   pct_below(pv_lbs,   5.0),
        },
    }

    out_json = args.output_dir / "real_beta_eval_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    np.savetxt(
        args.output_dir / "real_beta_per_pose_pve_mm.csv",
        np.column_stack([pve_model_real, pve_lbs_real, shape_delta_pv]),
        delimiter=",",
        header="zdecn_pve_mm,lbs_pve_mm,target_shape_delta_mm",
        comments="",
    )
    np.savetxt(
        args.output_dir / "real_beta_per_vertex_errors_mm.csv",
        pv_zdecn,
        delimiter=",",
        header="per_vertex_error_mm",
        comments="",
    )

    # ── print ─────────────────────────────────────────────────────────────
    pt = summary["perceptual_thresholds"]
    print()
    print("═" * 62)
    print("  ZD-ECN Real-Beta Generalisation Evaluation")
    print("═" * 62)
    print(f"  Poses         : {n}  |  beta std ≈ {betas.std():.3f} (natural population)")
    print()
    print(f"  Target shape-delta (real vs zero beta) : {shape_delta_pv.mean():.4f} mm mean")
    print()
    print(f"  ZD-ECN PVE — zero-beta target          : {pve_model_zero.mean():.6f} mm")
    print(f"  ZD-ECN PVE — real-beta target          : {pve_model_real.mean():.6f} mm")
    print(f"  Generalisation gap                     : {gap:+.6f} mm")
    print()
    print(f"  LBS baseline — real-beta target        : {pve_lbs_real.mean():.6f} mm")
    print(f"  Improvement over LBS (real betas)      : {summary['improvement_real_betas_pct']:.2f}%")
    print()
    print("  Per-vertex errors vs real-beta target:")
    print(f"    ZD-ECN  < 1 mm : {pt['zdecn_below_1mm_pct']:5.1f}%  |  LBS < 1 mm : {pt['lbs_below_1mm_pct']:5.1f}%")
    print(f"    ZD-ECN  < 2 mm : {pt['zdecn_below_2mm_pct']:5.1f}%  |  LBS < 2 mm : {pt['lbs_below_2mm_pct']:5.1f}%")
    print(f"    ZD-ECN  < 5 mm : {pt['zdecn_below_5mm_pct']:5.1f}%  |  LBS < 5 mm : {pt['lbs_below_5mm_pct']:5.1f}%")
    print()
    print(f"  Results: {out_json}")
    print("═" * 62)


if __name__ == "__main__":
    main()

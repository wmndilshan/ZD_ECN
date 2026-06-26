#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_zdecn_dqs import INPUT_SIZE, NUM_BETAS, VERTEX_COUNT, build_model, parse_weight_binary
from generate_smpl_lbs_validation_fixture import (
    JOINT_COUNT,
    batch_rigid_transform,
    batch_rodrigues,
    load_model,
)


def poseblend_target(
    model_data: dict[str, np.ndarray],
    poses: np.ndarray,
    v_lbs: np.ndarray,
    batch_size: int,
    betas: np.ndarray | None = None,
) -> np.ndarray:
    if betas is None:
        betas = np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)

    targets = []
    ident = np.eye(3, dtype=np.float32)
    for start in range(0, poses.shape[0], batch_size):
        pose = poses[start:start + batch_size].astype(np.float32)
        beta = betas[start:start + pose.shape[0]].astype(np.float32)
        v_shaped = model_data["v_template"][None, :, :] + np.einsum("bl,mkl->bmk", beta, model_data["shapedirs"])
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(pose.shape[0], JOINT_COUNT, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(pose.shape[0], -1)
        pose_offsets = np.matmul(pose_feature, model_data["posedirs"]).reshape(pose.shape[0], VERTEX_COUNT, 3)
        v_posecorrected = v_shaped + pose_offsets
        targets.append((v_posecorrected - v_lbs[start:start + pose.shape[0]]) * 1000.0)
    return np.concatenate(targets, axis=0).astype(np.float32)


def predict(model: torch.nn.Module, theta_rot6d: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, theta_rot6d.shape[0], batch_size):
            x = torch.from_numpy(theta_rot6d[start:start + batch_size].astype(np.float32))
            outputs.append(model(x).numpy().reshape(-1, VERTEX_COUNT, 3))
    return np.concatenate(outputs, axis=0).astype(np.float32)


def per_pose_norm_stats(values_mm: np.ndarray) -> dict[str, float | int]:
    per_vertex = np.linalg.norm(values_mm, axis=2)
    per_pose = per_vertex.mean(axis=1)
    return {
        "samples": int(per_pose.size),
        "mean_mm": float(np.mean(per_pose)),
        "std_mm": float(np.std(per_pose, ddof=1)),
        "p50_mm": float(np.percentile(per_pose, 50)),
        "p95_mm": float(np.percentile(per_pose, 95)),
        "p99_mm": float(np.percentile(per_pose, 99)),
        "max_mm": float(np.max(per_pose)),
        "per_vertex_p99_mm": float(np.percentile(per_vertex, 99)),
        "per_vertex_max_mm": float(np.max(per_vertex)),
    }


def error_stats(predicted_mm: np.ndarray, target_mm: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    per_vertex = np.linalg.norm(predicted_mm - target_mm, axis=2)
    per_pose = per_vertex.mean(axis=1)
    return {
        "samples": int(per_pose.size),
        "mean_mm": float(np.mean(per_pose)),
        "std_mm": float(np.std(per_pose, ddof=1)),
        "p50_mm": float(np.percentile(per_pose, 50)),
        "p95_mm": float(np.percentile(per_pose, 95)),
        "p99_mm": float(np.percentile(per_pose, 99)),
        "max_mm": float(np.max(per_pose)),
    }, per_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=Path("data/dataset_eval_3k.npz"))
    parser.add_argument("--model-path", type=Path, default=Path("models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"))
    parser.add_argument("--weights", type=Path, default=Path("models/weights_real_direct_fp16.bin"))
    parser.add_argument("--summary", type=Path, default=Path("results/paper/zdecn_poseblend_target_eval_summary.json"))
    parser.add_argument("--smpl-batch-size", type=int, default=20)
    parser.add_argument("--mlp-batch-size", type=int, default=16)
    args = parser.parse_args()

    data = np.load(args.eval_set, allow_pickle=False)
    poses = data["pose_axis_angle"].astype(np.float32)
    theta_rot6d = data["theta_rot6d"].astype(np.float32)
    v_lbs = data["v_lbs"].astype(np.float32)
    betas = data["betas"].astype(np.float32) if "betas" in data.files else np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)

    model_data = load_model(args.model_path)
    target_mm = poseblend_target(model_data, poses, v_lbs, args.smpl_batch_size, betas=betas)
    network = build_model(parse_weight_binary(args.weights))
    predicted_mm = predict(network, theta_rot6d, args.mlp_batch_size)

    pve, per_pose = error_stats(predicted_mm, target_mm)
    target_stats = per_pose_norm_stats(target_mm)
    pred_stats = per_pose_norm_stats(predicted_mm)

    per_pose_path = args.summary.with_name(args.summary.stem + "_per_pose_pve_mm.csv")
    target_path = args.summary.with_name(args.summary.stem + "_target_scale_per_pose_mm.csv")
    np.savetxt(per_pose_path, per_pose, delimiter=",", header="per_pose_pve_mm", comments="")
    target_per_pose = np.linalg.norm(target_mm, axis=2).mean(axis=1)
    np.savetxt(target_path, target_per_pose, delimiter=",", header="target_poseblend_mean_delta_mm", comments="")

    summary = {
        "eval_set": str(args.eval_set),
        "samples": int(poses.shape[0]),
        "target": "pose-corrected vertices before final LBS skinning minus v_lbs, converted meters to millimeters",
        "target_scale": target_stats,
        "predicted_delta_scale": pred_stats,
        "pve_against_poseblend_target": pve,
        "reference_training_distribution": {
            "mean_delta_v_mm": 5.99,
            "p99_delta_v_mm": 29.27,
            "source": "PROJECT_CONTEXT.md",
        },
        "artifacts": {
            "per_pose_pve_csv": str(per_pose_path),
            "target_scale_per_pose_csv": str(target_path),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

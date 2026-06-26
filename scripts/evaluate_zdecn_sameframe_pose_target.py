#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_zdecn_dqs import INPUT_SIZE, NUM_BETAS, VERTEX_COUNT, build_model, parse_weight_binary
from generate_smpl_lbs_validation_fixture import JOINT_COUNT, batch_rigid_transform, batch_rodrigues, load_model


def skin_with_transforms(vertices: np.ndarray, transforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    batch = vertices.shape[0]
    transforms_flat = transforms.reshape(batch, JOINT_COUNT, 16)
    blended = np.matmul(
        np.broadcast_to(weights[None, :, :], (batch, VERTEX_COUNT, JOINT_COUNT)),
        transforms_flat,
    ).reshape(batch, VERTEX_COUNT, 4, 4)
    v_homo = np.concatenate([vertices, np.ones((batch, VERTEX_COUNT, 1), dtype=np.float32)], axis=2)
    return np.matmul(blended, v_homo[:, :, :, None])[:, :, :3, 0].astype(np.float32)


def sameframe_target(model_data: dict[str, np.ndarray], poses: np.ndarray, existing_v_lbs: np.ndarray, batch_size: int, betas: np.ndarray | None = None):
    if betas is None:
        betas = np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)
    ident = np.eye(3, dtype=np.float32)
    targets = []
    ref_diffs = []
    nopose_vertices = []
    for start in range(0, poses.shape[0], batch_size):
        pose = poses[start:start + batch_size].astype(np.float32)
        beta = betas[start:start + pose.shape[0]].astype(np.float32)
        v_shaped = model_data["v_template"][None, :, :] + np.einsum("bl,mkl->bmk", beta, model_data["shapedirs"])
        joints = np.einsum("bik,ji->bjk", v_shaped, model_data["J_regressor"])
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(pose.shape[0], JOINT_COUNT, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(pose.shape[0], -1)
        pose_offsets = np.matmul(pose_feature, model_data["posedirs"]).reshape(pose.shape[0], VERTEX_COUNT, 3)
        v_posed_corrected = v_shaped + pose_offsets
        _, transforms = batch_rigid_transform(rot_mats, joints, model_data["parents"])
        v_ref_skinned = skin_with_transforms(v_posed_corrected, transforms, model_data["lbs_weights"])
        v_lbs_nopose = skin_with_transforms(v_shaped, transforms, model_data["lbs_weights"])
        ref_diffs.append(v_ref_skinned - existing_v_lbs[start:start + pose.shape[0]])
        nopose_vertices.append(v_lbs_nopose)
        targets.append((v_ref_skinned - v_lbs_nopose) * 1000.0)
    return np.concatenate(targets).astype(np.float32), np.concatenate(ref_diffs).astype(np.float32), np.concatenate(nopose_vertices).astype(np.float32)


def predict(model: torch.nn.Module, theta_rot6d: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, theta_rot6d.shape[0], batch_size):
            x = torch.from_numpy(theta_rot6d[start:start + batch_size].astype(np.float32))
            outputs.append(model(x).numpy().reshape(-1, VERTEX_COUNT, 3))
    return np.concatenate(outputs).astype(np.float32)


def stats_from_vectors(values: np.ndarray) -> dict[str, float | int]:
    per_vertex = np.linalg.norm(values, axis=2)
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


def error_stats(predicted: np.ndarray, target: np.ndarray):
    per_vertex = np.linalg.norm(predicted - target, axis=2)
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
    parser.add_argument("--summary", type=Path, default=Path("results/paper/zdecn_sameframe_pose_target_eval_summary.json"))
    parser.add_argument("--smpl-batch-size", type=int, default=20)
    parser.add_argument("--mlp-batch-size", type=int, default=16)
    args = parser.parse_args()

    data = np.load(args.eval_set, allow_pickle=False)
    poses = data["pose_axis_angle"].astype(np.float32)
    theta_rot6d = data["theta_rot6d"].astype(np.float32)
    existing_v_lbs = data["v_lbs"].astype(np.float32)
    betas = data["betas"].astype(np.float32) if "betas" in data.files else np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)

    model_data = load_model(args.model_path)
    target_mm, ref_diff_m, nopose_vertices = sameframe_target(model_data, poses, existing_v_lbs, args.smpl_batch_size, betas=betas)
    ref_diff_mm_stats = stats_from_vectors(ref_diff_m * 1000.0)
    target_scale = stats_from_vectors(target_mm)
    gate_pass = target_scale["mean_mm"] < 50.0 and target_scale["p99_mm"] < 150.0

    summary: dict[str, object] = {
        "eval_set": str(args.eval_set),
        "samples": int(poses.shape[0]),
        "existing_v_lbs_poseblend_status": {
            "status": "WITH pose blendshapes",
            "evidence": "scripts/evaluate_zdecn_dqs.py computes pose_offsets, sets v_posed=v_shaped+pose_offsets, then skins v_posed into v_lbs; smpl_skinning.comp also adds pose_features before skinning.",
            "recomputed_v_ref_skinned_vs_existing_v_lbs": ref_diff_mm_stats,
        },
        "target_formula": "skin_A(theta)(v_shaped + pose_bs(theta)) - skin_A(theta)(v_shaped), converted meters to millimeters",
        "target_scale": target_scale,
        "scale_gate": {
            "passed": bool(gate_pass),
            "criterion": "mean < 50 mm and p99 < 150 mm; expected single-digit mean and double-digit p99 from PROJECT_CONTEXT.md 5.99/29.27 mm",
        },
    }

    if gate_pass:
        network = build_model(parse_weight_binary(args.weights))
        predicted_mm = predict(network, theta_rot6d, args.mlp_batch_size)
        pve, per_pose = error_stats(predicted_mm, target_mm)
        per_pose_path = args.summary.with_name(args.summary.stem + "_per_pose_pve_mm.csv")
        np.savetxt(per_pose_path, per_pose, delimiter=",", header="per_pose_pve_mm", comments="")
        summary["pve_against_sameframe_pose_target"] = pve
        summary["per_pose_pve_csv"] = str(per_pose_path)
        summary["predicted_delta_scale"] = stats_from_vectors(predicted_mm)
    else:
        summary["pve_against_sameframe_pose_target"] = None

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

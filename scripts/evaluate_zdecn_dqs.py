#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_smpl_lbs_validation_fixture import (  # noqa: E402
    JOINT_COUNT,
    NUM_BETAS,
    VERTEX_COUNT,
    batch_rodrigues,
    batch_rigid_transform,
    load_model,
)


INPUT_SIZE = 144
HIDDEN_SIZE = 64
OUTPUT_SIZE = 20670
POSE_BASIS_COUNT = 207


def read_fp16(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float16)


def parse_weight_binary(path: Path) -> dict[str, np.ndarray]:
    raw = read_fp16(path)
    expected = (
        64 * 144 + 64 +
        64 * 64 + 64 +
        20670 * 64 + 20670
    )
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} fp16 values, expected {expected}")

    offset = 0

    def take(count: int) -> np.ndarray:
        nonlocal offset
        out = raw[offset:offset + count]
        offset += count
        return out

    arrays = {
        "W0": take(64 * 144).reshape(64, 144).astype(np.float32),
        "b0": take(64).astype(np.float32),
        "W1": take(64 * 64).reshape(64, 64).astype(np.float32),
        "b1": take(64).astype(np.float32),
        "W2": take(20670 * 64).reshape(20670, 64).astype(np.float32),
        "b2": take(20670).astype(np.float32),
    }
    if offset != raw.size:
        raise AssertionError("weight parser did not consume all values")
    return arrays


def verify_split_assets(arrays: dict[str, np.ndarray], asset_dir: Path) -> dict[str, float]:
    checks = {}
    names = [
        ("W0", "weights_l0.bin", (64, 144)),
        ("b0", "biases_l0.bin", (64,)),
        ("W1", "weights_l1.bin", (64, 64)),
        ("b1", "biases_l1.bin", (64,)),
        ("W2", "weights_l2.bin", (20670, 64)),
        ("b2", "biases_l2.bin", (20670,)),
    ]
    for key, filename, shape in names:
        split = read_fp16(asset_dir / filename).reshape(shape).astype(np.float32)
        checks[key] = float(np.max(np.abs(split - arrays[key])))
    return checks


def build_model(arrays: dict[str, np.ndarray]) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(144, 64), nn.GELU(approximate="tanh"),
        nn.Linear(64, 64), nn.GELU(approximate="tanh"),
        nn.Linear(64, 20670),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(arrays["W0"]))
        model[0].bias.copy_(torch.from_numpy(arrays["b0"]))
        model[2].weight.copy_(torch.from_numpy(arrays["W1"]))
        model[2].bias.copy_(torch.from_numpy(arrays["b1"]))
        model[4].weight.copy_(torch.from_numpy(arrays["W2"]))
        model[4].bias.copy_(torch.from_numpy(arrays["b2"]))
    model.eval()
    return model


def validate_reconstruction(model: nn.Module, fixture_dir: Path, samples: int) -> dict[str, float | int | str]:
    inputs = np.fromfile(fixture_dir / "inputs_rot6d_f32.bin", dtype=np.float32).reshape(-1, INPUT_SIZE)
    reference = np.fromfile(fixture_dir / "outputs_ref_f32.bin", dtype=np.float32).reshape(-1, OUTPUT_SIZE)
    n = min(samples, inputs.shape[0])
    with torch.no_grad():
        pred = model(torch.from_numpy(inputs[:n])).numpy()
    diff = np.abs(pred - reference[:n])
    return {
        "samples": int(n),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "fixture": str(fixture_dir),
        "reference": "same fixture consumed by Android zdecn_validate GLSL readback mode",
    }


def sample_surreal_poses(path: Path, count: int, seed: int) -> tuple[np.ndarray, list[tuple[str, int]]]:
    data = np.load(path, allow_pickle=False)
    candidates: list[tuple[str, int]] = []
    for key in sorted(data.files):
        parts = key.split("_")
        if key.startswith("pose_ung_") and len(parts) >= 3:
            subject = parts[2]
        elif key.startswith("pose_") and len(parts) >= 2:
            subject = parts[1]
        else:
            continue
        if subject in {"18", "19"}:
            continue
        rows = int(data[key].shape[0])
        candidates.extend((key, row) for row in range(rows))
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible poses, need {count}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=count, replace=False)
    selected = [candidates[int(i)] for i in indices]
    poses = np.empty((count, JOINT_COUNT * 3), dtype=np.float32)
    for out_idx, (key, row) in enumerate(selected):
        poses[out_idx] = np.asarray(data[key][row], dtype=np.float32)
    return poses, selected


def sample_surreal_betas(path: Path, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    male_shapes = np.asarray(data["maleshapes"], dtype=np.float32)
    female_shapes = np.asarray(data["femaleshapes"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    genders = rng.integers(0, 2, size=count, dtype=np.int32)
    male_indices = rng.integers(0, male_shapes.shape[0], size=count, dtype=np.int32)
    female_indices = rng.integers(0, female_shapes.shape[0], size=count, dtype=np.int32)
    betas = np.empty((count, NUM_BETAS), dtype=np.float32)
    male_mask = genders == 0
    female_mask = ~male_mask
    betas[male_mask] = male_shapes[male_indices[male_mask], :NUM_BETAS]
    betas[female_mask] = female_shapes[female_indices[female_mask], :NUM_BETAS]
    shape_indices = np.where(male_mask, male_indices, female_indices).astype(np.int32)
    return betas, genders, shape_indices


def axis_angle_to_rot6d(poses: np.ndarray) -> np.ndarray:
    rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(poses.shape[0], JOINT_COUNT, 3, 3)
    return rotations[:, :, :, :2].transpose(0, 1, 3, 2).reshape(poses.shape[0], INPUT_SIZE).astype(np.float32)


def quat_from_matrix(mats: np.ndarray) -> np.ndarray:
    m = mats.astype(np.float64, copy=False)
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    pos = trace > 0.0
    s = np.sqrt(np.maximum(trace[pos] + 1.0, 0.0)) * 2.0
    q[pos, 0] = 0.25 * s
    q[pos, 1] = (m[pos, 2, 1] - m[pos, 1, 2]) / s
    q[pos, 2] = (m[pos, 0, 2] - m[pos, 2, 0]) / s
    q[pos, 3] = (m[pos, 1, 0] - m[pos, 0, 1]) / s

    neg = ~pos
    if np.any(neg):
        mn = m[neg]
        qn = q[neg]
        c0 = (mn[:, 0, 0] > mn[:, 1, 1]) & (mn[:, 0, 0] > mn[:, 2, 2])
        c1 = ~c0 & (mn[:, 1, 1] > mn[:, 2, 2])
        c2 = ~(c0 | c1)
        s0 = np.sqrt(np.maximum(1.0 + mn[c0, 0, 0] - mn[c0, 1, 1] - mn[c0, 2, 2], 0.0)) * 2.0
        qn[c0, 0] = (mn[c0, 2, 1] - mn[c0, 1, 2]) / s0
        qn[c0, 1] = 0.25 * s0
        qn[c0, 2] = (mn[c0, 0, 1] + mn[c0, 1, 0]) / s0
        qn[c0, 3] = (mn[c0, 0, 2] + mn[c0, 2, 0]) / s0
        s1 = np.sqrt(np.maximum(1.0 + mn[c1, 1, 1] - mn[c1, 0, 0] - mn[c1, 2, 2], 0.0)) * 2.0
        qn[c1, 0] = (mn[c1, 0, 2] - mn[c1, 2, 0]) / s1
        qn[c1, 1] = (mn[c1, 0, 1] + mn[c1, 1, 0]) / s1
        qn[c1, 2] = 0.25 * s1
        qn[c1, 3] = (mn[c1, 1, 2] + mn[c1, 2, 1]) / s1
        s2 = np.sqrt(np.maximum(1.0 + mn[c2, 2, 2] - mn[c2, 0, 0] - mn[c2, 1, 1], 0.0)) * 2.0
        qn[c2, 0] = (mn[c2, 1, 0] - mn[c2, 0, 1]) / s2
        qn[c2, 1] = (mn[c2, 0, 2] + mn[c2, 2, 0]) / s2
        qn[c2, 2] = (mn[c2, 1, 2] + mn[c2, 2, 1]) / s2
        qn[c2, 3] = 0.25 * s2
        q[neg] = qn

    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def dqs_from_transforms(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qr = quat_from_matrix(transforms[:, :, :3, :3])
    translation = transforms[:, :, :3, 3]
    tquat = np.concatenate([np.zeros((*translation.shape[:-1], 1), dtype=np.float32), translation], axis=-1)
    qd = 0.5 * quat_mul(tquat, qr)
    return qr.astype(np.float32), qd.astype(np.float32)


def dqs_skin(v_posed: np.ndarray, transforms: np.ndarray, lbs_weights: np.ndarray) -> np.ndarray:
    batch = v_posed.shape[0]
    qr, qd = dqs_from_transforms(transforms)
    out = np.empty_like(v_posed)
    weights = lbs_weights.astype(np.float32)
    for b in range(batch):
        ref = qr[b, np.argmax(weights, axis=1)]
        signs = np.where(np.sum(qr[b][None, :, :] * ref[:, None, :], axis=-1) < 0.0, -1.0, 1.0).astype(np.float32)
        blend_r = np.sum(weights[:, :, None] * signs[:, :, None] * qr[b][None, :, :], axis=1)
        blend_d = np.sum(weights[:, :, None] * signs[:, :, None] * qd[b][None, :, :], axis=1)
        norm = np.linalg.norm(blend_r, axis=1, keepdims=True)
        blend_r /= norm
        blend_d /= norm
        conj = blend_r.copy()
        conj[:, 1:] *= -1.0
        point = np.concatenate([np.zeros((VERTEX_COUNT, 1), dtype=np.float32), v_posed[b]], axis=1)
        rotated = quat_mul(quat_mul(blend_r, point), conj)[:, 1:]
        translated = 2.0 * quat_mul(blend_d, conj)[:, 1:]
        out[b] = rotated + translated
    return out.astype(np.float32)


def lbs_and_dqs(
    model_data: dict[str, np.ndarray],
    poses: np.ndarray,
    batch_size: int,
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_lbs_all = []
    v_dqs_all = []
    rot6d_all = []
    if betas is None:
        betas = np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)
    ident = np.eye(3, dtype=np.float32)
    for start in range(0, poses.shape[0], batch_size):
        pose = poses[start:start + batch_size].astype(np.float32)
        batch = pose.shape[0]
        beta = betas[start:start + batch].astype(np.float32)
        v_shaped = model_data["v_template"][None, :, :] + np.einsum("bl,mkl->bmk", beta, model_data["shapedirs"])
        joints = np.einsum("bik,ji->bjk", v_shaped, model_data["J_regressor"])
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch, JOINT_COUNT, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch, -1)
        pose_offsets = np.matmul(pose_feature, model_data["posedirs"]).reshape(batch, VERTEX_COUNT, 3)
        v_posed = v_shaped + pose_offsets
        _, transforms = batch_rigid_transform(rot_mats, joints, model_data["parents"])

        transforms_flat = transforms.reshape(batch, JOINT_COUNT, 16)
        blended = np.matmul(
            np.broadcast_to(model_data["lbs_weights"][None, :, :], (batch, VERTEX_COUNT, JOINT_COUNT)),
            transforms_flat,
        ).reshape(batch, VERTEX_COUNT, 4, 4)
        v_homo = np.concatenate([v_posed, np.ones((batch, VERTEX_COUNT, 1), dtype=np.float32)], axis=2)
        v_lbs = np.matmul(blended, v_homo[:, :, :, None])[:, :, :3, 0]
        v_dqs = dqs_skin(v_posed, transforms, model_data["lbs_weights"])
        v_lbs_all.append(v_lbs.astype(np.float32))
        v_dqs_all.append(v_dqs.astype(np.float32))
        rot6d_all.append(rot_mats[:, :, :, :2].transpose(0, 1, 3, 2).reshape(batch, INPUT_SIZE).astype(np.float32))
    return np.concatenate(rot6d_all), np.concatenate(v_lbs_all), np.concatenate(v_dqs_all)


def evaluate_pve(
    model: nn.Module,
    theta_rot6d: np.ndarray,
    v_lbs: np.ndarray,
    v_dqs: np.ndarray,
    batch_size: int,
    model_output_scale_to_meters: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    errors = []
    with torch.no_grad():
        for start in range(0, theta_rot6d.shape[0], batch_size):
            x = torch.from_numpy(theta_rot6d[start:start + batch_size])
            pred = model(x).numpy().reshape(-1, VERTEX_COUNT, 3) * model_output_scale_to_meters
            corrected = v_lbs[start:start + pred.shape[0]] + pred
            per_vertex_mm = np.linalg.norm(corrected - v_dqs[start:start + pred.shape[0]], axis=2) * 1000.0
            errors.append(per_vertex_mm.mean(axis=1))
    pve = np.concatenate(errors)
    return {
        "samples": int(pve.size),
        "mean_mm": float(np.mean(pve)),
        "std_mm": float(np.std(pve, ddof=1)),
        "p50_mm": float(np.percentile(pve, 50)),
        "p95_mm": float(np.percentile(pve, 95)),
        "p99_mm": float(np.percentile(pve, 99)),
        "max_mm": float(np.max(pve)),
    }, pve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("models/weights_real_direct_fp16.bin"))
    parser.add_argument("--shader-assets", type=Path, default=Path("android_compute_benchmark/app/src/main/assets/shaders"))
    parser.add_argument("--fixture-dir", type=Path, default=Path("android_compute_benchmark/app/src/main/assets/zdecn_validation_fixture"))
    parser.add_argument("--surreal", type=Path, default=Path("surreal/zd_ecn_subset/smpl_data.npz"))
    parser.add_argument("--model-path", type=Path, default=Path("models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset_eval_3k.npz"))
    parser.add_argument("--summary", type=Path, default=Path("results/paper/zdecn_dqs_eval_summary.json"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--smpl-batch-size", type=int, default=20)
    parser.add_argument("--mlp-batch-size", type=int, default=16)
    parser.add_argument("--reuse-existing-eval-set", action="store_true")
    parser.add_argument(
        "--beta-mode",
        choices=["zero", "surreal_shapes"],
        default="zero",
        help="Shape coefficients for newly built eval sets. surreal_shapes samples maleshapes/femaleshapes from smpl_data.npz.",
    )
    parser.add_argument(
        "--model-output-units",
        choices=["mm", "m"],
        default="mm",
        help="Units of the ZD-ECN delta output. SMPL vertices are meters; production deltas are treated as millimeters by default.",
    )
    args = parser.parse_args()

    arrays = parse_weight_binary(args.weights)
    split_checks = verify_split_assets(arrays, args.shader_assets)
    model = build_model(arrays)
    reconstruction = validate_reconstruction(model, args.fixture_dir, samples=10)

    if args.reuse_existing_eval_set:
        existing = np.load(args.output, allow_pickle=False)
        theta_rot6d = existing["theta_rot6d"].astype(np.float32)
        v_lbs = existing["v_lbs"].astype(np.float32)
        v_dqs = existing["v_dqs"].astype(np.float32)
        delta_v_target = existing["delta_v_target"].astype(np.float32)
        poses = existing["pose_axis_angle"].astype(np.float32) if "pose_axis_angle" in existing.files else np.empty((theta_rot6d.shape[0], JOINT_COUNT * 3), dtype=np.float32)
        selected = []
    else:
        poses, selected = sample_surreal_poses(args.surreal, args.samples, args.seed)
        model_data = load_model(args.model_path)
        if args.beta_mode == "surreal_shapes":
            betas, genders, shape_indices = sample_surreal_betas(args.surreal, poses.shape[0], args.seed + 17)
        else:
            betas = np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)
            genders = np.full((poses.shape[0],), -1, dtype=np.int32)
            shape_indices = np.full((poses.shape[0],), -1, dtype=np.int32)
        theta_rot6d, v_lbs, v_dqs = lbs_and_dqs(model_data, poses, args.smpl_batch_size, betas=betas)
        delta_v_target = (v_dqs - v_lbs).astype(np.float32)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            theta_rot6d=theta_rot6d.astype(np.float32),
            pose_axis_angle=poses.astype(np.float32),
            betas=betas.astype(np.float32),
            gender_code=genders.astype(np.int32),
            shape_index=shape_indices.astype(np.int32),
            v_lbs=v_lbs.astype(np.float32),
            v_dqs=v_dqs.astype(np.float32),
            delta_v_target=delta_v_target,
            source_keys=np.asarray([k for k, _ in selected]),
            source_rows=np.asarray([r for _, r in selected], dtype=np.int32),
            reference_target=np.asarray("DQS_minus_LBS_evaluation_only"),
        )

    model_output_scale_to_meters = 0.001 if args.model_output_units == "mm" else 1.0
    pve, per_pose_pve = evaluate_pve(model, theta_rot6d, v_lbs, v_dqs, args.mlp_batch_size, model_output_scale_to_meters)
    per_pose_path = args.summary.with_name(args.summary.stem + "_per_pose_pve_mm.csv")
    np.savetxt(per_pose_path, per_pose_pve, delimiter=",", header="per_pose_pve_mm", comments="")
    summary = {
        "weight_binary": str(args.weights),
        "weight_order": "W0[64,144],b0[64],W1[64,64],b1[64],W2[20670,64],b2[20670], all fp16 row-major",
        "split_asset_max_abs_diffs": split_checks,
        "binary_reconstruction": reconstruction,
        "eval_set": {
            "path": str(args.output),
            "samples": args.samples,
            "actual_samples": int(theta_rot6d.shape[0]),
            "source": str(args.surreal),
            "reused_existing_eval_set": bool(args.reuse_existing_eval_set),
            "excluded_subjects": ["18", "19"],
            "betas": "sampled from SURREAL maleshapes/femaleshapes with deterministic 50/50 gender draw" if args.beta_mode == "surreal_shapes" and not args.reuse_existing_eval_set else "all zeros or loaded from existing eval set",
            "beta_mode": args.beta_mode,
            "target": "closed-form DQS blend from the same SMPL pose blend and joint transforms, minus LBS",
            "vertex_units": "meters",
            "model_output_units": args.model_output_units,
            "model_output_scale_to_meters": model_output_scale_to_meters,
            "methodology_caveat": "Evaluation-only DQS target; not the original dataset_real.npz methodology.",
        },
        "pve_against_dqs": pve,
        "per_pose_pve_csv": str(per_pose_path),
        "paper_reference_mean_pve_mm": 4.48,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build the canonical ZD-ECN training/evaluation dataset.

This script is intentionally explicit because the original dataset_real.npz
and its construction script are not present in this checkout. The target below
is the only versioned target-generation convention for new retraining runs.

Canonical decisions:
  - Raw pose source is hard-coded by default to
    surreal/zd_ecn_subset/smpl_data.npz.
  - Subjects 18 and 19 are excluded.
  - Betas are fixed to zero. ZD-ECN's production input is pose-only
    rot6d_144, so shape-dependent targets would be unobservable by the model.
  - Target is same-coordinate-frame pose correction:
      target_mm = (skin_A(theta)(v_shaped + pose_bs(theta))
                   - skin_A(theta)(v_shaped)) * 1000
    Both terms are skinned with the same posed joint transforms A(theta), so the
    delta is in the posed frame and is saved in millimeters.
  - Sampling is natural, unweighted random sampling without replacement from
    real CMU MoCap pose sequences. No synthetic stratification is applied.
  - Split is deterministic train/validation using a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_zdecn_dqs import INPUT_SIZE, NUM_BETAS, VERTEX_COUNT
from generate_smpl_lbs_validation_fixture import (
    JOINT_COUNT,
    batch_rigid_transform,
    batch_rodrigues,
    load_model,
)


RAW_POSE_SOURCE = Path("surreal/zd_ecn_subset/smpl_data.npz")
SMPL_MODEL_PATH = Path("models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
EXCLUDED_SUBJECTS = {18, 19}
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SAMPLE_COUNT = 30_000


def parse_subject_id(key: str) -> int | None:
    match = re.match(r"^pose(?:_ung)?_(\d+)_", key)
    return int(match.group(1)) if match else None


def pose_keys(raw: np.lib.npyio.NpzFile) -> list[str]:
    keys: list[str] = []
    for key in raw.files:
        subject_id = parse_subject_id(key)
        if subject_id is None or subject_id in EXCLUDED_SUBJECTS:
            continue
        value = raw[key]
        if value.ndim == 2 and value.shape[1] == JOINT_COUNT * 3:
            keys.append(key)
    return sorted(keys)


def load_poses_from_existing_eval(path: Path, samples: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    poses = data["pose_axis_angle"].astype(np.float32)
    theta_rot6d = data["theta_rot6d"].astype(np.float32)
    if samples is not None:
        poses = poses[:samples]
        theta_rot6d = theta_rot6d[:samples]
    source_keys = np.array([path.name] * poses.shape[0])
    source_rows = np.arange(poses.shape[0], dtype=np.int64)
    metadata = {
        "pose_source_mode": "existing_eval_set",
        "pose_source": str(path),
        "sampling": "first N rows from the existing eval set; no new pose sampling",
    }
    return poses, theta_rot6d, source_keys, source_rows, metadata


def load_poses_from_raw(path: Path, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.load(path, allow_pickle=False)
    keys = pose_keys(raw)
    if not keys:
        raise RuntimeError(f"No usable pose arrays found in {path}")

    lengths = np.array([raw[key].shape[0] for key in keys], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total_rows = int(offsets[-1])
    if samples > total_rows:
        raise ValueError(f"Requested {samples} samples, but only {total_rows} usable rows are available")

    rng = np.random.default_rng(seed)
    global_rows = rng.choice(total_rows, size=samples, replace=False)
    key_indices = np.searchsorted(offsets[1:], global_rows, side="right")
    local_rows = global_rows - offsets[key_indices]

    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for output_index, (key_index, row_index) in enumerate(zip(key_indices, local_rows, strict=True)):
        grouped[int(key_index)].append((output_index, int(row_index)))

    poses = np.empty((samples, JOINT_COUNT * 3), dtype=np.float32)
    source_keys = np.empty(samples, dtype=f"<U{max(len(key) for key in keys)}")
    source_rows = local_rows.astype(np.int64)
    for key_index, entries in grouped.items():
        key = keys[key_index]
        output_indices = np.array([entry[0] for entry in entries], dtype=np.int64)
        row_indices = np.array([entry[1] for entry in entries], dtype=np.int64)
        poses[output_indices] = raw[key][row_indices].astype(np.float32)
        source_keys[output_indices] = key

    theta_rot6d = axis_angle_to_rot6d(poses)
    metadata = {
        "pose_source_mode": "raw_surreal_subset",
        "pose_source": str(path),
        "usable_pose_arrays": len(keys),
        "usable_pose_rows": total_rows,
        "sampling": "natural unweighted random sampling without replacement",
    }
    return poses, theta_rot6d, source_keys, source_rows, metadata


def axis_angle_to_rot6d(poses: np.ndarray) -> np.ndarray:
    rot_mats = batch_rodrigues(poses.reshape(-1, 3)).reshape(poses.shape[0], JOINT_COUNT, 3, 3)
    return rot_mats[:, :, :, :2].transpose(0, 1, 3, 2).reshape(poses.shape[0], INPUT_SIZE).astype(np.float32)


def skin_with_transforms(vertices: np.ndarray, transforms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    batch = vertices.shape[0]
    transforms_flat = transforms.reshape(batch, JOINT_COUNT, 16)
    blended = np.matmul(
        np.broadcast_to(weights[None, :, :], (batch, VERTEX_COUNT, JOINT_COUNT)),
        transforms_flat,
    ).reshape(batch, VERTEX_COUNT, 4, 4)
    v_homo = np.concatenate([vertices, np.ones((batch, VERTEX_COUNT, 1), dtype=np.float32)], axis=2)
    return np.matmul(blended, v_homo[:, :, :, None])[:, :, :3, 0].astype(np.float32)


def build_sameframe_targets(
    model_data: dict[str, np.ndarray],
    poses: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    betas = np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    target_chunks: list[np.ndarray] = []
    target_magnitude_chunks: list[np.ndarray] = []

    for start in range(0, poses.shape[0], batch_size):
        pose = poses[start:start + batch_size].astype(np.float32)
        beta = betas[start:start + pose.shape[0]]
        v_shaped = model_data["v_template"][None, :, :] + np.einsum("bl,mkl->bmk", beta, model_data["shapedirs"])
        joints = np.einsum("bik,ji->bjk", v_shaped, model_data["J_regressor"])
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(pose.shape[0], JOINT_COUNT, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - identity).reshape(pose.shape[0], -1)
        pose_offsets = np.matmul(pose_feature, model_data["posedirs"]).reshape(pose.shape[0], VERTEX_COUNT, 3)
        _, transforms = batch_rigid_transform(rot_mats, joints, model_data["parents"])

        v_ref = skin_with_transforms(v_shaped + pose_offsets, transforms, model_data["lbs_weights"])
        v_base = skin_with_transforms(v_shaped, transforms, model_data["lbs_weights"])
        target_mm = (v_ref - v_base) * 1000.0
        target_chunks.append(target_mm.reshape(pose.shape[0], VERTEX_COUNT * 3).astype(np.float32))
        target_magnitude_chunks.append(np.linalg.norm(target_mm, axis=2).mean(axis=1).astype(np.float32))

    return np.concatenate(target_chunks), np.concatenate(target_magnitude_chunks)


def split_indices(samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(samples)
    val_count = int(round(samples * val_ratio))
    val_indices = np.sort(indices[:val_count]).astype(np.int64)
    train_indices = np.sort(indices[val_count:]).astype(np.int64)
    return train_indices, val_indices


def write_dataset(
    output: Path,
    metadata_path: Path,
    poses: np.ndarray,
    theta_rot6d: np.ndarray,
    target_delta_mm: np.ndarray,
    target_magnitude_mm: np.ndarray,
    source_keys: np.ndarray,
    source_rows: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    betas = np.zeros((poses.shape[0], NUM_BETAS), dtype=np.float32)
    np.savez_compressed(
        output,
        pose_axis_angle=poses.astype(np.float32),
        theta_rot6d=theta_rot6d.astype(np.float32),
        target_delta_mm=target_delta_mm.astype(np.float32),
        target_magnitude_mm=target_magnitude_mm.astype(np.float32),
        betas=betas,
        source_keys=source_keys,
        source_rows=source_rows.astype(np.int64),
        train_indices=train_indices.astype(np.int64),
        val_indices=val_indices.astype(np.int64),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pose-source", type=Path, default=RAW_POSE_SOURCE)
    parser.add_argument("--model-path", type=Path, default=SMPL_MODEL_PATH)
    parser.add_argument("--from-existing-eval-set", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--smpl-batch-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/dataset_sameframe_pose_30k.npz"))
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    if args.from_existing_eval_set is not None:
        poses, theta_rot6d, source_keys, source_rows, source_metadata = load_poses_from_existing_eval(
            args.from_existing_eval_set,
            args.samples,
        )
    else:
        poses, theta_rot6d, source_keys, source_rows, source_metadata = load_poses_from_raw(
            args.raw_pose_source,
            args.samples,
            args.seed,
        )

    model_data = load_model(args.model_path)
    target_delta_mm, target_magnitude_mm = build_sameframe_targets(model_data, poses, args.smpl_batch_size)
    train_indices, val_indices = split_indices(poses.shape[0], args.val_ratio, args.seed)

    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    target_stats = {
        "mean_mm": float(np.mean(target_magnitude_mm)),
        "std_mm": float(np.std(target_magnitude_mm, ddof=1)),
        "p50_mm": float(np.percentile(target_magnitude_mm, 50)),
        "p95_mm": float(np.percentile(target_magnitude_mm, 95)),
        "p99_mm": float(np.percentile(target_magnitude_mm, 99)),
        "max_mm": float(np.max(target_magnitude_mm)),
    }
    metadata: dict[str, Any] = {
        "format": "ZD-ECN canonical same-frame pose-correction dataset",
        "output": str(args.output),
        "samples": int(poses.shape[0]),
        "model_path": str(args.model_path),
        "excluded_subjects": sorted(EXCLUDED_SUBJECTS),
        "beta_handling": "zero-beta only; production input is pose-only rot6d_144, so beta-dependent targets are not observable",
        "input_representation": "24 joints x first two rotation-matrix columns = rot6d_144",
        "target_formula": "target_mm = (skin_A(theta)(v_shaped + pose_bs(theta)) - skin_A(theta)(v_shaped)) * 1000",
        "target_coordinate_frame": "posed frame; both sides use the same posed joint transforms",
        "target_units": "millimeters",
        "sample_method": "natural/unweighted random sampling; no synthetic stratification",
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "train_samples": int(train_indices.size),
        "val_samples": int(val_indices.size),
        "target_magnitude_stats": target_stats,
        **source_metadata,
    }
    write_dataset(
        args.output,
        metadata_path,
        poses,
        theta_rot6d,
        target_delta_mm,
        target_magnitude_mm,
        source_keys,
        source_rows,
        train_indices,
        val_indices,
        metadata,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

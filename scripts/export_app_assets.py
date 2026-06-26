#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from generate_smpl_lbs_validation_fixture import batch_rodrigues


ROOT = Path(__file__).resolve().parents[1]
SMPL_DATA = ROOT / "surreal/zd_ecn_subset/smpl_data.npz"
SMPL_REF = ROOT / "models/smpl_reference_data"
WEIGHTS_FP16 = ROOT / "models/weights_sameframe_fp16.bin"
ONNX_FP16 = ROOT / "models/zdecn_fp16.onnx"
APP_ASSETS = ROOT / "android_smpl_app/app/src/main/assets"
EXCLUDED_SUBJECTS = {18, 19}
JOINT_COUNT = 24
INPUT_SIZE = 144


REQUIRED_SMPL_FILES = [
    "v_template_f32.bin",
    "shapedirs_10_f16.bin",
    "posedirs_207_f16.bin",
    "j_regressor_sparse.npz",
    "lbs_weights_sparse.npz",
    "parents_i32.bin",
    "faces_u16.bin",
]


def subject_id(key: str) -> int | None:
    match = re.match(r"^pose(?:_ung)?_(\d+)_", key)
    return int(match.group(1)) if match else None


def axis_angle_to_rot6d(poses: np.ndarray) -> np.ndarray:
    rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(poses.shape[0], JOINT_COUNT, 3, 3)
    return rotations[:, :, :, :2].transpose(0, 1, 3, 2).reshape(poses.shape[0], INPUT_SIZE).astype(np.float32)


def ensure_sparse_npz() -> None:
    j_npz = SMPL_REF / "j_regressor_sparse.npz"
    w_npz = SMPL_REF / "lbs_weights_sparse.npz"
    if not j_npz.exists():
        np.savez(
            j_npz,
            offsets=np.fromfile(SMPL_REF / "joint_regressor_offsets_i32.bin", dtype=np.int32),
            indices=np.fromfile(SMPL_REF / "joint_regressor_indices_i32.bin", dtype=np.int32),
            weights=np.fromfile(SMPL_REF / "joint_regressor_weights_f32.bin", dtype=np.float32),
            shape=np.asarray([24, 6890], dtype=np.int32),
        )
    if not w_npz.exists():
        np.savez(
            w_npz,
            offsets=np.fromfile(SMPL_REF / "lbs_weights_offsets_i32.bin", dtype=np.int32),
            indices=np.fromfile(SMPL_REF / "lbs_weights_indices_i32.bin", dtype=np.int32),
            weights=np.fromfile(SMPL_REF / "lbs_weights_values_f32.bin", dtype=np.float32),
            shape=np.asarray([6890, 24], dtype=np.int32),
        )


def verify_smpl_reference() -> list[str]:
    missing = [name for name in REQUIRED_SMPL_FILES if not (SMPL_REF / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing SMPL reference files: {missing}")
    return [str(SMPL_REF / name) for name in REQUIRED_SMPL_FILES]


def collect_pose_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    key_rows: list[tuple[str, int, float]] = []
    for key in sorted(data.files):
        sid = subject_id(key)
        if sid is None or sid in EXCLUDED_SUBJECTS:
            continue
        arr = data[key]
        if arr.ndim != 2 or arr.shape[1] != 72:
            continue
        max_angles = np.linalg.norm(arr.reshape(-1, JOINT_COUNT, 3), axis=2).max(axis=1)
        for row, max_angle in enumerate(max_angles):
            key_rows.append((key, row, float(max_angle)))
    dtype = np.dtype([("key", "U64"), ("row", np.int32), ("max_angle", np.float32)])
    table = np.asarray(key_rows, dtype=dtype)
    return table, np.asarray([r[2] for r in key_rows], dtype=np.float32)


def sample_stratified_poses(path: Path, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    data = np.load(path, allow_pickle=False)
    table, max_angles = collect_pose_rows(path)
    buckets = [
        ("natural", max_angles < 1.5, 200),
        ("moderate", (max_angles >= 1.5) & (max_angles <= 2.5), 200),
        ("extreme", max_angles > 2.5, 200),
    ]
    selected = []
    bucket_meta = {}
    for name, mask, count in buckets:
        indices = np.flatnonzero(mask)
        if indices.size < count:
            raise RuntimeError(f"Bucket {name} has {indices.size} rows, need {count}")
        chosen = rng.choice(indices, size=count, replace=False)
        selected.append(table[chosen])
        bucket_meta[name] = {
            "available": int(indices.size),
            "selected": int(count),
            "min_max_angle": float(max_angles[chosen].min()),
            "max_max_angle": float(max_angles[chosen].max()),
        }
    selection = np.concatenate(selected)
    poses = np.empty((selection.shape[0], 72), dtype=np.float32)
    for out_index, item in enumerate(selection):
        poses[out_index] = data[str(item["key"])][int(item["row"])].astype(np.float32)
    return poses, {"seed": seed, "buckets": bucket_meta}


def copy_tree_files(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dest_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/render_demo")
    parser.add_argument("--app-assets", type=Path, default=APP_ASSETS)
    args = parser.parse_args()

    ensure_sparse_npz()
    smpl_files = verify_smpl_reference()
    for path in [WEIGHTS_FP16, ONNX_FP16, SMPL_DATA]:
        if not path.exists():
            raise FileNotFoundError(path)

    poses, sampling_meta = sample_stratified_poses(SMPL_DATA, args.seed)
    rot6d = axis_angle_to_rot6d(poses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "animation_poses_aa_f32.bin").write_bytes(np.ascontiguousarray(poses).tobytes())
    (args.output_dir / "animation_poses_rot6d_f32.bin").write_bytes(np.ascontiguousarray(rot6d).tobytes())

    app_assets = args.app_assets
    (app_assets / "animation").mkdir(parents=True, exist_ok=True)
    (app_assets / "smpl").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_dir / "animation_poses_aa_f32.bin", app_assets / "animation/animation_poses_aa_f32.bin")
    shutil.copy2(args.output_dir / "animation_poses_rot6d_f32.bin", app_assets / "animation/animation_poses_rot6d_f32.bin")
    shutil.copy2(WEIGHTS_FP16, app_assets / "zdecn_weights_fp16.bin")
    shutil.copy2(ONNX_FP16, app_assets / "zdecn_fp16.onnx")
    copy_tree_files(SMPL_REF, app_assets / "smpl")

    metadata = {
        "smpl_reference_files": smpl_files,
        "weights_fp16": str(WEIGHTS_FP16),
        "weights_fp16_bytes": WEIGHTS_FP16.stat().st_size,
        "onnx_fp16": str(ONNX_FP16),
        "onnx_fp16_bytes": ONNX_FP16.stat().st_size,
        "smpl_data": str(SMPL_DATA),
        "animation_samples": int(poses.shape[0]),
        "axis_angle_file": str(args.output_dir / "animation_poses_aa_f32.bin"),
        "rot6d_file": str(args.output_dir / "animation_poses_rot6d_f32.bin"),
        "sampling": sampling_meta,
        "app_assets": str(app_assets),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

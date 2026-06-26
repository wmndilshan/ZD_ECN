#!/usr/bin/env python3
"""
Generate deterministic SMPL LBS validation fixtures.

This is a NumPy port of the vchoutas/smplx `lbs.py` path used only to produce
golden data for Android shader validation. It does not add PyTorch, smplx, or
any ML runtime dependency to the Android app.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

for _alias, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _value)


VERTEX_COUNT = 6890
JOINT_COUNT = 24
NUM_BETAS = 10
POSE_BASIS_COUNT = 207


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "r"):
        value = value.r
    elif hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "todense"):
        value = value.todense()
    return np.asarray(value)


def load_model(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")

    v_template = to_numpy(data["v_template"]).astype(np.float32)
    shapedirs = to_numpy(data["shapedirs"]).astype(np.float32)
    posedirs = to_numpy(data["posedirs"]).astype(np.float32)
    j_regressor = to_numpy(data["J_regressor"]).astype(np.float32)
    lbs_weights = to_numpy(data["weights"]).astype(np.float32)
    parents = to_numpy(data["kintree_table"])[0].astype(np.int32)
    parents[0] = -1

    if shapedirs.shape != (VERTEX_COUNT, 3, NUM_BETAS):
        shapedirs = shapedirs[:, :, :NUM_BETAS]
    if posedirs.shape == (VERTEX_COUNT, 3, POSE_BASIS_COUNT):
        posedirs = np.transpose(posedirs, (2, 0, 1)).reshape(POSE_BASIS_COUNT, VERTEX_COUNT * 3)
    elif posedirs.shape == (POSE_BASIS_COUNT, VERTEX_COUNT * 3):
        posedirs = posedirs.reshape(POSE_BASIS_COUNT, VERTEX_COUNT * 3)
    else:
        raise ValueError(f"Unsupported posedirs shape: {posedirs.shape}")

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": j_regressor,
        "lbs_weights": lbs_weights,
        "parents": parents,
    }


def batch_rodrigues(rot_vecs: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
    rot_vecs = rot_vecs.astype(np.float32, copy=False)
    angle = np.linalg.norm(rot_vecs + epsilon, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle
    cos = np.cos(angle)[:, None, :]
    sin = np.sin(angle)[:, None, :]

    rx = rot_dir[:, 0:1]
    ry = rot_dir[:, 1:2]
    rz = rot_dir[:, 2:3]
    zeros = np.zeros_like(rx)
    k = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape(-1, 3, 3)
    ident = np.eye(3, dtype=np.float32)[None, :, :]
    return ident + sin * k + (1.0 - cos) * np.matmul(k, k)


def transform_mat(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    batch = rotation.shape[0]
    transform = np.zeros((batch, 4, 4), dtype=np.float32)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3:4] = translation
    transform[:, 3, 3] = 1.0
    return transform


def batch_rigid_transform(rot_mats: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints_col = joints[:, :, :, None]
    rel_joints = joints_col.copy()
    rel_joints[:, 1:] -= joints_col[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1),
    ).reshape(-1, JOINT_COUNT, 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for joint in range(1, JOINT_COUNT):
        transform_chain.append(np.matmul(transform_chain[int(parents[joint])], transforms_mat[:, joint]))
    transforms = np.stack(transform_chain, axis=1)

    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = np.concatenate(
        [joints_col, np.zeros((joints.shape[0], JOINT_COUNT, 1, 1), dtype=np.float32)],
        axis=2,
    )
    transformed_joints = np.matmul(transforms, joints_homogen)
    rel_transforms = transforms.copy()
    rel_transforms[:, :, :, 3:4] -= transformed_joints
    return posed_joints, rel_transforms


def lbs_numpy(
    model: dict[str, np.ndarray],
    betas: np.ndarray,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = betas.shape[0]
    v_template = model["v_template"]
    shapedirs = model["shapedirs"]
    posedirs = model["posedirs"]
    j_regressor = model["J_regressor"]
    lbs_weights = model["lbs_weights"]
    parents = model["parents"]

    v_shaped = v_template[None, :, :] + np.einsum("bl,mkl->bmk", betas, shapedirs)
    joints = np.einsum("bik,ji->bjk", v_shaped, j_regressor)
    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch, JOINT_COUNT, 3, 3)

    ident = np.eye(3, dtype=np.float32)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch, -1)
    pose_offsets = np.matmul(pose_feature, posedirs).reshape(batch, VERTEX_COUNT, 3)
    v_posed = v_shaped + pose_offsets

    posed_joints, transforms = batch_rigid_transform(rot_mats, joints, parents)
    transforms_flat = transforms.reshape(batch, JOINT_COUNT, 16)
    blended_transforms = np.matmul(
        np.broadcast_to(lbs_weights[None, :, :], (batch, VERTEX_COUNT, JOINT_COUNT)),
        transforms_flat,
    ).reshape(batch, VERTEX_COUNT, 4, 4)
    v_homo = np.concatenate([v_posed, np.ones((batch, VERTEX_COUNT, 1), dtype=np.float32)], axis=2)
    vertices = np.matmul(blended_transforms, v_homo[:, :, :, None])[:, :, :3, 0]
    return (
        vertices.astype(np.float32),
        posed_joints.astype(np.float32),
        pose_feature.astype(np.float32),
        transforms.astype(np.float32),
    )


def generate_fixture(model_path: Path, output_dir: Path, samples: int, seed: int, pose_std: float, beta_std: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model = load_model(model_path)
    betas = rng.normal(0.0, beta_std, size=(samples, NUM_BETAS)).astype(np.float32)
    poses = rng.normal(0.0, pose_std, size=(samples, JOINT_COUNT * 3)).astype(np.float32)
    betas[0, :] = 0.0
    poses[0, :] = 0.0

    vertices, joints, pose_features, transforms = lbs_numpy(model, betas, poses)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "poses_axis_angle_f32.bin").write_bytes(np.ascontiguousarray(poses).tobytes())
    (output_dir / "betas_f32.bin").write_bytes(np.ascontiguousarray(betas).tobytes())
    (output_dir / "vertices_ref_f32.bin").write_bytes(np.ascontiguousarray(vertices).tobytes())
    (output_dir / "joints_ref_f32.bin").write_bytes(np.ascontiguousarray(joints).tobytes())
    (output_dir / "pose_features_f32.bin").write_bytes(np.ascontiguousarray(pose_features).tobytes())
    (output_dir / "joint_transforms_f32.bin").write_bytes(np.ascontiguousarray(transforms).tobytes())

    zero_vertices = vertices[0]
    metadata = {
        "format": "SMPL LBS shader validation fixture",
        "reference_algorithm": "NumPy port of reference/smplx/smplx/lbs.py",
        "source_model": str(model_path),
        "samples": samples,
        "seed": seed,
        "pose_std_axis_angle_rad": pose_std,
        "beta_std": beta_std,
        "contains_zero_pose_at_index": 0,
        "vertex_count": VERTEX_COUNT,
        "joint_count": JOINT_COUNT,
        "num_betas": NUM_BETAS,
        "pose_basis_count": POSE_BASIS_COUNT,
        "zero_pose_vertex_min": zero_vertices.min(axis=0).tolist(),
        "zero_pose_vertex_max": zero_vertices.max(axis=0).tolist(),
        "files": {
            "poses_axis_angle_f32.bin": [samples, JOINT_COUNT * 3],
            "betas_f32.bin": [samples, NUM_BETAS],
            "vertices_ref_f32.bin": [samples, VERTEX_COUNT, 3],
            "joints_ref_f32.bin": [samples, JOINT_COUNT, 3],
            "pose_features_f32.bin": [samples, POSE_BASIS_COUNT],
            "joint_transforms_f32.bin": [samples, JOINT_COUNT, 4, 4],
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/smpl_lbs_validation_fixture"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--pose-std", type=float, default=0.35)
    parser.add_argument("--beta-std", type=float, default=1.0)
    args = parser.parse_args()
    metadata = generate_fixture(args.model_path, args.output_dir, args.samples, args.seed, args.pose_std, args.beta_std)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

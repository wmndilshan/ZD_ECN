#!/usr/bin/env python3
"""
Export SMPL static model tensors for the OpenGL ES compute-shader LBS baseline.

This script intentionally avoids adding an inference/runtime dependency. It reads
licensed SMPL/SMPL-X model files directly when provided as .npz or .pkl and writes
flat binary buffers suitable for SSBO upload by the Android benchmark.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "todense"):
        value = value.todense()
    return np.asarray(value)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        try:
            data = pickle.load(handle, encoding="latin1")
        except TypeError:
            handle.seek(0)
            data = pickle.load(handle)
    if not isinstance(data, dict):
        data = vars(data)
    return data


def load_smpl_model(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"SMPL model file not found: {path}")

    aliases = {
        "v_template": ("v_template",),
        "shapedirs": ("shapedirs",),
        "posedirs": ("posedirs",),
        "weights": ("weights", "lbs_weights"),
        "J_regressor": ("J_regressor", "j_regressor"),
        "kintree_table": ("kintree_table", "parents"),
    }

    if path.suffix.lower() == ".npz":
        loaded = np.load(path, allow_pickle=True)
        available = set(loaded.files)
        out: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for canonical, names in aliases.items():
            for name in names:
                if name in available:
                    out[canonical] = _to_numpy(loaded[name])
                    break
            else:
                missing.append(canonical)
        if missing:
            raise KeyError(
                f"SMPL model is missing required arrays: {', '.join(missing)}. "
                f"Available keys include: {', '.join(loaded.files[:20])}"
            )
        return out
    elif path.suffix.lower() == ".pkl":
        raw = _load_pickle(path)
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}. Expected .npz or .pkl")

    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for canonical, names in aliases.items():
        for name in names:
            if name in raw:
                out[canonical] = _to_numpy(raw[name])
                break
        else:
            missing.append(canonical)
    if missing:
        raise KeyError(f"SMPL model is missing required arrays: {', '.join(missing)}")
    return out


def normalize_model_arrays(model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    v_template = np.asarray(model["v_template"], dtype=np.float32)
    if v_template.shape != (VERTEX_COUNT, 3):
        raise ValueError(f"v_template shape must be {(VERTEX_COUNT, 3)}, got {v_template.shape}")

    shapedirs_raw = np.asarray(model["shapedirs"], dtype=np.float32)
    if shapedirs_raw.shape[0:2] != (VERTEX_COUNT, 3) or shapedirs_raw.shape[2] < NUM_BETAS:
        raise ValueError(
            f"shapedirs shape must start with {(VERTEX_COUNT, 3)} and include "
            f"{NUM_BETAS} betas, got {shapedirs_raw.shape}"
        )
    shapedirs = np.ascontiguousarray(np.transpose(shapedirs_raw[:, :, :NUM_BETAS], (2, 0, 1)))

    posedirs_raw = np.asarray(model["posedirs"], dtype=np.float32)
    if posedirs_raw.shape == (VERTEX_COUNT, 3, POSE_BASIS_COUNT):
        posedirs = np.ascontiguousarray(np.transpose(posedirs_raw, (2, 0, 1)))
    elif posedirs_raw.shape == (POSE_BASIS_COUNT, VERTEX_COUNT * 3):
        posedirs = np.ascontiguousarray(posedirs_raw.reshape(POSE_BASIS_COUNT, VERTEX_COUNT, 3))
    else:
        raise ValueError(
            "posedirs shape must be "
            f"{(VERTEX_COUNT, 3, POSE_BASIS_COUNT)} or "
            f"{(POSE_BASIS_COUNT, VERTEX_COUNT * 3)}, got {posedirs_raw.shape}"
        )

    weights = np.asarray(model["weights"], dtype=np.float32)
    if weights.shape != (VERTEX_COUNT, JOINT_COUNT):
        raise ValueError(f"weights shape must be {(VERTEX_COUNT, JOINT_COUNT)}, got {weights.shape}")

    j_regressor = np.asarray(model["J_regressor"], dtype=np.float32)
    if j_regressor.shape != (JOINT_COUNT, VERTEX_COUNT):
        raise ValueError(
            f"J_regressor shape must be {(JOINT_COUNT, VERTEX_COUNT)}, got {j_regressor.shape}"
        )

    kintree = np.asarray(model["kintree_table"])
    if kintree.shape == (2, JOINT_COUNT):
        parents = np.asarray(kintree[0], dtype=np.int32)
        parents[0] = -1
    elif kintree.shape == (JOINT_COUNT,):
        parents = np.asarray(kintree, dtype=np.int32)
        parents[0] = -1
    else:
        raise ValueError(f"kintree_table/parents shape must be {(2, JOINT_COUNT)} or {(JOINT_COUNT,)}, got {kintree.shape}")

    return {
        "v_template": np.ascontiguousarray(v_template),
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "weights": np.ascontiguousarray(weights),
        "J_regressor": np.ascontiguousarray(j_regressor),
        "parents": np.ascontiguousarray(parents),
    }


def topk_skinning(weights: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    indices = np.argsort(-weights, axis=1)[:, :k].astype(np.int32)
    sparse_weights = np.take_along_axis(weights, indices, axis=1).astype(np.float32)
    sums = sparse_weights.sum(axis=1, keepdims=True)
    sparse_weights = np.divide(
        sparse_weights,
        sums,
        out=np.zeros_like(sparse_weights),
        where=sums > 0.0,
    )
    reconstructed = np.zeros_like(weights)
    np.put_along_axis(reconstructed, indices, sparse_weights, axis=1)
    max_abs_error = float(np.max(np.abs(weights - reconstructed)))
    return np.ascontiguousarray(indices), np.ascontiguousarray(sparse_weights), max_abs_error


def write_binary(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)
    return {
        "file": path.name,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "bytes": path.stat().st_size,
    }


def export_smpl_shader_data(model_path: Path, output_dir: Path, sparse_influences: int) -> dict[str, Any]:
    arrays = normalize_model_arrays(load_smpl_model(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}
    files["v_template"] = write_binary(output_dir / "v_template_f32.bin", arrays["v_template"])
    files["shapedirs"] = write_binary(output_dir / "shapedirs_10_f32.bin", arrays["shapedirs"])
    files["posedirs"] = write_binary(output_dir / "posedirs_207_f32.bin", arrays["posedirs"])
    files["lbs_weights"] = write_binary(output_dir / "lbs_weights_f32.bin", arrays["weights"])
    files["joint_regressor"] = write_binary(output_dir / "joint_regressor_f32.bin", arrays["J_regressor"])
    files["parents"] = write_binary(output_dir / "parents_i32.bin", arrays["parents"])
    files["betas_zero"] = write_binary(output_dir / "betas_zero_f32.bin", np.zeros(NUM_BETAS, dtype=np.float32))
    files["pose_axis_angle_zero"] = write_binary(
        output_dir / "pose_axis_angle_zero_f32.bin",
        np.zeros((JOINT_COUNT, 3), dtype=np.float32),
    )

    sparse_error = None
    if sparse_influences > 0:
        indices, sparse_weights, sparse_error = topk_skinning(arrays["weights"], sparse_influences)
        files["lbs_indices_topk"] = write_binary(
            output_dir / f"lbs_indices_top{sparse_influences}_i32.bin", indices
        )
        files["lbs_weights_topk"] = write_binary(
            output_dir / f"lbs_weights_top{sparse_influences}_f32.bin", sparse_weights
        )

    metadata = {
        "source_model": str(model_path),
        "format": "SMPL static data for OpenGL ES 3.2 SSBO LBS benchmark",
        "vertex_count": VERTEX_COUNT,
        "joint_count": JOINT_COUNT,
        "num_betas": NUM_BETAS,
        "pose_basis_count": POSE_BASIS_COUNT,
        "layout": {
            "v_template": "[6890,3] float32",
            "shapedirs": "[10,6890,3] float32",
            "posedirs": "[207,6890,3] float32",
            "lbs_weights": "[6890,24] float32",
            "joint_regressor": "[24,6890] float32",
            "parents": "[24] int32, root=-1",
        },
        "sparse_influences": sparse_influences,
        "sparse_reconstruction_max_abs_weight_error": sparse_error,
        "files": files,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Licensed SMPL neutral/male/female .npz or .pkl model file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/smpl_shader_data"),
        help="Directory for exported flat binaries.",
    )
    parser.add_argument(
        "--sparse-influences",
        type=int,
        default=4,
        help="Also export top-k sparse skinning buffers. Use 0 to disable.",
    )
    args = parser.parse_args()

    try:
        metadata = export_smpl_shader_data(args.model_path, args.output_dir, args.sparse_influences)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

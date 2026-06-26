#!/usr/bin/env python3
"""
Export static SMPL reference data for the exact vchoutas/smplx LBS port.

The output is intentionally flat binary SSBO data for Android/OpenGL ES.  It
does not add smplx, PyTorch, chumpy, or scipy as APK/runtime dependencies.
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


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "r"):
        value = value.r
    elif hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "todense"):
        value = value.todense()
    return np.asarray(value)


def load_legacy_smpl_pickle(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_smplx_model_object(model_path: Path) -> dict[str, Any] | None:
    try:
        sys.path.insert(0, str(Path("reference/smplx").resolve()))
        from smplx.body_models import SMPL  # type: ignore

        model = SMPL(
            str(model_path),
            gender="neutral",
            num_betas=NUM_BETAS,
            batch_size=1,
        )
        return {
            "v_template": model.v_template,
            "shapedirs": model.shapedirs,
            "posedirs": model.posedirs,
            "J_regressor": model.J_regressor,
            "weights": model.lbs_weights,
            "parents": model.parents,
            "f": model.faces,
            "loader": "reference/smplx model object",
        }
    except Exception as exc:
        print(f"WARNING: smplx model-object load unavailable, falling back to licensed pickle fields: {exc}", file=sys.stderr)
        return None


def sparse_rows(matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    indices: list[int] = []
    weights: list[float] = []
    for row in matrix:
        nz = np.flatnonzero(np.abs(row) > threshold)
        indices.extend(int(i) for i in nz)
        weights.extend(float(row[i]) for i in nz)
        offsets.append(len(indices))
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
    )


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


def export_reference_data(model_path: Path, output_dir: Path, sparse_threshold: float) -> dict[str, Any]:
    data = load_smplx_model_object(model_path)
    loader = "reference/smplx model object"
    if data is None:
        data = load_legacy_smpl_pickle(model_path)
        loader = "licensed SMPL pickle fields matching smplx SMPL buffers"

    v_template = to_numpy(data["v_template"]).astype(np.float32)
    shapedirs_raw = to_numpy(data["shapedirs"]).astype(np.float32)
    posedirs_raw = to_numpy(data["posedirs"]).astype(np.float32)
    j_regressor = to_numpy(data["J_regressor"]).astype(np.float32)
    lbs_weights = to_numpy(data["weights"] if "weights" in data else data["lbs_weights"]).astype(np.float32)
    faces = to_numpy(data["f"] if "f" in data else data["faces"]).astype(np.uint16)

    if "parents" in data:
        parents = to_numpy(data["parents"]).astype(np.int32)
    else:
        parents = to_numpy(data["kintree_table"])[0].astype(np.int32)
        parents[0] = -1

    if shapedirs_raw.shape != (VERTEX_COUNT, 3, NUM_BETAS):
        shapedirs_raw = shapedirs_raw[:, :, :NUM_BETAS]
    shapedirs = np.transpose(shapedirs_raw, (2, 0, 1)).astype(np.float16)

    if posedirs_raw.shape == (VERTEX_COUNT, 3, POSE_BASIS_COUNT):
        posedirs = np.transpose(posedirs_raw, (2, 0, 1)).astype(np.float16)
    elif posedirs_raw.shape == (POSE_BASIS_COUNT, VERTEX_COUNT * 3):
        posedirs = posedirs_raw.reshape(POSE_BASIS_COUNT, VERTEX_COUNT, 3).astype(np.float16)
    else:
        raise ValueError(f"Unsupported posedirs shape: {posedirs_raw.shape}")

    if v_template.shape != (VERTEX_COUNT, 3):
        raise ValueError(f"Unexpected v_template shape: {v_template.shape}")
    if shapedirs.shape != (NUM_BETAS, VERTEX_COUNT, 3):
        raise ValueError(f"Unexpected shapedirs shape: {shapedirs.shape}")
    if posedirs.shape != (POSE_BASIS_COUNT, VERTEX_COUNT, 3):
        raise ValueError(f"Unexpected posedirs shape: {posedirs.shape}")
    if j_regressor.shape != (JOINT_COUNT, VERTEX_COUNT):
        raise ValueError(f"Unexpected J_regressor shape: {j_regressor.shape}")
    if lbs_weights.shape != (VERTEX_COUNT, JOINT_COUNT):
        raise ValueError(f"Unexpected lbs_weights shape: {lbs_weights.shape}")

    j_offsets, j_indices, j_weights = sparse_rows(j_regressor, sparse_threshold)
    w_offsets, w_indices, w_weights = sparse_rows(lbs_weights, sparse_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    files["v_template"] = write_binary(output_dir / "v_template_f32.bin", v_template)
    files["shapedirs"] = write_binary(output_dir / "shapedirs_10_f16.bin", shapedirs)
    files["posedirs"] = write_binary(output_dir / "posedirs_207_f16.bin", posedirs)
    files["joint_regressor_offsets"] = write_binary(output_dir / "joint_regressor_offsets_i32.bin", j_offsets)
    files["joint_regressor_indices"] = write_binary(output_dir / "joint_regressor_indices_i32.bin", j_indices)
    files["joint_regressor_weights"] = write_binary(output_dir / "joint_regressor_weights_f32.bin", j_weights)
    files["lbs_weights_offsets"] = write_binary(output_dir / "lbs_weights_offsets_i32.bin", w_offsets)
    files["lbs_weights_indices"] = write_binary(output_dir / "lbs_weights_indices_i32.bin", w_indices)
    files["lbs_weights_values"] = write_binary(output_dir / "lbs_weights_values_f32.bin", w_weights)
    files["parents"] = write_binary(output_dir / "parents_i32.bin", parents)
    files["faces"] = write_binary(output_dir / "faces_u16.bin", faces)

    metadata = {
        "source_model": str(model_path),
        "reference_repo": "reference/smplx",
        "reference_commit": read_git_commit(Path("reference/smplx")),
        "loader": loader,
        "vertex_count": VERTEX_COUNT,
        "joint_count": JOINT_COUNT,
        "num_betas": NUM_BETAS,
        "pose_basis_count": POSE_BASIS_COUNT,
        "face_count": int(faces.shape[0]),
        "sparse_threshold": sparse_threshold,
        "joint_regressor_nnz": int(j_indices.size),
        "lbs_weights_nnz": int(w_indices.size),
        "files": files,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def read_git_commit(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text().strip()
    if value.startswith("ref: "):
        ref_path = repo / ".git" / value.split(" ", 1)[1]
        return ref_path.read_text().strip() if ref_path.exists() else None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/smpl_reference_data"))
    parser.add_argument("--sparse-threshold", type=float, default=0.0)
    args = parser.parse_args()
    try:
        print(json.dumps(export_reference_data(args.model_path, args.output_dir, args.sparse_threshold), indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_zdecn_dqs import VERTEX_COUNT
from export_zdecn_checkpoint import load_checkpoint_model
from train_sameframe_small import load_dataset_arrays


def summarize(per_pose: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(per_pose.size),
        "mean_mm": float(np.mean(per_pose)),
        "std_mm": float(np.std(per_pose, ddof=1)),
        "p50_mm": float(np.percentile(per_pose, 50)),
        "p95_mm": float(np.percentile(per_pose, 95)),
        "p99_mm": float(np.percentile(per_pose, 99)),
        "max_mm": float(np.max(per_pose)),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: Path,
    split: str,
    batch_size: int,
    cache_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    x, y, train_indices, val_indices, dataset_metadata = load_dataset_arrays(dataset, cache_dir)
    if split == "val":
        indices = val_indices
    elif split == "train":
        indices = train_indices
    elif split == "all":
        indices = np.arange(x.shape[0], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported split: {split}")

    model = load_checkpoint_model(checkpoint)
    per_pose_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start:start + batch_size]
            pred = model(torch.from_numpy(np.ascontiguousarray(x[batch_indices], dtype=np.float32))).cpu().numpy().astype(np.float32)
            target = np.ascontiguousarray(y[batch_indices], dtype=np.float32)
            per_vertex = np.linalg.norm((pred - target).reshape(-1, VERTEX_COUNT, 3), axis=2)
            per_pose_chunks.append(per_vertex.mean(axis=1).astype(np.float32))

    per_pose = np.concatenate(per_pose_chunks)
    baseline_chunks: list[np.ndarray] = []
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start:start + batch_size]
        target = np.ascontiguousarray(y[batch_indices], dtype=np.float32)
        baseline_per_vertex = np.linalg.norm(target.reshape(-1, VERTEX_COUNT, 3), axis=2)
        baseline_chunks.append(baseline_per_vertex.mean(axis=1).astype(np.float32))
    baseline_per_pose = np.concatenate(baseline_chunks)
    summary: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "split": split,
        "dataset_metadata": dataset_metadata,
        "pve": summarize(per_pose),
        "lbs_zero_correction_baseline": summarize(baseline_per_pose),
    }
    return summary, per_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ZD-ECN same-frame checkpoint on a dataset split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_sameframe_pose_30k.npz"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-pose-csv", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    summary_path = args.summary or args.output
    if summary_path is None:
        parser.error("one of --summary or --output is required")

    cache_dir = args.cache_dir or Path("results/paper/full_retrain_sameframe/dataset_cache") / args.dataset.stem
    summary, per_pose = evaluate_checkpoint(args.checkpoint, args.dataset, args.split, args.batch_size, cache_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.per_pose_csv is not None:
        args.per_pose_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.per_pose_csv, per_pose, delimiter=",", header="per_pose_pve_mm", comments="")
        summary["per_pose_csv"] = str(args.per_pose_csv)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

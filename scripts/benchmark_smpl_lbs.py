#!/usr/bin/env python3
"""
SMPL LBS Baseline Benchmark
============================
Measures the cost of the computations in the ZD-ECN pipeline
EXCLUDING the neural correction — i.e., raw SMPL Linear Blend Skinning.

For the ZD-ECN paper this is the "no-correction" reference point.
We benchmark:
  1. rot6d encoding of axis-angle pose (24 joints x 72 -> 144)
  2. LBS pose-blend-shape matmul (the dominant SMPL forward cost)
  3. Full vertex-set output (6890 x 3)

All measured on host CPU via numpy (matches what a mobile CPU would do
if SMPL were run without GPU acceleration — gives an upper bound on
non-GPU SMPL cost per frame).

Author: ZD-ECN project
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# rot6d encoding (matches the ZD-ECN input pipeline exactly)
# ---------------------------------------------------------------------------
def axis_angle_to_rot_matrix(theta):
    """Convert axis-angle (N, 3) to rotation matrix (N, 3, 3) using Rodrigues."""
    N = theta.shape[0]
    angle = np.linalg.norm(theta, axis=1, keepdims=True)  # (N, 1)
    eps = 1e-8
    safe_angle = np.where(angle < eps, np.ones_like(angle), angle)
    axis = theta / safe_angle  # (N, 3)
    # Clamp for numerical stability
    angle = np.clip(angle.squeeze(-1), -np.pi, np.pi)
    
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    
    R = np.stack([
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    ], axis=1).reshape(N, 3, 3)
    return R


def rot_matrix_to_rot6d(R):
    """Extract 6D rotation representation from rotation matrices (N, 3, 3) -> (N, 6)."""
    return R[:, :, :2].transpose(0, 2, 1).reshape(R.shape[0], 6)


def encode_pose_to_rot6d(pose_aa):
    """
    Convert axis-angle pose vector (72,) to rot6d (144,).
    pose_aa: (72,) = 24 joints x 3 axis-angle
    Returns: (144,) = 24 joints x 6 rot6d
    """
    joints_aa = pose_aa.reshape(24, 3)
    R = axis_angle_to_rot_matrix(joints_aa)   # (24, 3, 3)
    rot6d = rot_matrix_to_rot6d(R)            # (24, 6)
    return rot6d.reshape(144)                  # (144,)


# ---------------------------------------------------------------------------
# Synthetic LBS (representative matmul cost)
# ---------------------------------------------------------------------------
def simulate_lbs(rot6d_input, W_lbs, J_regressor):
    """
    Simulate the dominant matmul cost of SMPL LBS:
      - Joint regression: (6890x3) x (6890xJ)^T
      - Pose blend shapes: (N_blend x 3) x pose_features
      - Vertex skinning: (6890 x J x 4) dot-product
    
    We approximate with representative matrix multiplications of the same
    dimensionality as the actual SMPL model.
    
    W_lbs: (6890, 24)    - blend weights
    J_regressor: (24, 6890) - joint regressor
    rot6d_input: (144,)  - pose input
    """
    # Step 1: Expand rot6d to per-joint rotation matrices for blend shapes
    rot_mats = rot6d_input.reshape(24, 6)  # (24, 6) proxy
    
    # Step 2: Joint positions from regressor (24 x 6890) @ template (6890 x 3)
    template_verts = np.random.randn(6890, 3).astype(np.float32)
    J = J_regressor @ template_verts  # (24, 3)
    
    # Step 3: Pose blend shapes — (207,) @ (207, 6890*3)
    # 207 = 23 joints * 9 (full rotation matrix entries, minus root)
    pose_features = np.random.randn(207).astype(np.float32)
    pose_bs_weights = np.random.randn(207, 6890 * 3).astype(np.float32)
    pose_disp = pose_bs_weights.T @ pose_features  # (6890*3,)
    pose_disp = pose_disp.reshape(6890, 3)
    
    # Step 4: LBS skinning — weighted sum of per-joint transforms
    # (6890, 24) @ (24, 3) -> (6890, 3)
    local_verts = template_verts + pose_disp
    skinned = W_lbs @ J  # (6890, 3) proxy
    
    return skinned


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def benchmark_smpl_lbs(smpl_data_path, sessions=5, iterations=200):
    print(f"Loading pose data from {smpl_data_path}...")
    data = np.load(smpl_data_path, allow_pickle=True)
    
    # Collect all pose frames
    pose_keys = [k for k in data.keys() if k.startswith('pose') and not k.startswith('pose_ung')]
    print(f"Found {len(pose_keys)} pose sequences")
    
    all_poses = []
    for k in pose_keys[:50]:  # Use first 50 sequences
        seq = data[k]  # (T, 72)
        all_poses.append(seq)
    all_poses = np.concatenate(all_poses, axis=0)
    print(f"Total frames: {len(all_poses)}")
    
    # Pre-build synthetic LBS matrices (representative SMPL sizes)
    np.random.seed(42)
    W_lbs = np.random.dirichlet(np.ones(24), size=6890).astype(np.float32)  # (6890, 24)
    J_regressor = np.random.randn(24, 6890).astype(np.float32)              # (24, 6890)
    J_regressor /= 6890  # normalize
    
    # ----------- WARM UP -----------
    print("Warming up (5 iterations)...")
    for i in range(5):
        pose = all_poses[i % len(all_poses)]
        rot6d = encode_pose_to_rot6d(pose)
        _ = simulate_lbs(rot6d, W_lbs, J_regressor)
    
    # ----------- BENCHMARK -----------
    results = {"sessions": [], "raw_ms": []}
    
    for session in range(1, sessions + 1):
        print(f"Session {session}/{sessions}...")
        session_times = []
        
        for i in range(iterations):
            pose = all_poses[(session * iterations + i) % len(all_poses)]
            
            t0 = time.perf_counter()
            # Step 1: rot6d encoding
            rot6d = encode_pose_to_rot6d(pose)
            # Step 2: LBS forward pass simulation
            _ = simulate_lbs(rot6d, W_lbs, J_regressor)
            t1 = time.perf_counter()
            
            elapsed_ms = (t1 - t0) * 1000.0
            session_times.append(elapsed_ms)
            results["raw_ms"].append(elapsed_ms)
        
        s_arr = np.array(session_times)
        session_summary = {
            "session": session,
            "n": len(session_times),
            "mean_ms": float(np.mean(s_arr)),
            "stddev_ms": float(np.std(s_arr)),
            "p50_ms": float(np.percentile(s_arr, 50)),
            "p95_ms": float(np.percentile(s_arr, 95)),
            "p99_ms": float(np.percentile(s_arr, 99)),
        }
        results["sessions"].append(session_summary)
        print(f"  mean={session_summary['mean_ms']:.3f}ms  "
              f"p99={session_summary['p99_ms']:.3f}ms")
    
    # Overall statistics (exclude first session as warmup)
    all_times = np.array(results["raw_ms"][iterations:])  # skip session 1
    
    results["overall"] = {
        "n": int(len(all_times)),
        "mean_ms": float(np.mean(all_times)),
        "stddev_ms": float(np.std(all_times)),
        "p50_ms": float(np.percentile(all_times, 50)),
        "p95_ms": float(np.percentile(all_times, 95)),
        "p99_ms": float(np.percentile(all_times, 99)),
        "fps_from_mean": float(1000.0 / np.mean(all_times)),
        "coefficient_of_variation": float(np.std(all_times) / np.mean(all_times)),
    }
    
    results["metadata"] = {
        "description": "SMPL LBS forward pass simulation on host CPU (NumPy)",
        "note": "Representative of per-frame SMPL cost without GPU acceleration",
        "components": [
            "rot6d encoding (axis-angle -> 6D rotation representation)",
            "Joint regression (24 x 6890 matmul)",
            "Pose blend shapes (207 x 6890*3 matmul)",
            "LBS skinning (6890 x 24 weighted sum)",
        ],
        "architecture": "SMPL (6890 vertices, 24 joints)",
        "precision": "float32",
        "platform": "Host CPU (Python/NumPy)",
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SMPL LBS Baseline Benchmark")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--smpl-data", required=True, help="Path to smpl_data.npz")
    args = parser.parse_args()
    
    results = benchmark_smpl_lbs(
        smpl_data_path=args.smpl_data,
        sessions=args.sessions,
        iterations=args.iterations,
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SMPL LBS Benchmark Results (Overall, sessions 2-{args.sessions}):")
    print(f"{'='*60}")
    o = results["overall"]
    print(f"  N samples:    {o['n']}")
    print(f"  Mean:         {o['mean_ms']:.3f} ms")
    print(f"  Std Dev:      {o['stddev_ms']:.3f} ms")
    print(f"  p50:          {o['p50_ms']:.3f} ms")
    print(f"  p95:          {o['p95_ms']:.3f} ms")
    print(f"  p99:          {o['p99_ms']:.3f} ms")
    print(f"  FPS:          {o['fps_from_mean']:.1f}")
    print(f"  CV:           {o['coefficient_of_variation']*100:.2f}%")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
SMPL-24 Kinematics & Closed-Loop Analytical Inverse Kinematics Engine.
Computes recursive forward kinematics (SE(3) transforms) and exact 3D two-bone leg IK
with ground-clamped foot locking.
"""
import math
import numpy as np
from typing import Tuple, Dict

JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

PARENT_MAP: Dict[int, int] = {
    0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14,
    18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21
}

PARENT_LIST = [PARENT_MAP[j] for j in range(24)]

# Canonical SMPL skeleton rest offsets (in meters)
REST_OFFSETS = np.array([
    [0.0, 0.0, 0.84],      # 0: Pelvis
    [0.12, -0.02, -0.06],  # 1: L_Hip
    [-0.12, -0.02, -0.06], # 2: R_Hip
    [0.0, 0.02, 0.14],     # 3: Spine1
    [0.0, 0.0, -0.38],     # 4: L_Knee (femur length = 0.38m)
    [0.0, 0.0, -0.38],     # 5: R_Knee (femur length = 0.38m)
    [0.0, 0.0, 0.14],      # 6: Spine2
    [0.0, 0.0, -0.38],     # 7: L_Ankle (tibia length = 0.38m)
    [0.0, 0.0, -0.38],     # 8: R_Ankle (tibia length = 0.38m)
    [0.0, 0.0, 0.14],      # 9: Spine3
    [0.0, 0.12, -0.05],    # 10: L_Foot (sole at Z - 0.05m relative to ankle)
    [0.0, 0.12, -0.05],    # 11: R_Foot
    [0.0, 0.0, 0.12],      # 12: Neck
    [0.08, -0.02, 0.08],   # 13: L_Collar
    [-0.08, -0.02, 0.08],  # 14: R_Collar
    [0.0, 0.0, 0.14],      # 15: Head
    [0.15, 0.0, 0.0],      # 16: L_Shoulder
    [-0.15, 0.0, 0.0],     # 17: R_Shoulder
    [0.24, 0.0, 0.0],      # 18: L_Elbow
    [-0.24, 0.0, 0.0],     # 19: R_Elbow
    [0.22, 0.0, 0.0],      # 20: L_Wrist
    [-0.22, 0.0, 0.0],     # 21: R_Wrist
    [0.08, 0.0, 0.0],      # 22: L_Hand
    [-0.08, 0.0, 0.0],     # 23: R_Hand
], dtype=np.float32)


def euler_to_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """
    Fast direct computation of SO(3) rotation matrix R = Rz(yaw) @ Ry(roll) @ Rx(-pitch).
    Convention: +pitch flexes knee backward (-Y), -pitch flexes hip forward (+Y).
    """
    cx, sx = math.cos(-pitch), math.sin(-pitch)
    cy, sy = math.cos(roll), math.sin(roll)
    cz, sz = math.cos(yaw), math.sin(yaw)

    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,     cy * sx,                cy * cx]
    ], dtype=np.float32)


def matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Extracts Euler angles [pitch, roll, yaw] matching euler_to_matrix(pitch, roll, yaw)."""
    sy = -max(-1.0, min(1.0, float(R[2, 0])))
    roll = math.asin(sy)
    if abs(R[2, 0]) < 0.99999:
        pitch = -math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        pitch = -math.atan2(-float(R[1, 2]), float(R[1, 1]))
        yaw = 0.0
    return np.array([pitch, roll, yaw], dtype=np.float32)


def solve_leg_ik(
    pelvis_pos: np.ndarray,
    R_pelvis: np.ndarray,
    foot_target_world: np.ndarray,
    foot_yaw: float,
    foot_pitch: float,
    pole_vector_local: np.ndarray,
    is_left: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form Analytical 3-DOF Two-Bone Inverse Kinematics for SMPL leg chain.
    Computes exact local Euler angles (hip_euler, knee_euler, ankle_euler) so that
    forward kinematics places the Foot joint (10 or 11) at foot_target_world (Z >= 0).
    """
    hip_idx = 1 if is_left else 2
    foot_idx = 10 if is_left else 11
    l1 = 0.38
    l2 = 0.38

    p_hip = pelvis_pos + R_pelvis @ REST_OFFSETS[hip_idx]

    R_ankle_glob = euler_to_matrix(foot_pitch, 0.0, foot_yaw)
    foot_offset = REST_OFFSETS[foot_idx]  # [0.0, 0.12, -0.05]

    safe_foot_target = np.array([
        foot_target_world[0],
        foot_target_world[1],
        max(0.0, float(foot_target_world[2]))
    ], dtype=np.float32)

    p_ankle_target = safe_foot_target - R_ankle_glob @ foot_offset
    if p_ankle_target[2] < 0.05:
        p_ankle_target[2] = 0.05

    v = p_ankle_target - p_hip
    d = float(np.linalg.norm(v))
    max_reach = (l1 + l2) * 0.998
    if d > max_reach:
        u_t = v / d
        d_c = max_reach
        p_ankle_reach = p_hip + u_t * d_c
    elif d < 1e-4:
        u_t = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d_c = 1e-4
        p_ankle_reach = p_hip + u_t * d_c
    else:
        u_t = v / d
        d_c = d
        p_ankle_reach = p_ankle_target

    cos_a = max(-1.0, min(1.0, (l1 * l1 + d_c * d_c - l2 * l2) / (2.0 * l1 * d_c)))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))

    pole_world = R_pelvis @ pole_vector_local
    normal = np.cross(u_t, pole_world)
    n_len = float(np.linalg.norm(normal))
    if n_len < 1e-5:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        normal = normal / n_len

    bend_dir = np.cross(normal, u_t)
    bend_dir = bend_dir / float(np.linalg.norm(bend_dir))

    p_knee = p_hip + l1 * (cos_a * u_t + sin_a * bend_dir)

    u_femur = (p_knee - p_hip) / l1
    u_tibia = (p_ankle_reach - p_knee) / l2

    z_hip = -u_femur
    cross_ft = np.cross(u_femur, u_tibia)
    c_len = float(np.linalg.norm(cross_ft))
    if c_len < 1e-5:
        x_hip = normal
    else:
        x_hip = -cross_ft / c_len
    y_hip = np.cross(z_hip, x_hip)
    R_hip_glob = np.column_stack([x_hip, y_hip, z_hip]).astype(np.float32)

    knee_pitch = math.acos(max(-1.0, min(1.0, float(np.dot(u_femur, u_tibia)))))
    R_knee_local = euler_to_matrix(knee_pitch, 0.0, 0.0)
    R_knee_glob = R_hip_glob @ R_knee_local

    R_hip_local = R_pelvis.T @ R_hip_glob
    R_ankle_local = R_knee_glob.T @ R_ankle_glob

    hip_euler = matrix_to_euler(R_hip_local)
    knee_euler = np.array([knee_pitch, 0.0, 0.0], dtype=np.float32)
    ankle_euler = matrix_to_euler(R_ankle_local)

    return hip_euler, knee_euler, ankle_euler


def compute_forward_kinematics(joint_rotations: np.ndarray, root_translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes global SE(3) transformation matrices (24, 4, 4) and 3D positions (24, 3)."""
    transforms = np.zeros((24, 4, 4), dtype=np.float32)
    positions = np.zeros((24, 3), dtype=np.float32)

    R0 = euler_to_matrix(float(joint_rotations[0, 0]), float(joint_rotations[0, 1]), float(joint_rotations[0, 2]))
    transforms[0, :3, :3] = R0
    transforms[0, :3, 3] = root_translation
    transforms[0, 3, 3] = 1.0
    positions[0] = root_translation

    for j in range(1, 24):
        parent = PARENT_LIST[j]
        R_parent = transforms[parent, :3, :3]
        p_parent = positions[parent]

        R_local = euler_to_matrix(float(joint_rotations[j, 0]), float(joint_rotations[j, 1]), float(joint_rotations[j, 2]))
        R_glob = R_parent @ R_local
        p_glob = p_parent + R_parent @ REST_OFFSETS[j]

        transforms[j, :3, :3] = R_glob
        transforms[j, :3, 3] = p_glob
        transforms[j, 3, 3] = 1.0
        positions[j] = p_glob

    return transforms, positions


def verify_transformation_dynamic_response() -> bool:
    rot_zero = np.zeros((24, 3), dtype=np.float32)
    trans_zero = np.array([0.0, 0.0, 0.84], dtype=np.float32)
    t_base, _ = compute_forward_kinematics(rot_zero, trans_zero)

    rot_test = np.zeros((24, 3), dtype=np.float32)
    rot_test[4, 0] = np.radians(45.0)
    t_test, _ = compute_forward_kinematics(rot_test, trans_zero)

    diff_knee = np.linalg.norm(t_test[4] - t_base[4])
    diff_ankle = np.linalg.norm(t_test[7] - t_base[7])
    diff_foot = np.linalg.norm(t_test[10] - t_base[10])
    diff_rknee = np.linalg.norm(t_test[5] - t_base[5])

    return bool((diff_knee > 1e-3) and (diff_ankle > 1e-3) and (diff_foot > 1e-3) and (diff_rknee < 1e-5))

"""
Reference converter: zd_ecn_motion generator output -> SMPL pose stream for the
prompt_motion_app renderer. The Kotlin port (SmplMotionConverter.kt) mirrors this
file line-for-line; tools/export_golden.py dumps its output as a parity fixture.

Frames
------
Generator world  : Z-up, character faces +Y at heading 0, semantic left = +X.
                   (left x forward = up, i.e. the generator's body is mirrored
                   relative to a right-handed human.)
Generator body   : rest pose legs along -Z, arms along +-X, feet point +Y.
SMPL body        : rest pose legs along -Y, arms along +-X, feet point +Z.
ZD-ECN training  : SURREAL/CMU roots are ~Rx(+90deg) with free yaw about world Z,
                   i.e. a right-handed Z-up world where heading 0 faces -Y.

Mappings (both are reflections; their composition is a proper rotation):
    M = swap(y, z)       SMPL body coords  <-> generator body coords
    W = diag(1, -1, 1)   generator world   ->  right-handed Z-up world

    global  G_smpl_j = W  G_gen_j  M
    local   L_smpl_j = M  L_gen_j  M     (j >= 1)
    root    L_smpl_0 = W  L_gen_0  M
    points  p_world  = W  p_gen

Feeding the generator's real root orientation keeps ZD-ECN's root input inside
its training distribution; the renderer only applies the fixed Z-up -> Y-up
display rotation and the root translation on top.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zd_ecn_motion.generator.kinematics import euler_to_matrix  # noqa: E402

SMPL_DATA = REPO / "prompt_motion_app/app/src/main/assets/smpl_shader_data"
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

M = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
W = np.diag([1.0, -1.0, 1.0])

# Generator straight-leg pelvis height above its ground plane (foot joint z=0):
# hip(-0.06) + femur(-0.38) + tibia(-0.38) + foot(-0.05)
GEN_STANDING_PELVIS_HEIGHT = 0.87


# Arm retargeting. zd_ecn_motion authors arm rotations for arms HANGING along -Z
# (shoulder +pitch swings the arm back, -pitch reaches forward, roll abducts), but
# its REST_OFFSETS put the arms along +-X (T-pose), where a pitch about X only
# twists the arm. We realise the authored intent on the T-pose rig with a fixed
# per-arm base rotation B (T-pose bone -> hanging), applied in generator space:
#     shoulder L = R B,   elbow/wrist/hand L = B^T R B
# The generator's elbow +pitch maps to hyperextension in that frame, so elbow
# pitch is negated to make its [0, 150 deg] limit range a forward flexion.
def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


B_LEFT = _ry(math.pi / 2)    # +X -> -Z
B_RIGHT = _ry(-math.pi / 2)  # -X -> -Z
ARM_SIDE = {16: B_LEFT, 18: B_LEFT, 20: B_LEFT, 22: B_LEFT,
            17: B_RIGHT, 19: B_RIGHT, 21: B_RIGHT, 23: B_RIGHT}


def gen_local_matrix(j: int, euler) -> np.ndarray:
    pitch, roll, yaw = euler
    if j in (18, 19):
        pitch = -pitch
    R = euler_to_matrix(pitch, roll, yaw).astype(np.float64)
    if j in (16, 17):
        return R @ ARM_SIDE[j]
    if j in ARM_SIDE:
        B = ARM_SIDE[j]
        return B.T @ R @ B
    return R


def load_smpl_rest():
    v = np.fromfile(SMPL_DATA / "v_template_f32.bin", dtype=np.float32).reshape(6890, 3).astype(np.float64)
    jr = np.fromfile(SMPL_DATA / "joint_regressor_f32.bin", dtype=np.float32).reshape(24, 6890).astype(np.float64)
    return v, jr @ v


def rest_constants(v, joints):
    """Scale gen->SMPL and per-joint sole heights (SMPL rest is Y-up)."""
    floor_y = float(v[:, 1].min())
    scale = (joints[0, 1] - floor_y) / GEN_STANDING_PELVIS_HEIGHT
    sole = {j: float(joints[j, 1] - floor_y) for j in (7, 8, 10, 11)}
    return scale, sole


LEGS = ((1, 4, 7, 10), (2, 5, 8, 11))  # (hip, knee, ankle, foot) left, right


def align_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a onto direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        return np.eye(3)  # antiparallel never happens for leg bones within one frame
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def smpl_fk(local, root, joints):
    G = np.zeros((24, 3, 3))
    P = np.zeros((24, 3))
    G[0], P[0] = local[0], root
    for j in range(1, 24):
        p = PARENTS[j]
        G[j] = G[p] @ local[j]
        P[j] = P[p] + G[p] @ (joints[j] - joints[p])
    return G, P


def retarget_legs(local, root, joints, foot_targets, sole):
    """Two-bone IK on the SMPL skeleton so each SMPL foot joint lands on the generator's
    foot target (scaled, Z-up world). Knees keep the generator's bend plane and the
    feet keep their global orientation, so only the SMPL/generator bone-length mismatch
    is absorbed. Modifies `local` in place."""
    G, P = smpl_fk(local, root, joints)
    for leg, (h, k, a, ft) in enumerate(LEGS):
        target_foot = foot_targets[leg] + np.array([0.0, 0.0, sole[ft]])
        target_ankle = target_foot - G[a] @ (joints[ft] - joints[a])
        l1 = np.linalg.norm(joints[k] - joints[h])
        l2 = np.linalg.norm(joints[a] - joints[k])
        d_vec = target_ankle - P[h]
        d = float(np.linalg.norm(d_vec))
        d = min(max(d, abs(l1 - l2) + 1e-4), (l1 + l2) * 0.9995)
        u = d_vec / np.linalg.norm(d_vec)
        ankle_new = P[h] + u * d
        # Bend plane from the generator's knee placement
        bend = (P[k] - P[h]) - u * float(np.dot(P[k] - P[h], u))
        if np.linalg.norm(bend) < 1e-6:
            bend = G[h] @ np.array([0.0, 0.0, 1.0])  # SMPL forward
            bend -= u * float(np.dot(bend, u))
        bend /= np.linalg.norm(bend)
        cos_a = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
        knee_new = P[h] + l1 * (cos_a * u + math.sqrt(max(0.0, 1.0 - cos_a * cos_a)) * bend)

        G_ankle = G[a].copy()
        G_hip = align_rotation(P[k] - P[h], knee_new - P[h]) @ G[h]
        shin_now = G_hip @ local[k] @ (joints[a] - joints[k])
        G_knee = align_rotation(shin_now, ankle_new - knee_new) @ G_hip @ local[k]
        local[h] = G[0].T @ G_hip  # hips are children of the pelvis
        local[k] = G_hip.T @ G_knee
        local[a] = G_knee.T @ G_ankle


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Shepperd's quaternion extraction -> axis-angle. Stable near theta = pi,
    where the 1/(2 sin theta) formula blows up (heading ~ 180 deg after a turn)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    vn = math.sqrt(x * x + y * y + z * z)
    if vn < 1e-12:
        return np.zeros(3)
    theta = 2.0 * math.atan2(vn, w)
    return np.array([x, y, z]) * (theta / vn)


def convert(motion: dict, smpl_rest=None):
    """Returns dict(aa[N,72], rot6d[N,144], trans[N,3]) — trans is the Z-up world
    offset to add to the LBS output (whose pelvis sits at the rest J0)."""
    v, joints = smpl_rest if smpl_rest is not None else load_smpl_rest()
    scale, sole = rest_constants(v, joints)
    frames = motion["frames"]
    n = len(frames)
    aa = np.zeros((n, 72), dtype=np.float32)
    rot6d = np.zeros((n, 144), dtype=np.float32)
    pelvis = np.zeros((n, 3))
    ground = np.full(n, np.nan)

    for f, fr in enumerate(frames):
        eul = fr["joint_rotations_euler"]
        local = np.zeros((24, 3, 3))
        for j in range(24):
            Rg = gen_local_matrix(j, eul[j])
            local[j] = (W @ Rg @ M) if j == 0 else (M @ Rg @ M)

        root = W @ (np.asarray(fr["root_translation"], dtype=np.float64) * scale)
        pelvis[f] = root

        # Leg IK on the SMPL skeleton towards the generator's (scaled) foot joints
        gen_pos = np.asarray(fr["joint_positions_3d"], dtype=np.float64)
        feet = [W @ (gen_pos[10] * scale), W @ (gen_pos[11] * scale)]
        retarget_legs(local, root, joints, feet, sole)

        for j in range(24):
            aa[f, 3 * j:3 * j + 3] = rotmat_to_axis_angle(local[j])
            rot6d[f, 6 * j:6 * j + 3] = local[j][:, 0]
            rot6d[f, 6 * j + 3:6 * j + 6] = local[j][:, 1]

        # Ground contact: every foot the generator has on the floor (planted or gliding)
        _, P = smpl_fk(local, root, joints)
        soles = [P[ft, 2] - sole[ft] for (_, _, _, ft), g in zip(LEGS, (gen_pos[10], gen_pos[11])) if g[2] <= 1e-3]
        if soles:
            ground[f] = min(soles)

    # Airborne frames inherit a linear blend of the neighbouring contact offsets
    idx = np.arange(n)
    have = ~np.isnan(ground)
    ground = np.interp(idx, idx[have], ground[have]) if have.any() else np.zeros(n)
    # Light 5-tap smoothing so heel/toe switching does not pop
    k = np.array([1, 2, 3, 2, 1], dtype=np.float64) / 9.0
    ground = np.convolve(np.pad(ground, 2, mode="edge"), k, mode="valid")

    trans = pelvis.copy()
    trans[:, 2] -= ground
    trans -= joints[0]  # LBS output keeps the pelvis at rest J0
    return {"aa": aa, "rot6d": rot6d, "trans": trans.astype(np.float32),
            "scale": scale, "ground_offset": ground}

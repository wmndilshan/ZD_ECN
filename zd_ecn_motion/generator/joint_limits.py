"""
Anatomical joint limits for SMPL-24 skeleton (in radians).
Enforces biomechanical Range of Motion (RoM) bounds with vectorized clamping.
"""
import numpy as np
from typing import Tuple

# SMPL 24 Joints: (min_pitch, max_pitch), (min_roll, max_roll), (min_yaw, max_yaw)
# Bounds accommodate 3D Euler decomposition of physiological spherical/hinge joints.
JOINT_LIMITS = {
    # Root yaw is the accumulated heading, not an anatomical angle: leave it unbounded
    # (a +-4pi clamp froze the body after more than two full spins).
    0: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.inf, np.inf)),
    1: ((-np.radians(135), np.radians(90)), (-np.radians(75), np.radians(75)), (-np.radians(90), np.radians(90))),
    2: ((-np.radians(135), np.radians(90)), (-np.radians(75), np.radians(75)), (-np.radians(90), np.radians(90))),
    3: ((-np.radians(35), np.radians(35)), (-np.radians(30), np.radians(30)), (-np.radians(35), np.radians(35))),
    4: ((0.0, np.radians(150)), (-np.radians(10), np.radians(10)), (-np.radians(15), np.radians(15))),
    5: ((0.0, np.radians(150)), (-np.radians(10), np.radians(10)), (-np.radians(15), np.radians(15))),
    6: ((-np.radians(30), np.radians(30)), (-np.radians(25), np.radians(25)), (-np.radians(30), np.radians(30))),
    7: ((-np.radians(95), np.radians(95)), (-np.radians(65), np.radians(65)), (-np.radians(85), np.radians(85))),
    8: ((-np.radians(95), np.radians(95)), (-np.radians(65), np.radians(65)), (-np.radians(85), np.radians(85))),
    9: ((-np.radians(30), np.radians(30)), (-np.radians(25), np.radians(25)), (-np.radians(30), np.radians(30))),
    10: ((-np.radians(70), np.radians(45)), (-np.radians(25), np.radians(25)), (-np.radians(30), np.radians(30))),  # toe dorsiflexion to 70 deg
    11: ((-np.radians(70), np.radians(45)), (-np.radians(25), np.radians(25)), (-np.radians(30), np.radians(30))),  # toe dorsiflexion to 70 deg
    12: ((-np.radians(45), np.radians(45)), (-np.radians(40), np.radians(40)), (-np.radians(60), np.radians(60))),
    13: ((-np.radians(25), np.radians(25)), (-np.radians(25), np.radians(25)), (-np.radians(25), np.radians(25))),
    14: ((-np.radians(25), np.radians(25)), (-np.radians(25), np.radians(25)), (-np.radians(25), np.radians(25))),
    15: ((-np.radians(50), np.radians(50)), (-np.radians(40), np.radians(40)), (-np.radians(70), np.radians(70))),
    16: ((-np.radians(120), np.radians(180)), (-np.radians(110), np.radians(110)), (-np.radians(110), np.radians(110))),
    17: ((-np.radians(120), np.radians(180)), (-np.radians(110), np.radians(110)), (-np.radians(110), np.radians(110))),
    18: ((0.0, np.radians(150)), (-np.radians(20), np.radians(20)), (-np.radians(20), np.radians(20))),
    19: ((0.0, np.radians(150)), (-np.radians(20), np.radians(20)), (-np.radians(20), np.radians(20))),
    20: ((-np.radians(80), np.radians(80)), (-np.radians(50), np.radians(50)), (-np.radians(45), np.radians(45))),
    21: ((-np.radians(80), np.radians(80)), (-np.radians(50), np.radians(50)), (-np.radians(45), np.radians(45))),
    22: ((-np.radians(40), np.radians(40)), (-np.radians(30), np.radians(30)), (-np.radians(30), np.radians(30))),
    23: ((-np.radians(40), np.radians(40)), (-np.radians(30), np.radians(30)), (-np.radians(30), np.radians(30)))
}

# Precomputed vectorized bounds arrays (24, 3)
MIN_LIMITS = np.zeros((24, 3), dtype=np.float32)
MAX_LIMITS = np.zeros((24, 3), dtype=np.float32)
for _j in range(24):
    _lp, _lr, _ly = JOINT_LIMITS[_j]
    MIN_LIMITS[_j] = [_lp[0], _lr[0], _ly[0]]
    MAX_LIMITS[_j] = [_lp[1], _lr[1], _ly[1]]


def clamp_joint_rotations(rotations_24x3: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Vectorized clamping of joint rotation angles to physiological limits.
    Returns (clamped_rotations, num_joints_clamped).
    """
    clamped = np.clip(rotations_24x3, MIN_LIMITS, MAX_LIMITS)
    diff = np.abs(clamped - rotations_24x3) > 1e-5
    violations = int(np.sum(np.any(diff, axis=-1)))
    return clamped, violations

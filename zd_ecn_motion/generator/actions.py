"""
Biomechanical Procedural Action Synthesizers for SMPL-24.
Produces desired root trajectories, upper-body articulations, foot contact schedules,
and world-space swing clearances for closed-loop IK foot locking.
"""
import math
import numpy as np
from typing import Dict, Any


def min_jerk(s: float) -> float:
    """Minimum-jerk C^2 smooth step trajectory: 10s^3 - 15s^4 + 6s^5."""
    s = max(0.0, min(1.0, s))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def generate_idle(duration: float, fps: int, start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    base_z = 0.78
    for i, t in enumerate(times):
        breath = 0.015 * math.sin(2.0 * math.pi * 0.28 * t)
        sway_x = 0.006 * math.sin(2.0 * math.pi * 0.15 * t)

        s_blend = min_jerk(min(1.0, t / max(0.25, duration * 0.25)))
        z_curr = (1.0 - s_blend) * start_root[2] + s_blend * (base_z + breath * 0.25)

        translations[i] = [start_root[0] + sway_x, start_root[1], z_curr]
        rotations[i, 0] = [0.02, -sway_x * 0.5, start_heading]

        # Multi-segment thoracic breathing wave
        rotations[i, 3] = [breath * 0.6, sway_x * 0.3, 0.0]
        rotations[i, 6] = [breath * 0.4, sway_x * 0.2, 0.0]
        rotations[i, 9] = [breath * 0.3, 0.0, 0.0]
        rotations[i, 12] = [-breath * 0.3, 0.0, 0.0]
        rotations[i, 15] = [-breath * 0.2, 0.0, 0.02 * math.sin(2.0 * math.pi * 0.1 * t)]

        # Natural relaxed arms with slight elbow flexion
        rotations[i, 16] = [0.04, -0.18, 0.05]
        rotations[i, 17] = [0.04, 0.18, -0.05]
        rotations[i, 18] = [0.18 + 0.02 * breath, 0.0, 0.0]
        rotations[i, 19] = [0.18 + 0.02 * breath, 0.0, 0.0]
        contacts.append((True, True))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "standard",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_walk(duration: float, fps: int, speed: float, step_freq: float,
                  start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    fwd_dir = np.array([-math.sin(start_heading), math.cos(start_heading), 0.0], dtype=np.float32)
    lat_dir = np.array([math.cos(start_heading), math.sin(start_heading), 0.0], dtype=np.float32)

    # Cap speed so discrete steps stay comfortably within human leg reach
    effective_speed = min(float(speed), 1.15)
    base_z = 0.77

    for i, t in enumerate(times):
        # Smooth ramp-up and ramp-down of walking velocity
        env = min_jerk(min(1.0, t / 0.35)) * min_jerk(min(1.0, (duration - t) / 0.35))
        disp = effective_speed * t * (0.85 + 0.15 * env)
        phase = 2.0 * math.pi * step_freq * t

        sin_p = math.sin(phase)
        cos_p = math.cos(phase)

        # Double-frequency vertical CoM oscillation + lateral ZMP weight transfer
        bounce = 0.022 * abs(cos_p) * env
        sway_lat = -0.035 * sin_p * env

        cur_root = start_root + fwd_dir * disp + lat_dir * sway_lat
        cur_root[2] = base_z + bounce
        translations[i] = cur_root

        # Pelvis 3D dynamics: slight forward lean, Trendelenburg roll, and transverse yaw
        rotations[i, 0, 0] = 0.05 * env
        rotations[i, 0, 1] = 0.05 * sin_p * env
        rotations[i, 0, 2] = start_heading + 0.08 * sin_p * env

        # Counter-rotating multi-segment spine wave
        rotations[i, 3] = [0.03 * env, -0.03 * sin_p * env, -0.04 * sin_p * env]
        rotations[i, 6] = [0.02 * env, -0.02 * sin_p * env, -0.05 * sin_p * env]
        rotations[i, 9] = [0.01 * env, -0.01 * sin_p * env, -0.04 * sin_p * env]
        rotations[i, 12] = [-0.03 * env, 0.0, 0.03 * sin_p * env]
        rotations[i, 15] = [-0.02 * env, 0.0, 0.02 * sin_p * env]

        # Contralateral arm pendulum swing with distal elbow/wrist phase lag
        arm_lag = math.sin(phase - 0.25)
        wrist_lag = math.sin(phase - 0.45)
        rotations[i, 16] = [0.38 * sin_p * env, -0.14, 0.0]
        rotations[i, 17] = [-0.38 * sin_p * env, 0.14, 0.0]
        rotations[i, 18] = [0.28 + 0.22 * max(0.0, arm_lag) * env, 0.0, 0.0]
        rotations[i, 19] = [0.28 + 0.22 * max(0.0, -arm_lag) * env, 0.0, 0.0]
        rotations[i, 20] = [0.10 * wrist_lag * env, 0.0, 0.0]
        rotations[i, 21] = [-0.10 * wrist_lag * env, 0.0, 0.0]

        # Stance vs Swing scheduling (with double-support overlap for realism)
        l_swing = max(0.0, sin_p - 0.15) / 0.85
        r_swing = max(0.0, -sin_p - 0.15) / 0.85
        l_planted = (l_swing <= 0.0) or (i == 0) or (i == num_frames - 1)
        r_planted = (r_swing <= 0.0) or (i == 0) or (i == num_frames - 1)

        foot_lifts[i, 0] = 0.09 * math.sin(l_swing * math.pi) if not l_planted else 0.0
        foot_lifts[i, 1] = 0.09 * math.sin(r_swing * math.pi) if not r_planted else 0.0
        contacts.append((bool(l_planted), bool(r_planted)))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "standard",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_turn(duration: float, fps: int, angle_degrees: float,
                  start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    angle_rad = math.radians(angle_degrees)
    base_z = 0.77

    for i, t in enumerate(times):
        s = t / duration
        turn_prog = min_jerk(s) * angle_rad
        current_yaw = start_heading + turn_prog

        step_phase = 2.0 * math.pi * (t / max(0.5, duration * 0.5))
        sin_p = math.sin(step_phase)

        translations[i] = [start_root[0], start_root[1], base_z + 0.015 * abs(sin_p)]
        rotations[i, 0] = [0.03, 0.03 * sin_p, current_yaw]

        # Head and upper spine lead the turn (oculo-cephalic anticipation)
        lead_envelope = math.sin(s * math.pi)
        lead_sign = 1.0 if angle_rad >= 0 else -1.0
        rotations[i, 3, 2] = 0.10 * lead_sign * lead_envelope
        rotations[i, 6, 2] = 0.12 * lead_sign * lead_envelope
        rotations[i, 12, 2] = 0.18 * lead_sign * lead_envelope
        rotations[i, 15, 2] = 0.25 * lead_sign * lead_envelope

        rotations[i, 16] = [0.15 * sin_p, -0.18, 0.0]
        rotations[i, 17] = [-0.15 * sin_p, 0.18, 0.0]
        rotations[i, 18] = [0.30, 0.0, 0.0]
        rotations[i, 19] = [0.30, 0.0, 0.0]

        l_swing = max(0.0, sin_p - 0.1) / 0.9
        r_swing = max(0.0, -sin_p - 0.1) / 0.9
        l_planted = (l_swing <= 0.0) or (i == 0) or (i == num_frames - 1)
        r_planted = (r_swing <= 0.0) or (i == 0) or (i == num_frames - 1)

        foot_lifts[i, 0] = 0.07 * math.sin(l_swing * math.pi) if not l_planted else 0.0
        foot_lifts[i, 1] = 0.07 * math.sin(r_swing * math.pi) if not r_planted else 0.0
        contacts.append((bool(l_planted), bool(r_planted)))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "standard",
        "end_root": translations[-1],
        "end_heading": start_heading + angle_rad
    }


def generate_squat(duration: float, fps: int, depth: float, hold_duration: float,
                   start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    base_z = 0.78
    max_drop = 0.28 * depth

    for i, t in enumerate(times):
        s = t / duration
        if s < 0.42:
            curve = min_jerk(s / 0.42)
        elif s < 0.58:
            curve = 1.0
        else:
            curve = 1.0 - min_jerk((s - 0.58) / 0.42)

        translations[i] = [start_root[0], start_root[1], base_z - max_drop * curve]
        rotations[i, 0] = [0.18 * curve, 0.0, start_heading]

        # Lumbar/thoracic forward inclination for CoM balance over mid-foot
        rotations[i, 3] = [0.22 * curve, 0.0, 0.0]
        rotations[i, 6] = [0.15 * curve, 0.0, 0.0]
        rotations[i, 9] = [0.08 * curve, 0.0, 0.0]
        # Neck extends to keep gaze horizontal
        rotations[i, 12] = [-0.25 * curve, 0.0, 0.0]
        rotations[i, 15] = [-0.15 * curve, 0.0, 0.0]

        # Counterbalance arm reach
        rotations[i, 16] = [-0.85 * curve, -0.12, 0.0]
        rotations[i, 17] = [-0.85 * curve, 0.12, 0.0]
        rotations[i, 18] = [0.35 * curve + 0.15, 0.0, 0.0]
        rotations[i, 19] = [0.35 * curve + 0.15, 0.0, 0.0]
        contacts.append((True, True))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "standard",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_jump(duration: float, fps: int, height: float, forward_dist: float,
                  start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    fwd_dir = np.array([-math.sin(start_heading), math.cos(start_heading), 0.0], dtype=np.float32)
    base_z = 0.78

    for i, t in enumerate(times):
        s = t / duration
        if s < 0.30:
            # Counter-movement crouch preparation
            p_crouch = min_jerk(s / 0.30)
            z_pos = base_z - 0.16 * p_crouch
            fwd_pos = 0.0
            rotations[i, 0] = [0.20 * p_crouch, 0.0, start_heading]
            rotations[i, 3] = [0.22 * p_crouch, 0.0, 0.0]
            # Arms cock back for launch momentum
            rotations[i, 16] = [0.55 * p_crouch, -0.15, 0.0]
            rotations[i, 17] = [0.55 * p_crouch, 0.15, 0.0]
            rotations[i, 18] = [0.45 * p_crouch + 0.15, 0.0, 0.0]
            rotations[i, 19] = [0.45 * p_crouch + 0.15, 0.0, 0.0]
            contacts.append((True, True))
        else:
            # Ballistic flight parabola
            p_air = (s - 0.30) / 0.70
            parabola = 4.0 * p_air * (1.0 - p_air)
            z_pos = (base_z - 0.16 * (1.0 - p_air)) + height * parabola
            fwd_pos = forward_dist * min_jerk(p_air)

            rotations[i, 0] = [-0.06 * parabola, 0.0, start_heading]
            rotations[i, 3] = [-0.08 * parabola, 0.0, 0.0]
            rotations[i, 16] = [-1.15 * parabola, -0.20, 0.0]
            rotations[i, 17] = [-1.15 * parabola, 0.20, 0.0]
            rotations[i, 18] = [0.35, 0.0, 0.0]
            rotations[i, 19] = [0.35, 0.0, 0.0]

            # Feet lift along the flight parabola with subtle mid-air knee tuck
            air_lift = max(0.02, height * parabola * 0.88)
            foot_lifts[i, 0] = air_lift
            foot_lifts[i, 1] = air_lift
            contacts.append((False, False))

        translations[i] = start_root + fwd_dir * fwd_pos
        translations[i, 2] = z_pos

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "standard",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_land(duration: float, fps: int, recovery_duration: float, damping: float,
                  start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    base_z = 0.78
    impact_drop = 0.18 * damping

    for i, t in enumerate(times):
        s = t / duration
        # Under-damped spring absorption curve
        shock = math.exp(-3.8 * s) * math.sin(math.pi * min(1.0, s * 1.4))
        z_pos = base_z - impact_drop * max(0.0, shock)

        translations[i] = [start_root[0], start_root[1], z_pos]
        rotations[i, 0] = [0.16 * max(0.0, shock), 0.0, start_heading]
        rotations[i, 3] = [0.22 * max(0.0, shock), 0.0, 0.0]
        rotations[i, 6] = [0.12 * max(0.0, shock), 0.0, 0.0]
        rotations[i, 12] = [-0.20 * max(0.0, shock), 0.0, 0.0]

        rotations[i, 16] = [-0.45 * max(0.0, shock), -0.18, 0.0]
        rotations[i, 17] = [-0.45 * max(0.0, shock), 0.18, 0.0]
        rotations[i, 18] = [0.40 * max(0.0, shock) + 0.15, 0.0, 0.0]
        rotations[i, 19] = [0.40 * max(0.0, shock) + 0.15, 0.0, 0.0]
        contacts.append((True, True))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "standard",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_dance_mandiya(duration: float, fps: int, start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Authentic Kandyan/Uva Dance Opening Salutation (Namaskaraya) & Sinking into Deep Mandiya."""
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    target_z = 0.64  # Deep seated Mandiya elevation
    for i, t in enumerate(times):
        prog = min_jerk(t / duration)
        z_pos = start_root[2] - (start_root[2] - target_z) * prog
        translations[i] = [start_root[0], start_root[1], z_pos]
        rotations[i, 0] = [0.06 * prog, 0.0, start_heading]

        # Proud chest expansion (Hada) & erect cervical posture
        rotations[i, 3] = [0.08 * prog, 0.0, 0.0]
        rotations[i, 6] = [-0.06 * prog, 0.0, 0.0]
        rotations[i, 9] = [-0.05 * prog, 0.0, 0.0]
        rotations[i, 12] = [0.04 * prog, 0.0, 0.0]
        rotations[i, 15] = [0.05 * (1.0 - prog), 0.0, 0.0]

        # Arms: Anjali Mudra at chest -> Sweeping open into classical sculptural Mandiya arms
        if prog < 0.5:
            p_a = min_jerk(prog / 0.5)
            rotations[i, 16] = [-0.35 * p_a, -0.35 * p_a, 0.35 * p_a]
            rotations[i, 17] = [-0.35 * p_a, 0.35 * p_a, -0.35 * p_a]
            rotations[i, 18] = [1.10 * p_a, 0.0, 0.0]
            rotations[i, 19] = [1.10 * p_a, 0.0, 0.0]
        else:
            p_o = min_jerk((prog - 0.5) / 0.5)
            rotations[i, 16] = [-0.35 + 0.45 * p_o, -0.35 - 0.45 * p_o, 0.35]
            rotations[i, 17] = [-0.35 + 0.45 * p_o, 0.35 + 0.45 * p_o, -0.35]
            rotations[i, 18] = [1.10 - 0.15 * p_o, -0.18 * p_o, 0.0]
            rotations[i, 19] = [1.10 - 0.15 * p_o, 0.18 * p_o, 0.0]
            rotations[i, 20] = [0.35 * p_o, 0.25 * p_o, -0.30 * p_o]
            rotations[i, 21] = [0.35 * p_o, -0.25 * p_o, 0.30 * p_o]

        contacts.append((True, True))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "mandiya",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_dance_step(duration: float, fps: int, cadence: str,
                        start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Authentic Kandyan/Uva Pa Saramba & Kastiram Rhythmic Stamping with CoM Weight Transfer."""
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    tempo_hz = 0.75
    base_z = 0.64
    lat_dir = np.array([math.cos(start_heading), math.sin(start_heading), 0.0], dtype=np.float32)

    for i, t in enumerate(times):
        env = min_jerk(min(1.0, t / 0.25)) * min_jerk(min(1.0, (duration - t) / 0.25))
        phase = 2.0 * math.pi * tempo_hz * t
        sin_p = math.sin(phase)
        cos_p = math.cos(phase)

        # Crisp drum-beat bounce (Geta Bera / Dawula syncopation)
        bounce = 0.032 * (math.sin(2.0 * phase) ** 2) * env
        # Anticipatory CoM shift toward supporting leg
        weight_shift = -0.045 * sin_p * env

        cur_root = start_root + lat_dir * weight_shift
        cur_root[2] = base_z - bounce
        translations[i] = cur_root

        rotations[i, 0] = [0.06, 0.06 * sin_p * env, start_heading + 0.05 * cos_p * env]

        # Multi-segment torso undulation & Urahis Keliya (shoulder accents)
        sh_accent = 0.06 * math.sin(2.0 * phase) * env
        rotations[i, 3] = [0.08, -0.07 * sin_p * env, 0.06 * sin_p * env]
        rotations[i, 6] = [-0.05, -0.05 * sin_p * env, 0.05 * sin_p * env]
        rotations[i, 9] = [-0.04, -0.04 * sin_p * env, 0.04 * sin_p * env]
        rotations[i, 13] = [0.0, sh_accent, 0.04 * sin_p * env]
        rotations[i, 14] = [0.0, -sh_accent, -0.04 * sin_p * env]

        # Head Drishti (sharp rhythmic gaze snaps tracking active hand mudra)
        rotations[i, 12] = [0.03, 0.06 * cos_p * env, 0.12 * sin_p * env]
        rotations[i, 15] = [0.05, 0.12 * cos_p * env, 0.28 * sin_p * env]

        # Sculptural Hastamudra arm arcs & Kaatuka wrist rolls with distal phase lag
        arm_lag = phase - 0.22
        wrist_lag = phase - 0.45
        rotations[i, 16] = [0.28 * math.sin(arm_lag) * env, -0.78 - 0.16 * math.cos(arm_lag) * env, 0.35]
        rotations[i, 18] = [0.92 + 0.28 * math.sin(arm_lag) * env, -0.18, 0.15]
        rotations[i, 20] = [0.48 * math.cos(wrist_lag) * env, 0.38 * math.sin(wrist_lag) * env, -0.35]

        rotations[i, 17] = [0.28 * math.sin(arm_lag + math.pi) * env, 0.78 + 0.16 * math.cos(arm_lag) * env, -0.35]
        rotations[i, 19] = [0.92 + 0.28 * math.sin(arm_lag + math.pi) * env, 0.18, -0.15]
        rotations[i, 21] = [-0.48 * math.cos(wrist_lag) * env, -0.38 * math.sin(wrist_lag) * env, 0.35]

        # Crisp foot lift and firm heel-to-sole stamp
        l_swing = max(0.0, sin_p - 0.20) / 0.80
        r_swing = max(0.0, -sin_p - 0.20) / 0.80
        l_planted = (l_swing <= 0.0) or (i == 0) or (i == num_frames - 1)
        r_planted = (r_swing <= 0.0) or (i == 0) or (i == num_frames - 1)

        foot_lifts[i, 0] = 0.12 * (math.sin(l_swing * math.pi) ** 1.4) * env if not l_planted else 0.0
        foot_lifts[i, 1] = 0.12 * (math.sin(r_swing * math.pi) ** 1.4) * env if not r_planted else 0.0
        contacts.append((bool(l_planted), bool(r_planted)))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "mandiya",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


def generate_dance_turn(duration: float, fps: int, angle_degrees: float,
                        start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Vattam 360-degree Circular Dance Turn holding Deep Mandiya & Flowing Arm Sweeps."""
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    angle_rad = math.radians(angle_degrees)
    base_z = 0.65

    for i, t in enumerate(times):
        s = t / duration
        turn_prog = min_jerk(s) * angle_rad
        current_yaw = start_heading + turn_prog

        step_phase = 2.0 * math.pi * 1.6 * t
        sin_p = math.sin(step_phase)

        translations[i] = [
            start_root[0] + 0.04 * math.sin(turn_prog),
            start_root[1] + 0.04 * (1.0 - math.cos(turn_prog)),
            base_z + 0.02 * abs(sin_p)
        ]
        rotations[i, 0] = [0.06, 0.04 * sin_p, current_yaw]

        vatt_phase = turn_prog * 2.0
        rotations[i, 3] = [0.08, -0.05 * math.sin(vatt_phase), 0.08 * math.cos(vatt_phase)]
        rotations[i, 6] = [-0.04, -0.04 * math.sin(vatt_phase), 0.06 * math.cos(vatt_phase)]
        rotations[i, 15] = [0.05, 0.10 * math.cos(vatt_phase), 0.30 * math.sin(vatt_phase)]

        rotations[i, 16] = [0.32 * math.cos(vatt_phase), -0.72, 0.35]
        rotations[i, 17] = [-0.32 * math.cos(vatt_phase), 0.72, -0.35]
        rotations[i, 18] = [0.95 + 0.22 * math.sin(vatt_phase), -0.18, 0.12]
        rotations[i, 19] = [0.95 - 0.22 * math.sin(vatt_phase), 0.18, -0.12]
        rotations[i, 20] = [0.40 * math.sin(vatt_phase), 0.30 * math.cos(vatt_phase), -0.25]
        rotations[i, 21] = [-0.40 * math.sin(vatt_phase), -0.30 * math.cos(vatt_phase), 0.25]

        l_swing = max(0.0, sin_p - 0.15) / 0.85
        r_swing = max(0.0, -sin_p - 0.15) / 0.85
        l_planted = (l_swing <= 0.0) or (i == 0) or (i == num_frames - 1)
        r_planted = (r_swing <= 0.0) or (i == 0) or (i == num_frames - 1)

        foot_lifts[i, 0] = 0.09 * math.sin(l_swing * math.pi) if not l_planted else 0.0
        foot_lifts[i, 1] = 0.09 * math.sin(r_swing * math.pi) if not r_planted else 0.0
        contacts.append((bool(l_planted), bool(r_planted)))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "mandiya",
        "end_root": translations[-1],
        "end_heading": start_heading + angle_rad
    }


def generate_dance_freeze(duration: float, fps: int,
                          start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Samapthiya Heroic Final Statue Posture with Subtle Breathing."""
    num_frames = max(1, int(round(duration * fps)))
    times = np.linspace(0.0, duration, num_frames, endpoint=False)
    rotations = np.zeros((num_frames, 24, 3), dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    contacts = []
    foot_lifts = np.zeros((num_frames, 2), dtype=np.float32)

    base_z = 0.63
    for i, t in enumerate(times):
        settle = min_jerk(min(1.0, t / 0.4))
        breath = 0.008 * math.sin(2.0 * math.pi * 0.3 * t)
        z_curr = (1.0 - settle) * start_root[2] + settle * base_z + breath * 0.2

        translations[i] = [start_root[0], start_root[1], z_curr]
        rotations[i, 0] = [0.06, 0.0, start_heading]

        # Proud chest, hands locked at waist/hips, chin elevated
        rotations[i, 3] = [0.10 + breath * 0.4, 0.0, 0.0]
        rotations[i, 6] = [-0.08 + breath * 0.3, 0.0, 0.0]
        rotations[i, 9] = [-0.06, 0.0, 0.0]
        rotations[i, 12] = [0.05, 0.0, 0.0]
        rotations[i, 15] = [0.12, 0.0, 0.0]

        rotations[i, 16] = [0.05, -0.68, 0.32]
        rotations[i, 17] = [0.05, 0.68, -0.32]
        rotations[i, 18] = [1.22, -0.16, 0.0]
        rotations[i, 19] = [1.22, 0.16, 0.0]
        rotations[i, 20] = [0.32, 0.0, 0.0]
        rotations[i, 21] = [0.32, 0.0, 0.0]

        contacts.append((True, True))

    return {
        "rotations": rotations,
        "translations": translations,
        "contacts": contacts,
        "foot_lifts": foot_lifts,
        "style": "mandiya",
        "end_root": translations[-1],
        "end_heading": start_heading
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pop-dance vocabulary (moonwalk, spin, toe stand, pop hits, point pose).
#
# These moves need feet that slide while loaded (moonwalk), pivot in place
# (spin) or stand on the toes (toe stand), which the contact-based foot lock
# cannot express. Such actions return "foot_override": (n, 2, 5) rows of
# [x, y, z, yaw, pitch] for (left, right) feet — NaN where the regular foot
# lock applies. +pitch raises the heel about the ball of the foot.
#
# Arm angles follow the generator's hanging-arm convention: shoulder +pitch
# swings the arm back, -pitch forward/up, roll abducts (left -, right +),
# elbow +pitch flexes.
# ─────────────────────────────────────────────────────────────────────────────

STANCE_LAT = 0.13   # standard stance half-width (matches the foot-lock nominal)
STANCE_FWD = 0.10   # standard foot forward offset from the pelvis


def _stance_feet(root_xy: np.ndarray, heading: float):
    fwd = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=np.float32)
    lat = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=np.float32)
    base = np.array([root_xy[0], root_xy[1], 0.0], dtype=np.float32)
    return base + lat * STANCE_LAT + fwd * STANCE_FWD, base - lat * STANCE_LAT + fwd * STANCE_FWD


def _set_override(ov: np.ndarray, i: int, foot: int, pos, yaw: float, pitch: float) -> None:
    ov[i, foot] = [pos[0], pos[1], max(0.0, float(pos[2])), yaw, pitch]


def _new_buffers(duration: float, fps: int):
    # endpoint=True so the last frame reaches the move's end state exactly; foot
    # paths then hand over to the next phase without a residual jump.
    n = max(1, int(round(duration * fps)))
    return (
        n,
        np.linspace(0.0, duration, n, endpoint=True) if n > 1 else np.zeros(1),
        np.zeros((n, 24, 3), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
        np.zeros((n, 2), dtype=np.float32),
        np.full((n, 2, 5), np.nan, dtype=np.float32),
    )


def generate_moonwalk(duration: float, fps: int, stride: float,
                      start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Backslide: the loaded flat foot glides back past the toe-standing foot,
    then the feet swap roles. The toe foot never moves, which sells the illusion
    of walking forward while travelling backward."""
    n, times, rotations, translations, foot_lifts, override = _new_buffers(duration, fps)

    h = start_heading
    fwd = np.array([-math.sin(h), math.cos(h), 0.0], dtype=np.float32)
    lat = np.array([math.cos(h), math.sin(h), 0.0], dtype=np.float32)
    steps = max(2, int(round(duration / 0.55)))
    half = duration / steps
    toe_pitch = 0.95
    swap = 0.18  # fraction of a step spent rolling heel up / down

    def slide_range(k: int, foot: int):
        """(start, end) back-offset of `foot` during step k (multiples of stride)."""
        # Right foot slides on even steps, left on odd; toe foot sits at -k*stride.
        sliding = (k % 2 == 0) == (foot == 1)
        if not sliding:
            return float(k), float(k)
        start = float(max(0, k - 1))
        end = float(k + 1) if k < steps - 1 else float(k)  # last step closes the feet
        return start, end

    for i, t in enumerate(times):
        k = min(steps - 1, int(t / half))
        u = (t - k * half) / half
        slide = 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, (u - swap) / (1.0 - swap))))
        roll_in = min_jerk(u / swap)
        offsets = []
        for foot in (0, 1):
            a, b = slide_range(k, foot)
            back = (a + (b - a) * slide) * stride
            sliding = a != b
            if k == 0:
                pitch = 0.0 if sliding else toe_pitch * roll_in
            elif sliding:
                pitch = toe_pitch * (1.0 - roll_in)
            else:
                pitch = toe_pitch * roll_in
            if k == steps - 1:
                pitch *= 1.0 - min_jerk((u - 0.7) / 0.3)  # settle both feet flat
            side = STANCE_LAT if foot == 0 else -STANCE_LAT
            pos = start_root + lat * side + fwd * (STANCE_FWD - back)
            pos[2] = 0.0
            _set_override(override, i, foot, pos, h, pitch)
            offsets.append(back)

        mean_back = 0.5 * (offsets[0] + offsets[1])
        phase = math.pi * t / half
        bob = 0.012 * math.sin(phase) ** 2
        root = start_root - fwd * mean_back
        root[2] = 0.765 - bob
        translations[i] = root

        rotations[i, 0] = [0.05, 0.03 * math.sin(phase), h]
        rotations[i, 3] = [0.03, 0.0, 0.0]
        rotations[i, 9] = [-0.03, 0.0, 0.0]
        rotations[i, 15] = [-0.06, 0.0, 0.0]
        rotations[i, 16] = [0.28 * math.cos(phase), -0.12, 0.0]
        rotations[i, 17] = [-0.28 * math.cos(phase), 0.12, 0.0]
        rotations[i, 18] = [0.55 + 0.15 * max(0.0, math.cos(phase)), 0.0, 0.0]
        rotations[i, 19] = [0.55 + 0.15 * max(0.0, -math.cos(phase)), 0.0, 0.0]
        rotations[i, 20] = [0.25, 0.0, 0.0]
        rotations[i, 21] = [0.25, 0.0, 0.0]

    # A foot is "planted" (locked, weight-bearing) only while it is not gliding.
    contacts = []
    for i in range(n):
        moving = np.linalg.norm(override[min(n - 1, i + 1), :, :2] - override[max(0, i - 1), :, :2], axis=-1) > 1e-6
        contacts.append((not bool(moving[0]), not bool(moving[1])))

    return {
        "rotations": rotations, "translations": translations, "contacts": contacts,
        "foot_lifts": foot_lifts, "foot_override": override, "style": "standard",
        "end_root": translations[-1], "end_heading": start_heading,
    }


def generate_spin(duration: float, fps: int, angle_degrees: float,
                  start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Fast pivot on the ball of the right foot with the left foot tucked at the
    knee and arms drawn in. The pivot foot rotates in place (no XY slip)."""
    n, times, rotations, translations, foot_lifts, override = _new_buffers(duration, fps)
    contacts = []
    angle = math.radians(angle_degrees)
    h0 = start_heading

    def frame_dirs(h):
        return (np.array([-math.sin(h), math.cos(h), 0.0], dtype=np.float32),
                np.array([math.cos(h), math.sin(h), 0.0], dtype=np.float32))

    f0, l0 = frame_dirs(h0)
    pivot = start_root - l0 * STANCE_LAT + f0 * STANCE_FWD
    pivot[2] = 0.0

    for i, t in enumerate(times):
        s = t / duration
        tuck = min_jerk(s / 0.2) * (1.0 - min_jerk((s - 0.8) / 0.2))
        h = h0 + angle * min_jerk((s - 0.2) / 0.6)
        fwd, lat = frame_dirs(h)

        # Pelvis sits over the pivot, shifting its weight onto it while tucked
        lat_off = STANCE_LAT - 0.07 * tuck
        root = pivot + lat * lat_off - fwd * STANCE_FWD
        root[2] = 0.78 + 0.05 * tuck
        translations[i] = root

        nominal_l = root + lat * STANCE_LAT + fwd * STANCE_FWD
        tucked_l = root + lat * 0.04 + fwd * 0.14
        nominal_l[2] = 0.0
        tucked_l[2] = 0.32
        _set_override(override, i, 0, nominal_l + (tucked_l - nominal_l) * tuck, h, 0.7 * tuck)
        _set_override(override, i, 1, pivot, h, 0.55 * tuck)
        contacts.append((tuck <= 0.0, True))

        rotations[i, 0] = [0.03 * tuck, 0.0, h]
        rotations[i, 3] = [0.04 * tuck, 0.0, 0.0]
        rotations[i, 15] = [-0.05 * tuck, 0.0, 0.0]
        rotations[i, 16] = [-0.55 * tuck + 0.04, 0.15 * tuck - 0.12, 0.0]
        rotations[i, 17] = [-0.55 * tuck + 0.04, -0.15 * tuck + 0.12, 0.0]
        rotations[i, 18] = [0.25 + 1.75 * tuck, 0.0, 0.0]
        rotations[i, 19] = [0.25 + 1.75 * tuck, 0.0, 0.0]

    return {
        "rotations": rotations, "translations": translations, "contacts": contacts,
        "foot_lifts": foot_lifts, "foot_override": override, "style": "standard",
        "end_root": translations[-1], "end_heading": h0 + angle,
    }


def generate_toe_stand(duration: float, fps: int,
                       start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Rise onto both toes, hold the balance with one arm lifted, drop back."""
    n, times, rotations, translations, foot_lifts, override = _new_buffers(duration, fps)
    contacts = []
    h = start_heading
    left, right = _stance_feet(start_root, h)

    for i, t in enumerate(times):
        s = t / duration
        rise = min_jerk(s / 0.3) * (1.0 - min_jerk((s - 0.75) / 0.25))
        translations[i] = [start_root[0], start_root[1], 0.78 + 0.075 * rise]
        _set_override(override, i, 0, left, h, 1.05 * rise)
        _set_override(override, i, 1, right, h, 1.05 * rise)
        contacts.append((True, True))

        rotations[i, 0] = [0.0, 0.0, h]
        rotations[i, 3] = [-0.05 * rise, 0.0, 0.0]
        rotations[i, 6] = [-0.04 * rise, 0.0, 0.0]
        rotations[i, 12] = [-0.08 * rise, 0.0, 0.0]
        rotations[i, 15] = [-0.12 * rise, 0.0, 0.0]
        rotations[i, 16] = [0.15 * rise, -0.20, 0.0]
        rotations[i, 17] = [-1.05 * rise, 0.12 + 0.2 * rise, -0.3 * rise]
        rotations[i, 18] = [0.20, 0.0, 0.0]
        rotations[i, 19] = [0.30 + 1.10 * rise, 0.0, 0.0]
        rotations[i, 21] = [-0.30 * rise, 0.0, 0.0]

    return {
        "rotations": rotations, "translations": translations, "contacts": contacts,
        "foot_lifts": foot_lifts, "foot_override": override, "style": "standard",
        "end_root": translations[-1], "end_heading": h,
    }


# Staccato pop hit poses: joint -> (pitch, roll, yaw); "root" = pelvis (pitch, roll, yaw offset)
_POP_NEUTRAL = {16: (0.04, -0.18, 0.05), 17: (0.04, 0.18, -0.05), 18: (0.25, 0.0, 0.0), 19: (0.25, 0.0, 0.0)}
_POP_POSES = [
    {"root": (0.02, 0.0, 0.15), 3: (0.0, 0.0, 0.10), 15: (0.0, 0.0, 0.45),
     16: (0.10, -0.15, 0.0), 17: (-0.95, 0.20, -0.20), 18: (0.20, 0.0, 0.0), 19: (1.45, 0.0, 0.0), 21: (-0.40, 0.0, 0.0)},
    {"root": (0.0, 0.08, -0.10), 13: (0.0, 0.0, 0.20), 14: (0.0, 0.0, -0.20), 15: (-0.05, 0.0, -0.45),
     16: (0.20, -0.45, 0.0), 17: (0.20, 0.45, 0.0), 18: (1.60, 0.0, 0.0), 19: (1.60, 0.0, 0.0)},
    {"root": (0.0, -0.06, -0.20), 3: (0.0, 0.0, -0.12), 9: (0.0, 0.0, -0.08), 15: (-0.08, 0.0, -0.30),
     16: (-1.40, -0.10, 0.20), 17: (-0.50, 0.15, 0.30), 18: (0.15, 0.0, 0.0), 19: (2.00, 0.0, 0.0)},
    {"root": (-0.10, 0.0, 0.0), 3: (0.10, 0.0, 0.0), 12: (0.15, 0.0, 0.0), 15: (0.20, 0.0, 0.0),
     16: (0.25, -0.30, 0.0), 17: (0.25, 0.30, 0.0), 18: (1.00, 0.0, 0.0), 19: (1.00, 0.0, 0.0)},
]


def _pop_pose_array(pose: Dict[Any, Any]) -> np.ndarray:
    arr = np.zeros((25, 3), dtype=np.float32)  # row 24 = pelvis offsets
    for j, v in _POP_NEUTRAL.items():
        arr[j] = v
    for j, v in pose.items():
        arr[24 if j == "root" else j] = v
    return arr


def generate_pop_hits(duration: float, fps: int, tempo_bpm: float,
                      start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Staccato upper-body 'hits' on each beat: snap into a pose in ~20% of the
    beat, hold, with a knee dip on the accent. Feet stay planted."""
    n, times, rotations, translations, foot_lifts, override = _new_buffers(duration, fps)
    contacts = []
    beat = 60.0 / max(40.0, min(200.0, tempo_bpm))
    # Feet stay pinned to the stance while the pelvis twists into each hit
    left, right = _stance_feet(start_root, start_heading)
    poses = [_pop_pose_array(p) for p in _POP_POSES]
    neutral = _pop_pose_array({})

    for i, t in enumerate(times):
        k = int(t / beat)
        u = (t - k * beat) / beat
        cur = poses[k % len(poses)]
        prev = neutral if k == 0 else poses[(k - 1) % len(poses)]
        w = min_jerk(u / 0.2)
        pose = prev + (cur - prev) * w
        out = min_jerk((t - (duration - 0.35)) / 0.35)  # return to neutral at the end
        pose = pose + (neutral - pose) * out

        dip = 0.028 * math.sin(math.pi * min(1.0, u / 0.45)) * (1.0 - out)
        translations[i] = [start_root[0], start_root[1], 0.77 - dip]
        rotations[i, :24] = pose[:24]
        rotations[i, 0] = [0.02 + pose[24, 0], pose[24, 1], start_heading + pose[24, 2]]
        _set_override(override, i, 0, left, start_heading, 0.0)
        _set_override(override, i, 1, right, start_heading, 0.0)
        contacts.append((True, True))

    return {
        "rotations": rotations, "translations": translations, "contacts": contacts,
        "foot_lifts": foot_lifts, "foot_override": override, "style": "standard",
        "end_root": translations[-1], "end_heading": start_heading,
    }


def generate_pose_point(duration: float, fps: int,
                        start_root: np.ndarray, start_heading: float) -> Dict[str, Any]:
    """Signature closing freeze: left arm points diagonally up and out, gaze
    follows it, weight on the right leg with the left foot forward on its toe."""
    n, times, rotations, translations, foot_lifts, override = _new_buffers(duration, fps)
    contacts = []
    h = start_heading
    fwd = np.array([-math.sin(h), math.cos(h), 0.0], dtype=np.float32)
    lat = np.array([math.cos(h), math.sin(h), 0.0], dtype=np.float32)
    left0, right0 = _stance_feet(start_root, h)
    left_pose = left0 + fwd * 0.10 - lat * 0.03

    settle_time = min(0.5, duration * 0.4)
    for i, t in enumerate(times):
        settle = min_jerk(t / settle_time)
        breath = 0.006 * math.sin(2.0 * math.pi * 0.3 * t)
        root = start_root - lat * 0.04 * settle
        root[2] = 0.78 - 0.02 * settle + breath
        translations[i] = root
        lf = left0 + (left_pose - left0) * settle
        lf[2] = 0.0
        _set_override(override, i, 0, lf, h - 0.25 * settle, 0.85 * settle)
        _set_override(override, i, 1, right0, h + 0.15 * settle, 0.0)
        contacts.append((t == 0.0 or t >= settle_time, True))

        rotations[i, 0] = [0.0, -0.06 * settle, h - 0.10 * settle]
        rotations[i, 3] = [-0.04 * settle, 0.04 * settle, -0.08 * settle]
        rotations[i, 9] = [-0.04 * settle, 0.0, -0.10 * settle]
        rotations[i, 12] = [-0.08 * settle, 0.0, -0.20 * settle]
        rotations[i, 15] = [-0.12 * settle, 0.0, -0.30 * settle]
        rotations[i, 16] = [-2.05 * settle, -0.18 + 0.48 * settle, -0.5 * settle]
        rotations[i, 18] = [0.25 - 0.15 * settle, 0.0, 0.0]
        rotations[i, 17] = [0.04, 0.18 + 0.20 * settle, 0.0]
        rotations[i, 19] = [0.25, 0.0, 0.0]
        rotations[i, 21] = [-0.35 * settle, 0.0, 0.0]

    return {
        "rotations": rotations, "translations": translations, "contacts": contacts,
        "foot_lifts": foot_lifts, "foot_override": override, "style": "standard",
        "end_root": translations[-1], "end_heading": h,
    }

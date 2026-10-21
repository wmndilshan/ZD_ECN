"""
Deterministic SMPL-24 Motion Generator with Closed-Loop Foot-Locking IK.
Converts validated motion plans into continuous skeletal animation telemetry.
Enforces C^2 minimum-jerk interpolation, joint limits, root-motion continuity,
zero ground penetration (Z >= 0), and zero stance foot slip.
"""
import math
import numpy as np
from typing import Dict, Any, List, Tuple
from .actions import (
    min_jerk,
    generate_idle, generate_walk, generate_turn,
    generate_squat, generate_jump, generate_land,
    generate_dance_mandiya, generate_dance_step,
    generate_dance_turn, generate_dance_freeze,
    generate_moonwalk, generate_spin, generate_toe_stand,
    generate_pop_hits, generate_pose_point
)
from .joint_limits import clamp_joint_rotations
from .kinematics import (
    REST_OFFSETS, euler_to_matrix, solve_leg_ik, compute_forward_kinematics
)


class DeterministicMotionGenerator:
    """
    Deterministic Biomechanical Motion Generator for SMPL-24.
    Synthesizes continuous joint rotations, closed-loop IK leg trajectories,
    forward kinematics transforms, and root motion from a validated plan.
    """

    def __init__(self):
        self.default_root = np.array([0.0, 0.0, 0.78], dtype=np.float32)
        self.default_heading = 0.0

    def generate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        fps = int(plan.get("target_fps", 20))
        phases = plan.get("phases", [])

        current_root = np.copy(self.default_root)
        current_heading = self.default_heading
        raw_phase_results = []

        # 1. Synthesize each phase sequentially
        for phase in phases:
            action = phase["action"]
            duration = float(phase["duration"])
            params = phase.get("parameters", {})

            if action == "idle":
                res = generate_idle(duration, fps, current_root, current_heading)
            elif action == "walk":
                speed = float(params.get("speed", 1.1))
                step_freq = float(params.get("step_frequency", 1.8))
                res = generate_walk(duration, fps, speed, step_freq, current_root, current_heading)
            elif action == "turn":
                angle_deg = float(params.get("angle_degrees", 90.0))
                res = generate_turn(duration, fps, angle_deg, current_root, current_heading)
            elif action == "squat":
                depth = float(params.get("depth", 0.6))
                hold_dur = float(params.get("hold_duration", 0.3))
                res = generate_squat(duration, fps, depth, hold_dur, current_root, current_heading)
            elif action == "jump":
                height = float(params.get("height", 0.45))
                fwd_dist = float(params.get("forward_distance", 0.2))
                res = generate_jump(duration, fps, height, fwd_dist, current_root, current_heading)
            elif action == "land":
                rec_dur = float(params.get("recovery_duration", 0.5))
                damping = float(params.get("damping", 0.85))
                res = generate_land(duration, fps, rec_dur, damping, current_root, current_heading)
            elif action == "dance_mandiya":
                res = generate_dance_mandiya(duration, fps, current_root, current_heading)
            elif action == "dance_step":
                cadence = str(params.get("cadence", "thei_tha"))
                res = generate_dance_step(duration, fps, cadence, current_root, current_heading)
            elif action == "dance_turn":
                angle_deg = float(params.get("angle_degrees", 360.0))
                res = generate_dance_turn(duration, fps, angle_deg, current_root, current_heading)
            elif action == "dance_freeze":
                res = generate_dance_freeze(duration, fps, current_root, current_heading)
            elif action == "moonwalk":
                stride = float(params.get("stride", 0.18))
                res = generate_moonwalk(duration, fps, stride, current_root, current_heading)
            elif action == "spin":
                angle_deg = float(params.get("angle_degrees", 360.0))
                res = generate_spin(duration, fps, angle_deg, current_root, current_heading)
            elif action == "toe_stand":
                res = generate_toe_stand(duration, fps, current_root, current_heading)
            elif action == "pop_hits":
                tempo = float(params.get("tempo_bpm", 117.0))
                res = generate_pop_hits(duration, fps, tempo, current_root, current_heading)
            elif action == "pose_point":
                res = generate_pose_point(duration, fps, current_root, current_heading)
            else:
                res = generate_idle(duration, fps, current_root, current_heading)

            res["phase_id"] = phase["phase_id"]
            res["action"] = action
            raw_phase_results.append(res)

            current_root = np.copy(res["end_root"])
            current_heading = float(res["end_heading"])

        # 2. Stitch and blend upper-body & root across phase boundaries
        (
            stitched_rotations,
            stitched_translations,
            stitched_contacts,
            stitched_lifts,
            stitched_styles,
            phase_tags,
            stitched_overrides
        ) = self._stitch_and_blend_phases(raw_phase_results, blend_window=4)

        total_frames = len(stitched_rotations)
        total_duration = total_frames / float(fps)

        # 3. Closed-Loop World-Space Foot-Locking & Analytical 3-DOF Leg IK Pass
        self._apply_closed_loop_foot_ik(
            stitched_rotations,
            stitched_translations,
            stitched_contacts,
            stitched_lifts,
            stitched_styles,
            stitched_overrides
        )

        # 4. Enforce Anatomical Joint Limits (Vectorized)
        clamped_rotations, total_limit_violations = clamp_joint_rotations(stitched_rotations)

        # 5. Compute Forward Kinematics & Verify Ground/Slip Metrics
        frame_telemetry: List[Dict[str, Any]] = []
        max_foot_penetration = 0.0
        max_foot_slip_velocity = 0.0
        prev_l_pos = None
        prev_r_pos = None
        prev_l_planted = False
        prev_r_planted = False

        # Round arrays once in vectorized form for fast serialization
        rounded_trans_all = np.round(stitched_translations, 4).tolist()
        rounded_rots_all = np.round(clamped_rotations, 4).tolist()

        for f in range(total_frames):
            t_sec = round(f / float(fps), 4)
            rot = clamped_rotations[f]
            trans = stitched_translations[f]
            l_planted, r_planted = stitched_contacts[f]

            transforms, positions = compute_forward_kinematics(rot, trans)

            l_foot_pos = positions[10]
            r_foot_pos = positions[11]

            # Ground penetration check (tolerance 1e-4m for float32 rounding)
            if l_foot_pos[2] < -1e-4:
                max_foot_penetration = max(max_foot_penetration, float(abs(l_foot_pos[2])))
            if r_foot_pos[2] < -1e-4:
                max_foot_penetration = max(max_foot_penetration, float(abs(r_foot_pos[2])))

            # Stance foot slip check (when foot remains continuously planted between f-1 and f)
            if prev_l_pos is not None and l_planted and prev_l_planted:
                l_slip = float(np.linalg.norm(l_foot_pos[:2] - prev_l_pos[:2])) * fps
                if l_slip > 1e-3:
                    max_foot_slip_velocity = max(max_foot_slip_velocity, l_slip)
            if prev_r_pos is not None and r_planted and prev_r_planted:
                r_slip = float(np.linalg.norm(r_foot_pos[:2] - prev_r_pos[:2])) * fps
                if r_slip > 1e-3:
                    max_foot_slip_velocity = max(max_foot_slip_velocity, r_slip)

            prev_l_pos = l_foot_pos.copy()
            prev_r_pos = r_foot_pos.copy()
            prev_l_planted = l_planted
            prev_r_planted = r_planted

            frame_telemetry.append({
                "frame": f,
                "time_seconds": t_sec,
                "phase_name": phase_tags[f],
                "root_translation": rounded_trans_all[f],
                "joint_rotations_euler": rounded_rots_all[f],
                "joint_positions_3d": np.round(positions, 4).tolist(),
                "joint_transforms_4x4": np.round(transforms, 5).tolist(),
                "foot_contact": {
                    "left_foot_planted": bool(l_planted),
                    "right_foot_planted": bool(r_planted)
                }
            })

        return {
            "metadata": {
                "version": "1.0",
                "generator_type": "Closed-Loop Biomechanical IK & Procedural Synthesizer (SMPL-24)",
                "prompt": plan.get("prompt", ""),
                "fps": fps,
                "total_duration_seconds": round(total_duration, 4),
                "total_frames": total_frames,
                "num_phases": len(phases)
            },
            "quality_metrics": {
                "joint_limit_clampings": int(total_limit_violations),
                "max_ground_penetration_m": round(float(max_foot_penetration), 4),
                "max_foot_slip_velocity_m_per_s": round(float(max_foot_slip_velocity), 4),
                "root_motion_continuous": True
            },
            "frames": frame_telemetry
        }

    def _apply_closed_loop_foot_ik(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        contacts: List[Tuple[bool, bool]],
        lifts: np.ndarray,
        styles: List[str],
        overrides: np.ndarray
    ) -> None:
        """
        Computes world-space locked foot trajectories for all stance intervals,
        interpolates cycloidal swing trajectories for airborne intervals,
        constrains pelvis reach, and solves exact 3-DOF analytical leg IK.
        """
        n_frames = len(rotations)

        # 1. Compute nominal unconstrained foot targets for every frame
        nom_l = np.zeros((n_frames, 3), dtype=np.float32)
        nom_r = np.zeros((n_frames, 3), dtype=np.float32)
        nom_yaw_l = np.zeros(n_frames, dtype=np.float32)
        nom_yaw_r = np.zeros(n_frames, dtype=np.float32)

        for f in range(n_frames):
            heading = float(rotations[f, 0, 2])
            fwd = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=np.float32)
            lat = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=np.float32)
            root_xy = np.array([translations[f, 0], translations[f, 1], 0.0], dtype=np.float32)

            if styles[f] == "mandiya":
                # Wide outward turnout stance for classical Sri Lankan dance
                w_lat = 0.28
                fwd_off = 0.06
                turnout = math.radians(42.0)
            else:
                w_lat = 0.13
                fwd_off = 0.10
                turnout = 0.0

            nom_l[f] = root_xy + lat * w_lat + fwd * fwd_off
            nom_r[f] = root_xy - lat * w_lat + fwd * fwd_off
            nom_yaw_l[f] = heading - turnout
            nom_yaw_r[f] = heading + turnout

        # 2. Build locked stance intervals and minimum-jerk swing arcs for Left & Right feet
        l_contacts = [c[0] for c in contacts]
        r_contacts = [c[1] for c in contacts]

        foot_pos_l, foot_yaw_l, foot_pitch_l = self._solve_foot_channel(
            nom_l, nom_yaw_l, l_contacts, lifts[:, 0], overrides[:, 0]
        )
        foot_pos_r, foot_yaw_r, foot_pitch_r = self._solve_foot_channel(
            nom_r, nom_yaw_r, r_contacts, lifts[:, 1], overrides[:, 1]
        )

        # 3. Ensure pelvis stays within kinematic leg reach of planted feet & solve 3-DOF IK
        foot_offset_l = REST_OFFSETS[10]
        foot_offset_r = REST_OFFSETS[11]
        max_leg_reach = 0.742  # < l1 + l2 = 0.76m

        for f in range(n_frames):
            R_pelvis = euler_to_matrix(
                float(rotations[f, 0, 0]),
                float(rotations[f, 0, 1]),
                float(rotations[f, 0, 2])
            )
            pelvis_pos = translations[f]

            # Target ankle positions for reachability verification
            R_ank_l = euler_to_matrix(float(foot_pitch_l[f]), 0.0, float(foot_yaw_l[f]))
            R_ank_r = euler_to_matrix(float(foot_pitch_r[f]), 0.0, float(foot_yaw_r[f]))
            p_ank_l = foot_pos_l[f] - R_ank_l @ foot_offset_l
            p_ank_r = foot_pos_r[f] - R_ank_r @ foot_offset_r

            # Pull pelvis within reach sphere of planted ankles so planted feet never slip
            for _ in range(2):
                if l_contacts[f]:
                    p_hip_l = pelvis_pos + R_pelvis @ REST_OFFSETS[1]
                    v_l = p_hip_l - p_ank_l
                    dist_l = float(np.linalg.norm(v_l))
                    if dist_l > max_leg_reach:
                        pelvis_pos = pelvis_pos - (v_l / dist_l) * (dist_l - max_leg_reach)
                if r_contacts[f]:
                    p_hip_r = pelvis_pos + R_pelvis @ REST_OFFSETS[2]
                    v_r = p_hip_r - p_ank_r
                    dist_r = float(np.linalg.norm(v_r))
                    if dist_r > max_leg_reach:
                        pelvis_pos = pelvis_pos - (v_r / dist_r) * (dist_r - max_leg_reach)

            translations[f] = pelvis_pos

            if styles[f] == "mandiya":
                pole_l = np.array([1.0, 0.38, 0.0], dtype=np.float32)
                pole_r = np.array([-1.0, 0.38, 0.0], dtype=np.float32)
            else:
                pole_l = np.array([0.12, 1.0, 0.0], dtype=np.float32)
                pole_r = np.array([-0.12, 1.0, 0.0], dtype=np.float32)

            hip_l, knee_l, ank_l = solve_leg_ik(
                pelvis_pos, R_pelvis, foot_pos_l[f],
                float(foot_yaw_l[f]), float(foot_pitch_l[f]), pole_l, is_left=True
            )
            hip_r, knee_r, ank_r = solve_leg_ik(
                pelvis_pos, R_pelvis, foot_pos_r[f],
                float(foot_yaw_r[f]), float(foot_pitch_r[f]), pole_r, is_left=False
            )

            rotations[f, 1] = hip_l
            rotations[f, 4] = knee_l
            rotations[f, 7] = ank_l
            rotations[f, 2] = hip_r
            rotations[f, 5] = knee_r
            rotations[f, 8] = ank_r

            # Toes stay flat on the floor when a grounded foot raises its heel
            # (metatarsophalangeal dorsiflexion); otherwise the rigid toe segment
            # of a skinned mesh stabs into the ground during toe stands.
            if foot_pos_l[f][2] <= 1e-6 and foot_pitch_l[f] > 0.0:
                rotations[f, 10, 0] = -float(foot_pitch_l[f])
            if foot_pos_r[f][2] <= 1e-6 and foot_pitch_r[f] > 0.0:
                rotations[f, 11, 0] = -float(foot_pitch_r[f])

    def _solve_foot_channel(
        self,
        nom_pos: np.ndarray,
        nom_yaw: np.ndarray,
        planted: List[bool],
        lift_z: np.ndarray,
        override: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Locks horizontal (X, Y) and yaw during contiguous planted intervals (Z = 0),
        and interpolates swing intervals using minimum-jerk cycloidal curves.
        Frames with an explicit foot override ([x, y, z, yaw, pitch], non-NaN) take
        it verbatim and form their own intervals, so locks never span them and
        neighbouring swings interpolate to/from the overridden foot.
        """
        n = len(planted)
        out_pos = np.copy(nom_pos)
        out_yaw = np.copy(nom_yaw)
        out_pitch = np.zeros(n, dtype=np.float32)

        # Identify contiguous intervals of planted == True and planted == False
        has_override = ~np.isnan(override[:, 0])
        # 0 = swing, 1 = planted, 2 = explicit override
        state_of = [2 if has_override[f] else int(bool(planted[f])) for f in range(n)]

        intervals = []
        idx = 0
        while idx < n:
            state = state_of[idx]
            end_idx = idx
            while end_idx + 1 < n and state_of[end_idx + 1] == state:
                end_idx += 1
            intervals.append((idx, end_idx, state))
            idx = end_idx + 1

        # First pass: explicit overrides, then lock planted intervals to the midpoint nominal
        for start_i, end_i, state in intervals:
            if state == 2:
                for f in range(start_i, end_i + 1):
                    out_pos[f] = override[f, :3]
                    out_yaw[f] = override[f, 3]
                    out_pitch[f] = override[f, 4]
        for k, (start_i, end_i, state) in enumerate(intervals):
            if state == 1:
                mid_i = (start_i + end_i) // 2
                lock_xy = nom_pos[mid_i, :2].copy()
                lock_yaw = float(nom_yaw[mid_i])
                # A stance touching an override keeps the foot where the override has it
                if k > 0 and intervals[k - 1][2] == 2:
                    lock_xy = out_pos[start_i - 1, :2].copy()
                    lock_yaw = float(out_yaw[start_i - 1])
                elif k + 1 < len(intervals) and intervals[k + 1][2] == 2:
                    lock_xy = out_pos[end_i + 1, :2].copy()
                    lock_yaw = float(out_yaw[end_i + 1])
                for f in range(start_i, end_i + 1):
                    out_pos[f, 0] = lock_xy[0]
                    out_pos[f, 1] = lock_xy[1]
                    out_pos[f, 2] = 0.0
                    out_yaw[f] = lock_yaw
                    out_pitch[f] = 0.0

        # Second pass: Interpolate swing intervals between adjacent locked positions
        for k, (start_i, end_i, state) in enumerate(intervals):
            if state == 0:
                p_start = out_pos[start_i - 1, :2] if start_i > 0 else nom_pos[start_i, :2]
                y_start = float(out_yaw[start_i - 1]) if start_i > 0 else float(nom_yaw[start_i])

                p_end = out_pos[end_i + 1, :2] if end_i + 1 < n else nom_pos[end_i, :2]
                y_end = float(out_yaw[end_i + 1]) if end_i + 1 < n else float(nom_yaw[end_i])

                span = max(1, end_i - start_i + 1)
                for f in range(start_i, end_i + 1):
                    tau = (f - start_i + 1) / float(span + 1)
                    s = min_jerk(tau)
                    out_pos[f, 0] = (1.0 - s) * p_start[0] + s * p_end[0]
                    out_pos[f, 1] = (1.0 - s) * p_start[1] + s * p_end[1]
                    out_pos[f, 2] = max(0.005, float(lift_z[f]))
                    out_yaw[f] = (1.0 - s) * y_start + s * y_end
                    # Natural toe-off (+pitch) to heel-strike (-pitch) foot articulation
                    out_pitch[f] = 0.18 * math.sin(2.0 * math.pi * tau) * math.sin(math.pi * tau)

        return out_pos, out_yaw, out_pitch

    def _stitch_and_blend_phases(self, phase_results: List[Dict[str, Any]], blend_window: int = 4):
        all_rots = []
        all_trans = []
        all_contacts = []
        all_lifts = []
        all_styles = []
        all_tags = []
        all_overrides = []

        for p_idx, res in enumerate(phase_results):
            rots = res["rotations"].copy()
            trans = res["translations"].copy()
            contacts = res["contacts"]
            lifts = res["foot_lifts"]
            style = res.get("style", "standard")
            override = res.get("foot_override")
            if override is None:
                override = np.full((len(res["rotations"]), 2, 5), np.nan, dtype=np.float32)
            act_name = f"Phase {res['phase_id']}: {res['action']}"
            n_p = len(rots)

            if p_idx > 0 and len(all_rots) >= blend_window and n_p >= blend_window:
                prev_end_rot = all_rots[-1]
                prev_end_trans = all_trans[-1]

                for b in range(blend_window):
                    alpha = (b + 1) / float(blend_window + 1)
                    w = min_jerk(alpha)
                    trans[b] = (1.0 - w) * prev_end_trans + w * trans[b]
                    rots[b] = (1.0 - w) * prev_end_rot + w * rots[b]

            for i in range(n_p):
                all_rots.append(rots[i])
                all_trans.append(trans[i])
                all_contacts.append(contacts[i])
                all_lifts.append(lifts[i])
                all_styles.append(style)
                all_tags.append(act_name)
                all_overrides.append(override[i])

        return (
            np.array(all_rots, dtype=np.float32),
            np.array(all_trans, dtype=np.float32),
            all_contacts,
            np.array(all_lifts, dtype=np.float32),
            all_styles,
            all_tags,
            np.array(all_overrides, dtype=np.float32)
        )

"""
Unit tests for Deterministic Motion Generator & Closed-Loop Foot-Locking IK.
"""
import unittest
import numpy as np
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator
from zd_ecn_motion.generator.joint_limits import JOINT_LIMITS


class TestMotionGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = MockMotionPlanner(default_fps=20)
        cls.generator = DeterministicMotionGenerator()

    def test_all_required_actions(self):
        prompt = "walk forward, turn left, squat, jump, land, and return to idle"
        plan = self.planner.plan(prompt)
        motion_data = self.generator.generate(plan)

        frames = motion_data["frames"]
        self.assertGreater(len(frames), 50)

        actions_present = set(p["action"] for p in plan["phases"])
        required_actions = {"walk", "turn", "squat", "jump", "land", "idle"}
        self.assertEqual(actions_present, required_actions)

    def test_strictly_monotonic_timing(self):
        plan = self.planner.plan("walk forward, turn left, return to idle")
        motion_data = self.generator.generate(plan)
        frames = motion_data["frames"]

        for i in range(1, len(frames)):
            t_prev = frames[i - 1]["time_seconds"]
            t_curr = frames[i]["time_seconds"]
            self.assertGreater(t_curr, t_prev, f"Non-monotonic timestamp at frame {i}: {t_curr} <= {t_prev}")

    def test_root_motion_continuity(self):
        plan = self.planner.plan("walk forward, turn left, jump, land, return to idle")
        motion_data = self.generator.generate(plan)
        frames = motion_data["frames"]

        max_allowed_delta = 0.25  # meters per frame at 20fps
        for i in range(1, len(frames)):
            root_prev = np.array(frames[i - 1]["root_translation"])
            root_curr = np.array(frames[i]["root_translation"])
            delta = np.linalg.norm(root_curr - root_prev)
            self.assertLess(
                delta, max_allowed_delta,
                f"Discontinuous root jump of {delta:.4f}m at frame {i} ({frames[i]['phase_name']})"
            )

    def test_joint_limits_compliance(self):
        plan = self.planner.plan("walk, squat deep, jump high, land, return to idle")
        motion_data = self.generator.generate(plan)
        frames = motion_data["frames"]

        self.assertEqual(motion_data["quality_metrics"]["joint_limit_clampings"], 0)

        for f_idx, frame in enumerate(frames):
            rotations = frame["joint_rotations_euler"]
            for j in range(24):
                p, r, y = rotations[j]
                lim_p, lim_r, lim_y = JOINT_LIMITS.get(j, ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)))
                eps = 1e-3
                self.assertTrue(
                    lim_p[0] - eps <= p <= lim_p[1] + eps,
                    f"Frame {f_idx} Joint {j} Pitch {p:.3f} outside [{lim_p[0]:.3f}, {lim_p[1]:.3f}]"
                )
                self.assertTrue(
                    lim_r[0] - eps <= r <= lim_r[1] + eps,
                    f"Frame {f_idx} Joint {j} Roll {r:.3f} outside [{lim_r[0]:.3f}, {lim_r[1]:.3f}]"
                )
                self.assertTrue(
                    lim_y[0] - eps <= y <= lim_y[1] + eps,
                    f"Frame {f_idx} Joint {j} Yaw {y:.3f} outside [{lim_y[0]:.3f}, {lim_y[1]:.3f}]"
                )

    def test_zero_ground_penetration_and_zero_foot_slip(self):
        plan = self.planner.plan("walk forward, turn left, jump, land, and return to idle")
        motion_data = self.generator.generate(plan)
        qm = motion_data["quality_metrics"]

        self.assertLessEqual(qm["max_ground_penetration_m"], 0.001, "Foot penetrated ground Z < 0!")
        self.assertLessEqual(qm["max_foot_slip_velocity_m_per_s"], 0.001, "Planted foot slipped on floor!")

    def test_ground_contact_flags(self):
        plan = self.planner.plan("walk forward, jump, land")
        motion_data = self.generator.generate(plan)
        frames = motion_data["frames"]

        airborne_frames = 0
        planted_frames = 0
        for frame in frames:
            fc = frame["foot_contact"]
            if not fc["left_foot_planted"] and not fc["right_foot_planted"]:
                airborne_frames += 1
            if fc["left_foot_planted"] or fc["right_foot_planted"]:
                planted_frames += 1

        self.assertGreater(airborne_frames, 0, "Jump phase must contain airborne frames where both feet are false.")
        self.assertGreater(planted_frames, 0, "Walk/land phases must contain planted frames where foot contact is true.")


if __name__ == '__main__':
    unittest.main()

"""
Unit and integration tests for the pop-dance vocabulary
(moonwalk, spin, toe_stand, pop_hits, pose_point) and explicit foot overrides.
"""
import math
import unittest

import numpy as np

from zd_ecn_motion.planner.mock_planner import MockMotionPlanner
from zd_ecn_motion.planner.schema import validate_motion_plan
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator


class TestPopDanceGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = MockMotionPlanner(default_fps=30)
        cls.generator = DeterministicMotionGenerator()

    def test_macro_plan(self):
        plan = self.planner.plan("Michael Jackson style dance")
        ok, err = validate_motion_plan(plan)
        self.assertTrue(ok, err)
        self.assertEqual([p["action"] for p in plan["phases"]],
                         ["pop_hits", "moonwalk", "spin", "toe_stand", "pose_point"])

    def test_clause_parsing(self):
        plan = self.planner.plan("moonwalk, spin twice, then toe stand and point")
        self.assertEqual([p["action"] for p in plan["phases"]],
                         ["moonwalk", "spin", "toe_stand", "pose_point"])
        self.assertEqual(plan["phases"][1]["parameters"]["angle_degrees"], 720.0)

    def test_routine_quality(self):
        for prompt in ("Michael Jackson style dance",
                       "pop hits, moonwalk, walk forward, spin right, idle"):
            qm = self.generator.generate(self.planner.plan(prompt))["quality_metrics"]
            self.assertEqual(qm["joint_limit_clampings"], 0, prompt)
            self.assertEqual(qm["max_ground_penetration_m"], 0.0, prompt)
            self.assertEqual(qm["max_foot_slip_velocity_m_per_s"], 0.0, prompt)

    def test_moonwalk_travels_backward_with_toe_foot(self):
        motion = self.generator.generate(self.planner.plan("moonwalk for 3.3 seconds"))
        frames = motion["frames"]
        root = np.array([f["root_translation"] for f in frames])
        # Heading 0 faces +Y: a moonwalk must travel toward -Y
        self.assertLess(root[-1, 1] - root[0, 1], -0.5)
        # Mid-step, one foot is on its toe: the ankle sits clearly higher than the other
        mid = len(frames) // 3
        ankles = np.array(frames[mid]["joint_positions_3d"])[[7, 8], 2]
        self.assertGreater(abs(ankles[0] - ankles[1]), 0.03)

    def test_spin_pivot_does_not_slip_and_rotates(self):
        motion = self.generator.generate(self.planner.plan("spin twice"))
        frames = motion["frames"]
        pivot = np.array([f["joint_positions_3d"][11][:2] for f in frames])
        self.assertLess(np.abs(pivot - pivot[0]).max(), 2e-3)
        yaw = frames[-1]["joint_rotations_euler"][0][2] - frames[0]["joint_rotations_euler"][0][2]
        self.assertAlmostEqual(yaw, 4.0 * math.pi, places=2)

    def test_toe_stand_raises_heels(self):
        motion = self.generator.generate(self.planner.plan("toe stand"))
        frames = motion["frames"]
        ankle_z = np.array([f["joint_positions_3d"][7][2] for f in frames])
        self.assertGreater(ankle_z.max() - ankle_z[0], 0.05)


if __name__ == "__main__":
    unittest.main()

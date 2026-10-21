"""
Automated unit and integration tests for Traditional Dance generation.
Verifies outward Mandiya knee turnout, zero ground penetration, and zero foot slip.
"""
import unittest
import json
import numpy as np
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner
from zd_ecn_motion.planner.schema import validate_motion_plan
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator
from zd_ecn_motion.native.native_bridge import NativeMotionPlaybackBridge


class TestTraditionalDanceGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = MockMotionPlanner(default_fps=20)
        cls.generator = DeterministicMotionGenerator()
        cls.bridge = NativeMotionPlaybackBridge()

    def test_traditional_dance_planning(self):
        prompt = "traditional dancer generation again"
        plan = self.planner.plan(prompt)

        is_valid, err = validate_motion_plan(plan)
        self.assertTrue(is_valid, f"Plan failed validation: {err}")

        actions = [p["action"] for p in plan["phases"]]
        expected = ["dance_mandiya", "dance_step", "dance_turn", "dance_freeze"]
        self.assertEqual(actions, expected)

    def test_traditional_dance_kinematics_and_zero_slip(self):
        plan = self.planner.plan("perform traditional dance in mandiya stance")
        motion_data = self.generator.generate(plan)

        frames = motion_data["frames"]
        self.assertGreaterEqual(len(frames), 100)

        qm = motion_data["quality_metrics"]
        self.assertEqual(qm["joint_limit_clampings"], 0, "Traditional dance exceeded anatomical joint limits!")
        self.assertLessEqual(qm["max_ground_penetration_m"], 0.001, "Traditional dance foot penetrated ground!")
        self.assertLessEqual(qm["max_foot_slip_velocity_m_per_s"], 0.001, "Traditional dance planted foot slipped!")

        mandiya_frames = [f for f in frames if "dance_mandiya" in f["phase_name"]]
        self.assertTrue(len(mandiya_frames) > 0)
        for f in mandiya_frames:
            root_z = f["root_translation"][2]
            self.assertLess(root_z, 0.85)
            self.assertGreater(root_z, 0.58)

            l_knee_pitch = f["joint_rotations_euler"][4][0]
            r_knee_pitch = f["joint_rotations_euler"][5][0]
            self.assertGreater(l_knee_pitch, 0.5)
            self.assertGreater(r_knee_pitch, 0.5)

            # Verify lateral knee spread (diamond Mandiya geometry: knees wider than hips)
            pos = np.array(f["joint_positions_3d"])
            hip_width = np.linalg.norm(pos[1, :2] - pos[2, :2])
            knee_width = np.linalg.norm(pos[4, :2] - pos[5, :2])
            self.assertGreater(knee_width, hip_width * 1.8, "Knees must flare outward wider than hips in Mandiya!")

    def test_native_playback_stream_dance(self):
        plan = self.planner.plan("traditional dance")
        motion_data = self.generator.generate(plan)

        json_str = json.dumps(motion_data)
        loaded = self.bridge.load_json(json_str)
        self.assertTrue(loaded, "Failed to load traditional dance JSON in native C++ engine!")

        dur = self.bridge.get_duration()
        self.assertGreater(dur, 8.0)

        self.bridge.seek(1.5)
        t_mandiya = self.bridge.get_current_transforms()
        self.assertEqual(t_mandiya.shape, (24, 4, 4))

        self.bridge.seek(4.5)
        t_step = self.bridge.get_current_transforms()

        diff = np.linalg.norm(t_step[0] - t_mandiya[0])
        self.assertGreater(diff, 0.01, "Transforms did not vary between dance phases")


if __name__ == '__main__':
    unittest.main()

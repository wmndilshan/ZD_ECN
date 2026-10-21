"""
Unit tests for Native C++ Playback Engine via ctypes bridge.
"""
import unittest
import json
import numpy as np
from zd_ecn_motion.native.native_bridge import NativeMotionPlaybackBridge
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator


class TestNativePlaybackEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = NativeMotionPlaybackBridge()
        cls.planner = MockMotionPlanner(default_fps=20)
        cls.generator = DeterministicMotionGenerator()

        # Generate a small sample motion
        plan = cls.planner.plan("walk forward, turn left, idle")
        cls.motion_data = cls.generator.generate(plan)
        cls.json_str = json.dumps(cls.motion_data)

    def test_load_and_duration(self):
        loaded = self.bridge.load_json(self.json_str)
        self.assertTrue(loaded, "Native engine failed to load motion JSON!")

        expected_frames = self.motion_data["metadata"]["total_frames"]
        expected_dur = self.motion_data["frames"][-1]["time_seconds"]

        self.assertEqual(self.bridge.get_frame_count(), expected_frames)
        self.assertAlmostEqual(self.bridge.get_duration(), expected_dur, delta=0.05)

    def test_timeline_advance_and_interpolation(self):
        self.bridge.load_json(self.json_str)
        self.bridge.seek(0.0)
        self.assertAlmostEqual(self.bridge.get_time(), 0.0, places=4)

        t_0_transforms = self.bridge.get_current_transforms()
        self.assertEqual(t_0_transforms.shape, (24, 4, 4))

        # Advance by 0.25s
        self.bridge.update(0.25)
        self.assertAlmostEqual(self.bridge.get_time(), 0.25, delta=1e-3)

        t_025_transforms = self.bridge.get_current_transforms()
        # Verify that transforms have smoothly updated
        diff = np.linalg.norm(t_025_transforms[0] - t_0_transforms[0])
        self.assertGreater(diff, 1e-4, "Root transform did not change after time advance!")

    def test_seek_and_boundary_loop(self):
        self.bridge.load_json(self.json_str)
        dur = self.bridge.get_duration()

        # Seek to halfway
        self.bridge.seek(dur * 0.5)
        self.assertAlmostEqual(self.bridge.get_time(), dur * 0.5, delta=1e-3)

        # Update past duration (loop test)
        self.bridge.update(dur * 0.8) # should wrap around
        wrapped_time = self.bridge.get_time()
        self.assertLess(wrapped_time, dur)
        self.assertGreaterEqual(wrapped_time, 0.0)


if __name__ == '__main__':
    unittest.main()

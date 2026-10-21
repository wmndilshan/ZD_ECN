"""
End-to-End Integration Test for ZD-ECN Text-to-Motion Pipeline.
Verifies: Prompt -> Motion Plan -> Kinematic Frames -> Forward Kinematics -> Native C++ Engine.
"""
import unittest
import json
import numpy as np
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner
from zd_ecn_motion.planner.schema import validate_motion_plan
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator
from zd_ecn_motion.native.native_bridge import NativeMotionPlaybackBridge


class TestEndToEndPipeline(unittest.TestCase):
    def test_full_pipeline_vertical_slice(self):
        # 1. Natural language user prompt
        prompt = "Walk forward, turn left, jump, land, and return to idle"

        # 2. LLM / Mock Motion Planner
        planner = MockMotionPlanner(default_fps=20)
        plan = planner.plan(prompt)

        # Verify plan schema
        is_valid, err = validate_motion_plan(plan)
        self.assertTrue(is_valid, f"Generated motion plan failed validation: {err}")
        self.assertEqual(len(plan["phases"]), 5)

        # 3. Deterministic Kinematic Generator
        generator = DeterministicMotionGenerator()
        motion_data = generator.generate(plan)

        frames = motion_data["frames"]
        self.assertGreater(len(frames), 100)
        self.assertTrue(motion_data["quality_metrics"]["root_motion_continuous"])

        # 4. JSON Serialization & Native C++ Playback Ingestion
        json_payload = json.dumps(motion_data)
        bridge = NativeMotionPlaybackBridge()
        loaded = bridge.load_json(json_payload)
        self.assertTrue(loaded, "Failed to load generated JSON payload into native C++ engine!")

        # 5. Playback Timeline Simulation
        total_duration = bridge.get_duration()
        self.assertGreater(total_duration, 5.0)

        # Sample transforms at key phase timestamps:
        # Phase 1: Walk (t = 1.0s)
        bridge.seek(1.0)
        t_walk = bridge.get_current_transforms()

        # Phase 2: Turn (t = 2.5s)
        bridge.seek(2.5)
        t_turn = bridge.get_current_transforms()

        # Phase 3: Jump (t = 3.5s)
        bridge.seek(3.5)
        t_jump = bridge.get_current_transforms()

        # Phase 4: Land (t = 4.4s)
        bridge.seek(4.4)
        t_land = bridge.get_current_transforms()

        # Phase 5: Idle (t = 5.5s)
        bridge.seek(5.5)
        t_idle = bridge.get_current_transforms()

        # Assert that transforms dynamically change across distinct actions
        self.assertGreater(np.linalg.norm(t_turn[0] - t_walk[0]), 0.1, "Root should turn between walk and turn")
        self.assertGreater(np.linalg.norm(t_jump[0, 2, 3] - t_walk[0, 2, 3]), 0.1, "Root Z elevation should rise in jump")
        self.assertGreater(np.linalg.norm(t_jump[4] - t_land[4]), 0.1, "Knee transforms should differ between jump and land")

        print("\n[End-to-End Test] Prompt successfully executed full pipeline vertical slice!")
        print(f"  Prompt: '{prompt}'")
        print(f"  Phases: {len(plan['phases'])} | Frames: {len(frames)} | Native Duration: {total_duration:.2f}s")


if __name__ == '__main__':
    unittest.main()

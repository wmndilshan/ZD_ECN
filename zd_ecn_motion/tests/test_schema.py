"""
Unit tests for Motion Plan Schema Validation.
"""
import unittest
from zd_ecn_motion.planner.schema import validate_motion_plan, SUPPORTED_ACTIONS


class TestMotionPlanSchema(unittest.TestCase):
    def test_valid_plan(self):
        plan = {
            "version": "1.0",
            "prompt": "walk and turn",
            "target_fps": 20,
            "total_duration": 3.0,
            "phases": [
                {"phase_id": 1, "action": "walk", "duration": 2.0, "parameters": {"speed": 1.2}},
                {"phase_id": 2, "action": "turn", "duration": 1.0, "parameters": {"angle_degrees": 90.0}}
            ]
        }
        is_valid, err = validate_motion_plan(plan)
        self.assertTrue(is_valid, f"Valid plan failed: {err}")

    def test_unsupported_action(self):
        plan = {
            "version": "1.0",
            "prompt": "fly away",
            "target_fps": 20,
            "total_duration": 2.0,
            "phases": [
                {"phase_id": 1, "action": "fly", "duration": 2.0, "parameters": {}}
            ]
        }
        is_valid, err = validate_motion_plan(plan)
        self.assertFalse(is_valid)
        self.assertIn("Schema validation error", err)

    def test_duration_sum_mismatch(self):
        plan = {
            "version": "1.0",
            "prompt": "walk",
            "target_fps": 20,
            "total_duration": 5.0, # Mismatch: phases sum to 2.0
            "phases": [
                {"phase_id": 1, "action": "walk", "duration": 2.0, "parameters": {"speed": 1.0}}
            ]
        }
        is_valid, err = validate_motion_plan(plan)
        self.assertFalse(is_valid)
        self.assertIn("Sum of phase durations", err)

    def test_invalid_parameters(self):
        # Negative walk speed
        plan = {
            "version": "1.0",
            "prompt": "walk",
            "target_fps": 20,
            "total_duration": 2.0,
            "phases": [
                {"phase_id": 1, "action": "walk", "duration": 2.0, "parameters": {"speed": -1.0}}
            ]
        }
        is_valid, err = validate_motion_plan(plan)
        self.assertFalse(is_valid)
        self.assertIn("speed must be > 0", err)

        # Turn missing angle_degrees
        plan_turn = {
            "version": "1.0",
            "prompt": "turn",
            "target_fps": 20,
            "total_duration": 1.0,
            "phases": [
                {"phase_id": 1, "action": "turn", "duration": 1.0, "parameters": {}}
            ]
        }
        is_valid_turn, err_turn = validate_motion_plan(plan_turn)
        self.assertFalse(is_valid_turn)
        self.assertIn("missing 'angle_degrees'", err_turn)


if __name__ == '__main__':
    unittest.main()

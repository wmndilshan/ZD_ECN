"""
Strict JSON Schema and validation for ZD-ECN Motion Plans.
"""
from typing import Dict, Any, Tuple
import jsonschema

SUPPORTED_ACTIONS = [
    "walk", "turn", "squat", "jump", "land", "idle",
    "dance_mandiya", "dance_step", "dance_turn", "dance_freeze",
    "moonwalk", "spin", "toe_stand", "pop_hits", "pose_point"
]

MOTION_PLAN_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ZDECNMotionPlan",
    "type": "object",
    "required": ["version", "prompt", "target_fps", "total_duration", "phases"],
    "additionalProperties": False,
    "properties": {
        "version": {
            "type": "string",
            "enum": ["1.0", "1.0.0"]
        },
        "prompt": {
            "type": "string",
            "minLength": 1
        },
        "target_fps": {
            "type": "integer",
            "minimum": 5,
            "maximum": 120
        },
        "total_duration": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 300.0
        },
        "phases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["phase_id", "action", "duration", "parameters"],
                "additionalProperties": False,
                "properties": {
                    "phase_id": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "action": {
                        "type": "string",
                        "enum": SUPPORTED_ACTIONS
                    },
                    "duration": {
                        "type": "number",
                        "exclusiveMinimum": 0.0,
                        "maximum": 60.0
                    },
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True
                    }
                }
            }
        }
    }
}


def validate_motion_plan(plan: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(plan, dict):
        return False, "Plan must be a JSON dictionary object."

    try:
        jsonschema.validate(instance=plan, schema=MOTION_PLAN_SCHEMA)
    except jsonschema.ValidationError as e:
        return False, f"Schema validation error: {e.message} at path {list(e.path)}"
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"

    phase_sum = sum(p["duration"] for p in plan["phases"])
    if abs(phase_sum - plan["total_duration"]) > 0.05:
        return False, (
            f"Semantic error: Sum of phase durations ({phase_sum:.3f}s) "
            f"does not match total_duration ({plan['total_duration']:.3f}s)"
        )

    for i, p in enumerate(plan["phases"], start=1):
        if p["phase_id"] != i:
            return False, f"Semantic error: Phase at index {i-1} has phase_id {p['phase_id']}, expected {i}"

        act = p["action"]
        params = p.get("parameters", {})
        if act == "turn":
            if "angle_degrees" not in params:
                return False, f"Semantic error: 'turn' action in phase {i} missing 'angle_degrees'"
        elif act == "walk":
            if params.get("speed", 1.0) <= 0:
                return False, f"Semantic error: 'walk' action in phase {i} speed must be > 0"
        elif act == "jump":
            if params.get("height", 0.5) <= 0:
                return False, f"Semantic error: 'jump' action in phase {i} height must be > 0"
        elif act == "squat":
            depth = params.get("depth", 0.5)
            if depth < 0.0 or depth > 1.0:
                return False, f"Semantic error: 'squat' depth must be between 0.0 and 1.0"

    return True, ""

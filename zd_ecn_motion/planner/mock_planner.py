import re
from typing import Dict, Any, List
from .schema import validate_motion_plan, SUPPORTED_ACTIONS


class MockMotionPlanner:
    DEFAULT_FPS = 20

    ACTION_PATTERNS = [
        (r"\b(moonwalk|moonwalking|backslide|moon walk)\b", "moonwalk"),
        (r"\b(spin|spins|spinning|twirl)\b", "spin"),
        (r"\b(toe stand|on (?:his |her |their )?toes|tiptoe|tip toe)\b", "toe_stand"),
        (r"\b(walk|walking|stride|step forward|walk forward)\b", "walk"),
        (r"\b(turn left|turn right|turn around|turn)\b", "turn"),
        (r"\b(squat|squatting|crouch|crouching|knee bend)\b", "squat"),
        (r"\b(jump|jumping|leap|hop)\b", "jump"),
        (r"\b(land|landing|absorb impact|ground contact)\b", "land"),
        (r"\b(idle|return to idle|stand|pause|wait|rest)\b", "idle"),
        (r"\b(dance mandiya|mandiya|namaskaraya|salutation|mandiya stance)\b", "dance_mandiya"),
        (r"\b(dance step|stepping|kastiram|thei-tha|footwork|rhythmic stamp)\b", "dance_step"),
        (r"\b(dance turn|vattam|pirouette|circular turn)\b", "dance_turn"),
        (r"\b(dance freeze|samapthiya|heroic freeze|statue)\b", "dance_freeze"),
        (r"\b(pop hits|popping|pop and lock|pop|hits|isolations)\b", "pop_hits"),
        (r"\b(point|pointing|signature pose|final pose|pose)\b", "pose_point"),
    ]

    UNSUPPORTED_ACTIONS = [
        "fly", "swim", "somersault", "backflip", "cartwheel", "crawl", "teleport", "climb"
    ]

    def __init__(self, default_fps: int = 20):
        self.default_fps = default_fps

    def plan(self, prompt: str) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        prompt_clean = prompt.strip().lower()

        # Check for explicitly unsupported actions
        for unsupp in self.UNSUPPORTED_ACTIONS:
            if re.search(rf"\b{unsupp}\b", prompt_clean):
                raise ValueError(
                    f"Unsupported action '{unsupp}' detected in prompt. "
                    f"Supported actions are: {', '.join(SUPPORTED_ACTIONS)}"
                )

        # High-level macro prompt detection for traditional dance
        if re.search(r"\b(traditional dance|dance traditionally|traditional dancer|kandyan dance|uva dance)\b", prompt_clean):
            # Check if prompt doesn't specify explicit clauses
            if not re.search(r",|;|\band\s+then\b|\bthen\b", prompt_clean) or "perform" in prompt_clean or "generation" in prompt_clean or "again" in prompt_clean:
                return self._generate_traditional_dance_plan(prompt)

        # High-level macro for pop / Michael Jackson style dance
        if re.search(r"\b(michael jackson|mj style|king of pop|pop dance|pop star dance)\b", prompt_clean):
            if not re.search(r",|;|\band\s+then\b|\bthen\b", prompt_clean) or re.search(r"\b(perform|style|like)\b", prompt_clean):
                return self._generate_pop_dance_plan(prompt)

        clauses = re.split(r",|;|\band\s+then\b|\bthen\b|\bafter\s+that\b|\band\s+return\s+to\b|\band\b", prompt_clean)
        clauses = [c.strip() for c in clauses if c.strip()]

        phases: List[Dict[str, Any]] = []
        phase_counter = 1

        for clause in clauses:
            detected_action = None
            for pattern, act_type in self.ACTION_PATTERNS:
                if re.search(pattern, clause):
                    detected_action = act_type
                    break

            if detected_action is None:
                if clause in ["please", "now", "forward", "cleanly", "smoothly", "again", "for traditional dancer"]:
                    continue
                raise ValueError(
                    f"Could not parse action from clause: '{clause}'. "
                    f"Supported actions are: {', '.join(SUPPORTED_ACTIONS)}"
                )

            duration, params = self._build_action_parameters(detected_action, clause)

            phases.append({
                "phase_id": phase_counter,
                "action": detected_action,
                "duration": round(duration, 3),
                "parameters": params
            })
            phase_counter += 1

        if not phases:
            raise ValueError("No valid motion phases could be extracted from prompt.")

        total_duration = round(sum(p["duration"] for p in phases), 3)

        plan = {
            "version": "1.0",
            "prompt": prompt.strip(),
            "target_fps": self.default_fps,
            "total_duration": total_duration,
            "phases": phases
        }

        is_valid, err_msg = validate_motion_plan(plan)
        if not is_valid:
            raise ValueError(f"Generated motion plan failed schema validation: {err_msg}")

        return plan

    def _generate_traditional_dance_plan(self, prompt: str) -> Dict[str, Any]:
        """Generates a complete 4-phase classical traditional dance choreography plan."""
        phases = [
            {
                "phase_id": 1,
                "action": "dance_mandiya",
                "duration": 2.5,
                "parameters": {"depth": 0.75, "arm_mudra": "anjali_to_open"}
            },
            {
                "phase_id": 2,
                "action": "dance_step",
                "duration": 4.0,
                "parameters": {"cadence": "thei_tha", "tempo_bpm": 80, "step_lift_m": 0.12}
            },
            {
                "phase_id": 3,
                "action": "dance_turn",
                "duration": 2.5,
                "parameters": {"turn_type": "vattam", "angle_degrees": 360.0}
            },
            {
                "phase_id": 4,
                "action": "dance_freeze",
                "duration": 2.0,
                "parameters": {"posture": "samapthiya", "chin_elevation": True}
            }
        ]
        total_duration = round(sum(p["duration"] for p in phases), 3)
        plan = {
            "version": "1.0",
            "prompt": prompt.strip(),
            "target_fps": self.default_fps,
            "total_duration": total_duration,
            "phases": phases
        }
        is_valid, err_msg = validate_motion_plan(plan)
        if not is_valid:
            raise ValueError(f"Traditional dance plan validation error: {err_msg}")
        return plan

    def _generate_pop_dance_plan(self, prompt: str) -> Dict[str, Any]:
        """Five-phase pop routine: hits -> moonwalk -> spin -> toe stand -> point pose."""
        phases = [
            {"phase_id": 1, "action": "pop_hits", "duration": 2.4, "parameters": {"tempo_bpm": 117}},
            {"phase_id": 2, "action": "moonwalk", "duration": 3.3, "parameters": {"stride": 0.18}},
            {"phase_id": 3, "action": "spin", "duration": 1.4, "parameters": {"angle_degrees": 360.0}},
            {"phase_id": 4, "action": "toe_stand", "duration": 1.5, "parameters": {}},
            {"phase_id": 5, "action": "pose_point", "duration": 2.4, "parameters": {}},
        ]
        plan = {
            "version": "1.0",
            "prompt": prompt.strip(),
            "target_fps": self.default_fps,
            "total_duration": round(sum(p["duration"] for p in phases), 3),
            "phases": phases
        }
        is_valid, err_msg = validate_motion_plan(plan)
        if not is_valid:
            raise ValueError(f"Pop dance plan validation error: {err_msg}")
        return plan

    def _build_action_parameters(self, action: str, clause: str):
        dur_match = re.search(r"for\s+([0-9]+(?:\.[0-9]+)?)\s*(?:seconds|second|s|sec)\b", clause)
        custom_dur = float(dur_match.group(1)) if dur_match else None

        if action == "walk":
            duration = custom_dur if custom_dur else 2.0
            speed_match = re.search(r"speed\s+([0-9]+(?:\.[0-9]+)?)\b", clause)
            speed = float(speed_match.group(1)) if speed_match else 1.2
            return duration, {"speed": speed, "step_frequency": 1.8, "direction": "forward"}

        elif action == "turn":
            duration = custom_dur if custom_dur else 1.0
            angle = 90.0
            direction = "left"
            if "right" in clause:
                angle = -90.0
                direction = "right"
            elif "around" in clause or "180" in clause:
                angle = 180.0
                direction = "around"

            deg_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:deg|degrees)\b", clause)
            if deg_match:
                val = float(deg_match.group(1))
                angle = -val if "right" in clause else val

            return duration, {"angle_degrees": angle, "direction": direction}

        elif action == "squat":
            duration = custom_dur if custom_dur else 1.5
            depth = 0.6
            if "deep" in clause:
                depth = 0.8
            elif "shallow" in clause:
                depth = 0.35
            return duration, {"depth": depth, "hold_duration": 0.3}

        elif action == "jump":
            duration = custom_dur if custom_dur else 1.0
            height = 0.45
            if "high" in clause:
                height = 0.65
            elif "small" in clause or "low" in clause:
                height = 0.25
            return duration, {"height": height, "forward_distance": 0.2}

        elif action == "land":
            duration = custom_dur if custom_dur else 0.8
            damping = 0.85
            return duration, {"recovery_duration": 0.5, "damping": damping}

        elif action == "idle":
            duration = custom_dur if custom_dur else 1.5
            posture = "neutral"
            if "mandiya" in clause:
                posture = "mandiya"
            return duration, {"posture": posture, "breathing": True}

        elif action == "dance_mandiya":
            duration = custom_dur if custom_dur else 2.5
            return duration, {"depth": 0.75, "arm_mudra": "anjali_to_open"}

        elif action == "dance_step":
            duration = custom_dur if custom_dur else 3.5
            return duration, {"cadence": "thei_tha", "tempo_bpm": 80, "step_lift_m": 0.12}

        elif action == "dance_turn":
            duration = custom_dur if custom_dur else 2.5
            return duration, {"turn_type": "vattam", "angle_degrees": 360.0}

        elif action == "dance_freeze":
            duration = custom_dur if custom_dur else 2.0
            return duration, {"posture": "samapthiya", "chin_elevation": True}

        elif action == "moonwalk":
            duration = custom_dur if custom_dur else 3.3
            return duration, {"stride": 0.18}

        elif action == "spin":
            duration = custom_dur if custom_dur else 1.4
            angle = 720.0 if re.search(r"\b(twice|double|two)\b", clause) else 360.0
            if re.search(r"\bright\b", clause):
                angle = -angle
            return duration, {"angle_degrees": angle}

        elif action == "toe_stand":
            duration = custom_dur if custom_dur else 1.5
            return duration, {}

        elif action == "pop_hits":
            duration = custom_dur if custom_dur else 2.4
            return duration, {"tempo_bpm": 117}

        elif action == "pose_point":
            duration = custom_dur if custom_dur else 2.4
            return duration, {}

        return 1.0, {}

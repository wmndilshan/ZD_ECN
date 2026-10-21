from typing import Dict, Any, Optional, Callable
from .mock_planner import MockMotionPlanner
from .schema import validate_motion_plan

class MotionPlannerAdapter:
    def __init__(self, llm_backend: Optional[Callable[[str], Dict[str, Any]]] = None, default_fps: int = 20):
        self.llm_backend = llm_backend
        self.mock_planner = MockMotionPlanner(default_fps=default_fps)

    def plan_motion(self, prompt: str) -> Dict[str, Any]:
        if self.llm_backend is not None:
            try:
                candidate_plan = self.llm_backend(prompt)
                is_valid, err = validate_motion_plan(candidate_plan)
                if is_valid:
                    return candidate_plan
            except Exception:
                pass

        return self.mock_planner.plan(prompt)

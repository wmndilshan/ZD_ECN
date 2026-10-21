"""Per-phase lowest mesh point (mm above the floor) for a prompt, using exact SMPL LBS +
pose blendshapes through the app's conversion path.

    .venv/bin/python prompt_motion_app/tools/floor_contact_by_phase.py "prompt"
"""
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path[:0] = [str(TOOLS), str(REPO / "scripts"), str(REPO)]
os.chdir(REPO)

from motion_to_smpl import convert  # noqa: E402
from build_training_dataset import SMPL_MODEL_PATH, skin_with_transforms  # noqa: E402
from generate_smpl_lbs_validation_fixture import batch_rigid_transform, batch_rodrigues, load_model  # noqa: E402
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner  # noqa: E402
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator  # noqa: E402

model = load_model(SMPL_MODEL_PATH)
m = DeterministicMotionGenerator().generate(MockMotionPlanner(default_fps=30).plan(sys.argv[1]))
s = convert(m)
n = len(m["frames"])
vs = model["v_template"][None]
J = np.einsum("bik,ji->bjk", vs, model["J_regressor"])
rot = batch_rodrigues(s["aa"].reshape(-1, 3)).reshape(n, 24, 3, 3)
_, A = batch_rigid_transform(rot, np.repeat(J, n, 0), model["parents"])
offs = ((rot[:, 1:] - np.eye(3, dtype=np.float32)).reshape(n, -1) @ model["posedirs"]).reshape(n, 6890, 3)
v = skin_with_transforms(vs + offs, A, model["lbs_weights"]) + s["trans"][:, None]
mz = v[:, :, 2].min(1) * 1000
planted = np.array([f["foot_contact"]["left_foot_planted"] or f["foot_contact"]["right_foot_planted"] for f in m["frames"]])
names = [f["phase_name"] for f in m["frames"]]
for ph in dict.fromkeys(names):
    idx = np.array([i for i, x in enumerate(names) if x == ph])
    pi = idx[planted[idx]]
    lo = f"{mz[pi].min():7.1f}" if len(pi) else "   n/a"
    hi = f"{mz[pi].max():7.1f}" if len(pi) else "   n/a"
    print(f"{ph:28s} planted-frame lowest point mm: min {lo}  max {hi}")

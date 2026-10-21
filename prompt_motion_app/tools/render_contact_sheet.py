"""
Host-side preview: renders SMPL meshes (LBS + exact pose blendshapes) for chosen
frames of a prompt through the same generator -> motion_to_smpl path the app uses.

    .venv/bin/python prompt_motion_app/tools/render_contact_sheet.py "prompt" out.png f1 f2 ...
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from motion_to_smpl import convert  # noqa: E402
from build_training_dataset import SMPL_MODEL_PATH, skin_with_transforms  # noqa: E402
from generate_smpl_lbs_validation_fixture import batch_rigid_transform, batch_rodrigues, load_model  # noqa: E402
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner  # noqa: E402
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator  # noqa: E402

prompt, out, frames = sys.argv[1], sys.argv[2], [int(f) for f in sys.argv[3:]]
model = load_model(SMPL_MODEL_PATH)
motion = DeterministicMotionGenerator().generate(MockMotionPlanner(default_fps=30).plan(prompt))
smpl = convert(motion)
frames = [min(f, len(motion["frames"]) - 1) for f in frames]
aa = smpl["aa"][frames]
n = len(frames)
v_shaped = model["v_template"][None]
joints = np.einsum("bik,ji->bjk", v_shaped, model["J_regressor"])
rot = batch_rodrigues(aa.reshape(-1, 3)).reshape(n, 24, 3, 3)
offs = ((rot[:, 1:] - np.eye(3, dtype=np.float32)).reshape(n, -1) @ model["posedirs"]).reshape(n, 6890, 3)
_, A = batch_rigid_transform(rot, np.repeat(joints, n, 0), model["parents"])
verts = skin_with_transforms(v_shaped + offs, A, model["lbs_weights"]) + smpl["trans"][frames][:, None, :]
faces = np.fromfile(REPO / "prompt_motion_app/app/src/main/assets/smpl_shader_data/faces_u16.bin", dtype=np.uint16).reshape(-1, 3).astype(np.int64)

fig = plt.figure(figsize=(1.9 * n, 4.0))
for k, f in enumerate(frames):
    ax = fig.add_subplot(1, n, k + 1, projection="3d")
    v = verts[k]
    c = v[:, :2].mean(0)
    tri = v[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    shade = 0.35 + 0.65 * np.clip(normals @ np.array([0.3, -0.6, 0.75]), 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.outer(shade, [0.85, 0.68, 0.55]), linewidths=0))
    ax.set_xlim(c[0] - 0.5, c[0] + 0.5); ax.set_ylim(c[1] - 0.5, c[1] + 0.5); ax.set_zlim(0.0, 1.8)
    ax.set_box_aspect((1, 1, 1.8), zoom=1.3)
    ax.view_init(elev=8, azim=-70)
    ax.set_axis_off()
    ax.set_title(f"{f}: {motion['frames'][f]['phase_name'].split(': ')[1]}", fontsize=9)
plt.subplots_adjust(wspace=0, left=0, right=1, top=0.92, bottom=0)
plt.savefig(out, dpi=80)
print("wrote", out)

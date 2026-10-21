"""
Validates the generator -> SMPL conversion on real SMPL geometry and measures
ZD-ECN accuracy on prompt-generated poses (are they in-distribution?).

    .venv/bin/python prompt_motion_app/tools/validate_motion_to_smpl.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(TOOLS))

from motion_to_smpl import convert, load_smpl_rest  # noqa: E402
from build_training_dataset import SMPL_MODEL_PATH, skin_with_transforms  # noqa: E402
from generate_smpl_lbs_validation_fixture import batch_rigid_transform, batch_rodrigues, load_model  # noqa: E402
from export_zdecn_checkpoint import load_checkpoint_model  # noqa: E402
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner  # noqa: E402
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator  # noqa: E402

CHECKPOINT = REPO / "results/paper/full_retrain_sameframe/checkpoint_epoch_0375.pt"
PROMPTS = [
    "Walk forward, turn left, jump, land, and return to idle",
    "Perform traditional dance",
    "squat deeply then jump high then land",
    "walk forward for 3 seconds, turn around, walk forward",
    "Michael Jackson style dance",
]


def main() -> None:
    import os
    os.chdir(REPO)
    model_data = load_model(SMPL_MODEL_PATH)
    v_rest, j_rest = load_smpl_rest()
    net = load_checkpoint_model(CHECKPOINT)
    planner = MockMotionPlanner(default_fps=30)
    gen = DeterministicMotionGenerator()
    report = []

    for prompt in PROMPTS:
        motion = gen.generate(planner.plan(prompt))
        smpl = convert(motion, (v_rest, j_rest))
        aa = smpl["aa"]
        n = aa.shape[0]

        v_shaped = model_data["v_template"][None]
        joints = np.einsum("bik,ji->bjk", v_shaped, model_data["J_regressor"])
        rot = batch_rodrigues(aa.reshape(-1, 3)).reshape(n, 24, 3, 3)
        feat = (rot[:, 1:] - np.eye(3, dtype=np.float32)).reshape(n, -1)
        offs = (feat @ model_data["posedirs"]).reshape(n, 6890, 3)
        _, A = batch_rigid_transform(rot, np.repeat(joints, n, 0), model_data["parents"])
        v_ref = skin_with_transforms(v_shaped + offs, A, model_data["lbs_weights"])
        v_lbs = skin_with_transforms(np.repeat(v_shaped, n, 0), A, model_data["lbs_weights"])
        target_mm = (v_ref - v_lbs) * 1000.0
        with torch.no_grad():
            pred_mm = net(torch.from_numpy(smpl["rot6d"])).numpy().reshape(n, 6890, 3)

        pve_zdecn = np.linalg.norm(pred_mm - target_mm, axis=-1).mean(1)
        pve_lbs = np.linalg.norm(target_mm, axis=-1).mean(1)

        world = v_ref + smpl["trans"][:, None, :]
        planted = np.array([fr["foot_contact"]["left_foot_planted"] or fr["foot_contact"]["right_foot_planted"]
                            for fr in motion["frames"]])
        min_z = world[:, :, 2].min(1)
        head_minus_pelvis = world[:, 411, 2] - world[:, 3500, 2]  # 411 ~ head top, 3500 ~ belly
        # Facing: generator forward at f=0 is +Y_gen -> -Y world. Nose vertex 332.
        nose_dir = world[0, 332] - world[0, :, :].mean(0)

        r = {
            "prompt": prompt,
            "frames": n,
            "scale_gen_to_smpl": round(float(smpl["scale"]), 4),
            "pve_lbs_mm": round(float(pve_lbs.mean()), 4),
            "pve_zdecn_mm": round(float(pve_zdecn.mean()), 4),
            "pve_zdecn_p95_mm": round(float(np.percentile(pve_zdecn, 95)), 4),
            "improvement_pct": round(float(100 * (1 - pve_zdecn.mean() / pve_lbs.mean())), 2),
            "planted_min_z_mm": [round(float(min_z[planted].min() * 1000), 1), round(float(min_z[planted].max() * 1000), 1)],
            "max_air_z_mm": round(float(min_z.max() * 1000), 1),
            "upright_all_frames": bool((head_minus_pelvis > 0.3).all()),
            "facing_world_dir_f0": np.round(nose_dir / np.linalg.norm(nose_dir), 2).tolist(),
        }
        report.append(r)
        print(json.dumps(r))

    out = TOOLS / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

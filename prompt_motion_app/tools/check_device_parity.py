"""
Compares the on-device Kotlin pipeline (plan -> generator -> SMPL converter) with the
Python reference (zd_ecn_motion generator + tools/motion_to_smpl.py) for the same plan.

    adb shell am start -n com.graphicsresearch.promptmotion/.MainActivity \
        --es prompt "..." --ez use_llm false --ez dump_clip true
    adb exec-out run-as com.graphicsresearch.promptmotion tar c files/dump | tar x -C <dir>
    .venv/bin/python prompt_motion_app/tools/check_device_parity.py <dir>/files/dump
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_to_smpl import convert  # noqa: E402
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator  # noqa: E402
from zd_ecn_motion.planner.schema import validate_motion_plan  # noqa: E402

d = Path(sys.argv[1])
plan = json.loads((d / "plan.json").read_text())
ok, err = validate_motion_plan(plan)
print("python schema validation of device plan:", "PASS" if ok else err)
ref = convert(DeterministicMotionGenerator().generate(plan))
n = ref["aa"].shape[0]
dev = {k: np.fromfile(d / f"{k}_f32.bin", dtype=np.float32) for k in ("aa", "rot6d", "trans")}
print("frames  python", n, " device", dev["aa"].size // 72)
for k, w in (("aa", 72), ("rot6d", 144), ("trans", 3)):
    a = dev[k].reshape(-1, w)
    diff = np.abs(a[:n] - ref[k][: a.shape[0]])
    print(f"{k:6s} max|diff| {diff.max():.3e}   mean {diff.mean():.3e}")

# Axis-angle is double-covered near theta = pi (n*pi == -n*pi), so compare rotations.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from generate_smpl_lbs_validation_fixture import batch_rodrigues  # noqa: E402
Rd = batch_rodrigues(dev["aa"].reshape(-1, 3)[: n * 24])
Rp = batch_rodrigues(ref["aa"].reshape(-1, 3))
print(f"aa as rotation matrices max|diff| {np.abs(Rd - Rp).max():.3e}")

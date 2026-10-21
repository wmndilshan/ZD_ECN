# ZD-ECN Prompt Motion (Android)

Text prompt → motion plan → biomechanical motion → SMPL body with ZD-ECN pose correction,
all on the phone. Package `com.graphicsresearch.promptmotion`.

```
prompt ─► planner (OpenAI gpt-4o-mini, JSON mode │ fallback: zd_ecn_motion rule-based)
       ─► ZDECNMotionPlan v1.0 (strict schema, same as zd_ecn_motion/planner/schema.py)
       ─► MotionGenerator (Kotlin port of zd_ecn_motion generator: min-jerk blending,
          world-space foot locking + analytical leg IK, joint limits)
       ─► SmplMotionConverter (generator skeleton → SMPL axis-angle / rot6d / root offset)
       ─► SmplGpuPipeline (unchanged): GPU LBS + ZD-ECN GLSL fp16 + combine + normals
       ─► wrap-lit skin shader + floor with contact shadow, follow camera
```

## Motion vocabulary

`idle walk turn squat jump land` · Kandyan: `dance_mandiya dance_step dance_turn dance_freeze` ·
pop / Michael-Jackson style: `pop_hits moonwalk spin toe_stand pose_point`
("Michael Jackson style dance" → hits → moonwalk → spin → toe stand → point).

The pop moves added an explicit **foot override** channel to `zd_ecn_motion` (per-frame foot
x, y, z, yaw, heel pitch) for feet that glide while loaded (moonwalk), pivot in place (spin) or
stand on the toes. Override frames form their own foot-lock intervals; a stance touching one keeps
that foot spot. Grounded heel-raised feet dorsiflex the toes (toe pitch limit widened to −70°), and
root yaw (accumulated heading) is no longer clamped at ±4π.

Only motion comes from `zd_ecn_motion/`; rendering, SMPL assets, canonical ZD-ECN weights
(`weights_sameframe_fp16.bin`, md5 54e5bafb…) and the ONNX model come from `skin_rendering/`.

## Build / run

```bash
cd prompt_motion_app
./gradlew --offline assembleDebug
adb -s 129305554R004031 install -r app/build/outputs/apk/debug/app-debug.apk
```

`OPENAI_API_KEY` (and optional `OPENAI_MODEL`, default `gpt-4o-mini`) are read from the repo-root
`.env` at build time into `BuildConfig`. **The key is inside the APK — do not share the APK.**

Headless extras: `--es prompt "..."`, `--ez use_llm false`, `--ei backend 0|1|2`
(ZD-ECN GPU / ONNX CPU / LBS only), `--ei fixed_frame N`, `--ez run_benchmark true`,
`--ez dump_clip true` (parity dump).

## Frame conventions (details in `tools/motion_to_smpl.py`)

* Generator: Z-up, faces +Y, mirrored body; SMPL: Y-up body. `M = swap(y,z)`, `W = diag(1,-1,1)`:
  root `W·R·M`, other joints `M·R·M`, points `W·p`.
* ZD-ECN was trained on SURREAL/CMU roots (~Rx(90°) + free yaw, i.e. a Z-up world), so the full
  root orientation is fed to the network; the renderer only applies `Rx(-90°)` for display.
* **Arm retargeting:** `zd_ecn_motion` authors arm rotations for arms hanging down, but its rest
  skeleton is a T-pose (arm pitch only twisted horizontal arms). Shoulders use `R·B`, distal arm
  joints `Bᵀ·R·B` with `B = Ry(±90°)`, and elbow pitch is negated so flexion bends forward.
* **Leg retargeting:** after conversion, two-bone IK on the *SMPL* skeleton puts each SMPL foot
  joint on the generator's scaled foot target (knee keeps the generator's bend plane, foot keeps its
  orientation). This absorbs the 0.38/0.38 m generator vs SMPL thigh/shin mismatch: planted-foot
  floor error went from ±10 mm to ≤ 2.5 mm (≤ 10.5 mm on heel-raised pop moves, toe-crease LBS).
* Axis-angle uses Shepperd's quaternion method (the `1/(2 sin θ)` formula broke at heading ≈ 180°).

## Verification (2026-10-21, TECNO KL4 / Mali-G57)

| Check | Result |
|---|---|
| Kotlin (on device) vs Python reference, 5 plans + OpenAI plans | rotation ≤ 5.0e-4 (1.3e-6 vs an unrounded reference; the reference JSON rounds positions to 1e-4 m), root ≤ 5.4e-5 m |
| Device plans re-validated by Python `validate_motion_plan` | PASS |
| Generator quality (all presets incl. pop routine) | 0 m ground penetration, 0 m/s stance slip, 0 joint clamps |
| `zd_ecn_motion` unit tests | 27/27 pass (21 existing + 6 in `tests/test_pop_dance.py`) |
| SMPL lowest point in planted frames (`tools/floor_contact_by_phase.py`) | walk/turn/jump/Kandyan within 2.5 mm; pop moves −10.5…−0.2 mm |
| Full frame GPU (LBS+ZD-ECN+combine+normals+floor+draw), N=300 | 9.288 ± 0.132 ms (walk clip), 9.290 ± 0.179 ms (pop clip), p99 ≤ 9.68, 0 disjoint |
| ZD-ECN dispatch (overlay) | ~1.13 ms |

ZD-ECN accuracy on prompt-generated poses vs the exact pose-blendshape target
(`tools/validation_report.json`, canonical checkpoint epoch 375):

| Prompt | LBS PVE mm | ZD-ECN PVE mm | Improvement |
|---|---:|---:|---:|
| Walk, turn left, jump, land, idle | 5.514 | 2.699 | 51.0% |
| Traditional dance | 6.712 | 4.587 | 31.7% |
| Squat deep, jump high, land | 5.793 | 2.953 | 49.0% |
| Walk 3 s, turn around, walk | 5.493 | 2.721 | 50.5% |
| Michael Jackson style dance | 6.139 | 3.516 | 42.7% |

Lower than the 64.06% on the CMU validation split: procedural poses (especially the deep
Kandyan *mandiya* stance) are partly outside the CMU training distribution. ZD-ECN still beats
LBS on every prompt. These are offline (PyTorch fp32) numbers, not on-device measurements.

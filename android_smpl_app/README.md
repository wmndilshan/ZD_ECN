# ZD-ECN SMPL Renderer App

This Android app renders animated SMPL geometry and is separate from `android_compute_benchmark/`.

## Current Status

Implemented:

- GPU SSBO SMPL LBS visual path.
- GPU GLSL fp16 ZD-ECN native visual path.
- Fixed-frame launch extras for deterministic screenshots.
- Portrait-first mobile UI with compact backend/pose controls.
- Touch camera controls: one-finger rotate, pinch zoom, reset view.
- ONNX Runtime Mobile CPU path and best-effort NNAPI session path upload deltas into the GL correction buffer.

Not implemented yet:

- Independent ShaderNN-style backend.
- No-readback render path.
- BenchmarkActivity CSV export.
- Final 15-screenshot backend comparison set.

## Important Architecture Notes

The app uses validated shader/source logic copied from `android_compute_benchmark/`, but the benchmark app must not be modified when working on this app unless explicitly requested.

Validated source locations used during wiring:

- ZD-ECN dispatch: `android_compute_benchmark/app/src/main/cpp/main.cpp` around line 719.
- ZD-ECN shader setup: same file around line 848.
- SMPL GPU dispatch: same file around line 2047.
- SMPL dense SSBO upload: same file around line 1817.

The app packages dense f32 SMPL assets in `app/src/main/assets/smpl_shader_data/` because the validated SMPL shaders consume dense f32 buffers, not the earlier sparse/f16 app assets.

## Build And Smoke Test

```bash
cd android_smpl_app
./gradlew assembleRelease
adb -s 129305554R004031 install -r app/build/outputs/apk/release/app-release.apk
```

LBS-only fixed extreme frame:

```bash
adb -s 129305554R004031 shell am start \
  -n com.graphicsresearch.smplrenderer/.MainActivity \
  --ei backend 4 --ei tier 2 --ei frame 500
```

ZD-ECN native fixed extreme frame:

```bash
adb -s 129305554R004031 shell am start \
  -n com.graphicsresearch.smplrenderer/.MainActivity \
  --ei backend 0 --ei tier 2 --ei frame 500
```

Expected logcat line:

```text
SmplRenderer: native init GPU SSBO pipeline initialized
```

Portrait UI smoke screenshot:

- `results/paper/screenshots/smpl_app_portrait_polish.png`

## Backend IDs

| ID | Backend | Current behavior |
|---:|---|---|
| 0 | ZD-ECN native | GPU SMPL LBS + GPU ZD-ECN correction |
| 1 | ShaderNN-style | Placeholder/same path pending independent implementation |
| 2 | ONNX CPU | ORT CPU delta uploaded to native GL correction buffer |
| 3 | ONNX NNAPI | Best-effort ORT NNAPI session; falls back if provider unavailable |
| 4 | LBS only | GPU SMPL LBS, zero correction |

## Pose Tier IDs

| ID | Tier | Representative frame |
|---:|---|---:|
| 0 | Natural | 100 |
| 1 | Moderate | 300 |
| 2 | Extreme | 500 |

## Documentation

Current authoritative report:

- `results/paper/SMPL_APP_BENCHMARK_REPORT.md`

Smoke screenshots:

- `results/paper/screenshots/smpl_gpu_lbs_extreme.png`
- `results/paper/screenshots/smpl_gpu_zdecn_extreme.png`

## ONNX CPU Backend Update

The ONNX Runtime Mobile CPU path is now wired into the renderer app:

- `OnnxBackend.kt` creates persistent CPU and best-effort NNAPI sessions once at GL surface creation.
- The current same-frame checkpoint was exported to a self-contained fp16 ONNX file with float32 external I/O: `models/zdecn_fp16.onnx`.
- The app packages the same asset at `android_smpl_app/app/src/main/assets/zdecn_fp16.onnx`.
- Per-frame ONNX output `FloatArray[20670]` is uploaded to the native delta SSBO via `nativeUploadExternalDelta()`.
- Smoke test passed for `[ONNX CPU]`, extreme tier, fixed frame 500; screenshot: `results/paper/screenshots/smpl_onnx_cpu_extreme.png`.

## fp16 Extension Guard

`smpl_gpu_pipeline.cpp` now logs the GL extension diagnostics but uses the fp16 packed-weight path unconditionally, matching `android_compute_benchmark`. The production shaders do not require `GL_EXT_shader_explicit_arithmetic_types_float16`; they store fp16 weights as packed `uint` words and decode with core `unpackHalf2x16()`. On TECNO KL4, `fp16_arith=0 fp16_storage=0 half=0`, but the renderer logs `fp16 packed-weight path enabled unconditionally via unpackHalf2x16` and ZD-ECN GPU timer samples are near `1.1 ms`.

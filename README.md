# ZD-ECN

Zero-Dependency Geometric Error Correction (ZD-ECN) is a GLSL ES 3.2 compute-shader-native MLP deformer for SMPL on Android edge devices.

Architecture: `144 → 64 → 64 → 20,670` with GELU activations. Input is 24 joints × 6D rotation; output is 6,890 vertices × 3 displacement values. On-device native inference uses packed fp16 weights in SSBOs and core `unpackHalf2x16`, with no TFLite, ONNX Runtime, libtorch, or TensorRT dependency for the ZD-ECN path.

## Canonical Results

Device: TECNO KL4, ARM Mali-G57, Android 14, OpenGL ES 3.2.

| Measurement | Result |
|---|---:|
| Full validation PVE | 1.947495 mm |
| LBS zero-correction baseline | 5.419208 mm |
| Isolated ZD-ECN GPU mean | 1.094395 ms |
| Isolated ZD-ECN CPU wall mean | 1.628550 ms |
| Combined app GPU mean | 9.108006 ms |
| Combined app GPU p99 | 9.656718 ms |
| Combined app CPU wall mean | 10.395140 ms |
| Combined app GPU throughput | 109.79 FPS |

The combined app timer includes SMPL LBS, ZD-ECN, GPU combine, smooth-normal generation, and indexed draw. It uses 5 sessions × 1,000 measured iterations, 100 warmups/session, and 30-second cooldowns.

## Repository Layout

- `android_compute_benchmark/` — isolated production compute-shader benchmark.
- `android_smpl_app/` — portrait SMPL renderer, ONNX visual comparison, and combined benchmark activity.
- `scripts/` — dataset construction, training, export, evaluation, parsing, and figure generation.
- `docs/` — current paper and benchmark status.
- `results/paper/figures/` — 14 final PDF/PNG figure pairs.
- `results/paper/screenshots/comparison_4x3/` — 4 real backends × 3 pose tiers.

Datasets, checkpoints, raw benchmark outputs, local model files, and external cloned repositories are intentionally excluded by `.gitignore`.

## Build The Android Renderer

```bash
cd android_smpl_app
./gradlew assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

## Run The Combined Benchmark

```bash
ADB_SERIAL=129305554R004031 ./android_smpl_app/run_combined_benchmark.sh
```

The script checks adb, builds and installs the release APK, launches `BenchmarkActivity`, monitors incremental progress, and pulls timestamped CSV/JSON/Markdown output under `benchmark_results/`.

## Capture The Visual Matrix

```bash
ADB_SERIAL=129305554R004031 ./android_smpl_app/capture_comparison_set.sh
```

This captures Native ZD-ECN, ONNX CPU, ONNX NNAPI, and LBS-only at natural, moderate, and extreme fixed poses. Existing output is backed up before replacement.

## Paper Figures

```bash
source .venv/bin/activate
python scripts/generate_paper_figures.py
```

Figure regeneration requires the ignored local training cache, validation CSVs, model binaries, and benchmark CSVs. `results/paper/figures/figure_manifest.txt` records all expected outputs.

## Important Limitations

- ShaderNN is excluded from the primary comparison. The available graph was a non-GELU surrogate and its internal alternating timer was not methodologically comparable.
- ONNX CPU and NNAPI are runtime baselines, not zero-dependency inference paths.
- Results currently cover one GPU family. Generalizing cache/SSBO findings requires a second Adreno or different-generation Mali device.
- The paper supervisor name still requires authoritative author input.

See `docs/PAPER_SUBMISSION_STATUS.md` and `docs/APP_COMBINED_BENCHMARK.md` for the current evidence boundary.

# Combined SMPL + ZD-ECN App Benchmark

Date: 2026-06-29  
Device: TECNO KL4, ARM Mali-G57, Android 14  
Samples: 5 sessions × 1,000 measured iterations = 5,000

## Timer Boundary

The GPU query and CPU wall timer wrap the complete app frame path:

1. SMPL shape/joint/kinematics/skinning compute dispatches.
2. Three ZD-ECN compute dispatches.
3. GPU LBS/correction combine.
4. GPU smooth-normal generation.
5. Indexed mesh draw.
6. `glFinish()` inside the CPU timing window.

Each session discards 100 warmup iterations. Sessions use a 30-second cooldown. All 5,000 samples reported `GL_GPU_DISJOINT_EXT = false`.

## Results

| Timer | Mean ms | Std ms | p50 | p95 | p99 | CV | 95% CI | FPS mean | FPS p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU full frame | 9.108006 | 0.158970 | 9.088481 | 9.334910 | 9.656718 | 1.745% | [9.103600, 9.112413] | 109.79 | 103.55 |
| CPU wall | 10.395140 | 0.267538 | 10.362308 | 10.804285 | 11.328310 | 2.574% | [10.387724, 10.402555] | 96.20 | 88.27 |

The combined GPU frame is `8.013611 ms` slower than the isolated ZD-ECN correction result (`1.094395 ms`) because it includes SMPL LBS, combine, smooth normals, and raster draw.

## Artifacts

- Raw CSV: `benchmark_results/app_combined_20260629_134830/20260629_134830/raw/benchmark_raw.csv`
- Device info: `benchmark_results/app_combined_20260629_134830/20260629_134830/raw/device_info.txt`
- JSON summary: `benchmark_results/app_combined_20260629_134830/20260629_134830/summary/benchmark_summary.json`
- Markdown summary: `benchmark_results/app_combined_20260629_134830/20260629_134830/summary/benchmark_summary.md`
- Visual matrix: `results/paper/screenshots/comparison_4x3/`

Reproduction command:

```bash
ADB_SERIAL=129305554R004031 ./android_smpl_app/run_combined_benchmark.sh
```

The historical “15 screenshot” target assumed five real backends. ShaderNN is excluded because no exact trained GELU ShaderNN backend exists. The honest final set is therefore 4 backends × 3 pose tiers = 12 screenshots.

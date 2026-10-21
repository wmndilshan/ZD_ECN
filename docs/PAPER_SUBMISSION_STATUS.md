# Paper Submission Status

Updated: 2026-06-29 (real-beta eval + perceptual threshold added)

## Resolved

- **Figures:** 14 PDF and 14 PNG figures are present under `results/paper/figures/` and explicitly unignored for version control. Quality figures use canonical validation PVE `1.947495 mm`; ONNX comparison uses CPU `5.256 ms` and NNAPI `5.277 ms`.
- **ShaderNN disposition:** the `0.010 ms` internal timer is invalid for the paper because it used an alternating timer and a surrogate non-GELU graph. ShaderNN is excluded from the primary latency table and retained as related-work discussion only.
- **ONNX visual backend:** execution is owned by `OnnxBackend.kt`, not `backend_onnx.cpp`. Persistent ORT CPU and best-effort NNAPI sessions produce `20670` corrections uploaded into the shared GL buffer.
- **No-readback rendering:** LBS and correction are combined by compute into a GPU buffer reused directly as the vertex buffer. The legacy buffer-mapping helper has been removed.
- **Smooth normals:** a compute pass accumulates adjacent face normals per vertex. Topology adjacency is constructed once at initialization; no per-frame CPU geometry processing occurs.
- **fp16 capability:** this implementation does not use native float16 arithmetic or storage types. It stores fp16 values in packed `uint` SSBO words and decodes them with core GLSL ES `unpackHalf2x16`; checking `GL_EXT_shader_explicit_arithmetic_types_float16` would incorrectly reject the validated Mali-G57 path.
- **Training safeguards:** future runs apply cosine LR decay over the final 100 epochs and checkpoint every five epochs in the final 50 epochs.

## Newly Completed

- **Combined app benchmark CSV:** 5 sessions × 1,000 measured iterations, 100 warmups/session, and 30-second cooldowns completed with 5,000 non-disjoint GPU samples. Full-frame GPU mean is `9.108006 ms`; CPU wall mean is `10.395140 ms`.
- **Visual comparison set:** 12 honest images cover Native, ONNX CPU, ONNX NNAPI, and LBS-only across natural, moderate, and extreme poses. The old count of 15 assumed a real ShaderNN backend and is not used.

## Newly Completed (2026-06-29)

- **Body-shape generalisation evaluated:** `scripts/evaluate_real_betas.py` runs the zero-beta-trained model
  on 3,000 CMU MoCap poses with real body shapes (beta std ≈ 1.0, 1,559 distinct subjects).
  Generalisation gap = **0.000 mm** — the same-frame pose-correction target
  `skin_A(θ)(pose_bs)` has zero homogeneous coordinate for `pose_bs`, so the joint
  translation components cancel under LBS and only the beta-independent rotation chain
  remains. Zero-beta training is therefore provably equivalent to beta-aware training
  for this target formulation. This eliminates the anticipated reviewer concern about
  shape generalisation and turns it into a theoretical contribution.
  Results: `results/paper/real_beta_eval/real_beta_eval_summary.json`.

- **Perceptual threshold analysis:** `fig_perceptual_threshold` and `fig_beta_generalisation`
  added to `scripts/generate_paper_figures.py` and generated to `results/paper/figures/`.
  Key numbers (3,000 real-beta poses, 20.67M vertex samples):
  - ZD-ECN < 1 mm: **31.4 %** of vertices  (LBS: 4.2 %)
  - ZD-ECN < 2 mm: **65.8 %** of vertices  (LBS: 18.7 %)
  - ZD-ECN < 5 mm: **94.6 %** of vertices  (LBS: 59.4 %)
  Improvement over LBS (real betas): **63.9 %**.

- **Dead code removed:** `SmplGpuFrameStats::readback_cpu_ms` and `::max_abs_delta_mm`
  removed from `smpl_gpu_pipeline.h` and their sole hardcoded assignment removed from
  `smpl_renderer.cpp`. Both fields were permanently zero after the no-readback render
  path was adopted.

## Open

- **Second GPU family:** requires physical access to an Adreno or another Mali generation.
- **Supervisor name:** requires authoritative author input; it cannot be inferred from repository files.

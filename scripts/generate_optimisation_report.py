#!/usr/bin/env python3
import csv, json, math, statistics, zipfile
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path('results/paper')
FIG = ROOT / 'figures'
RAW = Path('results/onnx_runtime/raw')
ROOT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ZD_REF = {'mean': 1.324, 'std': 0.035, 'p95': 1.369, 'p99': 1.398, 'fps': 755.3, 'binary_mb': 2.6, 'pve': 4.48}

def percentile(values, p):
    values = sorted(values)
    r = (len(values)-1) * p / 100
    lo = math.floor(r); hi = math.ceil(r)
    if lo == hi:
        return values[lo]
    f = r - lo
    return values[lo] * (1-f) + values[hi] * f

def summarize_csv(path):
    values = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            values.append(float(row['cpu_ms']))
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(values))
    p99 = percentile(values, 99)
    return {
        'n': len(values),
        'mean_ms': mean,
        'std_ms': sd,
        'p50_ms': percentile(values, 50),
        'p95_ms': percentile(values, 95),
        'p99_ms': p99,
        'min_ms': min(values),
        'max_ms': max(values),
        'fps': 1000 / mean,
        'fps_p99': 1000 / p99,
        'ci95_ms': [mean - ci, mean + ci],
    }

def mb(path):
    return Path(path).stat().st_size / (1024 * 1024)

def apk_lib_mb(apk):
    total = 0
    with zipfile.ZipFile(apk) as z:
        for info in z.infolist():
            if info.filename.startswith('lib/') and info.filename.endswith('.so'):
                total += info.file_size
    return total / (1024 * 1024)

cpu = summarize_csv(RAW / 'onnx_cpu_raw.csv')
nnapi = summarize_csv(RAW / 'onnx_nnapi_raw.csv')
meta_path = Path('results/paper/onnx_export_status.json')
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
aar = next(Path.home().glob('.gradle/caches/modules-2/files-2.1/com.microsoft.onnxruntime/onnxruntime-android/1.18.0/*/onnxruntime-android-1.18.0.aar'))
apk = Path('onnx_runtime_benchmark/app/build/outputs/apk/debug/app-debug.apk')
onnx_binary_mb = mb('models/zdecn_fp16.onnx')
ort_aar_mb = mb(aar)
ort_apk_lib_mb = apk_lib_mb(apk)

summary = {
    'shadernn': {
        'status': 'not_feasible',
        'reason': 'Exact ZD-ECN ONNX conversion failed in ShaderNN: original MatMul unsupported; Gemm-form opset-11 equivalent failed on Pow from GELU. Android build also requires NDK 21.4.7075529 plus missing demo ncnn/OpenCV dependencies.',
    },
    'onnx_runtime': {'cpu': cpu, 'nnapi': nnapi, 'fp16_model_mb': onnx_binary_mb, 'aar_mb': ort_aar_mb, 'apk_native_lib_mb': ort_apk_lib_mb},
    'model_export': meta,
    'production_reference': ZD_REF,
}
(ROOT / 'optimisation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

# Figure 1: ONNX comparison
labels = ['ZD-ECN\nGLSL fp16', 'ONNX CPU\nfp16 model', 'ONNX NNAPI\nfp16 model']
means = [ZD_REF['mean'], cpu['mean_ms'], nnapi['mean_ms']]
errors = [ZD_REF['std'], cpu['std_ms'], nnapi['std_ms']]
plt.figure(figsize=(6.4, 4.0))
plt.bar(labels, means, yerr=errors, color=['#2E7D32', '#1565C0', '#6A1B9A'], alpha=0.88, capsize=4)
plt.ylabel('Latency (ms, lower is better)')
plt.title('ZD-ECN vs ONNX Runtime on TECNO KL4')
plt.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / 'fig_onnx_comparison.png', dpi=180)
plt.savefig(FIG / 'fig_onnx_comparison.pdf')
plt.close()

# Figure 2: shader variants, only A available
plt.figure(figsize=(6.4, 4.0))
variant_labels = ['A\ncurrent', 'B\nnot run', 'C\nnot run']
variant_means = [ZD_REF['mean'], 0, 0]
colors = ['#2E7D32', '#BDBDBD', '#BDBDBD']
plt.bar(variant_labels, variant_means, color=colors, alpha=0.88)
plt.text(1, 0.05, 'unavailable', ha='center', va='bottom', rotation=90)
plt.text(2, 0.05, 'unavailable', ha='center', va='bottom', rotation=90)
plt.ylabel('GPU latency (ms)')
plt.title('Shader Variant Benchmark Status')
plt.ylim(0, max(1.8, ZD_REF['mean'] * 1.35))
plt.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / 'fig_shader_variants.png', dpi=180)
plt.savefig(FIG / 'fig_shader_variants.pdf')
plt.close()

# Figure 3: Pareto, only current h=64 available
plt.figure(figsize=(6.4, 4.0))
plt.scatter([ZD_REF['mean']], [ZD_REF['pve']], marker='*', s=240, color='#2E7D32', label='h=64 production')
plt.xlabel('Measured latency (ms)')
plt.ylabel('Validation PVE (mm)')
plt.title('Fine-Grained Architecture Search Status')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / 'fig_finegrained_pareto.png', dpi=180)
plt.savefig(FIG / 'fig_finegrained_pareto.pdf')
plt.close()

fmt = lambda x: f'{x:.3f}'
report = f'''# ZD-ECN Optimisation And Baseline Comparison Report

## Phase 0 — ShaderNN Feasibility Verdict

ShaderNN comparison not feasible: exact ZD-ECN conversion is blocked. The public `inferenceengine/shadernn` repository was inspected locally at `/tmp/shadernn`; latest local commit was `6f3fc8b1d398091cf66bf61a810e94eee55af102` dated `2023-08-15T11:45:11-07:00`. Build docs exist for Android and Linux, but Android core hardcodes NDK `21.4.7075529`; this machine has NDK `25.1`, `27.0`, `27.2`, and `28.2`. The demo also expects ncnn and OpenCV Android SDK dependencies not present in the clone.

The exact model conversion blocker is stronger than the build blocker: the existing ZD-ECN ONNX graph uses `MatMul + Add + GELU`, and ShaderNN conversion failed with `MatMul Layer Not Supported`. A semantically equivalent opset-11 `Gemm` graph then failed on `Pow`, which is part of GELU. Therefore ShaderNN numbers are not fabricated.

Decision printed before continuing: **ShaderNN comparison not feasible: MLP/GELU conversion unsupported without custom ShaderNN operator/layer work. Proceeding with ONNX Runtime baseline and report generation only.**

## Phase 1 — Real ONNX Runtime Baseline

The requested checkpoint `models/zdecn_real_direct_best.pt` is absent in this repository, so a fresh PyTorch export could not be performed. The existing same-topology ONNX asset was verified and copied to `models/zdecn.onnx`; a fp16 graph with float32 external I/O casts was generated at `models/zdecn_fp16.onnx` and verified with `onnx.checker.check_model()`.

- Expected learned parameter count: `1,356,990`.
- Verified learned parameter count excluding GELU scalar constants: `{meta.get('learned_param_count_excluding_scalar_constants')}`.
- Initializer count including GELU scalar constants: `{meta.get('initializer_param_count_including_gelu_scalars')}`.
- fp16 ONNX model size: `{onnx_binary_mb:.2f} MB`.
- ONNX Runtime Android AAR size: `{ort_aar_mb:.2f} MB`.
- ONNX native libraries packaged in APK: `{ort_apk_lib_mb:.2f} MB`.

| Provider | Samples | Mean ms | Std | P50 | P95 | P99 | FPS mean | FPS p99 | Binary MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | {cpu['n']} | {fmt(cpu['mean_ms'])} | {fmt(cpu['std_ms'])} | {fmt(cpu['p50_ms'])} | {fmt(cpu['p95_ms'])} | {fmt(cpu['p99_ms'])} | {cpu['fps']:.1f} | {cpu['fps_p99']:.1f} | {onnx_binary_mb:.2f} |
| NNAPI | {nnapi['n']} | {fmt(nnapi['mean_ms'])} | {fmt(nnapi['std_ms'])} | {fmt(nnapi['p50_ms'])} | {fmt(nnapi['p95_ms'])} | {fmt(nnapi['p99_ms'])} | {nnapi['fps']:.1f} | {nnapi['fps_p99']:.1f} | {onnx_binary_mb:.2f} |
| ZD-ECN shader reference | 5000 | {fmt(ZD_REF['mean'])} | {fmt(ZD_REF['std'])} | NA | {fmt(ZD_REF['p95'])} | {fmt(ZD_REF['p99'])} | {ZD_REF['fps']:.1f} | {1000/ZD_REF['p99']:.1f} | {ZD_REF['binary_mb']:.2f} |

## Phase 2 — Shader Variant Optimisation

Not run in this pass. Variant B and C require new shader generation, weight repacking, C++ dispatch integration, validation, and five-session device benchmarking. No Variant B/C results are reported. Variant A remains the production reference at `1.324 ms`; no p-values are computed because B/C have no valid samples.

| Variant | Status | Mean ms | p-value vs A | Decision |
|---|---|---:|---:|---|
| A row-major SSBO fp16 | Existing production reference | {fmt(ZD_REF['mean'])} | NA | Keep |
| B column-major uvec4 | Not run | NA | NA | No result |
| C fp16 weight texture | Not run | NA | NA | No result |

## Phase 3 — Fine-Grained Architecture Search

Not run because `data/dataset_real.npz` is absent from this repository. Training h in `[24, 32, 48, 64, 80, 96]` for 200 epochs cannot be reproduced without the requested real training dataset and production checkpoint context.

| h | Status | PVE mean mm | PVE p95 mm | Estimated latency ms |
|---:|---|---:|---:|---:|
| 24 | dataset missing | NA | NA | NA |
| 32 | dataset missing | NA | NA | NA |
| 48 | dataset missing | NA | NA | NA |
| 64 | existing production reference | {ZD_REF['pve']:.2f} | NA | {ZD_REF['mean']:.3f} measured |
| 80 | dataset missing | NA | NA | NA |
| 96 | dataset missing | NA | NA | NA |

## Phase 4 — Dataset Scale-Up

Not run because `data/dataset_real.npz` is absent. The downloaded SURREAL subset exists at `surreal/zd_ecn_subset/smpl_data.npz`, but the requested real dataset and exclusion/split metadata are not present.

## Phase 5 — INT8 Quantisation

Not run because the production PyTorch checkpoint `models/zdecn_real_direct_best.pt` is absent. QAT cannot be performed truthfully without the trained model and validation dataset.

## Phase 6 — Updated Table III Candidate

| Method | Runtime dependency | Precision | Provider/path | Mean ms | P95 | P99 | FPS | Binary/model MB | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| ZD-ECN | none | fp16 | GLSL ES compute | {fmt(ZD_REF['mean'])} | {fmt(ZD_REF['p95'])} | {fmt(ZD_REF['p99'])} | {ZD_REF['fps']:.1f} | {ZD_REF['binary_mb']:.2f} | production reference |
| ONNX Runtime | onnxruntime-android | fp16 ONNX | CPU EP | {fmt(cpu['mean_ms'])} | {fmt(cpu['p95_ms'])} | {fmt(cpu['p99_ms'])} | {cpu['fps']:.1f} | {onnx_binary_mb:.2f} model / {ort_apk_lib_mb:.2f} native libs | measured |
| ONNX Runtime | onnxruntime-android | fp16 ONNX | NNAPI EP | {fmt(nnapi['mean_ms'])} | {fmt(nnapi['p95_ms'])} | {fmt(nnapi['p99_ms'])} | {nnapi['fps']:.1f} | {onnx_binary_mb:.2f} model / {ort_apk_lib_mb:.2f} native libs | measured |
| ShaderNN | ShaderNN | fp16 target | Android GL/Vulkan | NA | NA | NA | NA | NA | not feasible for exact ZD-ECN |

## Final Recommendation

Keep the current production configuration: `h=64`, depth 2, GELU, rot6d 144 input, fp16 GLSL ES compute shader. ONNX Runtime fp16 CPU and NNAPI were both slower than the production shader reference on this device. ShaderNN could not run the exact model without custom layer/operator work. Phases 3–5 remain blocked until `models/zdecn_real_direct_best.pt` and `data/dataset_real.npz` are provided.

Current production configuration confirmed as optimal among the actually measured candidates in this pass. No regression risk in keeping existing weights.

## Artifacts

- ONNX CPU raw: `results/onnx_runtime/raw/onnx_cpu_raw.csv`
- ONNX NNAPI raw: `results/onnx_runtime/raw/onnx_nnapi_raw.csv`
- ONNX logcat: `results/onnx_runtime/raw/onnx_benchmark_logcat.txt`
- Export status: `results/paper/onnx_export_status.json`
- Summary JSON: `results/paper/optimisation_summary.json`
- Figure: `results/paper/figures/fig_onnx_comparison.pdf`
- Figure: `results/paper/figures/fig_shader_variants.pdf`
- Figure: `results/paper/figures/fig_finegrained_pareto.pdf`
'''
(ROOT / 'OPTIMISATION_REPORT.md').write_text(report)
print(report)

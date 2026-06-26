#!/usr/bin/env python3
import csv, json, math, statistics, zipfile
from pathlib import Path

ROOT = Path('runtime_comparison_results')
RAW = ROOT / 'raw'
SUMMARY = ROOT / 'summary'
SUMMARY.mkdir(parents=True, exist_ok=True)

def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    r = (len(values)-1) * p / 100.0
    lo = math.floor(r); hi = math.ceil(r)
    if lo == hi:
        return values[lo]
    f = r - lo
    return values[lo]*(1-f) + values[hi]*f

def summarize(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(values)) if values else 0.0
    p99 = percentile(values, 99)
    return {
        'n': len(values),
        'mean_ms': mean,
        'stddev_ms': sd,
        'p50_ms': percentile(values, 50),
        'p95_ms': percentile(values, 95),
        'p99_ms': p99,
        'coefficient_of_variation': sd / mean if mean else None,
        'confidence_interval_95_ms': [mean - ci, mean + ci],
        'fps_from_mean': 1000.0 / mean if mean else None,
        'fps_from_p99': 1000.0 / p99 if p99 else None,
    }

def read_shader():
    rows = list(csv.DictReader(open(RAW / 'shader_native_raw.csv')))
    return {
        'gpu': summarize([float(r['gpu_ms']) for r in rows if r['gpu_ms'] and r['gpu_ms'] != 'NA']),
        'cpu': summarize([float(r['cpu_ms']) for r in rows if r['cpu_ms'] and r['cpu_ms'] != 'NA']),
    }

def read_onnx():
    rows = list(csv.DictReader(open(RAW / 'onnx_runtime_raw.csv')))
    return {'cpu': summarize([float(r['cpu_ms']) for r in rows])}

def read_shadernn_status():
    path = RAW / 'shadernn_status.txt'
    return path.read_text().strip() if path.exists() else 'ShaderNN status file missing.'

def apk_info(path):
    p = Path(path)
    names = []
    with zipfile.ZipFile(p) as z:
        for info in z.infolist():
            names.append(info.filename)
    return {
        'apk_path': str(p),
        'apk_size_bytes': p.stat().st_size,
        'contains_ml_runtime': any(any(token in name.lower() for token in ['onnx', 'tensorflow', 'tflite', 'torch', 'tensorrt']) for name in names),
        'native_libraries': [name for name in names if name.endswith('.so')],
    }

shader = read_shader()
onnx = read_onnx()
shadernn_status = read_shadernn_status()
comparison = {
    'device': {
        'source': str(RAW / 'device_info.txt'),
        'summary': 'TECNO KL4, Android 14, arm64-v8a, ARM Mali-G57, OpenGL ES 3.2, GL_EXT_disjoint_timer_query available',
    },
    'dataset_input': {
        'source': 'surreal/zd_ecn_subset/smpl_data.npz',
        'pose_key': 'pose_01_01',
        'input': 'first SURREAL pose converted from 24x axis-angle to 144D rot6d',
    },
    'shader_native': {
        'description': 'ZD-ECN plain GLSL ES compute shader path; ShaderNN-related shader baseline but not the ShaderNN framework.',
        'gpu': shader['gpu'],
        'cpu': shader['cpu'],
        'apk': apk_info('android_compute_benchmark/app/build/outputs/apk/debug/app-debug.apk'),
    },
    'onnx_runtime': {
        'description': 'Separate ONNX Runtime Android CPU baseline app using matching 144->64->64->20670 ONNX model.',
        'cpu': onnx['cpu'],
        'apk': apk_info('onnx_runtime_benchmark/app/build/outputs/apk/debug/app-debug.apk'),
    },
    'shadernn': {
        'description': 'ShaderNN framework baseline requested using inferenceengine/shadernn.',
        'status': 'not_run',
        'reason': shadernn_status,
        'raw_status_file': str(RAW / 'shadernn_status.txt'),
    },
    'paper_reference': {
        'shader_gpu_mean_ms': 1.324,
        'shader_gpu_p95_ms': 1.369,
        'shader_gpu_p99_ms': 1.398,
        'shader_fps_mean': 755.3,
        'onnx_runtime_estimated_latency_ms': 14.0,
        'onnx_runtime_footprint_mb': 78.6,
        'shadernn_footprint_mb': 5.0,
    },
    'caveats': [
        'ShaderNN was inspected from inferenceengine/shadernn, but exact ZD-ECN conversion failed: original ONNX fails on MatMul; Gemm-form opset-11 equivalent fails on Pow in GELU.',
        'ONNX Runtime result is CPU execution through onnxruntime-android, not GPU/NNAPI unless ORT selects internal acceleration by default.',
        'Shader path uses fp16 SSBO weights; ONNX model stores fp32 weights for ONNX Runtime CPU compatibility.',
        'Both paths use the same SURREAL-derived 144D input and generated matching model topology.',
    ],
}
(SUMMARY / 'runtime_comparison_summary.json').write_text(json.dumps(comparison, indent=2) + '\n')

def f(x, d=3): return 'NA' if x is None else f'{x:.{d}f}'
lines = [
    '# ZD-ECN Runtime Comparison', '',
    '## Device And Input', '',
    f"- Device: {comparison['device']['summary']}.",
    "- Dataset input: `surreal/zd_ecn_subset/smpl_data.npz`, pose key `pose_01_01`, converted to 144D rot6d.",
    '', '## Results', '',
    '| Runtime | Timer | Samples | Mean ms | Std ms | p50 ms | p95 ms | p99 ms | FPS mean | FPS p99 | APK size | ML runtime libs |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
]
sg = comparison['shader_native']['gpu']; sc = comparison['shader_native']['cpu']; oc = comparison['onnx_runtime']['cpu']
shader_apk_mb = comparison['shader_native']['apk']['apk_size_bytes'] / (1024*1024)
onnx_apk_mb = comparison['onnx_runtime']['apk']['apk_size_bytes'] / (1024*1024)
lines.append(f"| ZD-ECN GLSL shader native | GPU timer | {sg['n']} | {f(sg['mean_ms'])} | {f(sg['stddev_ms'])} | {f(sg['p50_ms'])} | {f(sg['p95_ms'])} | {f(sg['p99_ms'])} | {f(sg['fps_from_mean'],1)} | {f(sg['fps_from_p99'],1)} | {shader_apk_mb:.2f} MB | no |")
lines.append(f"| ZD-ECN GLSL shader native | CPU+glFinish | {sc['n']} | {f(sc['mean_ms'])} | {f(sc['stddev_ms'])} | {f(sc['p50_ms'])} | {f(sc['p95_ms'])} | {f(sc['p99_ms'])} | {f(sc['fps_from_mean'],1)} | {f(sc['fps_from_p99'],1)} | {shader_apk_mb:.2f} MB | no |")
lines.append(f"| ONNX Runtime Android | CPU timer | {oc['n']} | {f(oc['mean_ms'])} | {f(oc['stddev_ms'])} | {f(oc['p50_ms'])} | {f(oc['p95_ms'])} | {f(oc['p99_ms'])} | {f(oc['fps_from_mean'],1)} | {f(oc['fps_from_p99'],1)} | {onnx_apk_mb:.2f} MB | yes |")
lines.append("| ShaderNN framework | not run | 0 | NA | NA | NA | NA | NA | NA | NA | ~5 MB paper reference | yes |")
lines += ['', '## Notes', '']
for caveat in comparison['caveats']:
    lines.append(f'- {caveat}')
lines += [
    '',
    '## ShaderNN Status',
    '',
    '- Exact ShaderNN run: not reported because conversion failed before a valid Android benchmark package could be produced.',
    '- Details: `runtime_comparison_results/raw/shadernn_status.txt`',
    '',
    '## Files',
    '',
    '- Raw shader CSV: `runtime_comparison_results/raw/shader_native_raw.csv`',
    '- Raw ONNX CSV: `runtime_comparison_results/raw/onnx_runtime_raw.csv`',
    '- ShaderNN status: `runtime_comparison_results/raw/shadernn_status.txt`',
    '- JSON summary: `runtime_comparison_results/summary/runtime_comparison_summary.json`',
]
(SUMMARY / 'runtime_comparison_summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

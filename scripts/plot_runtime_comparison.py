#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root = Path('runtime_comparison_results')
plots = root / 'plots'
plots.mkdir(parents=True, exist_ok=True)

def read_csv(path, column):
    with open(path, newline='') as f:
        return [float(r[column]) for r in csv.DictReader(f) if r[column] and r[column] != 'NA']

shader_gpu = read_csv(root / 'raw/shader_native_raw.csv', 'gpu_ms')
shader_cpu = read_csv(root / 'raw/shader_native_raw.csv', 'cpu_ms')
onnx_cpu = read_csv(root / 'raw/onnx_runtime_fp16_cpu_raw.csv', 'cpu_ms')
onnx_nnapi = read_csv(root / 'raw/onnx_runtime_fp16_nnapi_raw.csv', 'cpu_ms')

plt.figure(figsize=(8,5))
for label, values in [
    ('ZD-ECN GPU-only', shader_gpu),
    ('ZD-ECN CPU wall', shader_cpu),
    ('ONNX fp16 CPU EP', onnx_cpu),
    ('ONNX fp16 NNAPI EP', onnx_nnapi),
]:
    ordered = sorted(values)
    y = [(i + 1) / len(ordered) for i in range(len(ordered))]
    plt.plot(ordered, y, label=label)
plt.xlabel('Latency (ms)')
plt.ylabel('CDF')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(plots / 'runtime_latency_cdf.png', dpi=160)
plt.close()

plt.figure(figsize=(8,5))
plt.boxplot(
    [shader_gpu, shader_cpu, onnx_cpu, onnx_nnapi],
    tick_labels=['ZD-ECN GPU', 'ZD-ECN CPU', 'ONNX CPU', 'ONNX NNAPI'],
)
plt.ylabel('Latency (ms)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(plots / 'runtime_latency_boxplot.png', dpi=160)
plt.close()

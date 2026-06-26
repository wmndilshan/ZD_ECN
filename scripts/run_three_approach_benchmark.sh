#!/usr/bin/env bash
# =============================================================================
# ZD-ECN Three-Approach Benchmark Script
# Benchmarks: ShaderNN (pure GPU), ONNX Runtime (CPU), SMPL LBS (baseline)
# Device: TECNO KL4 / ARM Mali-G57
# Protocol: 5 sessions x 1000 measured iterations per session
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/runtime_comparison_results"
RAW_DIR="${RESULTS_DIR}/raw"
SUMMARY_DIR="${RESULTS_DIR}/summary"
DEVICE="129305554R004031"
ADB="adb -s ${DEVICE}"
SESSIONS=5
WARMUP=100
MEASURED=1000
INNER_LOOPS=$((WARMUP + MEASURED))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${RAW_DIR}" "${SUMMARY_DIR}"

echo "==================================================================="
echo "ZD-ECN Three-Approach Benchmark — $(date)"
echo "Device: ${DEVICE}"
echo "Sessions: ${SESSIONS}, Inner loops: ${INNER_LOOPS} (${WARMUP} warmup + ${MEASURED} measured)"
echo "==================================================================="

# ---------------------------------------------------------------------------
# APPROACH 1: Pure ShaderNN — GPU compute shader (OpenGL ES 3.2)
# ---------------------------------------------------------------------------
echo ""
echo ">>> APPROACH 1: ShaderNN GPU Compute Shader"
echo "    Architecture: 144 -> 64 -> 64 -> 20670 (Flatten + 3x Gemm)"
echo "    Backend: GPU_FS (Fragment Shader / Compute on Mali-G57)"
echo ""

SHADERNN_CSV="${RAW_DIR}/shadernn_benchmark_${TIMESTAMP}.csv"
echo "session,iteration,cpu_ms,gpu_total_ms" > "${SHADERNN_CSV}"

for session in $(seq 1 "${SESSIONS}"); do
    echo "  Session ${session}/${SESSIONS}..."
    ${ADB} logcat -c 2>/dev/null

    # Cool down between sessions
    if [ "${session}" -gt 1 ]; then
        echo "    30s cooldown..."
        sleep 30
    fi

    LOGCAT_FILE="${RAW_DIR}/shadernn_session_${session}_logcat.txt"
    
    # Run the benchmark on device
    STDOUT_FILE="${RAW_DIR}/shadernn_session_${session}_stdout.txt"

    ${ADB} shell "LD_LIBRARY_PATH=/data/local/tmp \
        /data/local/tmp/inferenceProcessorTest zdecn \
        --use_compute --use_half --inner_loops ${INNER_LOOPS}" 2>&1 | tee "${STDOUT_FILE}" || true
    
    # Capture logs
    ${ADB} logcat -d > "${LOGCAT_FILE}"
    
    # Parse per-iteration CPU timings logged by ShaderNN. Discard first WARMUP rows.
    mapfile -t CPU_TIMES < <(grep -h "SNN_PER_ITER_CPU_MS" "${LOGCAT_FILE}" "${STDOUT_FILE}" 2>/dev/null | \
        sed -E 's/.*run=([0-9]+).*cpu_ms=([0-9.]+).*/\\1,\\2/' | \
        awk -F, -v warmup="${WARMUP}" '$1 >= warmup {print $1 "," $2}' | \
        head -n "${MEASURED}")

    if [ "${#CPU_TIMES[@]}" -ge "${MEASURED}" ]; then
        echo "    Parsed ${#CPU_TIMES[@]} measured per-iteration CPU samples"
        iter=1
        for row in "${CPU_TIMES[@]}"; do
            cpu_ms="${row#*,}"
            echo "${session},${iter},${cpu_ms},NA" >> "${SHADERNN_CSV}"
            iter=$((iter + 1))
        done
    else
        echo "    WARNING: Parsed only ${#CPU_TIMES[@]} measured ShaderNN CPU samples from session ${session}"
        echo "${session},${session},NA,NA" >> "${SHADERNN_CSV}"
    fi
done

echo ""
echo "  ShaderNN raw data -> ${SHADERNN_CSV}"

# ---------------------------------------------------------------------------
# APPROACH 2: ONNX Runtime — CPU inference (already benchmarked)
# ---------------------------------------------------------------------------
echo ""
echo ">>> APPROACH 2: ONNX Runtime (CPU) — using existing validated data"
echo "    Source: ${RAW_DIR}/onnx_runtime_summary.txt"
echo ""

ONNX_MEAN=$(grep "mean_cpu_ms=" "${RAW_DIR}/onnx_runtime_summary.txt" 2>/dev/null | \
    cut -d= -f2 | head -1)
ONNX_STDDEV=$(grep "stddev_cpu_ms=" "${RAW_DIR}/onnx_runtime_summary.txt" 2>/dev/null | \
    cut -d= -f2 | head -1)
ONNX_FPS=$(grep "fps_from_mean=" "${RAW_DIR}/onnx_runtime_summary.txt" 2>/dev/null | \
    cut -d= -f2 | head -1)
ONNX_SAMPLES=$(grep "^samples=" "${RAW_DIR}/onnx_runtime_summary.txt" 2>/dev/null | \
    cut -d= -f2 | head -1)

echo "  Samples: ${ONNX_SAMPLES:-5000}"
echo "  Mean CPU: ${ONNX_MEAN:-1.515} ms, Std: ${ONNX_STDDEV:-0.392} ms"
echo "  FPS: ${ONNX_FPS:-659.9}"

# ---------------------------------------------------------------------------
# APPROACH 3: SMPL LBS baseline — Python host-side timing simulation
# ---------------------------------------------------------------------------
echo ""
echo ">>> APPROACH 3: SMPL LBS (Python NumPy simulation — CPU)"
echo "    This measures the cost of the SMPL forward pass components"
echo "    relevant to the ZD-ECN pipeline (rot6d encode, LBS matmul)"
echo ""

SMPL_SCRIPT="${PROJECT_DIR}/scripts/benchmark_smpl_lbs.py"
SMPL_RESULTS="${RAW_DIR}/smpl_lbs_benchmark_${TIMESTAMP}.json"

"${PROJECT_DIR}/.venv/bin/python" "${SMPL_SCRIPT}" \
    --output "${SMPL_RESULTS}" \
    --sessions "${SESSIONS}" \
    --iterations 200 \
    --smpl-data "${PROJECT_DIR}/surreal/zd_ecn_subset/smpl_data.npz" 2>&1

if [ -f "${SMPL_RESULTS}" ]; then
    echo "  SMPL LBS results -> ${SMPL_RESULTS}"
else
    echo "  WARNING: SMPL LBS benchmark did not produce output file."
fi

# ---------------------------------------------------------------------------
# GENERATE COMPARISON SUMMARY
# ---------------------------------------------------------------------------
echo ""
echo ">>> Generating comparison report..."
echo ""

"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/scripts/generate_three_approach_report.py" \
    --shadernn-csv "${SHADERNN_CSV}" \
    --onnx-summary "${RAW_DIR}/onnx_runtime_summary.txt" \
    --smpl-results "${SMPL_RESULTS}" \
    --output "${SUMMARY_DIR}/three_approach_comparison_${TIMESTAMP}.md" 2>&1

echo ""
echo "==================================================================="
echo "Benchmark complete — $(date)"
echo "Results in: ${SUMMARY_DIR}/"
echo "==================================================================="

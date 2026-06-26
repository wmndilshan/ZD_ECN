#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$PROJECT_DIR/android_compute_benchmark"
RESULTS_ROOT="$PROJECT_DIR/benchmark_results"
PACKAGE="com.graphicsresearch.androidcomputebenchmark"
ACTIVITY="$PACKAGE/.BenchmarkActivity"
SESSIONS="${SESSIONS:-5}"
EXPECTED_ROWS="${EXPECTED_ROWS:-1000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SESSION_COOLDOWN_SECONDS="${SESSION_COOLDOWN_SECONDS:-30}"
RUN_SHADERNN="${RUN_SHADERNN:-0}"
THERMAL_CHECK_INTERVAL_SECONDS="${THERMAL_CHECK_INTERVAL_SECONDS:-60}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COMMAND_LOG=""

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

log_command() {
  printf '%s\n' "$*" | tee -a "$COMMAND_LOG" >/dev/null
}

run_logged() {
  log_command "$*"
  "$@"
}

backup_existing_outputs() {
  mkdir -p "$RESULTS_ROOT"
  local backup_dir="$RESULTS_ROOT/backups/$TIMESTAMP"
  local moved=0
  for name in raw summary plots; do
    if [[ -e "$RESULTS_ROOT/$name" ]]; then
      mkdir -p "$backup_dir"
      mv "$RESULTS_ROOT/$name" "$backup_dir/$name"
      moved=1
    fi
  done
  if [[ "$moved" == "1" ]]; then
    echo "Previous benchmark outputs moved to $backup_dir"
  fi
}

adb_shell() {
  adb shell "$@"
}

thermal_status_line() {
  adb shell dumpsys thermalservice | grep "Thermal Status" | tr -d '\r' || true
}

wait_for_thermal_none() {
  local context="$1"
  while true; do
    local line
    line="$(thermal_status_line)"
    log_command "adb shell dumpsys thermalservice | grep \"Thermal Status\"  # $context -> ${line:-unavailable}"
    echo "Thermal check ($context): ${line:-unavailable}"
    if [[ "$line" == *"Thermal Status: 0"* ]]; then
      return 0
    fi
    echo "Thermal status is not NONE; waiting ${THERMAL_CHECK_INTERVAL_SECONDS}s"
    log_command "sleep $THERMAL_CHECK_INTERVAL_SECONDS"
    sleep "$THERMAL_CHECK_INTERVAL_SECONDS"
  done
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || fail "Required tool not found: $1"
}

line_count_on_device() {
  adb shell "run-as $PACKAGE sh -c 'if [ -f files/inference_log.txt ]; then wc -l < files/inference_log.txt; else echo 0; fi'" \
    | tr -dc '0-9'
}

pull_app_file() {
  local remote="$1"
  local local_path="$2"
  log_command "adb shell run-as $PACKAGE cat $remote > $local_path"
  adb shell run-as "$PACKAGE" cat "$remote" > "$local_path"
}

collect_device_info() {
  local out="$RESULTS_ROOT/raw/device_info.txt"
  {
    echo "adb devices:"
    adb devices
    echo
    echo "getprop:"
    for prop in \
      ro.product.manufacturer \
      ro.product.model \
      ro.product.device \
      ro.build.version.release \
      ro.build.version.sdk \
      ro.product.cpu.abi \
      ro.hardware \
      ro.board.platform; do
      printf '%s=' "$prop"
      adb shell getprop "$prop" | tr -d '\r'
    done
    echo
    echo "SurfaceFlinger GLES hints:"
    adb shell dumpsys SurfaceFlinger 2>/dev/null | grep -Ei 'GLES|OpenGL|GPU' || true
  } > "$out"
}

append_gl_info_from_logs() {
  local out="$RESULTS_ROOT/raw/device_info.txt"
  {
    echo
    echo "App OpenGL logs:"
    grep -hE 'Initialized benchmark|OpenGL ES|gpu_timer|renderer=|vendor=|version=' "$RESULTS_ROOT"/raw/session_*_logcat.txt 2>/dev/null || true
  } >> "$out"
}

check_zero_dependency_apk() {
  local apk="$1"
  local out="$RESULTS_ROOT/raw/zero_dependency_check.txt"
  {
    echo "APK: $apk"
    echo
    echo "Native libraries in APK:"
    if command -v unzip >/dev/null 2>&1; then
      unzip -l "$apk" | awk '/\\.so$/ {print $4}'
    else
      echo "unzip unavailable"
    fi
    echo
    echo "ML runtime library name scan:"
    if command -v unzip >/dev/null 2>&1; then
      if unzip -l "$apk" | grep -Eiq 'onnx|tensorflow|tflite|torch|tensorrt|nvinfer|libonnx|libtensorflow|libtorch'; then
        echo "Potential ML runtime dependency found."
      else
        echo "No ONNX/TensorFlow/TFLite/Torch/TensorRT library names found in APK contents."
      fi
    fi
    echo
    echo "Dynamic dependencies for libandroid_compute_benchmark.so:"
    if command -v unzip >/dev/null 2>&1 && command -v readelf >/dev/null 2>&1; then
      local tmp_so
      tmp_so="$(mktemp -p "${TMPDIR:-/tmp}" zd_ecn_so_XXXXXX)"
      unzip -p "$apk" 'lib/arm64-v8a/libandroid_compute_benchmark.so' > "$tmp_so" || true
      if [[ -s "$tmp_so" ]]; then
        readelf -d "$tmp_so" | grep NEEDED || true
      else
        echo "Unable to extract arm64 native library."
      fi
      rm -f "$tmp_so"
    else
      echo "unzip/readelf unavailable"
    fi
  } > "$out"
}

run_shadernn_benchmark() {
  local shadernn_dir="$PROJECT_DIR/shadernn"
  local shadernn_android="$shadernn_dir/demo/android"
  local shadernn_out="$PROJECT_DIR/results/shadernn"
  local shadernn_raw="$shadernn_out/raw"
  local shadernn_summary="$shadernn_out/summary"
  local status_file="$RESULTS_ROOT/summary/shadernn_status.md"

  mkdir -p "$shadernn_raw" "$shadernn_summary" "$RESULTS_ROOT/summary"
  [[ -d "$shadernn_android" ]] || fail "RUN_SHADERNN=1 but ShaderNN Android project not found: $shadernn_android"
  [[ -d "$shadernn_dir/modelzoo/ZDECN" ]] || fail "RUN_SHADERNN=1 but ShaderNN ZDECN model not found: $shadernn_dir/modelzoo/ZDECN"

  local shadernn_gradle
  if [[ -x "$shadernn_android/../../../android_compute_benchmark/gradlew" ]]; then
    shadernn_gradle="$shadernn_android/../../../android_compute_benchmark/gradlew"
  elif [[ -x "$shadernn_android/gradlew" ]]; then
    shadernn_gradle="$shadernn_android/gradlew"
  else
    fail "No usable Gradle wrapper found for ShaderNN Android project"
  fi

  echo "Running optional ShaderNN benchmark stage"
  log_command "cd $shadernn_android && $shadernn_gradle -p . assembleRelease assembleReleaseAndroidTest"
  (cd "$shadernn_android" && "$shadernn_gradle" -p . assembleRelease assembleReleaseAndroidTest)

  local app_apk="$shadernn_android/app/build/outputs/apk/release/app-release.apk"
  local test_apk="$shadernn_android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
  [[ -f "$app_apk" ]] || fail "ShaderNN app APK not found: $app_apk"
  [[ -f "$test_apk" ]] || fail "ShaderNN test APK not found: $test_apk"

  run_logged adb shell mkdir -p /data/local/tmp/jsonModel
  run_logged adb push "$shadernn_dir/modelzoo/ZDECN" /data/local/tmp/jsonModel/ZDECN
  run_logged adb install -r "$app_apk"
  run_logged adb install -r "$test_apk"
  run_logged adb logcat -c

  set +e
  log_command "adb shell am instrument -w -e class com.oppo.seattle.snndemo.NativeTests#zdecnCS32 com.innopeaktech.seattle.snndemo.test/com.oppo.seattle.snndemo.CustomInstrumentationRunner"
  adb shell am instrument -w \
    -e class com.oppo.seattle.snndemo.NativeTests#zdecnCS32 \
    com.innopeaktech.seattle.snndemo.test/com.oppo.seattle.snndemo.CustomInstrumentationRunner \
    | tee "$shadernn_raw/zdecn_instrumentation_stdout_${TIMESTAMP}.txt"
  local shadernn_rc="${PIPESTATUS[0]}"
  set -e

  log_command "adb logcat -d -v time > $shadernn_raw/zdecn_instrumentation_logcat_${TIMESTAMP}.txt"
  adb logcat -d -v time > "$shadernn_raw/zdecn_instrumentation_logcat_${TIMESTAMP}.txt"

  {
    echo "# ShaderNN Benchmark Status"
    echo
    echo "- Timestamp: $TIMESTAMP"
    echo "- Exit code: $shadernn_rc"
    echo "- App APK: \`$app_apk\`"
    echo "- Test APK: \`$test_apk\`"
    echo "- Stdout: \`$shadernn_raw/zdecn_instrumentation_stdout_${TIMESTAMP}.txt\`"
    echo "- Logcat: \`$shadernn_raw/zdecn_instrumentation_logcat_${TIMESTAMP}.txt\`"
    echo
    if grep -q 'Final Time Stats\\|Benchmark end\\|Time returned via GPU elapsed time query' "$shadernn_raw/zdecn_instrumentation_logcat_${TIMESTAMP}.txt"; then
      echo "Timing output was found in logcat; parse manually before reporting results."
    elif grep -qi 'lowmemorykiller\\|Fatal signal\\|SIGSEGV\\|Process crashed' "$shadernn_raw/zdecn_instrumentation_logcat_${TIMESTAMP}.txt" "$shadernn_raw/zdecn_instrumentation_stdout_${TIMESTAMP}.txt"; then
      echo "No valid ShaderNN timing output was found. Logcat/stdout contain crash or low-memory evidence."
    else
      echo "No valid ShaderNN timing output was found. Review logs for the exact failure."
    fi
    echo
    echo "Do not report ShaderNN latency/FPS unless per-iteration or benchmark timing samples are present."
  } > "$status_file"
  cp "$status_file" "$shadernn_summary/shadernn_status_${TIMESTAMP}.md"
}

prepare_python() {
  if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    log_command "python3 -m venv .venv"
    python3 -m venv "$PROJECT_DIR/.venv"
  fi
  # requirements.txt intentionally has no mandatory third-party dependencies.
  log_command ".venv/bin/python -m pip install -r requirements.txt"
  "$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
}

wait_for_session() {
  local session="$1"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    local lines
    lines="$(line_count_on_device || echo 0)"
    lines="${lines:-0}"
    if (( lines >= EXPECTED_ROWS + 1 )); then
      return 0
    fi
    sleep 2
  done
  fail "Session $session timed out waiting for $EXPECTED_ROWS measured rows"
}

main() {
  backup_existing_outputs
  mkdir -p "$RESULTS_ROOT/raw" "$RESULTS_ROOT/summary" "$RESULTS_ROOT/plots" "$RESULTS_ROOT/runs/$TIMESTAMP"
  COMMAND_LOG="$RESULTS_ROOT/raw/commands_executed.txt"
  : > "$COMMAND_LOG"

  require_tool adb
  require_tool python3
  if [[ -n "${GRADLE_BIN:-}" ]]; then
    [[ -x "$GRADLE_BIN" ]] || fail "GRADLE_BIN is not executable: $GRADLE_BIN"
    GRADLE_CMD=("$GRADLE_BIN" "-p" "$APP_DIR")
  elif [[ -x "$APP_DIR/gradlew" ]]; then
    GRADLE_CMD=("$APP_DIR/gradlew" "-p" "$APP_DIR")
  else
    require_tool gradle
    GRADLE_CMD=("gradle" "-p" "$APP_DIR")
  fi

  prepare_python

  log_command "adb devices"
  adb devices | tee "$RESULTS_ROOT/raw/adb_devices.txt"
  adb get-state >/dev/null || fail "No adb device detected or device is unauthorized"
  collect_device_info

  log_command "rm -rf $APP_DIR/app/.cxx"
  rm -rf "$APP_DIR/app/.cxx"
  run_logged "${GRADLE_CMD[@]}" assembleRelease
  APK="$APP_DIR/app/build/outputs/apk/release/app-release.apk"
  if [[ ! -f "$APK" && -f "$APP_DIR/app/build/outputs/apk/release/app-release-unsigned.apk" ]]; then
    APK="$APP_DIR/app/build/outputs/apk/release/app-release-unsigned.apk"
  fi
  [[ -f "$APK" ]] || fail "APK not found after build: $APK"
  check_zero_dependency_apk "$APK"

  if ! run_logged adb install -r "$APK"; then
    echo "Initial install failed. Uninstalling existing $PACKAGE and retrying install."
    run_logged adb uninstall "$PACKAGE"
    run_logged adb install "$APK"
  fi
  run_logged adb shell run-as "$PACKAGE" pwd >/dev/null

  for session in $(seq 1 "$SESSIONS"); do
    session_padded="$(printf '%02d' "$session")"
    echo "Running benchmark session $session/$SESSIONS"
    wait_for_thermal_none "before session $session"
    run_logged adb shell am force-stop "$PACKAGE"
    log_command "adb shell run-as $PACKAGE rm -f files/inference_log.txt files/inference_log.txt.summary.txt"
    adb shell run-as "$PACKAGE" rm -f files/inference_log.txt files/inference_log.txt.summary.txt
    run_logged adb logcat -c
    run_logged adb shell am start -n "$ACTIVITY" --es benchmark_mode zdecn
    wait_for_session "$session"
    pull_app_file "files/inference_log.txt" "$RESULTS_ROOT/raw/session_${session_padded}.csv"
    pull_app_file "files/inference_log.txt.summary.txt" "$RESULTS_ROOT/raw/session_${session_padded}.summary.txt"
    log_command "adb logcat -d -v time > $RESULTS_ROOT/raw/session_${session_padded}_logcat.txt"
    adb logcat -d -v time > "$RESULTS_ROOT/raw/session_${session_padded}_logcat.txt"
    run_logged adb shell am force-stop "$PACKAGE"
    if (( session < SESSIONS && SESSION_COOLDOWN_SECONDS > 0 )); then
      echo "Cooling down for ${SESSION_COOLDOWN_SECONDS}s before next session"
      log_command "sleep $SESSION_COOLDOWN_SECONDS"
      sleep "$SESSION_COOLDOWN_SECONDS"
      wait_for_thermal_none "after cooldown $session"
    fi
  done

  append_gl_info_from_logs

  mapfile -t session_csvs < <(find "$RESULTS_ROOT/raw" -maxdepth 1 -name 'session_*.csv' | sort)
  [[ "${#session_csvs[@]}" -eq "$SESSIONS" ]] || fail "Expected $SESSIONS session CSVs, found ${#session_csvs[@]}"
  log_command ".venv/bin/python scripts/parse_benchmark_results.py --output-root benchmark_results --session-csv ${session_csvs[*]}"
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/parse_benchmark_results.py" \
    --output-root "$RESULTS_ROOT" \
    --session-csv "${session_csvs[@]}" \
    | tee "$RESULTS_ROOT/raw/parser_output.json"

  if [[ "$RUN_SHADERNN" == "1" ]]; then
    run_shadernn_benchmark
  fi

  cp -a "$RESULTS_ROOT/raw" "$RESULTS_ROOT/summary" "$RESULTS_ROOT/plots" "$RESULTS_ROOT/runs/$TIMESTAMP/"
  echo "Benchmark complete. Results: $RESULTS_ROOT"
}

main "$@"

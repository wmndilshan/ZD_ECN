#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/android_smpl_app"
PACKAGE="com.graphicsresearch.smplrenderer"
ACTIVITY="$PACKAGE/.BenchmarkActivity"
SESSIONS="${SESSIONS:-5}"
WARMUP="${WARMUP:-100}"
ITERATIONS="${ITERATIONS:-1000}"
COOLDOWN_MS="${COOLDOWN_MS:-30000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"
BUILD_VARIANT="${BUILD_VARIANT:-release}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v adb >/dev/null 2>&1 || fail "adb is not installed or not on PATH"

if [[ -n "${ADB_SERIAL:-}" ]]; then
  SERIAL="$ADB_SERIAL"
else
  mapfile -t DEVICES < <(adb devices | awk 'NR>1 && $2=="device" {print $1}')
  [[ ${#DEVICES[@]} -eq 1 ]] || fail "expected exactly one authorized adb device; found ${#DEVICES[@]}"
  SERIAL="${DEVICES[0]}"
fi

adb -s "$SERIAL" get-state >/dev/null 2>&1 || fail "device $SERIAL is not available"

case "$BUILD_VARIANT" in
  release)
    GRADLE_TASK="assembleRelease"
    APK="$APP_DIR/app/build/outputs/apk/release/app-release.apk"
    ;;
  debug)
    GRADLE_TASK="assembleDebug"
    APK="$APP_DIR/app/build/outputs/apk/debug/app-debug.apk"
    ;;
  *) fail "BUILD_VARIANT must be release or debug" ;;
esac

printf 'Device: %s\n' "$SERIAL"
printf 'Protocol: %s sessions, %s warmups, %s iterations, %sms cooldown\n' \
  "$SESSIONS" "$WARMUP" "$ITERATIONS" "$COOLDOWN_MS"

(cd "$APP_DIR" && ./gradlew "$GRADLE_TASK")
[[ -f "$APK" ]] || fail "APK was not generated: $APK"
adb -s "$SERIAL" install -r "$APK"

REMOTE_ROOT="/sdcard/Android/data/$PACKAGE/files/benchmark_results"
BEFORE="$(adb -s "$SERIAL" shell "ls -1 '$REMOTE_ROOT' 2>/dev/null || true" | tr -d '\r')"

adb -s "$SERIAL" shell am force-stop "$PACKAGE" || true
adb -s "$SERIAL" shell am start -n "$ACTIVITY" \
  --ei sessions "$SESSIONS" \
  --ei warmup "$WARMUP" \
  --ei iterations "$ITERATIONS" \
  --el cooldown_ms "$COOLDOWN_MS" >/dev/null

RUN_ID=""
START_TIME="$(date +%s)"
while true; do
  NOW="$(date +%s)"
  (( NOW - START_TIME <= TIMEOUT_SECONDS )) || fail "benchmark timed out after ${TIMEOUT_SECONDS}s"
  LATEST="$(adb -s "$SERIAL" shell "ls -1 '$REMOTE_ROOT' 2>/dev/null | tail -1" | tr -d '\r')"
  if [[ -n "$LATEST" ]] && ! grep -qxF "$LATEST" <<<"$BEFORE"; then
    RUN_ID="$LATEST"
    break
  fi
  sleep "$POLL_SECONDS"
done

PROGRESS="$REMOTE_ROOT/$RUN_ID/benchmark_progress.txt"
while true; do
  NOW="$(date +%s)"
  (( NOW - START_TIME <= TIMEOUT_SECONDS )) || fail "benchmark timed out after ${TIMEOUT_SECONDS}s"
  STATUS="$(adb -s "$SERIAL" shell "cat '$PROGRESS' 2>/dev/null || true" | tr -d '\r')"
  [[ -n "$STATUS" ]] && printf '[%s] %s\n' "$(date +%H:%M:%S)" "$STATUS"
  if grep -q '^complete samples=' <<<"$STATUS"; then
    break
  fi
  if grep -q '^failed:' <<<"$STATUS"; then
    fail "$STATUS"
  fi
  sleep "$POLL_SECONDS"
done

LOCAL_ROOT="$ROOT_DIR/benchmark_results/app_combined_$RUN_ID"
if [[ -e "$LOCAL_ROOT" ]]; then
  LOCAL_ROOT="${LOCAL_ROOT}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOCAL_ROOT"
adb -s "$SERIAL" pull "$REMOTE_ROOT/$RUN_ID" "$LOCAL_ROOT"

SUMMARY="$LOCAL_ROOT/$RUN_ID/summary/benchmark_summary.md"
[[ -f "$SUMMARY" ]] || fail "summary was not pulled: $SUMMARY"
printf '\nCompleted: %s\n' "$LOCAL_ROOT/$RUN_ID"
cat "$SUMMARY"

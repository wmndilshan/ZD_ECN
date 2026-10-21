#!/usr/bin/env bash
# ZD-ECN cross-GPU validation benchmark — Vivo Y85 (Adreno 506, Android 9).
#
# Mirrors the canonical Mali-G57 protocol (run_device_benchmark.sh /
# benchmark_results/raw/commands_executed.txt):
#   5 sessions x (100 warmup discarded + 1000 measured), 30 s cooldowns,
#   thermal gate before each session.
#
# Android 9 deltas vs the canonical script:
#   - thermalservice has no "Thermal Status" on SDK 28 -> gate on raw
#     /sys/class/thermal battery zone (< BATT_MAX_MC milli-C).
#   - vivo kills backgrounded activities -> screen is forced on (svc power
#     stayon usb) and the activity relaunched per session.
#   - USB link is flaky (observed corrupt pushes) -> every pulled file is
#     re-read and byte-compared until two consecutive reads agree.
set -Eeuo pipefail

SERIAL="${SERIAL:-8e33900f}"
PACKAGE="com.graphicsresearch.androidcomputebenchmark"
ACTIVITY="$PACKAGE/.BenchmarkActivity"
SESSIONS="${SESSIONS:-5}"
EXPECTED_ROWS="${EXPECTED_ROWS:-1000}"
COOLDOWN_S="${COOLDOWN_S:-30}"
TIMEOUT_S="${TIMEOUT_S:-300}"
BATT_MAX_MC="${BATT_MAX_MC:-37000}"   # gate: battery below 37.0 C
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/adreno506"

mkdir -p "$OUT_DIR"

adbv() { adb -s "$SERIAL" "$@"; }

battery_temp_mc() {
    # vivo Y85 battery zone: type "battery", value in milli-C
    adbv shell "for z in /sys/class/thermal/thermal_zone*; do \
        if [ \"\$(cat \$z/type 2>/dev/null)\" = battery ]; then cat \$z/temp; fi; done" \
        | tr -d '\r' | head -1
}

thermal_gate() {
    local ctx="$1"
    while true; do
        local t
        t="$(battery_temp_mc)"
        if [ -z "$t" ]; then
            echo "[$ctx] battery thermal zone unreadable — proceeding (logged)" | tee -a "$OUT_DIR/thermal_log.txt"
            return
        fi
        echo "[$ctx] battery ${t} mC" | tee -a "$OUT_DIR/thermal_log.txt"
        if [ "$t" -lt "$BATT_MAX_MC" ]; then return; fi
        echo "[$ctx] above gate (${BATT_MAX_MC} mC) — waiting 60 s"
        sleep 60
    done
}

pull_verified() {
    # cat the on-device file twice; accept only when both reads agree
    # (the USB link on this host/cable pair corrupts ~50% of large transfers).
    local remote="$1" local_path="$2"
    for attempt in 1 2 3 4 5; do
        adbv shell cat "$remote" > "$local_path.a" 2>/dev/null
        adbv shell cat "$remote" > "$local_path.b" 2>/dev/null
        if [ -s "$local_path.a" ] && cmp -s "$local_path.a" "$local_path.b"; then
            mv "$local_path.a" "$local_path"; rm -f "$local_path.b"
            return 0
        fi
        echo "  pull mismatch for $remote (attempt $attempt) — retrying"
    done
    echo "ERROR: could not verify pull of $remote" >&2
    return 1
}

adbv shell svc power stayon usb
adbv shell input keyevent KEYCODE_WAKEUP || true

for s in $(seq 1 "$SESSIONS"); do
    tag=$(printf "session_%02d" "$s")
    echo "=== $tag ==="
    thermal_gate "before $tag"

    adbv shell am force-stop "$PACKAGE"
    adbv shell rm -f /sdcard/inference_log.txt /sdcard/inference_log.txt.summary.txt
    adbv shell am start -n "$ACTIVITY" --es benchmark_mode zdecn >/dev/null

    # The app rewrites the summary periodically DURING the run; completion is
    # marked by "Phase: realtime" in the summary, not by file existence.
    elapsed=0
    until adbv shell cat /sdcard/inference_log.txt.summary.txt 2>/dev/null | grep -q "Phase: realtime"; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$TIMEOUT_S" ]; then
            echo "ERROR: $tag timed out after ${TIMEOUT_S}s" >&2
            exit 1
        fi
    done
    sleep 2   # let the writer finish flushing

    pull_verified /sdcard/inference_log.txt             "$OUT_DIR/$tag.csv"
    pull_verified /sdcard/inference_log.txt.summary.txt "$OUT_DIR/$tag.summary.txt"

    rows=$(( $(wc -l < "$OUT_DIR/$tag.csv") ))
    echo "  rows: $rows (header included)"
    grep -E "GPU last/mean|GPU timer" "$OUT_DIR/$tag.summary.txt" || true

    adbv shell am force-stop "$PACKAGE"
    if [ "$s" -lt "$SESSIONS" ]; then
        echo "  cooldown ${COOLDOWN_S}s"
        sleep "$COOLDOWN_S"
    fi
done

adbv shell svc power stayon false

echo "All $SESSIONS sessions complete -> $OUT_DIR"

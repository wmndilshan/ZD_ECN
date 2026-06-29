#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/android_smpl_app"
PACKAGE="com.graphicsresearch.smplrenderer"
ACTIVITY="$PACKAGE/.MainActivity"
BUILD_VARIANT="${BUILD_VARIANT:-release}"
CAPTURE_WAIT_SECONDS="${CAPTURE_WAIT_SECONDS:-6}"
OUTPUT_DIR="$ROOT_DIR/results/paper/screenshots/comparison_4x3"

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

case "$BUILD_VARIANT" in
  release) GRADLE_TASK="assembleRelease"; APK="$APP_DIR/app/build/outputs/apk/release/app-release.apk" ;;
  debug) GRADLE_TASK="assembleDebug"; APK="$APP_DIR/app/build/outputs/apk/debug/app-debug.apk" ;;
  *) fail "BUILD_VARIANT must be release or debug" ;;
esac

(cd "$APP_DIR" && ./gradlew "$GRADLE_TASK")
[[ -f "$APK" ]] || fail "APK was not generated: $APK"
adb -s "$SERIAL" install -r "$APK"

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  BACKUP="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
  mv "$OUTPUT_DIR" "$BACKUP"
  printf 'Existing comparison set moved to %s\n' "$BACKUP"
fi
mkdir -p "$OUTPUT_DIR"
adb -s "$SERIAL" logcat -c

BACKEND_NAMES=(native onnx_cpu onnx_nnapi lbs_only)
BACKEND_IDS=(0 2 3 4)
TIER_NAMES=(natural moderate extreme)
TIER_IDS=(0 1 2)
FRAMES=(80 280 470)

for backend_index in "${!BACKEND_NAMES[@]}"; do
  backend_name="${BACKEND_NAMES[$backend_index]}"
  backend_id="${BACKEND_IDS[$backend_index]}"
  for tier_index in "${!TIER_NAMES[@]}"; do
    tier_name="${TIER_NAMES[$tier_index]}"
    tier_id="${TIER_IDS[$tier_index]}"
    frame="${FRAMES[$tier_index]}"
    adb -s "$SERIAL" shell am force-stop "$PACKAGE" || true
    adb -s "$SERIAL" shell am start -n "$ACTIVITY" \
      --ei backend "$backend_id" --ei tier "$tier_id" --ei frame "$frame" >/dev/null
    sleep "$CAPTURE_WAIT_SECONDS"
    remote="/sdcard/zdecn_${backend_name}_${tier_name}.png"
    local_path="$OUTPUT_DIR/${backend_name}_${tier_name}.png"
    adb -s "$SERIAL" shell screencap -p "$remote"
    adb -s "$SERIAL" pull "$remote" "$local_path" >/dev/null
    printf 'Captured %s / %s\n' "$backend_name" "$tier_name"
  done
done

adb -s "$SERIAL" logcat -d | grep -E 'OnnxBackend|SmplRenderer|FATAL|AndroidRuntime' \
  > "$OUTPUT_DIR/logcat_backend_init.txt" || true

NNAPI_STATUS="not confirmed"
if grep -q 'sessions initialized: cpu=true nnapi=true' "$OUTPUT_DIR/logcat_backend_init.txt"; then
  NNAPI_STATUS="confirmed"
fi

cat > "$OUTPUT_DIR/comparison_manifest.txt" <<EOF
Protocol: fixed-frame portrait screenshots, smooth GPU normals, no CPU vertex readback
Pose frames: natural=80, moderate=280, extreme=470
Backends: native ZD-ECN, ONNX CPU, ONNX NNAPI requested, LBS-only
NNAPI session creation: $NNAPI_STATUS
Image count: 12
EOF

if command -v python3 >/dev/null 2>&1 && python3 -c 'from PIL import Image' >/dev/null 2>&1; then
  OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(os.environ['OUTPUT_DIR'])
backends = ['native', 'onnx_cpu', 'onnx_nnapi', 'lbs_only']
tiers = ['natural', 'moderate', 'extreme']
cell_w, cell_h = 180, 400
sheet = Image.new('RGB', (cell_w * 3, cell_h * 4), (8, 12, 18))
for row, backend in enumerate(backends):
    for column, tier in enumerate(tiers):
        image = Image.open(root / f'{backend}_{tier}.png').convert('RGB').resize((cell_w, cell_h))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, cell_w, 24), fill=(0, 0, 0))
        draw.text((6, 6), f'{backend} / {tier}', fill=(255, 255, 255))
        sheet.paste(image, (column * cell_w, row * cell_h))
sheet.save(root / 'comparison_contact_sheet.png')
PY
fi

COUNT="$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.png' ! -name 'comparison_contact_sheet.png' | wc -l)"
[[ "$COUNT" -eq 12 ]] || fail "expected 12 core screenshots; found $COUNT"
printf 'Comparison set complete: %s\n' "$OUTPUT_DIR"

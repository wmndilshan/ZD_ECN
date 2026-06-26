#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$PROJECT_DIR/surreal/zd_ecn_subset}"
MANIFEST_DIR="$PROJECT_DIR/surreal/manifests"
MANIFEST="$MANIFEST_DIR/zd_ecn_smpl_subset.txt"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || fail "Required tool not found: $1"
}

write_manifest() {
  mkdir -p "$MANIFEST_DIR"
  cat > "$MANIFEST" <<'EOF'
https://lsh.paris.inria.fr/SURREAL/smpl_data/smpl_data.npz
https://lsh.paris.inria.fr/SURREAL/smpl_data/female_beta_stds.npy
https://lsh.paris.inria.fr/SURREAL/smpl_data/male_beta_stds.npy
EOF
}

main() {
  require_tool wget

  [[ -n "${SURREAL_USERNAME:-}" ]] || fail "Set SURREAL_USERNAME in the environment"
  [[ -n "${SURREAL_PASSWORD:-}" ]] || fail "Set SURREAL_PASSWORD in the environment"

  write_manifest
  mkdir -p "$OUT_DIR"

  echo "Downloading ZD-ECN-relevant SURREAL subset only:"
  echo "- smpl_data/smpl_data.npz"
  echo "- smpl_data/female_beta_stds.npy"
  echo "- smpl_data/male_beta_stds.npy"
  echo "Output: $OUT_DIR"
  echo "Manifest: $MANIFEST"

  download_one "https://lsh.paris.inria.fr/SURREAL/smpl_data/smpl_data.npz" \
    "$OUT_DIR/smpl_data.npz" \
    "2595025134"
  download_one "https://lsh.paris.inria.fr/SURREAL/smpl_data/female_beta_stds.npy" \
    "$OUT_DIR/female_beta_stds.npy" \
    "2480"
  download_one "https://lsh.paris.inria.fr/SURREAL/smpl_data/male_beta_stds.npy" \
    "$OUT_DIR/male_beta_stds.npy" \
    "2480"

  {
    echo "Downloaded files:"
    find "$OUT_DIR" -type f -printf '%s %p\n' | sort -n
    echo
    echo "Total size:"
    du -sh "$OUT_DIR"
  } | tee "$OUT_DIR/download_report.txt"
}

download_one() {
  local url="$1"
  local target="$2"
  local expected_bytes="$3"
  local temp="${target}.download"

  if [[ -f "$target" ]]; then
    local current_size
    current_size="$(stat -c '%s' "$target")"
    if [[ "$current_size" == "$expected_bytes" ]]; then
      echo "Already complete: $target ($current_size bytes)"
      return 0
    fi
    local backup="${target}.incomplete.${TIMESTAMP}"
    echo "Moving incomplete file to: $backup ($current_size of $expected_bytes bytes)"
    mv "$target" "$backup"
  fi

  if [[ -f "$temp" ]]; then
    local temp_size
    temp_size="$(stat -c '%s' "$temp")"
    local backup_temp="${temp}.incomplete.${TIMESTAMP}"
    echo "Moving previous temporary file to: $backup_temp ($temp_size bytes)"
    mv "$temp" "$backup_temp"
  fi

  echo "Downloading: $url"
  wget \
    --user="$SURREAL_USERNAME" \
    --password="$SURREAL_PASSWORD" \
    --tries=5 \
    --timeout=60 \
    --read-timeout=60 \
    --retry-connrefused \
    --progress=dot:giga \
    --output-document="$temp" \
    "$url"

  local downloaded_size
  downloaded_size="$(stat -c '%s' "$temp")"
  if [[ "$downloaded_size" != "$expected_bytes" ]]; then
    fail "Downloaded size mismatch for $target: expected $expected_bytes bytes, got $downloaded_size bytes"
  fi
  mv "$temp" "$target"
  echo "Complete: $target ($downloaded_size bytes)"
}

main "$@"

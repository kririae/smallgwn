#!/usr/bin/env bash
# Optional visual smoke capture (not a correctness gate).

set -euo pipefail

APP_PATH="${1:-}"
if [[ -z "${APP_PATH}" ]]; then
    echo "Usage: $(basename "$0") <path-to-winding_studio>" >&2
    exit 1
fi
if [[ ! -x "${APP_PATH}" ]]; then
    echo "Not executable: ${APP_PATH}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
E2E_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_DIR="${E2E_DIR}/artifacts"
mkdir -p "${ARTIFACT_DIR}"

command -v Xvfb >/dev/null 2>&1 || { echo "Missing Xvfb" >&2; exit 1; }
command -v xdotool >/dev/null 2>&1 || { echo "Missing xdotool" >&2; exit 1; }
command -v import >/dev/null 2>&1 || { echo "Missing import" >&2; exit 1; }

DISPLAY_NUM=131
WIDTH=1280
HEIGHT=720

cleanup_lock() {
    kill "$(cat "/tmp/.X${DISPLAY_NUM}-lock" 2>/dev/null)" 2>/dev/null || true
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null || true
}

cleanup() {
    kill "${APP_PID:-0}" 2>/dev/null || true
    kill "${XVFB_PID:-0}" 2>/dev/null || true
    cleanup_lock
}
trap cleanup EXIT

cleanup_lock
Xvfb ":${DISPLAY_NUM}" -screen 0 "${WIDTH}x${HEIGHT}x24" >/dev/null 2>&1 &
XVFB_PID=$!
export DISPLAY=":${DISPLAY_NUM}"
sleep 0.5

"${APP_PATH}" --width "${WIDTH}" --height "${HEIGHT}" &
APP_PID=$!
sleep 2.5

WID=""
for _ in $(seq 1 40); do
    WID="$(xdotool search --name "Winding Studio" 2>/dev/null | head -1)"
    [[ -n "${WID}" ]] && break
    sleep 0.2
done
[[ -n "${WID}" ]] || { echo "window not found" >&2; exit 1; }

xdotool windowfocus --sync "${WID}" 2>/dev/null || true
xdotool windowraise "${WID}" 2>/dev/null || true
sleep 0.3

capture() {
    local name="$1"
    import -window root "${ARTIFACT_DIR}/${name}.png"
    echo "[visual] captured ${ARTIFACT_DIR}/${name}.png"
}

click_toolbar() {
    local idx="$1" # 0 split, 1 raster, 2 harnack, 3 voxel
    local button_w=124
    local spacing=8
    local right_pad=8
    local start_x=$((WIDTH - right_pad - (button_w * 4 + spacing * 3)))
    local x=$((start_x + idx * (button_w + spacing) + button_w / 2))
    local y=27
    xdotool mousemove --window "${WID}" "${x}" "${y}"
    xdotool click --window "${WID}" 1
}

capture "visual_01_split"
click_toolbar 1; sleep 0.4; capture "visual_02_raster"
click_toolbar 2; sleep 0.6; capture "visual_03_harnack"
click_toolbar 3; sleep 0.6; capture "visual_04_voxel"
click_toolbar 0; sleep 0.4; capture "visual_05_split_back"

echo "[visual] smoke capture complete"

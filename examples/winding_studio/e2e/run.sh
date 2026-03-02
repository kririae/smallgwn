#!/usr/bin/env bash
# Winding Studio end-to-end pipeline.
# Layers:
#   1) unit (pure logic)
#   2) ui_contract (ImGui interaction + state assertions)
#   3) visual (optional screenshot smoke)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
APP_DIR="${REPO_ROOT}/examples/winding_studio"

BUILD_DIR="${APP_DIR}/build"
RUN_VISUAL=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --visual)
            RUN_VISUAL=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --build-dir <dir>  CMake build directory (default: examples/winding_studio/build)
  --visual           Run optional visual smoke capture after test layers
  --skip-build       Skip configure/build step
  -h, --help         Show this help
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    cmake -S "${APP_DIR}" -B "${BUILD_DIR}" -DBUILD_TESTING=ON -DWINDING_STUDIO_ENABLE_IMGUI_TEST_ENGINE=ON
    cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

echo "[e2e] Running layer: unit"
ctest --test-dir "${BUILD_DIR}" --output-on-failure -L unit

echo "[e2e] Running layer: ui_contract"
ctest --test-dir "${BUILD_DIR}" --output-on-failure -L ui_contract

if [[ "${RUN_VISUAL}" -eq 1 ]]; then
    echo "[e2e] Running layer: visual_smoke"
    "${SCRIPT_DIR}/visual/smoke_modes.sh" "${BUILD_DIR}/winding_studio"
fi

echo "[e2e] Pipeline completed"

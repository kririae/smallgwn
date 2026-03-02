#!/usr/bin/env bash
# CI entrypoint: run deterministic e2e gates (unit + ui_contract only).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-cubql-comparison}"
: "${CUBQL_SOURCE:?Set CUBQL_SOURCE to a local cuBQL source checkout}"

cmake \
  -S "${repo_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSMALLGWN_BUILD_EIGEN_BRIDGE=OFF \
  -DSMALLGWN_BUILD_CUBQL_BENCHMARK=ON \
  -DSMALLGWN_CUBQL_SOURCE="${CUBQL_SOURCE}"

cmake --build "${build_dir}" --target smallgwn_benchmark_cubql -j "${JOBS:-$(nproc)}"

"${build_dir}/tests/smallgwn_benchmark_cubql" "$@"

#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-cubql-comparison}"

cmake \
  -S "${repo_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSMALLGWN_BUILD_EIGEN_BRIDGE=OFF \
  -DSMALLGWN_BUILD_TESTS=OFF \
  -DSMALLGWN_BUILD_BENCHMARKS=ON

cmake --build "${build_dir}" --target smallgwn_benchmark_cubql -j "${JOBS:-$(nproc)}"

"${build_dir}/tests/smallgwn_benchmark_cubql" "$@"

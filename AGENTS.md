# AGENTS.md

## Scope
These instructions apply to the `smallgwn/` project tree.

## Project Snapshot
- `smallgwn` is a header-only CUDA/C++ library for winding number evaluation on triangle meshes.
- Public surface is centered under `include/gwn/` with umbrella include `include/gwn/gwn.cuh`.
- This codebase is independent from the parent `WindingNumber` source; parent code is reference-only.

## Language and Build Baseline
- Use C++20 and CUDA 12+.
- Target `sm_86+`.
- Keep the project header-only at the library level.

## File Naming Rules
- If a header contains CUDA callable annotations (`__device__`, `__global__`, or mixed host/device kernel interfaces), use the `.cuh` extension.
- Keep pure host-side helpers in `.hpp`.

## Naming and API Rules
- Use `gwn::` namespace and `gwn_` prefix for public symbols.
- Avoid `_wide` naming in public API types.
- Keep width as template parameter where relevant; default aliases are width=4 (`gwn_bvh_accessor`, `gwn_bvh_data4_accessor`).

## Formatting Rules
- Use `clang-format` with Chromium style via `smallgwn/.clang-format`.
- Run format after each code change touching C++/CUDA files.

## Architecture Rules
- Keep public include surface focused on runtime components; CPU reference helpers belong under `tests/`.
- Use SoA layout for geometry (`x/y/z`, `i0/i1/i2`).
- Prefer TBB for trivially parallel CPU-side batch work.
- Host owning objects should stay non-copyable and use accessor-centric RAII.
- Keep stream allocator path only (`cudaMallocAsync`/`cudaFreeAsync`), no legacy fallback path.

## Current BVH/Taylor Design
- Topology and data are separated:
  - Topology: `gwn_bvh_topology_accessor<Width,...>` / `gwn_bvh_topology_object<Width,...>`
  - Data tree: `gwn_bvh_data_tree_accessor<Width,...>` / `gwn_bvh_data_tree_object<Width,...>`
- LBVH topology build is GPU-centric (`gwn_build_bvh_lbvh<Width,...>` + `gwn_build_bvh4_lbvh`).
- Taylor build currently supports `Order=0/1`:
  - Production path: `gwn_build_bvh4_lbvh_taylor<Order,...>` uses fully GPU async upward propagation with atomics.
  - Reference path: `gwn_build_bvh4_lbvh_taylor_levelwise<Order,...>` is compiled only when `GWN_ENABLE_TAYLOR_LEVELWISE_REFERENCE` is defined (tests only).
- Winding number query APIs:
  - Exact: `gwn_compute_winding_number_batch_bvh_exact`
  - Taylor: `gwn_compute_winding_number_batch_bvh_taylor<Order,...>`

## Error Handling Rules
- Prefer internal C++ exceptions only inside implementation details; public APIs should return `gwn_status` error codes.
- For executable boundaries, prefer a single top-level exception translation block:
  - `int main() try { ... } catch (...) { ... }`
- Use a dedicated `catch` block to translate exceptions to status/diagnostics, rather than scattered local catches.

## Test Structure
- `tests/smoke_compile.cu`
  - API compile/smoke sanity for geometry/BVH/taylor entry points.
- `tests/parity_scaffold.cu`
  - Small synthetic parity checks (CPU exact vs GPU exact/taylor).
  - Includes vendored HDK headers for oracle helpers.
- `tests/bvh_model_parity.cu`
  - Integration tests on OBJ models (exact/taylor consistency, model-level parity, optional benchmark).
  - Uses env:
    - `SMALLGWN_MODEL_DATA_DIR` (default fallback: `/tmp/common-3d-test-models/data`)
    - `SMALLGWN_RUN_TAYLOR_BENCHMARK=1`
    - `SMALLGWN_TAYLOR_BENCH_REPEATS`
- `tests/gwn_correctness_models.cu`
  - Model correctness with voxelized query coverage and CPU/GPU consistency checks.
  - Uses env:
    - `SMALLGWN_MODEL_PATH`, `SMALLGWN_MODEL_DATA_DIR`
    - `SMALLGWN_VOXEL_TOTAL_POINTS`, `SMALLGWN_VOXEL_QUERY_CHUNK_SIZE`
    - `SMALLGWN_CPU_WORK_BUDGET`, `SMALLGWN_CPU_MAX_SAMPLES`, `SMALLGWN_CPU_MIN_SAMPLES`
- `tests/reference_cpu.hpp`
  - TBB-parallel CPU exact reference implementation.
- `tests/reference_hdk/*`
  - Vendored HDK reference sources for parity and regression checks; keep under `tests/`, not public include.

## Validation
- Configure/build: `cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j`
- Run tests: `ctest --test-dir smallgwn/build --output-on-failure`
- Note: test targets explicitly define `GWN_ENABLE_TAYLOR_LEVELWISE_REFERENCE` in `smallgwn/CMakeLists.txt`.

## Git Commit Style
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).

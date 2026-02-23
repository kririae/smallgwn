# AGENTS.md

## Scope
These instructions apply to the `smallgwn/` project tree.

## Project Snapshot
- `smallgwn` is a header-only CUDA/C++ library for winding number evaluation on triangle meshes.
- Public surface is centered under `include/gwn/` with umbrella include `include/gwn/gwn.cuh`.
- This codebase is independent from the `WindingNumber` source; `tests/reference_hdk/` contains vendored HDK sources for parity checks and code reference.

## Language and Build Baseline
- Use C++20 and CUDA 12+.
- Target `sm_86+`.
- Keep the project header-only at the library level.

## File Naming Rules
- **`.cuh` (CUDA Headers)**: Must be used if the file contains ANY of the following:
  - CUDA execution space specifiers (`__device__`, `__global__`, `__host__ __device__`).
  - CUDA kernel launches (`<<<...>>>`).
  - CUDA Runtime API calls (e.g., `cudaMallocAsync`, `cudaStream_t`).
  - Includes of other `.cuh` files.
- **`.hpp` (Pure Host Headers)**: Strictly reserved for pure C++ CPU-side code. Must compile cleanly with standard C++ compilers without NVCC.
- *Note*: `include/gwn/gwn_utils.cuh` is CUDA-aware and must stay `.cuh`.

## Naming and API Rules
- Use `gwn::` namespace and `gwn_` prefix for public symbols.
- Avoid `_wide` naming in public API types.
- Keep width as template parameter where relevant; default aliases are width=4 (`gwn_bvh_accessor`, `gwn_bvh_data4_accessor`).

## Formatting Rules
- Use `clang-format` with Chromium style via `smallgwn/.clang-format`.
- Run format after each code change touching C++/CUDA files.

## Architecture & Design Rules
- Keep public include surface focused on runtime components; CPU reference helpers belong under `tests/`.
- Use SoA (Structure of Arrays) layout for geometry (`x/y/z`, `i0/i1/i2`).
- Prefer TBB for trivially parallel CPU-side batch work.
- **Stream Binding**: Stream binding is explicit:
  - owning objects (`gwn_geometry_object`, `gwn_bvh_topology_object`, `gwn_bvh_data_tree_object`) use `gwn_stream_mixin`;
  - `clear()`/destructor release on the currently bound stream;
  - successful stream-parameterized mutations update the bound stream.
- **Memory Allocation**: Keep stream allocator path ONLY (`cudaMallocAsync`/`cudaFreeAsync`). Never use legacy synchronous fallback paths. `gwn_device_array<T>` remembers stream and releases on its bound stream by default.

## C++ Resource Management & Idioms
- **RAII & Non-copyable**: Host owning objects should generally stay non-copyable (delete copy constructor/assignment) to prevent accidental deep copies of GPU memory. Use accessor-centric RAII.
- **Copy/Move-and-Swap Idiom**: If custom assignment operators must be implemented for resource-managing classes, strictly adhere to the `copy-and-swap` (or `move-and-swap`) idiom. Provide a `noexcept` swap member function to ensure strong exception safety, avoid self-assignment bugs, and ensure streams are handled correctly during assignment.
- **Rule of Five**: If you define a destructor, explicitly define or delete the other four special member functions.

## Current BVH/Taylor Design
- Topology and data are separated:
  - Topology: `gwn_bvh_topology_accessor<Width,...>` / `gwn_bvh_topology_object<Width,...>`
  - Data tree: `gwn_bvh_data_tree_accessor<Width,...>` / `gwn_bvh_data_tree_object<Width,...>`
- `include/gwn/gwn_bvh_build.cuh` keeps public build APIs, while internal build passes/functors live in flat `include/gwn/detail/` headers.
- LBVH topology build is GPU-centric (`gwn_build_bvh_lbvh<Width,...>` + `gwn_build_bvh4_lbvh`).
- LBVH build path uses CUB for scene reduction and radix sort (NO Thrust in runtime build path).
- Taylor build currently supports `Order=0/1`:
  - Production path: `gwn_build_bvh4_lbvh_taylor<Order,...>` uses fully GPU async upward propagation with atomics.
  - Reference path: `gwn_build_bvh4_lbvh_taylor_levelwise<Order,...>` is always compiled and available.
  - Async Taylor temporary buffers (parent/slot/arity/arrivals/pending moments) use `gwn_device_array`.
- Winding number query APIs:
  - Exact: `gwn_compute_winding_number_batch_bvh_exact`
  - Taylor: `gwn_compute_winding_number_batch_bvh_taylor<Order,...>`

## Error Handling Rules
- Prefer internal C++ exceptions only inside implementation details; public APIs should return `gwn_status` error codes.
- For executable boundaries, prefer a single top-level exception translation block:
  - `int main() try { ... } catch (...) { ... }`
- Use a dedicated `catch` block to translate exceptions to status/diagnostics, rather than scattered local catches.

## Test Structure & Environments
- `tests/smoke_compile.cu`: API compile/smoke sanity for geometry/BVH/taylor entry points.
- `tests/parity_scaffold.cu`: Small synthetic parity checks (CPU exact vs GPU exact/taylor). Includes stream-binding regression checks and vendored HDK headers for oracle helpers.
- `tests/bvh_model_parity.cu`: Integration tests on OBJ models. 
  - *Env vars*: `SMALLGWN_MODEL_DATA_DIR` (default `/tmp/common-3d-test-models/data`), `SMALLGWN_RUN_TAYLOR_BENCHMARK=1`, `SMALLGWN_TAYLOR_BENCH_REPEATS`.
- `tests/gwn_correctness_models.cu`: Model correctness with voxelized query coverage.
  - *Env vars*: `SMALLGWN_MODEL_PATH`, `SMALLGWN_MODEL_DATA_DIR`, `SMALLGWN_VOXEL_TOTAL_POINTS`, `SMALLGWN_VOXEL_QUERY_CHUNK_SIZE`, `SMALLGWN_CPU_WORK_BUDGET`, `SMALLGWN_CPU_MAX_SAMPLES`, `SMALLGWN_CPU_MIN_SAMPLES`.
- `tests/reference_cpu.hpp`: TBB-parallel CPU exact reference implementation.
- `tests/reference_hdk/*`: Vendored HDK reference sources for parity/regression checks (keep under `tests/`, not public).

## Validation Workflow
- Configure/build: `cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j`
- Run tests: `ctest --test-dir smallgwn/build --output-on-failure`

## Git & Maintenance Rules
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).
- **AGENTS Maintenance**: Keep this file current with architecture/API/testing changes. When behavior, naming, file layout, stream/memory semantics, or test entrypoints change, update this `AGENTS.md` in the same commit. Re-check and refresh periodically rather than batching stale updates.
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
- Keep width as template parameter where relevant; width=4 aliases are `gwn_bvh_accessor`, `gwn_bvh_aabb4_accessor`, and `gwn_bvh_moment4_accessor`.

## Formatting Rules
- Use `clang-format` with Chromium style via `smallgwn/.clang-format`.
- Run format after each code change touching C++/CUDA files.

## Architecture & Design Rules
- Keep public include surface focused on runtime components; CPU reference helpers belong under `tests/`.
- Use SoA (Structure of Arrays) layout for geometry (`x/y/z`, `i0/i1/i2`).
- Prefer TBB for trivially parallel CPU-side batch work.
- **Stream Binding**: Stream binding is explicit:
  - owning objects (`gwn_geometry_object`, `gwn_bvh_topology_object`, `gwn_bvh_aabb_object`, `gwn_bvh_moment_object`) use `gwn_stream_mixin`;
  - `clear()`/destructor release on the currently bound stream;
  - successful stream-parameterized mutations update the bound stream;
  - object-first build/refit overloads in `gwn_bvh_build.cuh` must update bound stream on success.
- **Memory Allocation**: Keep stream allocator path ONLY (`cudaMallocAsync`/`cudaFreeAsync`). Never use legacy synchronous fallback paths. `gwn_device_array<T>` remembers stream and releases on its bound stream by default.
  - All `gwn_device_array` methods (`resize`, `clear`, `zero`, `copy_from_host`, `copy_to_host`) and free helpers (`gwn_cuda_free`, `gwn_free_span`, etc.) are `noexcept`.
  - `resize(count, stream)` rebinds the bound stream even when `count` matches the current size (no alloc/free occurs in that case).
  - `resize(count, stream)` allocates new storage on `stream` but releases previous storage on the previously bound stream; bound stream switches to `stream` only after successful commit.
  - `clear(stream)` releases on the currently bound stream, then rebinds to `stream` on success; no-arg `clear()` and destructor release on the bound stream.
  - For span handoff to accessors/objects, use span primitives (`gwn_allocate_span`, `gwn_free_span`, `gwn_copy_h2d`, `gwn_copy_d2h`, `gwn_copy_d2d`) instead of `gwn_device_array`.

## C++ Resource Management & Idioms
- **RAII & Non-copyable**: Host owning objects should generally stay non-copyable (delete copy constructor/assignment) to prevent accidental deep copies of GPU memory. Use accessor-centric RAII.
- **Copy/Move-and-Swap Idiom**: If custom assignment operators must be implemented for resource-managing classes, strictly adhere to the `copy-and-swap` (or `move-and-swap`) idiom. Provide a `noexcept` swap member function to ensure strong exception safety, avoid self-assignment bugs, and ensure streams are handled correctly during assignment.
- **Rule of Five**: If you define a destructor, explicitly define or delete the other four special member functions.

## Current BVH/Taylor Design
- Topology and data are separated:
  - Topology: `gwn_bvh_topology_accessor<Width,...>` / `gwn_bvh_topology_object<Width,...>`
  - AABB tree: `gwn_bvh_aabb_accessor<Width,...>` / `gwn_bvh_aabb_tree_object<Width,...>`
  - Moment tree: `gwn_bvh_moment_accessor<Width,...>` / `gwn_bvh_moment_tree_object<Width,...>`
- `include/gwn/gwn_bvh_build.cuh` keeps public build APIs, while internal build passes/functors live in flat `include/gwn/detail/` headers.
- Build/refit orchestration is split into detail pipeline headers:
  - `gwn_bvh_pipeline_topology.cuh`
  - `gwn_bvh_pipeline_aabb.cuh`
  - `gwn_bvh_pipeline_moment.cuh`
  - `gwn_bvh_pipeline_orchestrator.cuh`
  - shared helpers in `gwn_bvh_pipeline_common.cuh`
- Internal pipeline entrypoints use `_impl` suffix (e.g. `gwn_build_bvh_topology_lbvh_impl`) to avoid public/internal naming collisions.
- Public BVH build/refit entrypoints are object-based (`gwn_geometry_object` + BVH object types); accessor-based build/refit APIs are internal-only under `gwn::detail`.
- No `bvh4_*` convenience wrappers — use `gwn_build_bvh_topology_lbvh<4,...>` directly; the `gwn_bvh_object` / `gwn_bvh_aabb_object` / `gwn_bvh_moment_object` aliases already fix Width=4.
- Public object-based APIs are plain `noexcept` (no `try`/`catch`); exception translation is done once in the `detail` layer.
- LBVH topology build is pass-composed (all internal under `gwn::detail`):
  - binary LBVH pass (`gwn_build_binary_lbvh_topology` in `detail/gwn_bvh_build_lbvh.cuh`)
  - collapse pass (`gwn_collapse_binary_lbvh_topology` in `detail/gwn_bvh_build_lbvh.cuh`)
  - exposed detail entry: `detail::gwn_build_bvh_topology_lbvh_impl<Width,...>`
- Generic async refit kernels/traits live in `include/gwn/detail/gwn_bvh_refit_async.cuh`.
- LBVH build path uses CUB for scene reduction and radix sort (NO Thrust in runtime build path).
- Taylor build currently supports `Order=0/1`:
  - Production path uses fully GPU async upward propagation with atomics through `gwn_refit_bvh_moment`.
  - Async Taylor temporary buffers (parent/slot/arity/arrivals/pending moments) use `gwn_device_array`.
  - Each `gwn_refit_bvh_moment<Order,...>` call replaces the entire moment accessor (full replace, not merge). To maintain multiple Taylor orders simultaneously, use separate moment objects.
- Winding number query APIs:
  - Exact: `gwn_compute_winding_number_batch_bvh_exact`
  - Taylor: `gwn_compute_winding_number_batch_bvh_taylor<Order,...>`

## Error Handling Rules
- Prefer internal C++ exceptions only inside implementation details; public APIs should return `gwn_status` error codes.
- **`gwn_status` is fully `noexcept`**: Factory methods (`invalid_argument`, `internal_error`, `cuda_runtime_error`) and all helpers (`gwn_cuda_to_status`, `gwn_cuda_malloc`, `gwn_cuda_free`) are `noexcept`. `std::string`/`std::format` OOM is treated as a fatal precondition violation (same policy as `operator new` in embedded/GPU contexts).
- **`GWN_RETURN_ON_ERROR(expr)`**: Stores the result in `gwn_status_result_` (not `gwn_status`) to avoid shadowing the type name in the enclosing scope.
- For executable boundaries, prefer a single top-level exception translation block:
  - `int main() try { ... } catch (...) { ... }`
- Use a dedicated `catch` block to translate exceptions to status/diagnostics, rather than scattered local catches.

## Test Structure & Environments
- Unit tests:
  - `tests/unit_status.cu`
  - `tests/unit_device_array.cu`
  - `tests/unit_stream_mixin.cu`
  - `tests/unit_geometry.cu`
  - `tests/unit_bvh_topology.cu`
  - `tests/unit_bvh_taylor.cu`
  - `tests/unit_winding_exact.cu`
  - `tests/unit_winding_taylor.cu`
- Integration tests:
  - `tests/integration_model_parity.cu`
  - `tests/integration_correctness.cu`
  - Exact-heavy model parity/correctness cases are intentionally compiled out with `#if 0` during current refactor cycle.
  - Model-based integration tests are fail-fast on missing model data (no `GTEST_SKIP` on missing dataset path).
  - Taylor parity includes GPU order-0/order-1 vs HDK CPU order-0/order-1 comparisons.
- `tests/reference_cpu.cuh`: TBB-parallel CPU exact reference implementation.
- `tests/reference_hdk/*`: Vendored HDK reference sources for parity/regression checks (keep under `tests/`, not public).

## Validation Workflow
- Configure/build: `cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j`
- Run tests: `ctest --test-dir smallgwn/build --output-on-failure`

## Git & Maintenance Rules
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).
- **AGENTS Maintenance**: Keep this file current with architecture/API/testing changes. When behavior, naming, file layout, stream/memory semantics, or test entrypoints change, update this `AGENTS.md` in the same commit. Re-check and refresh periodically rather than batching stale updates.

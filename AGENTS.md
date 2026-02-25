# AGENTS.md

## Scope
These instructions apply to the `smallgwn/` project tree.

## Project Snapshot
- `smallgwn` is a header-only CUDA/C++ library for winding number evaluation on triangle meshes.
- Public surface is centered under `include/gwn/` with umbrella include `include/gwn/gwn.cuh`.
  The umbrella include is runtime-focused and does **not** include `gwn_eigen_bridge.hpp`.
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
- Default public index type is `std::uint32_t` for geometry/BVH/query templates unless explicitly overridden.
- Keep width as template parameter where relevant; width=4 accessor aliases are `gwn_bvh_accessor`, `gwn_bvh_aabb4_accessor`, and `gwn_bvh_moment4_accessor` (no `gwn_bvh4_*` duplicates).
- Owning-object state query uses a single unified predicate `has_data()` across all BVH object types (`gwn_bvh_topology_tree_object`, `gwn_bvh_aabb_tree_object`, `gwn_bvh_moment_tree_object`).

## Formatting Rules
- Use `clang-format` with Chromium style via `smallgwn/.clang-format`.
- Run format after each code change touching C++/CUDA files.
- **Internal `#include` paths must always be relative**:
  - Headers inside `include/gwn/` (top-level) reference siblings with bare names, e.g. `"gwn_bvh.cuh"`.
  - Headers inside `include/gwn/detail/` reference sibling detail headers with bare names (e.g. `"gwn_bvh_status_helpers.cuh"`) and parent-level headers with `../` prefix (e.g. `"../gwn_bvh.cuh"`).
  - Never use the rooted `"gwn/..."` form inside library headers; that form is only valid for *consumers* of the library.

## Architecture & Design Rules
- Keep public include surface focused on runtime components; CPU reference helpers belong under `tests/`.
- `include/gwn/gwn_eigen_bridge.hpp` is opt-in and must be explicitly included by users that need Eigen interop.
  Runtime headers (`*.cuh` under `include/gwn/`, except the bridge) must not require Eigen.
- Use SoA (Structure of Arrays) layout for geometry (`x/y/z`, `i0/i1/i2`).
- Prefer TBB for trivially parallel CPU-side batch work.
- **Stream Binding**: Stream binding is explicit:
  - owning objects (`gwn_geometry_object`, `gwn_bvh_topology_object`, `gwn_bvh_aabb_object`, `gwn_bvh_moment_object`) use `gwn_stream_mixin`;
  - `clear()`/destructor release on the currently bound stream;
  - successful stream-parameterized mutations update the bound stream;
  - object-first build/refit/facade overloads in `gwn_bvh_topology_build.cuh`, `gwn_bvh_refit.cuh`,
    and `gwn_bvh_facade.cuh` must update bound stream on success.
- **Memory Allocation**: Keep stream allocator path ONLY (`cudaMallocAsync`/`cudaFreeAsync`). Never use legacy synchronous fallback paths. `gwn_device_array<T>` remembers stream and releases on its bound stream by default.
- **Index Sentinels**: Never hardcode signed `< 0` checks in shared code paths; use `gwn_invalid_index`, `gwn_is_invalid_index`, and `gwn_index_in_bounds` from `gwn_utils.cuh` so both signed and unsigned `Index` instantiations are valid.
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
- Public BVH API is split by responsibility:
  - topology build: `include/gwn/gwn_bvh_topology_build.cuh`
  - payload refit: `include/gwn/gwn_bvh_refit.cuh`
  - composed workflows: `include/gwn/gwn_bvh_facade.cuh`
- Topology build currently exposes both:
  - `gwn_bvh_topology_build_lbvh`
  - `gwn_bvh_topology_build_hploc`
- Facade build currently exposes both LBVH and H-PLOC variants:
  - `gwn_bvh_facade_build_topology_aabb_{lbvh,hploc}`
  - `gwn_bvh_facade_build_topology_aabb_moment_{lbvh,hploc}`
- Internal implementation remains under flat `include/gwn/detail/` headers.
- Query implementation is split similarly:
  - public surface in `include/gwn/gwn_query.cuh`
  - internal traversal/math in:
    - `include/gwn/detail/gwn_query_vec3_impl.cuh`
    - `include/gwn/detail/gwn_query_geometry_impl.cuh`
    - `include/gwn/detail/gwn_query_winding_impl.cuh`
    - `include/gwn/detail/gwn_query_distance_impl.cuh`
- Query geometry math uses internal `gwn_query_vec3` (no Eigen dependency).
- Triangle solid-angle and point-triangle distance primitives are detail-only
  (`gwn::detail::*_impl`), not public `gwn::` API symbols.
- Detail entrypoints use `_impl` suffix (e.g. `gwn_bvh_topology_build_lbvh_impl`) to avoid public/internal naming collisions.
- Public BVH entrypoints are object-based (`gwn_geometry_object` + BVH object types); accessor-based routines are internal-only under `gwn::detail`.
- No `bvh4_*` convenience wrappers — use `gwn_bvh_topology_build_lbvh<4,...>` directly; the `gwn_bvh_object` / `gwn_bvh_aabb_object` / `gwn_bvh_moment_object` aliases already fix Width=4.  `gwn_bvh_object` is topology-only (does **not** include AABB or moment data).
- BVH SoA node structs (`gwn_bvh_topology_node_soa`, `gwn_bvh_aabb_node_soa`, `gwn_bvh_taylor_node_soa`) are 128-byte aligned for `Width=4` to keep node loads cacheline-aligned; non-4 widths keep natural element alignment.
- Device-side BVH traversal stacks call `gwn_trap()` on overflow (device `asm("trap;")`)
  rather than silently skipping nodes.
- Public object-based APIs are plain `noexcept` (no `try`/`catch`); exception translation is done once in the `detail` layer.
- Topology build uses a shared skeleton in `detail/gwn_bvh_topology_build_impl.cuh`:
  - shared preprocess pass (`gwn_bvh_topology_build_preprocess`) computes scene bounds + Morton + radix sort once;
  - builder strategy selects binary construction (`gwn_bvh_topology_build_binary_lbvh` or `gwn_bvh_topology_build_binary_hploc`);
  - shared collapse pass (`gwn_bvh_topology_build_collapse_binary_lbvh`) emits final wide topology.
- Generic async refit kernels/traits live in `include/gwn/detail/gwn_bvh_refit_async.cuh`.
- LBVH build path uses CUB for scene reduction and radix sort (NO Thrust in runtime build path).
- Morton radix sort end-bit tracks `MortonCode` width (`sizeof(MortonCode) * 8`) so `uint32` and
  `uint64` specializations both sort on the correct bit range.
- `detail/gwn_bvh_topology_build_hploc.cuh` contains the production H-PLOC GPU topology path
  (no experimental runtime toggle / no LBVH fallback branch inside H-PLOC builder).
- H-PLOC nearest-neighbour `delta` tie-break keeps a dedicated `uint32 Morton` path using a
  combined `(morton << 32) | primitive_index` key to maintain stable parent selection under equal
  Morton codes.
- H-PLOC kernel includes convergence guards (inner/outer iteration caps + failure flag) and returns
  explicit `gwn_status::internal_error` on non-convergence instead of hanging.
- Taylor build currently supports `Order=0/1`:
  - Production path uses fully GPU async upward propagation with atomics through `gwn_bvh_refit_moment`.
  - Async Taylor temporary buffers (parent/slot/arity/arrivals/pending moments) use `gwn_device_array`.
  - Each `gwn_bvh_refit_moment<Order,...>` call replaces the entire moment accessor (full replace, not merge). To maintain multiple Taylor orders simultaneously, use separate moment objects.
- Winding number query APIs:
  - Exact: `gwn_compute_winding_number_batch_bvh_exact`
  - Taylor: `gwn_compute_winding_number_batch_bvh_taylor<Order,...>`
  - Traversal stack capacity is template-configurable via `StackCapacity` (default:
    `k_gwn_default_traversal_stack_capacity = 64`).
- Distance point-query APIs:
  - `gwn_unsigned_distance_point_bvh` supports world-unit narrow-band culling via
    `culling_band` (default `+infinity` disables culling); return value is clamped
    to `culling_band` when outside the band.
  - `gwn_signed_distance_point_bvh` additionally supports
    `winding_number_threshold` (default `0.5`) for inside/outside sign choice.
  - Signed-distance sign is determined by exact winding traversal even when
    distance magnitude is culling-band clamped.

## Error Handling Rules
- Prefer internal C++ exceptions only inside implementation details; public APIs should return `gwn_status` error codes.
- **`gwn_status` is fully `noexcept`**: Factory methods (`invalid_argument`, `internal_error`, `cuda_runtime_error`) and all helpers (`gwn_cuda_to_status`, `gwn_cuda_malloc`, `gwn_cuda_free`) are `noexcept`. `std::string`/`std::format` OOM is treated as a fatal precondition violation (same policy as `operator new` in embedded/GPU contexts).
- **`GWN_RETURN_ON_ERROR(expr)`**: Stores the result in `gwn_status_result_` (not `gwn_status`) to avoid shadowing the type name in the enclosing scope.
- Build/refit diagnostics should use phase-prefixed helpers from `detail/gwn_bvh_status_helpers.cuh`
  (e.g. `bvh.topology.preprocess`, `bvh.topology.binary.hploc`, `bvh.refit.moment`) for consistent logs.
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
    - includes H-PLOC topology build coverage (`hploc_*` cases), including `uint32 Morton`
      duplicate-centroid stress checks
  - `tests/unit_uint64_compile.cu`
    - explicit template-instantiation compile coverage for `Index=uint64_t` topology builders
  - `tests/unit_bvh_taylor.cu`
    - includes H-PLOC facade build coverage for Taylor order 0/1
  - `tests/unit_winding_exact.cu`
    - includes exact-query parity coverage on H-PLOC topology
  - `tests/unit_winding_taylor.cu`
    - includes Taylor order-0 far-field parity coverage on H-PLOC topology
  - `tests/unit_sdf.cu`
    - host-side analytic checks call `gwn::detail::gwn_point_triangle_distance_squared_impl`
      from `include/gwn/detail/gwn_query_geometry_impl.cuh`
- Integration tests:
  - `tests/integration_model_parity.cu`
  - `tests/integration_correctness.cu`
  - `tests/integration_hploc_performance.cu`
  - Exact-heavy model parity/correctness cases are intentionally compiled out with `#if 0` during current refactor cycle.
  - Model-based integration tests are fail-fast on missing model data (no `GTEST_SKIP` on missing dataset path).
  - Taylor parity includes GPU order-0/order-1 vs HDK CPU order-0/order-1 comparisons.
  - Correctness now includes sampled model checks for order-1 Taylor LBVH/H-PLOC consistency.
  - H-PLOC performance gate enforces topology-build `p50(hploc)/p50(lbvh)` ratio on sampled
    model inputs (default `<= 2.00`, configurable via
    `SMALLGWN_HPLOC_TOPOLOGY_RATIO_LIMIT`; model count via `SMALLGWN_HPLOC_PERF_MODEL_LIMIT`).
- `tests/reference_cpu.cuh`: TBB-parallel CPU exact reference implementation.
- `tests/reference_hdk/*`: Vendored HDK reference sources for parity/regression checks (keep under `tests/`, not public).
- Benchmark executable:
  - `smallgwn_benchmark` from `benchmarks/benchmark_main.cu`
  - Topology builder selection via `--builder <lbvh|hploc>` (recorded in CSV as `topology_builder`)
  - Measures: `topology_build_lbvh`, `refit_aabb`, `refit_moment_o0`, `refit_moment_o1`,
    `facade_o0`, `facade_o1`, `query_taylor_o0`, `query_taylor_o1`
  - Output: console summary + CSV rows (`--csv <path>`)

## Validation Workflow
- Configure/build: `cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j`
- Run tests: `ctest --test-dir smallgwn/build --output-on-failure`
- Run benchmark (single model): `smallgwn/build/smallgwn_benchmark --model /path/to/model.obj --queries 1000000 --warmup 5 --iters 20 --csv smallgwn_bench.csv`
- Run benchmark (model dir): `smallgwn/build/smallgwn_benchmark --model-dir /path/to/models --queries 1000000 --warmup 5 --iters 20 --csv smallgwn_bench.csv`

## Git & Maintenance Rules
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).
- **AGENTS Maintenance**: Keep this file current with architecture/API/testing changes. When behavior, naming, file layout, stream/memory semantics, or test entrypoints change, update this `AGENTS.md` in the same commit. Re-check and refresh periodically rather than batching stale updates.

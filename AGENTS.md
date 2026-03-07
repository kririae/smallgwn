# AGENTS.md

## Scope
These instructions apply to the `smallgwn/` project tree.

## Project Snapshot
- `smallgwn` is a header-only CUDA/C++ library for geometric queries on triangle meshes,
  including winding numbers, winding-number gradients, signed/unsigned distances, edge distances,
  ray first-hit, and Harnack sphere-march ray tracing.
- Public surface is under `include/gwn/` with umbrella include `include/gwn/gwn.cuh`.
  The umbrella include is runtime-focused and does **not** include `gwn_eigen_bridge.cuh`.
- `examples/` contains standalone demo projects that consume `smallgwn` as a third-party library.
  The main library build must **not** be aware of `examples/`; each example has its own `CMakeLists.txt`.
- `tests/reference_hdk/` contains vendored HDK sources for parity checks and code reference.

## Language and Build Baseline
- Use C++20 and CUDA 12+.
- CMake defaults `CMAKE_CUDA_ARCHITECTURES` to `native` unless overridden.
- Keep the project header-only at the library level.

## File Naming Rules
- **`.cuh`**: Must be used if the file contains CUDA execution space specifiers, kernel launches,
  CUDA Runtime API calls, or includes other `.cuh` files.
- **`.hpp`**: Strictly for pure C++ CPU-side code; must compile without NVCC.

## Naming and API Rules
- Use `gwn::` namespace and `gwn_` prefix for public symbols.
- Default public index type is `std::uint32_t` unless explicitly overridden.
- Width-4 convenience aliases: `gwn_bvh4_<kind>_<role>` (e.g. `gwn_bvh4_topology_object`).
- Moment types carry `Order` as a compile-time template parameter (after `Width`).
- Owning-object state queries are converging on a unified `has_data()` predicate; BVH owning
  objects already expose `has_data()`, while `gwn_geometry_object` currently reports state via
  accessor validity and count accessors.
- Detail entrypoints use `_impl` suffix to avoid public/internal naming collisions.

## Formatting Rules
- Format with `clang-format` using the project `.clang-format` (LLVM-based).
  **MUST** run `clang-format` on every changed C++/CUDA file before committing.
- **Internal `#include` paths must always be relative**:
  - `include/gwn/`: bare names (`"gwn_bvh.cuh"`).
  - `include/gwn/detail/`: bare names for siblings, `../` for parent-level.
  - Never use the rooted `"gwn/..."` form inside library headers.

## Architecture & Design Rules

### Public / Detail Split
- Public headers under `include/gwn/` expose the minimal API surface.
  Implementation lives in `include/gwn/detail/`; public headers may `#include` detail headers
  (required for header-only templates), but users should never `#include` detail headers directly.

### Geometry
- SoA layout (`x/y/z`, `i0/i1/i2`).
- `gwn_geometry_object` is the owning GPU geometry container.

### BVH
- **Topology / data separation**: topology, AABB tree, and moment tree are independent types
  (accessor vs object pattern, width-parameterized with width-4 aliases).
- **Public API split**: `gwn_bvh_topology_build.cuh` (topology), `gwn_bvh_refit.cuh` (payload refit),
  `gwn_bvh_facade.cuh` (composed build workflows).
- Topology builders: LBVH and H-PLOC. Facade exposes both variants.
- Taylor moment supports `Order=0/1/2`.
  Each `gwn_bvh_refit_moment<Order,...>` call does a full replace of the moment accessor.
- Public BVH entrypoints are object-based; accessor-based routines are detail-only.

### Query
- Public surface: `include/gwn/gwn_query.cuh`. Query families:
  - `gwn_compute_winding_number_batch_bvh_taylor<Order,...>` — Taylor winding number
  - `gwn_compute_winding_gradient_batch_bvh_taylor<Order,...>` — winding gradient
  - `gwn_unsigned_distance_point_bvh` / `gwn_signed_distance_point_bvh` — point distance (device)
  - `gwn_compute_unsigned_boundary_edge_distance_batch_bvh` — unsigned boundary-edge distance
    (batch)
  - `gwn_ray_first_hit_bvh` / `gwn_compute_ray_first_hit_batch_bvh` — ray first-hit
  - `gwn_harnack_trace_ray_bvh_taylor` / `gwn_compute_harnack_trace_batch_bvh_taylor` — Harnack trace
  - `gwn_hybrid_trace_ray_bvh_taylor` / `gwn_compute_hybrid_trace_batch_bvh_taylor` — hybrid
    first-hit + conditioned Harnack trace
- Internal math uses `gwn_query_vec3` (no Eigen dependency); public alias `gwn::gwn_vec3<Real>`.
- Traversal stack capacity: template `StackCapacity` (default `k_gwn_default_traversal_stack_capacity = 64`).
  Query APIs accept an optional device-side overflow callback; default callback traps via `gwn_trap()`.

### Stream & Memory
- Stream binding is explicit via `gwn_stream_mixin`.
  `clear()`/destructor release on the currently bound stream;
  successful stream-parameterized mutations update the bound stream.
- Memory: stream allocator path only (`cudaMallocAsync`/`cudaFreeAsync`), no synchronous fallback.
- `gwn_device_array<T>`: all methods are `noexcept`; remembers the bound stream; same-size
  `resize()` is a no-op that preserves the current stream binding, while successful reallocating
  mutations update the bound stream.
- For span handoff to accessors/objects, prefer span primitives over `gwn_device_array`.

### Span Rules
- Always use `cuda::std::span`, never bare `std::span`.
- East const on element type: `span<T const>`, never `span<const T>`.
- Pass spans **by value** with top-level `const`:
  - Input: `cuda::std::span<T const> const`
  - Output: `cuda::std::span<T> const`
  - Exception: `gwn_allocate_span` / `gwn_free_span` take `cuda::std::span<T> &` (must mutate the span itself).
- Accessor struct members store mutable spans (`span<T>`). Functor members distinguish
  `span<T const>` (input) vs `span<T>` (output).
- `const_cast` is prohibited except where `cuda::atomic_ref` mandates a mutable reference for load-only operations.

### Index Type
- Default public `Index` is `std::uint32_t`, but all templates must work with `uint64_t` too.
  Never hardcode index type; always use the `Index` template parameter.
- For `uint32_t` paths, check that counts/offsets fit before narrowing and return an error early
  (e.g., triangle count exceeding `std::numeric_limits<Index>::max()`).
- Use `gwn_invalid_index`, `gwn_is_invalid_index`, `gwn_index_in_bounds` from `gwn_utils.cuh`.
  Never hardcode signed `< 0` checks.

## C++ Resource Management & Idioms
- **RAII & Non-copyable**: Owning objects delete copy ctor/assignment.
- **Copy/Move-and-Swap**: Provide `noexcept` swap for strong exception safety.
- **Rule of Five**: Explicit on all five if destructor is defined.

## Error Handling Rules
- Public APIs return `gwn_status`; internal details may use exceptions.
- `gwn_status` and all helpers are fully `noexcept`.
- `GWN_RETURN_ON_ERROR(expr)`: uses `gwn_status_result_` variable name to avoid shadowing.
- Build/refit diagnostics use phase-prefixed helpers from `detail/gwn_bvh_status_helpers.cuh`.
- Executable main: `int main() try { ... } catch (...) { ... }` single translation block.

## Test Structure

### Unit tests
| File | Coverage |
|------|----------|
| `unit_assert.cu` | `GWN_ASSERT` host/device assertion macros |
| `unit_status.cu` | `gwn_status` error code type |
| `unit_device_array.cu` | `gwn_device_array` lifecycle / stream semantics |
| `unit_kernel_utils.cu` | `gwn_launch_linear_kernel` and block/grid helpers |
| `unit_stream_mixin.cu` | `gwn_stream_mixin` stream binding |
| `unit_geometry.cu` | `gwn_geometry_object` upload/accessors |
| `unit_bvh_topology.cu` | LBVH + H-PLOC topology build, Morton stress checks |
| `unit_uint64_compile.cu` | `Index=uint64_t` compile coverage |
| `unit_bvh_taylor.cu` | Taylor moment refit + H-PLOC facade, order 0/1/2 |
| `unit_winding_taylor.cu` | Taylor winding far-field parity on H-PLOC topology |
| `unit_winding_gradient.cu` | Winding gradient batch query |
| `unit_harnack_trace.cu` | Harnack sphere-march traversal |
| `unit_ray_query.cu` | Ray first-hit point and batch queries |
| `unit_hybrid_trace.cu` | Hybrid first-hit + conditioned Harnack trace |
| `unit_sdf.cu` | Point-triangle distance (requires libigl for parity) |

### Integration tests
| File | Coverage |
|------|----------|
| `integration_model_parity.cu` | GPU vs HDK CPU winding parity on model files |
| `integration_correctness.cu` | LBVH/H-PLOC cross-consistency on sampled models |
| `integration_gradient.cu` | Gradient batch on model files |
| `integration_harnack_trace.cu` | Harnack trace on model files |
| `integration_harnack_behavior_match.cu` | Harnack behavior consistency checks |
| `integration_harnack_iteration_stats.cu` | Harnack convergence iteration statistics |
| `integration_hploc_performance.cu` | H-PLOC topology-build ratio gate vs LBVH |
| `integration_taylor_matrix.cu` | Taylor order 0/1/2 matrix (`NO_CTEST`; split into `light`/`heavy` CTest entries) |

### Test support
- `tests/reference_cpu.cuh`: TBB-parallel CPU exact reference.
- `tests/reference_hdk/*`: Vendored HDK sources (keep under `tests/`).
- `tests/test_fixtures.hpp`, `test_utils.hpp`, `test_harnack_meshes.hpp`: shared test utilities.
- `tests/libigl_reference.cpp/.hpp`: libigl CPU parity for SDF tests.
- Model datasets are optional; tests `GTEST_SKIP()` when absent.

### Benchmark
- Executable: `smallgwn_benchmark` from `benchmarks/benchmark_main.cu`.
- Builder selection: `--builder <lbvh|hploc>` (CSV column `topology_builder`).
- Stages: `topology_build_<builder>`, `refit_aabb`, `refit_moment_o{0,1,2}`,
  `facade_o{0,1,2}`, `query_taylor_o{0,1,2}`, `query_ray_first_hit`.
- Output: console summary + CSV via `--csv <path>`.

## Validation Workflow
```bash
cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j
ctest --test-dir smallgwn/build --output-on-failure
```

## Contribution Workflow
1. Create a `git worktree` for non-trivial changes to isolate from the working branch.
2. Commit with Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).
3. Open a PR for review; CI must pass (build + tests).
4. Author addresses review feedback; maintainer merges on pass.

## AGENTS.md Maintenance
Keep this file current with architecture/API/testing changes.
Update in the same commit when behavior, naming, file layout, or test entrypoints change.

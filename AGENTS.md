# AGENTS.md

## Scope
These instructions apply to the `smallgwn/` project tree.

## Language and Build Baseline
- Use C++20 and CUDA 12+.
- CMake defaults `CMAKE_CUDA_ARCHITECTURES` to `native` unless overridden.
- Keep the project header-only at the library level.
- Repo build targets use strict warnings by default; do not add opt-in warning-profile toggles for
  benchmark/test coverage.
- When a build toggle enables a dependency-bearing path, every `find_package(...)` in that path
  must be `REQUIRED` and fail configure immediately on missing packages. Do not use `QUIET`,
  `*_FOUND` fallback branches, skip-on-missing behavior, or local search hacks/workarounds.
- Example consumers should keep normal third-party `CPMAddPackage(...)` flows for `smallgwn`;
  avoid repo-relative source overrides in example `CMakeLists.txt` files.

## File Naming Rules
- **`.cuh`**: Must be used if the file contains CUDA execution space specifiers, kernel launches,
  CUDA Runtime API calls, or includes other `.cuh` files.
- **`.hpp`**: Strictly for pure C++ CPU-side code; must compile without NVCC.

## Naming and API Rules
- Use `gwn::` namespace and `gwn_` prefix for public symbols.
- Default public index type is `std::uint32_t` unless explicitly overridden.
- Width-4 convenience aliases: `gwn_bvh4_<kind>_<role>` (e.g. `gwn_bvh4_topology_object`).
- Moment types carry `Order` as a compile-time template parameter (after `Width`).
- Owning-object state query: unified `has_data()` predicate.
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
- `gwn_boundary_chain_object` stores the algebraic boundary of an oriented triangle-index chain.
  It is geometry-derived data, separate from BVH payloads and Taylor moments.

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
- Public surface: `include/gwn/gwn_query.cuh`. Query families provide device point APIs (for use
  in custom kernels) and batch APIs (host-callable launchers).
- Internal math uses `gwn_query_vec3` (no Eigen dependency); public alias `gwn::gwn_vec3<Real>`.
- Antipodal winding and gradient queries retry coordinate axes when a projected boundary edge is
  singular.
- Traversal stack capacity: template `StackCapacity` (default `k_gwn_default_traversal_stack_capacity = 64`).
  Query APIs accept an optional device-side overflow callback; default callback traps via `gwn_trap()`.

### Stream & Memory
- Stream binding is explicit via `gwn_stream_mixin`.
  `clear()`/destructor release on the currently bound stream;
  successful stream-parameterized mutations update the bound stream.
- Memory: stream allocator path only (`cudaMallocAsync`/`cudaFreeAsync`), no synchronous fallback.
- Dynamic vertex-position updates use `gwn_update_geometry(...)`, then
  `gwn_bvh_refit_aabb_moment<Order>(...)`. Topology changes use topology
  rebuild. Host spans passed to update APIs follow `cudaMemcpyAsync` lifetime
  rules.
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

## C++ Coding Rules

### Vocabulary discipline
Any new concept in the codebase must be named using a term that already exists in the project
(via grep search of `include/`, `tests/`). If no existing term fits and you must introduce one,
call this out explicitly in the commit message or in a `//` code comment so it can be reviewed.

### No implicit fallback
When an operation fails, do not silently dispatch to a different function to paper over the failure.
Either propagate the error or let the caller decide. This applies to error paths, unsupported
configurations, and missing data. All fallback logic must be explicit and caller-driven.

### Helper discipline
Every file-level helper must name a real algorithm step, shared production behavior, tested
semantic unit, numerical convention, precondition, or traversal boundary.

Use a local lambda for function-local repeated scalar logic. Inline one-line forwarding, field
extraction, trivial predicates, and wrappers that only rename an expression. Delete helpers you
cannot defend in review.

### Return shape
- Public host APIs return `gwn_status`; results go to output spans or owning objects.
- Public device point APIs return scalar or vector query values directly.
- Public result structs are for multi-field semantic records, such as hits and traces.
- Detail helpers may use small result structs for value plus control state across a helper boundary.
- Reference outputs are detail-only for tiny secondary values beside a primary return value.

### Use existing components
Do not reinvent project-provided mechanisms. Use `gwn_noncopyable`, `gwn_stream_mixin`,
`gwn_status`, `GWN_RETURN_ON_ERROR`, `gwn_device_array`, `gwn_scope_exit`, and `gwn_assert`
where applicable. If a pattern already exists in the codebase (factory function, accessor/object
split, copy-and-swap), follow it.

### KISS
Prefer the simplest design that works. Do not add abstractions, virtual dispatch, or type erasure
unless measured or proven necessary. A flat function with a switch is better than a polymorphic
hierarchy that exists only for testability.

### [[nodiscard]] discipline
All `gwn_status`-returning functions and all value-returning query functions must be marked
`[[nodiscard]]`.

### Template constraints
All public templates must constrain type parameters with project-provided concepts
(`gwn_real_type`, `gwn_index_type`). Do not use unconstrained `typename` for `Real` or `Index`.

### Comment discipline
Comments must use the code's own vocabulary: never invent new terms for concepts that already
have names in the source. Comments describe what the code does and why it matters here.
Do not use em dashes, `\tparam` lines that restate the parameter name, or defensive negatives
("No X") where X is absent from the signature.

Implementation code is not required to be trivial; if a non-trivial section benefits from an
explanatory comment, add one.

Public interface declarations must have doxygen (`\brief` minimum). Detail code may omit doxygen
for trivial forwarding wrappers. Functions with non-obvious behavior, preconditions, or side
effects should have a `\brief` or full doxygen block.

## C++ Resource Management & Idioms
- **RAII & Non-copyable**: Owning objects delete copy ctor/assignment.
- **Copy/Move-and-Swap**: Provide `noexcept` swap for strong exception safety.

## Error Handling Rules
- Public APIs return `gwn_status`; internal details may use exceptions.
  Use `gwn_assert` for invariant violations (fatal, debug-only). Use `gwn_status` for
  recoverable or user-facing errors.
- `gwn_status` and all helpers are fully `noexcept`.
- `GWN_RETURN_ON_ERROR(expr)`: uses `gwn_status_result_` variable name to avoid shadowing.
- Build/refit diagnostics use phase-prefixed helpers from `detail/gwn_bvh_status_helpers.cuh`.
- Executable main: `int main() try { ... } catch (...) { ... }` function-try-block.

## Test Structure

### Test policy
- Tests lock behavior, bugs, and user-visible contracts. They must not lock helper placement,
  file layout, call counts, wrapper existence, or first-pass implementation shape.
- Unit tests cover local semantic choke points: input rejection, invariants, numerical kernels,
  stream/memory semantics, and small cases that would fail from one clear defect.
- Integration tests cover end-to-end geometry workflows: upload/build/refit/query parity,
  model fixtures, CPU/GPU reference checks, and performance-sensitive paths.
- Before ending a change, review each new or changed test and ask: would this still pass if the
  implementation were wrong? Replace weak tests with choke-point checks.
- TDD is for behavior and bug reproduction. Do not use TDD to check compilation progress or to
  freeze exploratory first-development scaffolding.

### Test support
- `tests/reference_cpu.cuh`: TBB-parallel CPU exact reference.
- `tests/reference_hdk/*`: Vendored HDK sources (keep under `tests/`).
- `tests/test_fixtures.hpp`, `test_utils.hpp`, `test_harnack_meshes.hpp`: shared test utilities.
- `tests/libigl_reference.cpp/.hpp`: libigl CPU parity for SDF tests.
- Repo-local `fixtures` coverage remains available on clean machines via vendored OBJ meshes under
  `tests/data/`.
- `ctest -L fixtures` also includes the filtered
  `smallgwn_integration_correctness_dynamic_refit` entrypoint so the vendored
  `closed_cube.obj` dynamic-refit path stays covered on clean machines.
- `smallgwn_refit_fidelity`: explicit dataset-backed release validation. It uses libigl
  Laplacian-smoothed frames and compares vertex-update refit against fresh geometry upload on
  the same topology for LBVH/H-PLOC, Taylor winding/gradient, and Antipodal winding/gradient.

### Benchmark
- Executable: `smallgwn_benchmark` from `benchmarks/benchmark_main.cu`.
- Output: console summary + CSV via `--csv <path>`.

## Validation Workflow
```bash
cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j
ctest --test-dir smallgwn/build --output-on-failure
```

For clean-machine validation without external model data, prefer:

```bash
ctest --test-dir smallgwn/build -L unit --output-on-failure
ctest --test-dir smallgwn/build -L fixtures --output-on-failure
```

## Contribution Workflow
1. Create a `git worktree` for non-trivial changes to isolate from the working branch.
2. Commit with Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).
3. Open a PR for review; CI must pass (build + tests).
4. Author addresses review feedback; maintainer merges on pass.

## AGENTS.md Maintenance
Keep this file current with architecture/API/testing changes.
Update in the same commit when behavior, naming, file layout, or test entrypoints change.

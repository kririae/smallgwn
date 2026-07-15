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

## File Naming Rules
- **`.cuh`**: Must be used if the file contains CUDA execution space specifiers, kernel launches,
  CUDA Runtime API calls, or includes other `.cuh` files.
- **`.hpp`**: Strictly for pure C++ CPU-side code; must compile without NVCC.

## Naming and API Rules
- Use `gwn::` namespace and `gwn_` prefix for public symbols.
- Default public index type is `std::uint32_t` unless explicitly overridden.
- Width-4 convenience aliases: `gwn_bvh4_object`, `gwn_bvh4_accessor`, and
  `gwn_bvh4_moment_<role>`.
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

Four decisions generate most rules below. (1) Owning host objects pair with trivially copyable
device accessors: objects manage lifetime, accessors are the device-facing view. (2) Public host
entrypoints that can fail are `noexcept` and return `gwn_status`; detail host execution code throws
and is translated at the seam. (3) No lifetime operation or stream-binding transfer synchronizes
implicitly; allocation, use, and release remain stream-ordered. (4) Prefer exact contracts over
conservative ones (topology-exact stack bounds, closed-interval slab tests, staging-then-swap
replacement) so validity is checkable instead of assumed. When a situation is not covered by a
rule below, derive the answer from these four decisions.

### Public / Detail Split
- Public headers under `include/gwn/` expose the minimal API surface.
  Implementation lives in `include/gwn/detail/`; public headers may `#include` detail headers
  (required for header-only templates), but users should never `#include` detail headers directly.
- `gwn_utils.cuh` is the public foundation for template constraints, status values, stream binding,
  index conventions, and stream-ordered raw allocation. Exception adapters, span ownership, and
  kernel launch machinery live in `include/gwn/detail/`.

### Geometry
- SoA layout (`x/y/z`, `i0/i1/i2`).
- `gwn_geometry_object` owns only vertex positions and oriented triangle indices.
- `gwn_boundary_chain_object` stores the algebraic boundary of an oriented triangle-index chain.
  It is geometry-derived data, separate from BVH payloads and Taylor moments.
- The uploaded-geometry boundary builder is object-based; device point queries consume accessors.

### BVH
- `gwn_bvh_object` owns one canonical query structure: child-AoS bounds and references, primitive
  order, and leaf-ordered triangle records. Moment objects remain independent field-SoA payloads
  aligned to one BVH revision.
- Public API split: `gwn_bvh_build.cuh` builds the canonical BVH and `gwn_bvh_refit.cuh` refits
  geometry-derived BVH data or one moment order.
- `gwn_build_bvh` exposes H-PLOC by default and LBVH through `gwn_bvh_build_options`; H-PLOC's
  nearest-neighbor search radius is a runtime option in the inclusive range `[1, 8]`.
- BVH accessors store topology-exact `internal_stack_bound` and `packed_stack_bound` values computed
  during wide collapse. Host batch query launchers reject `StackCapacity` below the internal bound;
  packed ray traversal is selected only when the packed bound also fits.
- Taylor moment supports `Order=0/1/2`.
  Each `gwn_refit_bvh_moment<Order,...>` call does a full replace of the moment accessor.
- A successful BVH refit advances its revision and makes existing moment objects stale. Refit each
  required moment order explicitly before its next query.
- Accessors are trivially copyable snapshots of an object's device state. Any successful build,
  refit, clear, or destruction invalidates previously copied accessors. `revision` is the
  alignment identity: zero means no published queryable BVH state, and moment validity requires
  `bvh_revision == bvh.revision` in addition to structural checks.
- `accessor()` is mutable by design: it is the expert escape hatch for assembling custom
  accessors (tests rely on it). Any non-empty span installed through this path must be the base of a
  distinct allocation owned by the object and releasable through the stream allocator; aliases,
  subspans, and shared allocations are not valid owner state. `clear()` and the destructor release
  exactly those owned allocations.
- Public BVH entrypoints are object-based; accessor-based routines are detail-only.

### Query
- Public surface: `include/gwn/gwn_query.cuh`. Query families provide device point APIs (for use
  in custom kernels) and batch APIs (host-callable launchers).
- Internal math uses `gwn_query_vec3` (no Eigen dependency); public alias `gwn::gwn_vec3<Real>`.
- Ray first-hit returns `gwn_ray_first_hit_result`: `t`, original primitive ID, second/third
  barycentric weights, three scalar components of the unnormalized oriented geometric normal, and
  hit/miss/overflow status.
- Antipodal winding and gradient queries retry coordinate axes when a projected boundary edge is
  singular.
- Taylor batch queries reject negative and NaN `accuracy_scale` values. Increasing a valid scale
  descends more children.
- Query batch launch configuration is the structural NTTP `gwn_query_batch_config`. Its default
  `block_size` is `k_gwn_default_query_batch_block_size = 256`; traversal batch APIs additionally
  validate `stack_capacity` against the exact BVH bounds, with default
  `k_gwn_default_traversal_stack_capacity = 64`. The stack field is not applicable to exact winding
  or Antipodal gradient batches. Device point APIs retain an independent `StackCapacity` template.
  Query APIs accept an optional device-side overflow callback; default callback traps via `gwn_trap()`.
  For library-built BVHs with intact stack-bound metadata, host validation makes stack overflow
  unreachable in batch launches; the overflow-callback contract is exercised through device point
  APIs, where no host validation runs.

### Stream & Memory
- Lifetime-stream invariant: an owning object's bound stream is the stream on which the current
  allocation's prior uses are ordered. `clear()` and the destructor release on that stream. This
  remains safe only while callers maintain the invariant for custom kernels and cross-stream batch
  queries by ordering all later mutation or release after those uses.
- `set_stream()` is an unsynchronized invariant transfer: the caller asserts that all prior uses
  are already ordered on the new stream (event/wait established beforehand). It never inserts
  synchronization and matches the CCCL `cuda::buffer` model. Call `clear()` before `set_stream()`
  to release on the current binding, or establish cross-stream ordering before rebinding an
  allocation that remains owned.
- Replacement operations preserve the invariant through staging and swap: the displaced
  allocation keeps its old stream binding and is destroyed there, while the replacement adopts the
  operation's stream. New replacement-style operations must follow this pattern.
- Batch queries enqueue on the supplied stream and never synchronize with an object's bound stream.
  The caller orders producing work before the query and orders the query before any later mutation
  or release, or transfers the lifetime binding after establishing that dependency.
- Memory: stream allocator path only (`cudaMallocAsync`/`cudaFreeAsync`), no synchronous fallback.
- Dynamic vertex-position updates use `gwn_update_geometry(...)`, then `gwn_refit_bvh(...)`, then
  `gwn_refit_bvh_moment<Order>(...)` for every moment order in use. Triangle-index or primitive-count
  changes use `gwn_build_bvh(...)`. Host spans passed to update APIs follow `cudaMemcpyAsync`
  lifetime rules.
- A refit failure before mutation preserves the BVH and its stream binding. A later failure makes
  the BVH unqueryable and binds its retained allocations to the refit stream for clear or rebuild.
- `detail::gwn_device_array<T>` is an exception-based implementation buffer. It owns uninitialized
  typed device storage, remembers the lifetime stream, and releases through the stream allocator.
  Same-size `resize()` is a no-op that preserves the stream binding. Its stream-parameterized
  operations (`clear(stream)`, `resize(count, stream)`) release or replace on the supplied stream;
  the caller establishes ordering before supplying a different stream. Deallocation and lifetime
  operations are `noexcept`; allocation, copy, and memset operations report failure by exception.
- Public interfaces accept non-owning spans and never expose `detail::gwn_device_array`.

### Span Rules
- Host-source public parameters use `gwn_host_span`; query batch parameters use `gwn_device_span`.
  These nominal views state caller intent but do not inspect pointer attributes. They have no
  implicit conversion from `cuda::std::span` or across memory spaces. The detail friend adapter
  extracts their backing `cuda::std::span` at the public seam.
- Detail functions, accessors, functors, and owning-object state always use `cuda::std::span`, never
  bare `std::span` or the public memory-space views.
- East const on element type: `span<T const>`, never `span<const T>`.
- Pass spans **by value** with top-level `const`:
  - Public host input: `gwn_host_span<T const> const`
  - Public device input: `gwn_device_span<T const> const`
  - Public device output: `gwn_device_span<T> const`
  - Detail input/output: `cuda::std::span<T const> const` / `cuda::std::span<T> const`
  - Exception: `gwn_allocate_span` / `gwn_free_span` take `cuda::std::span<T> &` (must mutate the span itself).
- Accessor struct members store mutable spans (`span<T>`). Functor members distinguish
  `span<T const>` (input) vs `span<T>` (output).
- `const_cast` is prohibited except where `cuda::atomic_ref` mandates a mutable reference for load-only operations.

### Index Type
- Default public `Index` is `std::uint32_t`, but all templates must work with `uint64_t` too.
  Never hardcode index type; always use the `Index` template parameter.
- For `uint32_t` paths, check that counts/offsets fit before narrowing and return an error early
  (e.g., triangle count exceeding `std::numeric_limits<Index>::max()`).
- Use `gwn_invalid_index`, `gwn_is_invalid_index`, and `gwn_index_in_bounds` from `gwn_utils.cuh`.
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
Prefer a local lambda when behavior is used within one function; introduce a file-level helper only
when a lambda cannot express the behavior cleanly or the behavior is genuinely shared. Inline
one-line forwarding, field extraction, trivial predicates, and wrappers that only rename an
expression. Delete helpers you cannot defend in review.

Keep indentation shallow as logical complexity grows. Prefer early returns, explicit phase
boundaries, and local lambdas over deeply nested control flow.

### Return shape
- Public host APIs return `gwn_status`; results go to output spans or owning objects.
- Public device point APIs return scalar or vector query values directly.
- Public result structs are for multi-field semantic records, such as hits and traces.
- Detail helpers may use small result structs for value plus control state across a helper boundary.
- Reference outputs are detail-only for tiny secondary values beside a primary return value.

### Use existing components
Do not reinvent project-provided mechanisms. Use `gwn_noncopyable`, `gwn_stream_mixin`,
`gwn_status`, `GWN_RETURN_ON_ERROR`, `detail::gwn_device_array`, `gwn_scope_exit`, and `gwn_assert`
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

Important implementation details must have explanatory comments that describe the algorithmic
step or invariant and why it matters. Detail declarations may use a regular comment, `\brief`, or
full doxygen according to complexity. Permission to omit doxygen from trivial detail declarations
must not be treated as a reason to leave non-trivial detail behavior undocumented.

Every public interface declaration must have doxygen (`\brief` minimum). Functions with
non-obvious behavior, preconditions, or side effects should have a full doxygen block. Detail code
may omit doxygen for trivial forwarding wrappers.

## Error Handling Rules
- Public host functions return `gwn_status`; detail host execution code uses exceptions. A public
  function implementation may use status-returning primitives, but its `noexcept` seam must catch
  and translate every exception. Use `gwn_assert` for invariant violations (fatal, debug-only).
  Use `gwn_status` for recoverable or user-facing errors.
- State guarantees: operations that replace whole objects (upload, build, moment refit, boundary
  build) give the strong guarantee via staging and swap; failure preserves the previous object,
  its data, and its stream binding. In-place mutations document their weaker contract explicitly
  (`gwn_update_geometry` may leave some position components updated; a mid-refit failure leaves
  the BVH unqueryable but releasable). New operations must state which guarantee they give.
- `gwn_status` factories allocate their message strings inside `noexcept` functions: host allocation
  failure during error reporting terminates by design and is not a recoverable state.
- `gwn_status` accessors and factories are `noexcept`; `gwn_throw_if_error` is the explicit
  exception-raising convenience.
- `GWN_RETURN_ON_ERROR(expr)`: uses `gwn_status_result_` variable name to avoid shadowing.
- Build/refit diagnostics use phase-prefixed helpers from `detail/gwn_bvh_status_helpers.cuh`.
- Executable main: `int main() try { ... } catch (...) { ... }` function-try-block.

## Test Structure

### Test policy
- Tests lock behavior, bugs, and user-visible contracts. They must not lock helper placement,
  file layout, call counts, wrapper existence, or first-pass implementation shape.
- Test quality is measured by sensitivity, not quantity. A strong test fails when a relevant
  contract is implemented incorrectly; delete or replace tests that still pass when the behavior
  they claim to protect is broken.
- Unit tests cover local semantic choke points: input rejection, invariants, numerical kernels,
  stream/memory semantics, and small cases that would fail from one clear defect.
- Integration tests cover end-to-end geometry workflows: upload/build/refit/query parity,
  model fixtures, and bounded CPU/GPU reference checks. Benchmarks, not correctness tests, cover
  performance-sensitive paths.
- CPU references with O(queries * triangles) work must use a fixed, small query count on arbitrary
  mesh directories. Do not turn integration or E2E coverage into an accidental large-mesh exact
  benchmark.
- Before ending a change, review each new or changed test and ask: would this still pass if the
  implementation were wrong? Replace weak tests with choke-point checks.
- TDD is for behavior and bug reproduction. Do not use TDD to check compilation progress or to
  freeze exploratory first-development scaffolding.

### Test support
- `tests/reference_cpu.cuh`: TBB-parallel CPU exact reference.
- `tests/reference_hdk.cuh`: HDK Taylor reference adapter.
- `tests/reference_hdk/*`: vendored HDK sources, compiled only for E2E validation.
- `tests/test_fixtures.cuh`, `tests/test_meshes.hpp`, `tests/test_utils.cuh`: shared fixtures and
  explicit mesh-directory loading.
- `smallgwn_unit`: aggregate `unit_gwn_*` executable for local semantic choke points.
- `smallgwn_integration`: aggregate builder/static/dynamic workflow executable. CTest passes
  `tests/data/` through `--mesh-dir`.
- `smallgwn_e2e`: explicit dataset runner, absent from default CTest and requiring `--mesh-dir`.
- Dataset integration and E2E targets accept `SMALLGWN_TEST_STACK_CAPACITY=64|128` (default `64`).
- Mesh parse failures are reported and skipped. Every dataset run must still test at least one
  successfully parsed model.

### Benchmark
- Executable: `smallgwn_benchmark` from `tests/benchmark_main.cu`.
- The benchmark is built with the test tooling but remains a manual runner, not a CTest entry.
- Output: console summary + CSV via `--csv <path>`.
- `--skip-exact` excludes the O(queries * triangles) exact stage from large traversal runs.
- `smallgwn_benchmark_cubql` is opt-in through `tests/CMakeLists.txt`. It requires
  `SMALLGWN_CUBQL_SOURCE` to name a local checkout and must not download or vendor cuBQL.
- `tests/run_cubql_comparison.sh` sequentially configures, builds, and runs the comparison.

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

For explicit dataset validation:

```bash
smallgwn/build/tests/smallgwn_e2e --mesh-dir /path/to/obj-directory
```

## Contribution Workflow
1. Create a `git worktree` for non-trivial changes to isolate from the working branch.
2. Commit with Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).
3. Run the relevant build and test workflow before review.

## AGENTS.md Maintenance
Keep this file current with architecture/API/testing changes.
Update in the same commit when behavior, naming, file layout, or test entrypoints change.

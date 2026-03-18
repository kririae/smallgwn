# Instance Acceleration Structure (IAS) — Design Specification

## 1. Problem Statement

`smallgwn` currently operates as a single-mesh pipeline:
`geometry → BVH(topology + AABB + moment) → query`.
There is no mechanism to render or query scenes composed of multiple mesh
instances, each with its own rigid or similarity transform.

This spec adds a two-level acceleration structure (IAS over BLAS) that enables
multi-instance scene queries while maximising reuse of existing BVH types and
traversal infrastructure.

### Scope (v1)

| In scope | Out of scope |
|----------|-------------|
| Pre-built BLAS (user builds per-mesh BVH upfront) | Build-time BLAS within scene build |
| Similarity transforms (rigid + uniform scale) | Per-instance non-uniform scale / shear |
| Dynamic transform refit (same topology) | Topology rebuild on transform update |
| Ray first-hit query | Taylor winding / gradient / SDF / Harnack |
| Exact winding-number query | Nested IAS (multi-level instancing) |
| Unified public API (`gwn_ray_first_hit`, `gwn_winding_number_point`) | IAS-level Taylor moment tree |
| LBVH + H-PLOC IAS topology builders | |

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Transform type | `gwn_similarity_transform<Real>` | 1 extra float/instance vs rigid; avoids future semantic break |
| BLAS accessor | Variadic `gwn_blas_accessor<W, R, I, DataTrees...>` | Composable payload via `cuda::std::tuple`; `get<T>()` extraction |
| Scene accessor | `gwn_scene_accessor<W, R, I, BlasT>` parameterised on BLAS type | Compile-time knowledge of available data trees |
| IAS topology type | Reuse `gwn_bvh_topology_tree_object` + `gwn_bvh_aabb_tree_object` | Existing types are fully generic (no triangle semantics) |
| Build pipeline | Extract `from_preprocess_impl`; mesh + IAS share stages 2-4 | Orthogonal source × builder decomposition |
| Refit | Standalone IAS AABB refit (no moment tree) | Taylor moment transformation deferred |
| Refit generalisation | `Traits::source_type` in existing refit traits | ~20-line diff; isomorphic to existing `output_context` pattern |
| Query API | Unified `gwn_ray_first_hit(AccelT, ...)` via `if constexpr` | User-facing API identical for BLAS and scene |
| Return type | Unified `gwn_ray_hit_result` with `instance_id` | BLAS-only: `instance_id == gwn_invalid_index` |
| Stack capacity | Single `StackCapacity` param; scene uses 2 stacks of same size | Simple tuning; IAS stack oversized but negligible cost |
| Old API | Deprecated (kept for backward compat) | New unified API is the recommended entry point |
| v1 queries | Ray first-hit + exact winding number | Core capabilities; Taylor winding deferred to v2 |

---

## 3. Mathematical Foundations

### 3.1 Ray Parameter Invariance Under Similarity

For transform `x' = s·R·x + t` (scale `s`, rotation `R`, translation `t`):

```
World ray: P(τ) = o_w + τ · d_w
Local ray: P(τ') = o_l + τ' · d_l
  where  o_l = R^T · (o_w - t) / s
         d_l = R^T · d_w / s
```

A point at local parameter `τ'` maps to world parameter `τ = τ'`.
**Proof**: substituting `R·d_l = d_w/s` into `s·τ'·R·d_l = τ·d_w` yields `τ' = τ`. ∎

**Consequence**: `t_best` is globally consistent across IAS and BLAS traversal.
No `t` correction is needed when crossing transform boundaries.

The AABB slab test (`gwn_ray_aabb_intersect_interval_impl`) uses `inv_dir`
components and does not require normalised direction. The local ray's
un-normalised direction `d_l = R^T·d_w/s` works correctly.

### 3.2 Winding Number Additivity

For orientation-preserving similarity (`s > 0`, `det(R) > 0`):

```
wn(scene, q) = Σ_i wn(mesh_i, T_i^{-1}(q))
```

Each instance contribution is computed in local space and summed.
The winding number is invariant under orientation-preserving similarity. ∎

### 3.3 Taylor Moment Transformation (Deferred)

Transforming Taylor moments under similarity is tractable but complex:
- Order 0: ~10 ops (area scales as `s²`, normal rotates)
- Order 1: ~30 ops (3×3 tensor rotation + translation correction)
- Order 2: ~200+ ops (3rd-rank tensor)

Deferred to v2. V1 uses exact winding for scene queries.

---

## 4. Type Definitions

### 4.1 Similarity Transform

```cpp
// include/gwn/detail/gwn_similarity_transform.cuh
template <gwn_real_type Real>
struct gwn_similarity_transform {
    Real rotation[3][3];   // 3×3 orthogonal matrix, row-major
    Real translation[3];   // translation vector
    Real scale{Real(1)};   // uniform scale (must be > 0)

    /// Apply transform: x' = scale * R * x + t
    __host__ __device__ constexpr void
    apply_point(Real px, Real py, Real pz,
                Real &ox, Real &oy, Real &oz) const noexcept;

    /// Apply to direction: d' = scale * R * d
    __host__ __device__ constexpr void
    apply_direction(Real dx, Real dy, Real dz,
                    Real &ox, Real &oy, Real &oz) const noexcept;

    /// Inverse transform point: x = R^T * (x' - t) / scale
    __host__ __device__ constexpr void
    inverse_apply_point(Real px, Real py, Real pz,
                        Real &ox, Real &oy, Real &oz) const noexcept;

    /// Inverse transform direction: d = R^T * d' / scale
    __host__ __device__ constexpr void
    inverse_apply_direction(Real dx, Real dy, Real dz,
                            Real &ox, Real &oy, Real &oz) const noexcept;

    /// Compute world-space AABB from local-space AABB
    __host__ __device__ constexpr gwn_aabb<Real>
    transform_aabb(gwn_aabb<Real> const &local) const noexcept;

    /// Identity transform
    __host__ __device__ static constexpr gwn_similarity_transform identity() noexcept;
};
```

### 4.2 BLAS Accessor (Variadic)

```cpp
// include/gwn/gwn_scene.cuh

/// Non-owning bundle of per-mesh accessors. DataTrees... allows composable
/// payload (e.g. moment tree for Taylor queries).
template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename... DataTrees>
struct gwn_blas_accessor {
    gwn_geometry_accessor<Real, Index>                   geometry;
    gwn_bvh_topology_accessor<Width, Real, Index>        topology;
    gwn_bvh_aabb_accessor<Width, Real, Index>            aabb;
    cuda::std::tuple<DataTrees...>                       data;

    /// Type-safe extraction from data tuple.
    template <typename T>
    __host__ __device__ constexpr T const &get() const noexcept;

    __host__ __device__ constexpr bool is_valid() const noexcept;
};
```

### 4.3 BLAS Object (Owning)

```cpp
/// RAII owning container for a single-mesh BVH (geometry + topology + AABB).
/// Does not own DataTrees; those are managed separately.
template <int Width, gwn_real_type Real, gwn_index_type Index>
struct gwn_blas_object : gwn_noncopyable, gwn_stream_mixin {
    gwn_geometry_object<Real, Index>              geometry_;
    gwn_bvh_topology_tree_object<Width, Real, Index> topology_;
    gwn_bvh_aabb_tree_object<Width, Real, Index>  aabb_;

    /// Produce a base accessor (no DataTrees).
    gwn_blas_accessor<Width, Real, Index> accessor() const noexcept;

    bool has_data() const noexcept;
    // Rule of Five: move, swap, destructor
};
```

### 4.4 Instance Record

```cpp
template <gwn_real_type Real, gwn_index_type Index>
struct gwn_instance_record {
    Index                              blas_index;
    gwn_similarity_transform<Real>     transform;
};
```

### 4.5 Scene Accessor

```cpp
/// Non-owning view of a two-level acceleration structure.
/// BlasT is typically gwn_blas_accessor<Width, Real, Index, DataTrees...>.
template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename BlasT>
struct gwn_scene_accessor {
    gwn_bvh_topology_accessor<Width, Real, Index>              ias_topology;
    gwn_bvh_aabb_accessor<Width, Real, Index>                  ias_aabb;
    cuda::std::span<BlasT const>                               blas_table;
    cuda::std::span<gwn_instance_record<Real, Index> const>    instances;

    __host__ __device__ constexpr bool is_valid() const noexcept;
};
```

### 4.6 Scene Object (Owning)

```cpp
template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename BlasT>
struct gwn_scene_object : gwn_noncopyable, gwn_stream_mixin {
    gwn_bvh_topology_tree_object<Width, Real, Index>           ias_topology_;
    gwn_bvh_aabb_tree_object<Width, Real, Index>               ias_aabb_;
    gwn_device_array<BlasT>                                    blas_table_;
    gwn_device_array<gwn_instance_record<Real, Index>>         instances_;

    gwn_scene_accessor<Width, Real, Index, BlasT> accessor() const noexcept;
    bool has_data() const noexcept;
    // Rule of Five: move, swap, destructor
};
```

### 4.7 Type Traits

```cpp
/// True if AccelT is a gwn_blas_accessor instantiation.
template <typename T> inline constexpr bool is_blas_accessor_v = /* ... */;

/// True if AccelT is a gwn_scene_accessor instantiation.
template <typename T> inline constexpr bool is_scene_accessor_v = /* ... */;

/// True if AccelT is any traversable type (blas or scene).
template <typename T> inline constexpr bool is_traversable_v =
    is_blas_accessor_v<T> || is_scene_accessor_v<T>;
```

### 4.8 Convenience Aliases (Width=4)

```cpp
template <gwn_real_type R, gwn_index_type I, typename... D>
using gwn_blas4_accessor = gwn_blas_accessor<4, R, I, D...>;

template <gwn_real_type R, gwn_index_type I>
using gwn_blas4_object = gwn_blas_object<4, R, I>;

template <gwn_real_type R, gwn_index_type I, typename BlasT>
using gwn_scene4_accessor = gwn_scene_accessor<4, R, I, BlasT>;

template <gwn_real_type R, gwn_index_type I, typename BlasT>
using gwn_scene4_object = gwn_scene_object<4, R, I, BlasT>;
```

---

## 5. Build Pipeline

### 5.1 Refactoring: Extract `from_preprocess_impl`

Currently `gwn_bvh_topology_build_from_binary_impl` (in
`detail/gwn_bvh_topology_build_impl.cuh`) bundles preprocessing with stages 2-4.
This refactoring splits it:

```
[Stage 1: Source-specific preprocessing]
  Triangle mesh: compute triangle AABBs + centroids → Morton → sort
  IAS:           compute instance world AABBs + centroids → Morton → sort
                    ↓
  gwn_topology_build_preprocess<Real, Index, MortonCode>
                    ↓
[Stages 2-4: Source-agnostic pipeline]         ← gwn_bvh_topology_build_from_preprocess_impl
  2. Binary tree build (LBVH or H-PLOC)
  3. Binary→wide collapse
  4. BFS node reorder
```

**New internal function:**

```cpp
template <int Width, gwn_real_type Real, gwn_index_type Index,
          class MortonCode, typename BuildBinaryFn>
gwn_status gwn_bvh_topology_build_from_preprocess_impl(
    char const *entry_name,
    gwn_topology_build_preprocess<Real, Index, MortonCode> &preprocess,
    std::size_t primitive_count,
    gwn_bvh_topology_tree_accessor<Width, Real, Index> &topology,
    gwn_bvh_aabb_tree_accessor<Width, Real, Index> &aabb_tree,
    BuildBinaryFn &&build_binary_fn,
    cudaStream_t stream) noexcept;
```

Existing mesh build paths become: validate geometry → fill preprocess → call
`from_preprocess_impl`. Old public functions are preserved as wrappers.

### 5.2 IAS Build

```cpp
// Public API
template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_lbvh(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene,
    cudaStream_t stream) noexcept;

template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_build_hploc(
    cuda::std::span<BlasT const> const blas_table,
    cuda::std::span<gwn_instance_record<Real, Index> const> const instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene,
    cudaStream_t stream) noexcept;
```

**IAS preprocessing** (Stage 1 for instances):

1. For each instance: read BLAS root AABB → apply `transform_aabb` → world AABB.
2. Compute centroid from world AABB.
3. Morton encode centroids.
4. Radix sort by Morton code.
5. Call `from_preprocess_impl` with sorted data.

The IAS "primitives" are instances. `topology.primitive_indices` maps to
instance indices into the `instances` span.

### 5.3 IAS Refit (Transform Update)

```cpp
template <int Width, gwn_real_type Real, gwn_index_type Index, typename BlasT>
gwn_status gwn_scene_refit_transforms(
    cuda::std::span<gwn_instance_record<Real, Index> const> const updated_instances,
    gwn_scene_object<Width, Real, Index, BlasT> &scene,
    cudaStream_t stream) noexcept;
```

This is a standalone refit (not using `gwn_run_refit_pass`) because:
- IAS AABB refit from instances is simpler than the mesh AABB refit
  (instance AABB = `transform_aabb(blas_root_aabb)`, no per-triangle fan-out)
- Avoids invasive changes to the generic refit functor for source type
  generalisation in v1

The refit:
1. Kernel: for each instance, compute world AABB from BLAS root + transform.
   Write to leaf positions in `ias_aabb_`.
2. Bottom-up propagation: union child AABBs to parent, same atomic-flag pattern
   as existing refit.

**Future (v2):** generalise `gwn_run_refit_pass` via `Traits::source_type` to
unify mesh and IAS AABB refit into a single code path.

---

## 6. Query API

### 6.1 Unified Result Type

```cpp
template <gwn_real_type Real, gwn_index_type Index>
struct gwn_ray_hit_result {
    Real  t{Real(-1)};
    Index instance_id{gwn_invalid_index<Index>()};
    Index primitive_id{gwn_invalid_index<Index>()};
    Real  u{Real(0)};
    Real  v{Real(0)};
    gwn_ray_first_hit_status status{gwn_ray_first_hit_status::k_miss};

    __host__ __device__ constexpr bool hit() const noexcept {
        return status == gwn_ray_first_hit_status::k_hit;
    }
};
```

For BLAS-only queries, `instance_id` remains `gwn_invalid_index<Index>()`.

### 6.2 Unified Device Point APIs

```cpp
/// Ray first-hit: works for both BLAS and scene.
template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename AccelT,
          int StackCapacity = k_gwn_default_traversal_stack_capacity,
          typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline gwn_ray_hit_result<Real, Index>
gwn_ray_first_hit(
    AccelT const &accel,
    Real ray_ox, Real ray_oy, Real ray_oz,
    Real ray_dx, Real ray_dy, Real ray_dz,
    Real t_min = Real(0),
    Real t_max = std::numeric_limits<Real>::infinity(),
    OverflowCallback const &overflow_callback = {}) noexcept;

/// Exact winding number: works for both BLAS and scene.
template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename AccelT,
          int StackCapacity = k_gwn_default_traversal_stack_capacity,
          typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline Real
gwn_winding_number_point(
    AccelT const &accel,
    Real qx, Real qy, Real qz,
    OverflowCallback const &overflow_callback = {}) noexcept;
```

**Internal dispatch via `if constexpr`:**

```cpp
template <...>
__device__ inline gwn_ray_hit_result<Real, Index>
gwn_ray_first_hit(AccelT const &accel, ...) noexcept {
    if constexpr (is_blas_accessor_v<AccelT>) {
        // Delegate to existing single-mesh traversal (4B stack entries)
        auto inner = detail::gwn_ray_first_hit_bvh_impl<...>(...);
        // Convert gwn_ray_first_hit_result → gwn_ray_hit_result
        return {inner.t, gwn_invalid_index<Index>(), inner.primitive_id,
                inner.u, inner.v, inner.status};
    } else if constexpr (is_scene_accessor_v<AccelT>) {
        // Two-level traversal (IAS stack + BLAS stack)
        return detail::gwn_scene_ray_first_hit_impl<...>(...);
    } else {
        static_assert(is_traversable_v<AccelT>,
            "AccelT must be gwn_blas_accessor or gwn_scene_accessor.");
    }
}
```

### 6.3 Two-Level Traversal (Scene Path)

The scene ray traversal uses two separate stacks:

```cpp
Index ias_stack[StackCapacity];   // IAS node indices
Index blas_stack[StackCapacity];  // BLAS node indices (reused per instance)
```

**Algorithm (ray first-hit):**

```
ias_stack.push(ias_root)
t_best = t_max
result = miss

while ias_stack not empty:
    node = ias_stack.pop()
    read IAS topology node + AABB node

    for each child (sorted by t_near):
        if internal:
            if aabb_hit(child, world_ray, t_best):
                ias_stack.push(child)
        if leaf:
            for each instance_idx in primitive_range:
                instance = instances[instance_idx]
                blas = blas_table[instance.blas_index]
                local_ray = inverse_transform(world_ray, instance.transform)

                // Full BLAS traversal for this instance
                blas_result = gwn_ray_first_hit_bvh_impl(
                    blas.geometry, blas.topology, blas.aabb,
                    local_ray, t_min, t_best)

                if blas_result.hit() and blas_result.t < t_best:
                    t_best = blas_result.t
                    result = {t_best, instance_idx, blas_result.primitive_id,
                              blas_result.u, blas_result.v, k_hit}

return result
```

**Key properties:**
- `t_best` is shared across all instances (invariant under similarity).
- IAS children sorted by `t_near` ensures nearest-first traversal.
- BLAS traversal reuses the existing `gwn_ray_first_hit_bvh_impl` directly.
- The BLAS stack is local to each `gwn_ray_first_hit_bvh_impl` call (allocated
  inside that function), so no explicit second stack is needed at the scene level.

**Winding number (exact):**

```
ias_stack.push(ias_root)
omega_sum = 0

while ias_stack not empty:
    node = ias_stack.pop()
    for each child:
        if internal: ias_stack.push(child)
        if leaf:
            for each instance_idx in primitive_range:
                instance = instances[instance_idx]
                local_q = inverse_transform(query_point, instance.transform)
                omega_sum += gwn_winding_number_point_bvh_exact_impl(
                    blas.geometry, blas.topology, local_q)

return omega_sum
```

### 6.4 Unified Batch APIs

```cpp
template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename AccelT,
          int StackCapacity = k_gwn_default_traversal_stack_capacity,
          typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_ray_first_hit_batch(
    AccelT const &accel,
    cuda::std::span<Real const> const ray_origin_x,
    cuda::std::span<Real const> const ray_origin_y,
    cuda::std::span<Real const> const ray_origin_z,
    cuda::std::span<Real const> const ray_dir_x,
    cuda::std::span<Real const> const ray_dir_y,
    cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<gwn_ray_hit_result<Real, Index>> const results,
    Real t_min = Real(0),
    Real t_max = std::numeric_limits<Real>::infinity(),
    cudaStream_t stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}) noexcept;

template <int Width, gwn_real_type Real, gwn_index_type Index,
          typename AccelT,
          int StackCapacity = k_gwn_default_traversal_stack_capacity,
          typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
gwn_status gwn_compute_winding_number_batch(
    AccelT const &accel,
    cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output,
    cudaStream_t stream = gwn_default_stream(),
    OverflowCallback const &overflow_callback = {}) noexcept;
```

### 6.5 Overflow Callback

The overflow callback is unified — no IAS/BLAS distinction at the callback level:

```cpp
struct gwn_traversal_overflow_trap_callback {
    __device__ void operator()() const noexcept { gwn_trap(); }
};
```

For scene queries, if the IAS stack overflows the callback fires. If a BLAS
stack overflows during an inner `gwn_ray_first_hit_bvh_impl` call, that
function's own overflow callback fires. Both use the same callback instance
passed through from the unified API.

### 6.6 Deprecation of Old APIs

Existing per-component APIs are deprecated but retained:

```cpp
// Deprecated: use gwn_ray_first_hit(blas_accessor, ...) instead.
template <int Width, ...>
[[deprecated("Use gwn_ray_first_hit with gwn_blas_accessor instead.")]]
__device__ inline gwn_ray_first_hit_result<Real, Index>
gwn_ray_first_hit_bvh(geometry, topology, aabb, ...);
```

New code should use the unified `gwn_ray_first_hit` / `gwn_winding_number_point`
with a `gwn_blas_accessor` or `gwn_scene_accessor`.

---

## 7. Reuse Summary

| Existing Component | Reuse in IAS |
|-------------------|-------------|
| `gwn_bvh_topology_tree_accessor/object` | IAS top-level topology (zero changes) |
| `gwn_bvh_aabb_tree_accessor/object` | IAS AABB tree (zero changes) |
| `gwn_bvh_topology_build_from_preprocess_impl` | Shared stages 2-4 (mesh + IAS) |
| `gwn_ray_first_hit_bvh_impl` | Called per-instance from scene traversal |
| `gwn_winding_number_point_bvh_exact_impl` | Called per-instance from scene traversal |
| `gwn_device_array<T>` | All device buffers in scene_object |
| `gwn_stream_mixin` / `gwn_noncopyable` | scene_object RAII protocol |
| `gwn_replace_accessor_with_staging` | IAS build staging |
| `gwn_aabb` / `gwn_aabb_union` | Instance world AABB computation |
| Morton encoding + radix sort | IAS build preprocessing |
| `gwn_launch_linear_kernel` | Batch query launchers |

---

## 8. File Layout

### New Files

| File | Contents |
|------|----------|
| `include/gwn/detail/gwn_similarity_transform.cuh` | `gwn_similarity_transform<Real>` + device helpers |
| `include/gwn/gwn_scene.cuh` | Public types (`gwn_blas_accessor`, `gwn_blas_object`, `gwn_instance_record`, `gwn_scene_accessor`, `gwn_scene_object`, aliases), unified query API (`gwn_ray_first_hit`, `gwn_winding_number_point`), build API, refit API |
| `include/gwn/detail/gwn_scene_build_impl.cuh` | IAS preprocess + build orchestration |
| `include/gwn/detail/gwn_scene_query_impl.cuh` | Two-level traversal kernels |
| `tests/unit_scene.cu` | Unit tests for all IAS functionality |

### Modified Files

| File | Change |
|------|--------|
| `include/gwn/gwn.cuh` | Add `#include "gwn_scene.cuh"` |
| `include/gwn/detail/gwn_bvh_topology_build_impl.cuh` | Extract `from_preprocess_impl`; existing paths become wrappers |
| `include/gwn/gwn_query.cuh` | Add deprecation attributes to old APIs |
| `AGENTS.md` | Update with IAS types, build changes, new test entries |
| `tests/CMakeLists.txt` | Add `unit_scene.cu` |

---

## 9. Testing Plan

### Unit Tests (`unit_scene.cu`)

| Test | Description |
|------|-------------|
| `SimilarityTransformIdentity` | Identity transform round-trips points and directions |
| `SimilarityTransformInverse` | `apply(inverse_apply(x)) == x` for arbitrary transforms |
| `SimilarityTransformAABB` | `transform_aabb` produces correct world AABB |
| `BlasAccessorValid` | `gwn_blas_accessor::is_valid()` checks |
| `BlasAccessorDataGet` | `get<MomentTree>()` extraction from variadic tuple |
| `SceneBuildLBVH` | Build IAS via LBVH; verify topology node count, AABB validity |
| `SceneBuildHPLOC` | Build IAS via H-PLOC; same checks |
| `SceneRefitTransforms` | Update transforms, refit, verify root AABB changes |
| `SceneRayFirstHit_SingleInstance` | 1 instance, identity transform → matches BLAS result |
| `SceneRayFirstHit_MultiInstance` | 3+ instances, verify `instance_id` + `primitive_id` |
| `SceneRayFirstHit_ScaledInstance` | Similarity transform with scale ≠ 1; verify `t` correctness |
| `SceneWindingExact_SingleInstance` | 1 instance, identity → matches BLAS winding |
| `SceneWindingExact_MultiInstance` | 2 overlapping meshes → winding sum correctness |
| `UnifiedAPI_BlasPath` | `gwn_ray_first_hit(blas_accessor, ...)` returns correct result |
| `UnifiedAPI_ScenePath` | `gwn_ray_first_hit(scene_accessor, ...)` returns correct result |
| `UnifiedBatch_BlasPath` | `gwn_compute_ray_first_hit_batch(blas, ...)` matches old API |
| `UnifiedBatch_ScenePath` | `gwn_compute_ray_first_hit_batch(scene, ...)` multi-instance batch |

### Integration Tests (Future)

- Scene with vendored OBJ fixtures (cube, sphere).
- Large instance counts (1K, 10K) for IAS build + query.
- Cross-builder consistency (LBVH vs H-PLOC IAS).

---

## 10. Usage Example

```cpp
#include <gwn/gwn.cuh>

int main() try {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- Build per-mesh BLASes ---
    gwn_blas4_object<float, uint32_t> mesh_a, mesh_b;
    // (upload geometry, build topology + AABB for each mesh)

    // --- Pack BLAS accessors ---
    using Blas = gwn_blas4_accessor<float, uint32_t>;
    std::vector<Blas> h_blas = {mesh_a.accessor(), mesh_b.accessor()};

    gwn::gwn_device_array<Blas> d_blas;
    d_blas.resize(h_blas.size(), stream);
    cudaMemcpyAsync(d_blas.data(), h_blas.data(),
                    h_blas.size() * sizeof(Blas),
                    cudaMemcpyHostToDevice, stream);

    // --- Define instances ---
    using Inst = gwn::gwn_instance_record<float, uint32_t>;
    std::vector<Inst> h_instances = {
        {0, gwn::gwn_similarity_transform<float>::identity()},    // mesh_a at origin
        {1, {rotation_45_y, {2, 0, 0}, 1.5f}},                   // mesh_b translated+scaled
    };

    gwn::gwn_device_array<Inst> d_instances;
    d_instances.resize(h_instances.size(), stream);
    cudaMemcpyAsync(d_instances.data(), h_instances.data(),
                    h_instances.size() * sizeof(Inst),
                    cudaMemcpyHostToDevice, stream);

    // --- Build scene ---
    gwn::gwn_scene4_object<float, uint32_t, Blas> scene;
    gwn::gwn_scene_build_hploc<4>(
        cuda::std::span{d_blas.data(), d_blas.size()},
        cuda::std::span{d_instances.data(), d_instances.size()},
        scene, stream);

    // --- Query (unified API) ---
    auto scene_acc = scene.accessor();
    // In a kernel:
    //   auto hit = gwn::gwn_ray_first_hit<4, float, uint32_t>(
    //       scene_acc, ox, oy, oz, dx, dy, dz, 0.f, 1e30f);
    //   if (hit.hit()) {
    //       printf("instance %u, triangle %u, t=%f\n",
    //              hit.instance_id, hit.primitive_id, hit.t);
    //   }

    // --- Update transforms and refit ---
    h_instances[1].transform.translation[0] = 5.0f;
    cudaMemcpyAsync(d_instances.data(), h_instances.data(),
                    h_instances.size() * sizeof(Inst),
                    cudaMemcpyHostToDevice, stream);
    gwn::gwn_scene_refit_transforms<4>(
        cuda::std::span{d_instances.data(), d_instances.size()},
        scene, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
} catch (std::exception const &e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
}
```

---

## 11. Open Items for v2

1. **Taylor moment IAS tree**: Transform moments under similarity for
   Taylor-accelerated winding/gradient/SDF scene queries.
2. **Generalised refit via `Traits::source_type`**: Unify mesh and IAS AABB
   refit into a single `gwn_run_refit_pass` code path.
3. **Query sorting**: Sort queries by nearest instance to reduce warp divergence.
4. **Two-phase traversal**: IAS-only pass to identify hit instances, then
   batched per-instance BLAS queries for better memory coalescing.
5. **Additional scene queries**: unsigned/signed distance, boundary edge
   distance, Harnack trace.

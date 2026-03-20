# Instance Acceleration Structure (IAS) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OptiX-style two-level acceleration structure (IAS over BLAS) to smallgwn, enabling multi-instance scene queries with similarity transforms. V1 covers ray first-hit and exact winding number.

**Architecture:** New public header `gwn_scene.cuh` and three detail headers introduce scene/BLAS types, IAS build pipeline (reusing extracted `from_preprocess_impl` shared with mesh path), standalone IAS AABB refit, and unified query API dispatching BLAS vs scene via `if constexpr`. Existing BVH topology/AABB types are reused directly as IAS top-level structures.

**Tech Stack:** C++20, CUDA 12+, header-only, GTest, existing smallgwn infrastructure (`gwn_device_array`, `gwn_stream_mixin`, `gwn_noncopyable`, `gwn_launch_linear_kernel`).

**Spec:** `docs/superpowers/specs/2026-03-18-ias-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `include/gwn/detail/gwn_similarity_transform.cuh` | `gwn_similarity_transform<Real>` struct + device helpers (apply/inverse point/direction, transform_aabb, identity) |
| `include/gwn/gwn_scene.cuh` | Public header: `gwn_blas_accessor`, `gwn_blas_object`, `gwn_instance_record`, `gwn_scene_accessor`, `gwn_scene_object`, type traits (`gwn_accel_traits`, `is_blas_accessor_v`, `is_scene_accessor_v`), `gwn_ray_hit_result`, convenience aliases, unified device APIs (`gwn_ray_first_hit`, `gwn_winding_number_point`), unified batch APIs, build APIs (`gwn_scene_build_lbvh/hploc`), refit API (`gwn_scene_refit_transforms`), update API (`gwn_scene_update_blas_table`) |
| `include/gwn/detail/gwn_scene_build_impl.cuh` | IAS preprocess functor, IAS build orchestration calling `from_preprocess_impl`, standalone IAS AABB refit kernels, BLAS table update |
| `include/gwn/detail/gwn_scene_query_impl.cuh` | Two-level ray traversal (`gwn_scene_ray_first_hit_impl`), flat-loop exact winding (`gwn_scene_winding_exact_impl`), batch kernel launchers |
| `tests/unit_scene.cu` | All 20 unit tests from spec §9 |

### Modified Files

| File | Change |
|------|--------|
| `include/gwn/detail/gwn_bvh_topology_build_impl.cuh` | Extract `gwn_bvh_topology_build_from_preprocess_impl`; existing `gwn_bvh_topology_build_from_binary_impl` becomes a wrapper |
| `include/gwn/gwn.cuh` | Add `#include "gwn_scene.cuh"` |
| `include/gwn/gwn_query.cuh` | Add `[[deprecated]]` attributes to old ray/winding APIs |
| `CMakeLists.txt` | Register `unit_scene.cu` test |
| `AGENTS.md` | Update with IAS types, build changes, new test entries |

---

## Task 1: Similarity Transform

**Files:**
- Create: `include/gwn/detail/gwn_similarity_transform.cuh`
- Test: `tests/unit_scene.cu` (partial — transform tests only)
- Modify: `CMakeLists.txt` (register test target)

### Step-by-step

- [ ] **Step 1.1: Create test file with transform tests (red)**

Create `tests/unit_scene.cu` with the three transform tests.
The test file uses GTest and the existing `CudaFixture` pattern.

```cpp
// tests/unit_scene.cu
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"
#include "test_utils.hpp"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CudaFixture;

namespace {

// ---------------------------------------------------------------
// Transform tests (host-only — no CUDA kernel needed).
// ---------------------------------------------------------------

TEST(smallgwn_unit_scene, SimilarityTransformIdentity) {
    auto const id = gwn::gwn_similarity_transform<Real>::identity();
    Real ox{}, oy{}, oz{};

    // Point round-trip.
    id.apply_point(Real(1), Real(2), Real(3), ox, oy, oz);
    EXPECT_FLOAT_EQ(ox, Real(1));
    EXPECT_FLOAT_EQ(oy, Real(2));
    EXPECT_FLOAT_EQ(oz, Real(3));

    // Direction round-trip.
    id.apply_direction(Real(4), Real(5), Real(6), ox, oy, oz);
    EXPECT_FLOAT_EQ(ox, Real(4));
    EXPECT_FLOAT_EQ(oy, Real(5));
    EXPECT_FLOAT_EQ(oz, Real(6));
}

TEST(smallgwn_unit_scene, SimilarityTransformInverse) {
    // 90° rotation around Z + scale 2 + translation (10, 20, 30).
    gwn::gwn_similarity_transform<Real> t{};
    t.rotation[0][0] = Real(0);  t.rotation[0][1] = Real(-1); t.rotation[0][2] = Real(0);
    t.rotation[1][0] = Real(1);  t.rotation[1][1] = Real(0);  t.rotation[1][2] = Real(0);
    t.rotation[2][0] = Real(0);  t.rotation[2][1] = Real(0);  t.rotation[2][2] = Real(1);
    t.translation[0] = Real(10);
    t.translation[1] = Real(20);
    t.translation[2] = Real(30);
    t.scale = Real(2);

    Real const px = Real(1), py = Real(2), pz = Real(3);
    Real wx{}, wy{}, wz{};
    t.apply_point(px, py, pz, wx, wy, wz);

    Real rx{}, ry{}, rz{};
    t.inverse_apply_point(wx, wy, wz, rx, ry, rz);

    EXPECT_NEAR(rx, px, Real(1e-5));
    EXPECT_NEAR(ry, py, Real(1e-5));
    EXPECT_NEAR(rz, pz, Real(1e-5));

    // Direction round-trip.
    Real const dx = Real(4), dy = Real(5), dz = Real(6);
    t.apply_direction(dx, dy, dz, wx, wy, wz);
    t.inverse_apply_direction(wx, wy, wz, rx, ry, rz);

    EXPECT_NEAR(rx, dx, Real(1e-5));
    EXPECT_NEAR(ry, dy, Real(1e-5));
    EXPECT_NEAR(rz, dz, Real(1e-5));
}

TEST(smallgwn_unit_scene, SimilarityTransformAABB) {
    // Identity → AABB unchanged.
    auto const id = gwn::gwn_similarity_transform<Real>::identity();
    gwn::gwn_aabb<Real> const local{Real(-1), Real(-1), Real(-1), Real(1), Real(1), Real(1)};
    auto const world = id.transform_aabb(local);
    EXPECT_FLOAT_EQ(world.min_x, Real(-1));
    EXPECT_FLOAT_EQ(world.max_x, Real(1));

    // Scale-only → AABB scales.
    gwn::gwn_similarity_transform<Real> s = gwn::gwn_similarity_transform<Real>::identity();
    s.scale = Real(3);
    auto const scaled = s.transform_aabb(local);
    EXPECT_FLOAT_EQ(scaled.min_x, Real(-3));
    EXPECT_FLOAT_EQ(scaled.max_x, Real(3));
}

} // anonymous namespace
```

- [ ] **Step 1.2: Register test in CMakeLists.txt**

Add to `CMakeLists.txt` after the existing unit tests (after line 182):

```cmake
    smallgwn_add_cuda_test(smallgwn_unit_scene          LABELS "unit;cuda" SOURCES tests/unit_scene.cu)
```

- [ ] **Step 1.3: Verify tests fail (compilation error — header not found)**

Run: `cmake --build build --target smallgwn_unit_scene -j 2>&1 | head -20`
Expected: Compilation succeeds (gwn.cuh doesn't include gwn_scene.cuh yet, so transform struct not available). Actually, the tests reference `gwn::gwn_similarity_transform` which doesn't exist yet → compilation error. Good — red.

- [ ] **Step 1.4: Implement `gwn_similarity_transform`**

Create `include/gwn/detail/gwn_similarity_transform.cuh`:

```cpp
#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "../gwn_bvh.cuh"  // gwn_aabb
#include "../gwn_utils.cuh" // gwn_real_type

namespace gwn {

template <gwn_real_type Real>
struct gwn_similarity_transform {
    Real rotation[3][3]{};
    Real translation[3]{};
    Real scale{Real(1)};

    __host__ __device__ constexpr void
    apply_point(Real const px, Real const py, Real const pz,
                Real &ox, Real &oy, Real &oz) const noexcept {
        ox = scale * (rotation[0][0] * px + rotation[0][1] * py + rotation[0][2] * pz) + translation[0];
        oy = scale * (rotation[1][0] * px + rotation[1][1] * py + rotation[1][2] * pz) + translation[1];
        oz = scale * (rotation[2][0] * px + rotation[2][1] * py + rotation[2][2] * pz) + translation[2];
    }

    __host__ __device__ constexpr void
    apply_direction(Real const dx, Real const dy, Real const dz,
                    Real &ox, Real &oy, Real &oz) const noexcept {
        ox = scale * (rotation[0][0] * dx + rotation[0][1] * dy + rotation[0][2] * dz);
        oy = scale * (rotation[1][0] * dx + rotation[1][1] * dy + rotation[1][2] * dz);
        oz = scale * (rotation[2][0] * dx + rotation[2][1] * dy + rotation[2][2] * dz);
    }

    __host__ __device__ constexpr void
    inverse_apply_point(Real const px, Real const py, Real const pz,
                        Real &ox, Real &oy, Real &oz) const noexcept {
        Real const tx = px - translation[0];
        Real const ty = py - translation[1];
        Real const tz = pz - translation[2];
        Real const inv_scale = Real(1) / scale;
        // R^T * (p - t) / s
        ox = inv_scale * (rotation[0][0] * tx + rotation[1][0] * ty + rotation[2][0] * tz);
        oy = inv_scale * (rotation[0][1] * tx + rotation[1][1] * ty + rotation[2][1] * tz);
        oz = inv_scale * (rotation[0][2] * tx + rotation[1][2] * ty + rotation[2][2] * tz);
    }

    __host__ __device__ constexpr void
    inverse_apply_direction(Real const dx, Real const dy, Real const dz,
                            Real &ox, Real &oy, Real &oz) const noexcept {
        Real const inv_scale = Real(1) / scale;
        // R^T * d / s
        ox = inv_scale * (rotation[0][0] * dx + rotation[1][0] * dy + rotation[2][0] * dz);
        oy = inv_scale * (rotation[0][1] * dx + rotation[1][1] * dy + rotation[2][1] * dz);
        oz = inv_scale * (rotation[0][2] * dx + rotation[1][2] * dy + rotation[2][2] * dz);
    }

    __host__ __device__ constexpr gwn_aabb<Real>
    transform_aabb(gwn_aabb<Real> const &local) const noexcept {
        // Transform all 8 corners, take min/max.
        // Optimised: use the scaled rotation matrix column decomposition.
        Real const cx = (local.min_x + local.max_x) * Real(0.5);
        Real const cy = (local.min_y + local.max_y) * Real(0.5);
        Real const cz = (local.min_z + local.max_z) * Real(0.5);
        Real const ex = (local.max_x - local.min_x) * Real(0.5);
        Real const ey = (local.max_y - local.min_y) * Real(0.5);
        Real const ez = (local.max_z - local.min_z) * Real(0.5);

        Real tcx{}, tcy{}, tcz{};
        apply_point(cx, cy, cz, tcx, tcy, tcz);

        // Half-extents in world: |s*R| * e (element-wise absolute value).
        auto const abs_val = [](Real const v) noexcept -> Real {
            return v < Real(0) ? -v : v;
        };
        Real const new_ex = scale * (abs_val(rotation[0][0]) * ex + abs_val(rotation[0][1]) * ey + abs_val(rotation[0][2]) * ez);
        Real const new_ey = scale * (abs_val(rotation[1][0]) * ex + abs_val(rotation[1][1]) * ey + abs_val(rotation[1][2]) * ez);
        Real const new_ez = scale * (abs_val(rotation[2][0]) * ex + abs_val(rotation[2][1]) * ey + abs_val(rotation[2][2]) * ez);

        return gwn_aabb<Real>{
            tcx - new_ex, tcy - new_ey, tcz - new_ez,
            tcx + new_ex, tcy + new_ey, tcz + new_ez,
        };
    }

    __host__ __device__ static constexpr gwn_similarity_transform identity() noexcept {
        gwn_similarity_transform t{};
        t.rotation[0][0] = Real(1);
        t.rotation[1][1] = Real(1);
        t.rotation[2][2] = Real(1);
        t.scale = Real(1);
        return t;
    }
};

} // namespace gwn
```

- [ ] **Step 1.5: Create minimal `gwn_scene.cuh` stub that includes the transform**

Create `include/gwn/gwn_scene.cuh` with just enough to expose the transform:

```cpp
#pragma once

#include "detail/gwn_similarity_transform.cuh"
```

- [ ] **Step 1.6: Add `gwn_scene.cuh` to umbrella include**

In `include/gwn/gwn.cuh`, add after the `gwn_query.cuh` line:

```cpp
#include "gwn_scene.cuh"
```

- [ ] **Step 1.7: Build and run transform tests (green)**

Run: `cmake -S . -B build && cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*SimilarityTransform*'`
Expected: 3 tests PASS.

- [ ] **Step 1.8: clang-format changed files**

Run: `clang-format -i include/gwn/detail/gwn_similarity_transform.cuh include/gwn/gwn_scene.cuh include/gwn/gwn.cuh tests/unit_scene.cu`

- [ ] **Step 1.9: Commit**

```bash
git add include/gwn/detail/gwn_similarity_transform.cuh include/gwn/gwn_scene.cuh include/gwn/gwn.cuh tests/unit_scene.cu CMakeLists.txt
git commit -m "feat: add gwn_similarity_transform and unit_scene test target"
```

---

## Task 2: BLAS Accessor + BLAS Object + Type Traits

**Files:**
- Modify: `include/gwn/gwn_scene.cuh` (add types)
- Add tests to: `tests/unit_scene.cu`

### Step-by-step

- [ ] **Step 2.1: Write BLAS accessor/object tests (red)**

Append to `tests/unit_scene.cu`:

```cpp
// ---------------------------------------------------------------
// BLAS accessor + object tests.
// ---------------------------------------------------------------

TEST_F(CudaFixture, BlasAccessorValid) {
    // Default-constructed accessor should be invalid.
    gwn::gwn_blas_accessor<4, Real, Index> empty{};
    EXPECT_FALSE(empty.is_valid());
}

TEST(smallgwn_unit_scene, BlasAccessorDataGet) {
    // Verify get<T>() extracts from variadic tuple.
    // Use a dummy int as a stand-in for a data tree accessor.
    gwn::gwn_blas_accessor<4, Real, Index, int> blas{};
    blas.data = cuda::std::make_tuple(42);
    EXPECT_EQ(blas.get<int>(), 42);
}
```

- [ ] **Step 2.2: Implement `gwn_blas_accessor`, `gwn_blas_object`, type traits in `gwn_scene.cuh`**

Expand `include/gwn/gwn_scene.cuh` to include:
- `gwn_blas_accessor<Width, Real, Index, DataTrees...>` with `is_valid()` and `get<T>()`
- `gwn_blas_object<Width, Real, Index>` with private members, mutable/const sub-object accessors, `has_data()`, `accessor()`, Rule-of-Five
- `gwn_accel_traits` primary + BLAS specialisation
- `is_blas_accessor_v` trait
- Width-4 convenience aliases for BLAS

Follow existing patterns from `gwn_geometry_object` and `gwn_bvh_topology_tree_object`:
- `class ... final : public gwn_noncopyable, public gwn_stream_mixin`
- Move-and-swap idiom
- Private `accessor_` member (or in this case, three sub-objects)
- `friend void swap(...)`

Key references:
- `gwn_geometry.cuh:100-157` for object pattern
- `gwn_bvh.cuh:330-425` for topology/aabb object pattern
- `gwn_utils.cuh:18-35` for concepts and `gwn_noncopyable`

- [ ] **Step 2.3: Build and run tests (green)**

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*Blas*'`
Expected: 2 tests PASS.

- [ ] **Step 2.4: clang-format + commit**

```bash
clang-format -i include/gwn/gwn_scene.cuh tests/unit_scene.cu
git add include/gwn/gwn_scene.cuh tests/unit_scene.cu
git commit -m "feat: add gwn_blas_accessor, gwn_blas_object, and type traits"
```

---

## Task 3: Instance Record + Scene Accessor + Scene Object

**Files:**
- Modify: `include/gwn/gwn_scene.cuh`
- Add tests to: `tests/unit_scene.cu`

### Step-by-step

- [ ] **Step 3.1: Write scene type tests (red)**

Append tests that verify `gwn_scene_accessor::is_valid()` returns false for default-constructed, and `gwn_scene_object::has_data()` returns false for default.

- [ ] **Step 3.2: Implement scene types in `gwn_scene.cuh`**

Add to `include/gwn/gwn_scene.cuh`:
- `gwn_instance_record<Real, Index>` struct
- `gwn_scene_accessor<Width, Real, Index, BlasT>` with `is_valid()` (validates ias_topology, ias_aabb, blas_table non-empty, instances non-empty, each blas_table[i].is_valid())
- `gwn_scene_object<Width, Real, Index, BlasT>` with private members, `accessor()`, `has_data()`, Rule-of-Five
- `gwn_accel_traits` specialisation for scene accessor
- `is_scene_accessor_v`, `is_traversable_v` traits
- `gwn_ray_hit_result<Real, Index>` struct
- Width-4 convenience aliases for scene types

Follow patterns from `gwn_bvh_topology_tree_object` and `gwn_bvh_aabb_tree_object`.

Key references:
- Spec §4.4-4.8 for type definitions
- Spec §6.1 for `gwn_ray_hit_result`

- [ ] **Step 3.3: Build and run tests (green)**

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene`
Expected: All tests so far PASS.

- [ ] **Step 3.4: clang-format + commit**

```bash
clang-format -i include/gwn/gwn_scene.cuh tests/unit_scene.cu
git add include/gwn/gwn_scene.cuh tests/unit_scene.cu
git commit -m "feat: add gwn_instance_record, gwn_scene_accessor, gwn_scene_object, gwn_ray_hit_result"
```

---

## Task 4: Build Pipeline Refactoring — Extract `from_preprocess_impl`

**Files:**
- Modify: `include/gwn/detail/gwn_bvh_topology_build_impl.cuh`

This is a pure refactoring: extract the stages 2-4 code (binary build → collapse → BFS reorder) into `gwn_bvh_topology_build_from_preprocess_impl`, then make the existing `gwn_bvh_topology_build_from_binary_impl` call it as a wrapper.

### Step-by-step

- [ ] **Step 4.1: Run existing topology build tests (baseline green)**

Run: `cmake --build build -j && ctest --test-dir build -L unit --output-on-failure`
Expected: All existing tests PASS.

- [ ] **Step 4.2: Extract `gwn_bvh_topology_build_from_preprocess_impl`**

In `include/gwn/detail/gwn_bvh_topology_build_impl.cuh`:

1. Create a new function template `gwn_bvh_topology_build_from_preprocess_impl` that takes:
   - `char const *entry_name`
   - `gwn_topology_build_preprocess<Real, Index, MortonCode> &preprocess`
   - `std::size_t primitive_count`
   - `gwn_bvh_topology_tree_accessor<Width, Real, Index> &topology` (staging)
   - `gwn_bvh_aabb_tree_accessor<Width, Real, Index> &aabb_tree` (staging)
   - `BuildBinaryFn &&build_binary_fn`
   - `cudaStream_t const stream`

2. Move the body of the `build_topology` lambda from `gwn_bvh_topology_build_from_binary_impl` into this new function. The new function:
   - Calls `build_binary_fn(preprocess, binary_nodes, ...)` 
   - Copies sorted_primitive_indices → `topology.primitive_indices`
   - Handles single-primitive case (leaf root)
   - Calls collapse + BFS reorder
   - Writes leaf AABBs from `preprocess.sorted_primitive_aabbs` into `aabb_tree` (allocate aabb_tree nodes, then copy the sorted_primitive_aabbs as leaf data — this is the leaf AABB setup step)
   - Sets root metadata

3. Rewrite `gwn_bvh_topology_build_from_binary_impl` to:
   - Validate geometry
   - Call `gwn_bvh_topology_build_preprocess` to fill preprocess
   - Call `gwn_bvh_topology_build_from_preprocess_impl` with the preprocess data
   - Use `gwn_replace_accessor_with_staging` as before

Key reference: `gwn_bvh_topology_build_impl.cuh:18-129` (current orchestrator).

Note: The existing code writes leaf AABBs as part of the AABB refit pass (not in the topology build). Check `gwn_bvh_refit_aabb_impl.cuh` — the refit pass reads triangles from geometry to compute leaf AABBs. For the extracted function, `aabb_tree` allocation is needed so that a subsequent refit can write into it. The `from_preprocess_impl` should allocate the `aabb_tree.nodes` span to match `topology.nodes.size()` so the caller can immediately run AABB refit after. It does NOT write leaf AABB values — that's the refit's job.

Correction: Looking more carefully at the spec §5.1 note — "`from_preprocess_impl` writes leaf-level AABBs from `preprocess.sorted_primitive_aabbs`". This means the sorted primitive AABBs from the preprocess struct are copied into the aabb_tree as initial leaf data. However, in the existing code, leaf AABBs are computed during refit, not during topology build. The `from_preprocess_impl` must at minimum allocate the aabb_tree nodes. Whether it also writes leaf AABBs depends on the IAS use case — IAS needs leaf AABBs (instance world AABBs) available before bottom-up propagation. For mesh, the existing refit writes leaf AABBs from geometry.

**Decision**: `from_preprocess_impl` allocates `aabb_tree.nodes` to match `topology.nodes.size()`. It does NOT write leaf data — that is source-specific (mesh → refit from geometry; IAS → refit from instances). This keeps the function source-agnostic.

- [ ] **Step 4.3: Verify existing tests still pass (green)**

Run: `cmake --build build -j && ctest --test-dir build -L unit --output-on-failure`
Expected: All existing tests PASS (pure refactoring — no behavioral change).

- [ ] **Step 4.4: clang-format + commit**

```bash
clang-format -i include/gwn/detail/gwn_bvh_topology_build_impl.cuh
git add include/gwn/detail/gwn_bvh_topology_build_impl.cuh
git commit -m "refactor: extract gwn_bvh_topology_build_from_preprocess_impl"
```

---

## Task 5: IAS Build

**Files:**
- Create: `include/gwn/detail/gwn_scene_build_impl.cuh`
- Modify: `include/gwn/gwn_scene.cuh` (add build API declarations + include detail)
- Add tests to: `tests/unit_scene.cu`

### Step-by-step

- [ ] **Step 5.1: Write IAS build tests (red)**

Append to `tests/unit_scene.cu`:

```cpp
// ---------------------------------------------------------------
// IAS build tests.
// ---------------------------------------------------------------

// Helper: build a single-triangle BLAS.
struct SceneTestHelper {
    gwn::gwn_geometry_object<Real, Index> geometry{};
    gwn::gwn_bvh4_topology_object<Real, Index> topology{};
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb{};

    using Blas = gwn::gwn_blas_accessor<4, Real, Index>;

    gwn_status build(cudaStream_t stream, /* SoA vertex + index data */) {
        // upload geometry, build topology+aabb via facade
    }

    Blas accessor() const {
        return Blas{geometry.accessor(), topology.accessor(), aabb.accessor(), {}};
    }
};

TEST_F(CudaFixture, SceneBuildLBVH) {
    // Build 2 BLASes (simple meshes), 3 instances.
    // Call gwn_scene_build_lbvh.
    // Verify: scene.has_data(), topology node count > 0, root AABB is valid.
}

TEST_F(CudaFixture, SceneBuildHPLOC) {
    // Same as above but with gwn_scene_build_hploc.
}
```

(Exact mesh data and assertion values to be filled during implementation, using inline triangle meshes similar to `SingleTriangleMesh` / `OctahedronMesh` patterns from existing tests.)

- [ ] **Step 5.2: Implement IAS build in `gwn_scene_build_impl.cuh`**

Create `include/gwn/detail/gwn_scene_build_impl.cuh`:

1. **IAS preprocess functor** — `gwn_compute_instance_aabbs_and_morton_functor<Real, Index, BlasT, MortonCode>`:
   - For each instance: read `blas_table[instances[i].blas_index]` root AABB → `transform_aabb` → world AABB → centroid → Morton encode
   - Fill `primitive_aabbs[i]`, `morton_codes[i]`, `primitive_indices[i] = i`

2. **IAS preprocess function** — `gwn_scene_build_preprocess<MortonCode, ...>`:
   - Validate `blas_index` bounds (all < blas_table.size())
   - Compute scene AABB from instance world AABBs
   - Launch Morton encode + sort (reuse `gwn_compute_and_sort_morton` pattern or inline equivalent)
   - Fill `gwn_topology_build_preprocess` struct

3. **IAS build orchestrator** — `gwn_scene_build_impl<Width, ...>`:
   - Call IAS preprocess
   - Call `gwn_bvh_topology_build_from_preprocess_impl` with LBVH or H-PLOC binary build function
   - Run AABB refit (standalone — see Task 6, or inline the simple bottom-up here)
   - Copy `blas_table` and `instances` into `scene_object` internal storage
   - Use `gwn_replace_accessor_with_staging` for topology/aabb

Key references:
- `gwn_bvh_topology_build_common.cuh:156-224` for Morton functor pattern
- `gwn_bvh_topology_build_common.cuh:298-377` for preprocess function pattern
- `gwn_bvh_topology_build_impl.cuh` for orchestrator pattern

- [ ] **Step 5.3: Add build API declarations to `gwn_scene.cuh`**

Add `gwn_scene_build_lbvh` and `gwn_scene_build_hploc` function signatures and implementations that delegate to detail.

- [ ] **Step 5.4: Build and run tests (green)**

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*SceneBuild*'`
Expected: 2 tests PASS.

- [ ] **Step 5.5: clang-format + commit**

```bash
clang-format -i include/gwn/detail/gwn_scene_build_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git add include/gwn/detail/gwn_scene_build_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git commit -m "feat: add IAS build pipeline (LBVH + H-PLOC)"
```

---

## Task 6: IAS Refit + BLAS Table Update

**Files:**
- Modify: `include/gwn/detail/gwn_scene_build_impl.cuh` (add refit + update)
- Modify: `include/gwn/gwn_scene.cuh` (add refit/update API declarations)
- Add tests to: `tests/unit_scene.cu`

### Step-by-step

- [ ] **Step 6.1: Write refit + update tests (red)**

```cpp
TEST_F(CudaFixture, SceneRefitTransforms) {
    // Build scene with 2 instances.
    // Read root AABB (copy to host).
    // Change instance[1] translation.
    // Call gwn_scene_refit_transforms.
    // Read new root AABB → verify it changed.
}

TEST_F(CudaFixture, SceneUpdateBlasTable) {
    // Build scene.
    // Rebuild one BLAS with different geometry.
    // Call gwn_scene_update_blas_table with new accessor.
    // Verify root AABB reflects new BLAS bounds.
}
```

- [ ] **Step 6.2: Implement standalone IAS refit**

In `gwn_scene_build_impl.cuh`, add:

1. **Leaf AABB kernel** — `gwn_scene_refit_leaf_aabb_functor`:
   - For each instance (using primitive_indices mapping): read BLAS root AABB → `transform_aabb` → write leaf AABB into `ias_aabb.nodes[leaf_node_idx]`.

2. **Bottom-up propagation kernel** — reuse the atomic-flag parent-notification pattern from existing refit, or implement a simpler version since IAS trees are small:
   - Each leaf atomically increments parent's child-ready counter
   - When all children ready, compute parent AABB as union of children, then propagate up

3. **`gwn_scene_refit_transforms_impl`**:
   - Copy `updated_instances` into `scene.instances_`
   - Launch leaf AABB kernel
   - Launch bottom-up propagation
   - Return `gwn_status::ok()`

4. **`gwn_scene_update_blas_table_impl`**:
   - Copy `updated_blas_table` into `scene.blas_table_`
   - Recompute all leaf AABBs (same leaf kernel as refit)
   - Bottom-up propagation

- [ ] **Step 6.3: Add API declarations to `gwn_scene.cuh`**

Add `gwn_scene_refit_transforms` and `gwn_scene_update_blas_table` signatures.

- [ ] **Step 6.4: Build and run tests (green)**

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*Refit*:*Update*'`
Expected: 2 tests PASS.

- [ ] **Step 6.5: clang-format + commit**

```bash
clang-format -i include/gwn/detail/gwn_scene_build_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git add include/gwn/detail/gwn_scene_build_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git commit -m "feat: add IAS refit and BLAS table update"
```

---

## Task 7: Scene Ray First-Hit Query

**Files:**
- Create: `include/gwn/detail/gwn_scene_query_impl.cuh`
- Modify: `include/gwn/gwn_scene.cuh` (add unified ray API)
- Add tests to: `tests/unit_scene.cu`

### Step-by-step

- [ ] **Step 7.1: Write scene ray query tests (red)**

```cpp
TEST_F(CudaFixture, SceneRayFirstHit_SingleInstance) {
    // 1 instance, identity transform.
    // Fire ray that hits the mesh.
    // Verify result matches BLAS-only query.
    // Verify instance_id == 0.
}

TEST_F(CudaFixture, SceneRayFirstHit_MultiInstance) {
    // 3 instances at different positions.
    // Fire ray through them → verify nearest hit has correct instance_id + primitive_id.
}

TEST_F(CudaFixture, SceneRayFirstHit_ScaledInstance) {
    // 1 instance with scale != 1.
    // Fire ray → verify t is correct in world space.
}
```

- [ ] **Step 7.2: Implement two-level ray traversal in `gwn_scene_query_impl.cuh`**

Create `include/gwn/detail/gwn_scene_query_impl.cuh`:

1. **`gwn_scene_ray_first_hit_impl`** — device function:
   - IAS stack: `Index ias_stack[StackCapacity]; int ias_stack_size = 0;`
   - Push IAS root
   - IAS traversal loop (mirrors existing `gwn_ray_first_hit_bvh_impl` structure)
   - At IAS leaf: iterate primitive range → for each instance:
     - Read transform, compute local ray via `inverse_apply_point`/`inverse_apply_direction`
     - Call `gwn_ray_first_hit_bvh_impl` with BLAS geometry/topology/aabb
     - Update `t_best` and result if closer hit found
   - Return `gwn_ray_hit_result<Real, Index>`

2. **Batch kernel functor** — `gwn_scene_ray_first_hit_batch_functor`:
   - Reads SoA ray inputs, calls `gwn_scene_ray_first_hit_impl`, writes SoA outputs

Key references:
- `gwn_query_ray_impl.cuh:335-475` for traversal loop structure
- Spec §6.3 for algorithm pseudocode

- [ ] **Step 7.3: Add unified `gwn_ray_first_hit` to `gwn_scene.cuh`**

Add the unified device API:
```cpp
template <typename AccelT, int StackCapacity = k_gwn_default_traversal_stack_capacity,
          typename OverflowCallback = detail::gwn_traversal_overflow_trap_callback>
__device__ inline gwn_ray_hit_result<...> gwn_ray_first_hit(AccelT const &accel, ...) noexcept;
```

With `if constexpr` dispatch (spec §6.2):
- `is_blas_accessor_v<AccelT>` → delegate to existing `gwn_ray_first_hit_bvh_impl`, wrap result
- `is_scene_accessor_v<AccelT>` → delegate to `gwn_scene_ray_first_hit_impl`

Add unified batch API `gwn_compute_ray_first_hit_batch`.

- [ ] **Step 7.4: Build and run tests (green)**

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*SceneRay*'`
Expected: 3 tests PASS.

- [ ] **Step 7.5: clang-format + commit**

```bash
clang-format -i include/gwn/detail/gwn_scene_query_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git add include/gwn/detail/gwn_scene_query_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git commit -m "feat: add scene ray first-hit query (point + batch)"
```

---

## Task 8: Scene Exact Winding Number Query

**Files:**
- Modify: `include/gwn/detail/gwn_scene_query_impl.cuh` (add winding)
- Modify: `include/gwn/gwn_scene.cuh` (add unified winding API)
- Add tests to: `tests/unit_scene.cu`

### Step-by-step

- [ ] **Step 8.1: Write scene winding tests (red)**

```cpp
TEST_F(CudaFixture, SceneWindingExact_SingleInstance) {
    // 1 closed mesh instance, identity transform.
    // Query point inside → wn ≈ 1.0.
    // Verify matches BLAS-only winding.
}

TEST_F(CudaFixture, SceneWindingExact_MultiInstance) {
    // 2 overlapping closed mesh instances.
    // Query point inside both → wn ≈ 2.0.
}
```

- [ ] **Step 8.2: Implement flat-loop exact winding in `gwn_scene_query_impl.cuh`**

Add `gwn_scene_winding_exact_impl`:
- Flat loop over all instances (no IAS BVH traversal, per spec §6.3)
- For each instance: `inverse_apply_point(query)` → call `gwn_winding_number_point_bvh_exact_impl`
- Sum contributions

Add batch kernel functor.

Key reference: Spec §6.3 winding pseudocode.

- [ ] **Step 8.3: Add unified `gwn_winding_number_point` to `gwn_scene.cuh`**

Unified device API with `if constexpr`:
- `is_blas_accessor_v` → delegate to existing `gwn_winding_number_point_bvh_exact_impl`
- `is_scene_accessor_v` → delegate to `gwn_scene_winding_exact_impl`

Add unified batch API `gwn_compute_winding_number_batch`.

- [ ] **Step 8.4: Build and run tests (green)**

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*Winding*'`
Expected: 2 tests PASS.

- [ ] **Step 8.5: clang-format + commit**

```bash
clang-format -i include/gwn/detail/gwn_scene_query_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git add include/gwn/detail/gwn_scene_query_impl.cuh include/gwn/gwn_scene.cuh tests/unit_scene.cu
git commit -m "feat: add scene exact winding number query (point + batch)"
```

---

## Task 9: Unified API Tests (BLAS Path + Batch)

**Files:**
- Modify: `tests/unit_scene.cu` (add unified API tests)

### Step-by-step

- [ ] **Step 9.1: Write unified API BLAS-path tests (red)**

```cpp
TEST_F(CudaFixture, UnifiedAPI_BlasPath) {
    // Build a BLAS. Call gwn_ray_first_hit(blas_accessor, ...).
    // Verify result matches old gwn_ray_first_hit_bvh.
    // Verify instance_id == gwn_invalid_index.
}

TEST_F(CudaFixture, UnifiedAPI_ScenePath) {
    // Build scene. Call gwn_ray_first_hit(scene_accessor, ...).
    // Verify result has valid instance_id and primitive_id.
}

TEST_F(CudaFixture, UnifiedBatch_BlasPath) {
    // Call gwn_compute_ray_first_hit_batch(blas_accessor, ...).
    // Verify output_instance_id all == gwn_invalid_index.
    // Verify output_t and output_primitive_id match old batch API.
}

TEST_F(CudaFixture, UnifiedBatch_ScenePath) {
    // Call gwn_compute_ray_first_hit_batch(scene_accessor, ...).
    // Verify multi-instance batch results.
}

TEST_F(CudaFixture, UnifiedBatchWinding_ScenePath) {
    // Call gwn_compute_winding_number_batch(scene_accessor, ...).
    // Verify multi-instance winding results.
}
```

- [ ] **Step 9.2: Build and run tests (green)**

All unified API tests should pass since the underlying implementations from Tasks 7-8 are already done.

Run: `cmake --build build --target smallgwn_unit_scene -j && ./build/smallgwn_unit_scene --gtest_filter='*Unified*'`
Expected: 5 tests PASS.

- [ ] **Step 9.3: clang-format + commit**

```bash
clang-format -i tests/unit_scene.cu
git add tests/unit_scene.cu
git commit -m "test: add unified API tests (BLAS + scene path, point + batch)"
```

---

## Task 10: Deprecate Old APIs

**Files:**
- Modify: `include/gwn/gwn_query.cuh`

### Step-by-step

- [ ] **Step 10.1: Add `[[deprecated]]` attributes to old APIs**

In `include/gwn/gwn_query.cuh`, add deprecation to:
- `gwn_ray_first_hit_bvh` (device function)
- `gwn_compute_ray_first_hit_batch_bvh` (batch function)
- `gwn_winding_number_point_bvh_exact` (device function)
- `gwn_compute_winding_number_batch_bvh_exact` (batch function)

Pattern:
```cpp
template <int Width, ...>
[[deprecated("Use gwn_ray_first_hit with gwn_blas_accessor instead.")]]
__device__ inline gwn_ray_first_hit_result<Real, Index>
gwn_ray_first_hit_bvh(...) noexcept { ... }
```

- [ ] **Step 10.2: Verify build succeeds (deprecation warnings OK, no errors)**

Run: `cmake --build build -j 2>&1 | grep -c "deprecated"` → should show warnings from test files using old APIs.
Run: `cmake --build build -j` → must succeed with exit code 0.

- [ ] **Step 10.3: clang-format + commit**

```bash
clang-format -i include/gwn/gwn_query.cuh
git add include/gwn/gwn_query.cuh
git commit -m "refactor: deprecate old per-component query APIs"
```

---

## Task 11: Full Test Suite + Documentation

**Files:**
- Modify: `AGENTS.md`
- Run: full test suite

### Step-by-step

- [ ] **Step 11.1: Run full test suite**

Run: `cmake -S . -B build && cmake --build build -j && ctest --test-dir build -L unit --output-on-failure`
Expected: All unit tests PASS (including all existing + new unit_scene).

- [ ] **Step 11.2: Run fixtures integration tests**

Run: `ctest --test-dir build -L fixtures --output-on-failure`
Expected: All fixtures tests PASS (no regression).

- [ ] **Step 11.3: Update AGENTS.md**

Add/update the following sections:

**Under "Naming and API Rules":**
- `gwn_blas_accessor<Width, Real, Index, DataTrees...>`: variadic BLAS bundle
- `gwn_scene_accessor<Width, Real, Index, BlasT>`: parameterised scene type
- Width-4 aliases: `gwn_blas4_accessor`, `gwn_scene4_accessor`, `gwn_blas4_object`, `gwn_scene4_object`

**Under "Architecture & Design Rules" → new subsection "### Scene (IAS)":**
- Two-level acceleration: IAS topology/AABB reuse existing BVH types
- Scene owns copies of blas_table and instances
- BLAS objects must outlive scene (span-based references)
- Unified query API: `gwn_ray_first_hit(AccelT, ...)`, `gwn_winding_number_point(AccelT, ...)` with `if constexpr` dispatch
- IAS refit standalone (v1); generalised refit deferred to v2

**Under "Query" subsection:**
- Add unified APIs to the list
- Note old APIs deprecated

**Under "Unit tests" table:**
- Add `unit_scene.cu` row with coverage description

**Under "File Layout" or at top:**
- Add new files to the file listing

- [ ] **Step 11.4: clang-format any remaining files + commit**

```bash
clang-format -i include/gwn/gwn_scene.cuh include/gwn/detail/gwn_similarity_transform.cuh include/gwn/detail/gwn_scene_build_impl.cuh include/gwn/detail/gwn_scene_query_impl.cuh
git add AGENTS.md
git commit -m "docs: update AGENTS.md with IAS types, APIs, and tests"
```

---

## Task Dependencies

```
Task 1 (Transform)
    ↓
Task 2 (BLAS Accessor/Object)
    ↓
Task 3 (Scene Accessor/Object)
    ↓
Task 4 (from_preprocess_impl refactor) ← independent of Tasks 2-3, but ordered here for clean commits
    ↓
Task 5 (IAS Build) ← depends on Tasks 3 + 4
    ↓
Task 6 (IAS Refit) ← depends on Task 5
    ↓
Task 7 (Scene Ray Query) ← depends on Tasks 5 + 6
    ↓
Task 8 (Scene Winding Query) ← depends on Task 5
    ↓
Task 9 (Unified API Tests) ← depends on Tasks 7 + 8
    ↓
Task 10 (Deprecate Old APIs) ← depends on Tasks 7 + 8
    ↓
Task 11 (Full Test + Docs) ← depends on all above
```

Note: Tasks 4 and 2-3 are independent and can be parallelized. Tasks 7 and 8 are independent of each other (both depend on 5). Task 10 is independent of Task 9 but ordered after for clarity.

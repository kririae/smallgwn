# Integration Light/Heavy Width-Index Matrix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Taylor-only integration coverage for `Index={uint32_t,uint64_t}` and `Width={2,3,4,8}` (including odd width 3), exposed as CTest `light` and `heavy` layers.

**Architecture:** Introduce one new integration executable (`smallgwn_integration_taylor_matrix`) with two gtests (`light_*`, `heavy_*`) that run matrix combinations over builders (LBVH/H-PLOC), widths, and index types, without using exact APIs. Register two CTest entries that call the same binary with different `--gtest_filter` and different environment budgets, then label them with `light` / `heavy` for `ctest -L` selection.

**Tech Stack:** CMake/CTest, CUDA C++20, GTest, existing `smallgwn` Taylor APIs.

---

### Task 1: Add CTest light/heavy registration plumbing

**Files:**
- Modify: `CMakeLists.txt`
- Test: `build/CTestTestfile.cmake` (generated)

**Step 1: Write the failing test**

Run:
```bash
ctest -N -L light
ctest -N -L heavy
```
Expected: no dedicated `smallgwn_*_light` / `smallgwn_*_heavy` integration entries yet.

**Step 2: Run test to verify it fails**

Run:
```bash
ctest -N | rg "smallgwn_integration.*(light|heavy)"
```
Expected: empty output.

**Step 3: Write minimal implementation**

Add a `NO_CTEST` option to `smallgwn_add_test` and register a new target + split CTest entries:

```cmake
function(smallgwn_add_test TARGET_NAME)
    cmake_parse_arguments(ARG "NO_CTEST" "" "SOURCES;EXTRA_SOURCES" ${ARGN})
    add_executable(${TARGET_NAME} ${ARG_SOURCES} ${ARG_EXTRA_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE gwn::smallgwn Eigen3::Eigen GTest::gtest GTest::gtest_main)
    if(TBB_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE GWN_ENABLE_ASSERTS)
    target_include_directories(${TARGET_NAME} PRIVATE ${SMALLGWN_TEST_INCLUDE_DIRS})
    if(NOT ARG_NO_CTEST)
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    endif()
endfunction()

smallgwn_add_test(smallgwn_integration_taylor_matrix NO_CTEST
    SOURCES tests/integration_taylor_matrix.cu
    EXTRA_SOURCES tests/reference_hdk/UT_Array.cpp tests/reference_hdk/UT_SolidAngle.cpp)

add_test(NAME smallgwn_integration_taylor_matrix_light
    COMMAND smallgwn_integration_taylor_matrix --gtest_color=no
            --gtest_filter=smallgwn_integration_taylor_matrix.light_*)
set_tests_properties(smallgwn_integration_taylor_matrix_light PROPERTIES
    LABELS "integration;light"
    ENVIRONMENT "SMALLGWN_MATRIX_MODEL_LIMIT=2;SMALLGWN_MATRIX_TOTAL_POINTS=200000;SMALLGWN_MATRIX_ORDER_MAX=1")

add_test(NAME smallgwn_integration_taylor_matrix_heavy
    COMMAND smallgwn_integration_taylor_matrix --gtest_color=no
            --gtest_filter=smallgwn_integration_taylor_matrix.heavy_*)
set_tests_properties(smallgwn_integration_taylor_matrix_heavy PROPERTIES
    LABELS "integration;heavy"
    ENVIRONMENT "SMALLGWN_MATRIX_MODEL_LIMIT=0;SMALLGWN_MATRIX_TOTAL_POINTS=2000000;SMALLGWN_MATRIX_ORDER_MAX=2")
```

**Step 4: Run test to verify it passes**

Run:
```bash
cmake -S . -B build
ctest --test-dir build -N -L light
ctest --test-dir build -N -L heavy
```
Expected: new tests `smallgwn_integration_taylor_matrix_light` and `smallgwn_integration_taylor_matrix_heavy` appear under respective labels.

**Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "test: add ctest light/heavy registration for integration matrix"
```

### Task 2: Create Taylor-only matrix integration skeleton (light profile)

**Files:**
- Create: `tests/integration_taylor_matrix.cu`
- Test: `tests/integration_taylor_matrix.cu`

**Step 1: Write the failing test**

Create a test that requires width set `{2,3,4,8}` and index set `{uint32_t,uint64_t}` to be exercised, but initially wire only width 4 / uint32 and assert full counts:

```cpp
TEST(smallgwn_integration_taylor_matrix, light_order1_width_index_builder_matrix) {
    auto summary = run_light_profile();
    EXPECT_EQ(summary.width_count, 4u);
    EXPECT_EQ(summary.index_count, 2u);
    EXPECT_EQ(summary.combo_count, 16u); // 4 widths * 2 indices * 2 builders
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cmake --build build --target smallgwn_integration_taylor_matrix
./build/smallgwn_integration_taylor_matrix --gtest_filter=smallgwn_integration_taylor_matrix.light_order1_width_index_builder_matrix --gtest_color=no
```
Expected: FAIL on expected combo counts.

**Step 3: Write minimal implementation**

Implement minimal infrastructure in `tests/integration_taylor_matrix.cu`:
- model collection using existing `gwn::tests::collect_model_paths()`
- query generation (reuse lattice strategy from existing integration files)
- `run_matrix_profile(/*order_max=*/1, /*model_limit=*/env, /*total_points=*/env)`
- builders: LBVH + H-PLOC
- Taylor query only (`gwn_compute_winding_number_batch_bvh_taylor`)

**Step 4: Run test to verify it passes**

Run:
```bash
SMALLGWN_MODEL_DATA_DIR=/tmp/common-3d-test-models-subset \
SMALLGWN_MATRIX_MODEL_LIMIT=2 \
SMALLGWN_MATRIX_TOTAL_POINTS=200000 \
SMALLGWN_MATRIX_ORDER_MAX=1 \
./build/smallgwn_integration_taylor_matrix --gtest_filter=smallgwn_integration_taylor_matrix.light_order1_width_index_builder_matrix --gtest_color=no
```
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/integration_taylor_matrix.cu
git commit -m "test: add taylor integration matrix skeleton for light profile"
```

### Task 3: Expand matrix to full width/index coverage with odd width 3

**Files:**
- Modify: `tests/integration_taylor_matrix.cu`
- Test: `tests/integration_taylor_matrix.cu`

**Step 1: Write the failing test**

Add assertions that verify all required combinations are executed and validated:

```cpp
EXPECT_TRUE(summary.seen_width_2);
EXPECT_TRUE(summary.seen_width_3);
EXPECT_TRUE(summary.seen_width_4);
EXPECT_TRUE(summary.seen_width_8);
EXPECT_TRUE(summary.seen_index_u32);
EXPECT_TRUE(summary.seen_index_u64);
EXPECT_EQ(summary.combo_count, 16u);
```

**Step 2: Run test to verify it fails**

Run same light test as Task 2.
Expected: FAIL until width 2/3/8 and uint64 paths are fully wired.

**Step 3: Write minimal implementation**

Add templated helpers to run combinations:

```cpp
template <int Width, typename IndexT>
ComboResult run_combo(
    topology_builder builder,
    int order,
    HostMesh const& mesh,
    QuerySoA const& queries,
    float accuracy_scale);
```

Include:
- index casting helper (`uint32_t -> uint64_t`) for triangle buffers
- width loop over `2,3,4,8`
- index loop over `uint32_t,uint64_t`
- builder loop over `LBVH,H-PLOC`
- per-combo checks against reference width/index baseline (same order, same model)

**Step 4: Run test to verify it passes**

Run:
```bash
SMALLGWN_MODEL_DATA_DIR=/tmp/common-3d-test-models-subset \
SMALLGWN_MATRIX_MODEL_LIMIT=2 \
SMALLGWN_MATRIX_TOTAL_POINTS=200000 \
SMALLGWN_MATRIX_ORDER_MAX=1 \
./build/smallgwn_integration_taylor_matrix --gtest_filter=smallgwn_integration_taylor_matrix.light_order1_width_index_builder_matrix --gtest_color=no
```
Expected: PASS with all required width/index combinations exercised.

**Step 5: Commit**

```bash
git add tests/integration_taylor_matrix.cu
git commit -m "test: cover width 2/3/4/8 and uint32/uint64 in light integration matrix"
```

### Task 4: Add heavy profile (orders 1 and 2) on the same matrix

**Files:**
- Modify: `tests/integration_taylor_matrix.cu`
- Test: `tests/integration_taylor_matrix.cu`

**Step 1: Write the failing test**

Add heavy test requiring both order 1 and order 2 execution:

```cpp
TEST(smallgwn_integration_taylor_matrix, heavy_order1_order2_width_index_builder_matrix) {
    auto summary = run_heavy_profile();
    EXPECT_TRUE(summary.seen_order_1);
    EXPECT_TRUE(summary.seen_order_2);
    EXPECT_EQ(summary.combo_count_per_order, 16u);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
SMALLGWN_MODEL_DATA_DIR=/tmp/common-3d-test-models-subset \
SMALLGWN_MATRIX_MODEL_LIMIT=2 \
SMALLGWN_MATRIX_TOTAL_POINTS=300000 \
SMALLGWN_MATRIX_ORDER_MAX=2 \
./build/smallgwn_integration_taylor_matrix --gtest_filter=smallgwn_integration_taylor_matrix.heavy_order1_order2_width_index_builder_matrix --gtest_color=no
```
Expected: FAIL before order-2 path is fully enforced.

**Step 3: Write minimal implementation**

Implement heavy profile loop:
- order set `{1,2}`
- same width/index/builder matrix
- strict no-exact policy (Taylor only)
- numerical checks:
  - builder consistency (LBVH vs H-PLOC): `p99`/`max` thresholds
  - width consistency relative to width-4 baseline
  - index consistency (`uint64_t` vs `uint32_t`) under tolerance

**Step 4: Run test to verify it passes**

Run:
```bash
ctest --test-dir build -L heavy --output-on-failure
```
Expected: `smallgwn_integration_taylor_matrix_heavy` PASS.

**Step 5: Commit**

```bash
git add tests/integration_taylor_matrix.cu
git commit -m "test: add heavy order1+order2 matrix coverage for width/index combinations"
```

### Task 5: End-to-end verification for layered execution

**Files:**
- Modify: `README.md` (testing section; add light/heavy commands)
- Test: `build` CTest registry and integration binaries

**Step 1: Write the failing test**

Define expected command UX in README first, then verify commands fail before docs and env cleanup are complete.

**Step 2: Run test to verify it fails**

Run:
```bash
ctest --test-dir build -L light --output-on-failure
ctest --test-dir build -L heavy --output-on-failure
```
Expected: at least one failure until all prior tasks are complete and docs/commands match reality.

**Step 3: Write minimal implementation**

Update `README.md` with explicit layered commands:

```bash
ctest --test-dir build -L light --output-on-failure
ctest --test-dir build -L heavy --output-on-failure
```

Plus optional model-dir override example.

**Step 4: Run test to verify it passes**

Run:
```bash
ctest --test-dir build -N -L light
ctest --test-dir build -N -L heavy
ctest --test-dir build -L light --output-on-failure
ctest --test-dir build -L heavy --output-on-failure
```
Expected: both labels discover and execute the new layered matrix tests successfully.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document ctest light/heavy integration layers"
```

### Task 6: Final quality gate (no exact, no performance coupling)

**Files:**
- Modify: `tests/integration_taylor_matrix.cu` (if needed)
- Test: `tests/integration_taylor_matrix.cu`, `CMakeLists.txt`

**Step 1: Write the failing test**

Add assertions/guards ensuring the matrix suite never calls exact/performance APIs.

```cpp
static_assert(true, "Taylor-only integration matrix: exact API intentionally excluded");
```

**Step 2: Run test to verify it fails**

Run:
```bash
rg -n "bvh_exact|integration_hploc_performance|EXACT" tests/integration_taylor_matrix.cu CMakeLists.txt
```
Expected: if exact/perf coupling appears, treat as failure.

**Step 3: Write minimal implementation**

Keep only Taylor query entrypoints and correctness-style checks in new matrix suite.

**Step 4: Run test to verify it passes**

Run:
```bash
rg -n "gwn_compute_winding_number_batch_bvh_taylor" tests/integration_taylor_matrix.cu
rg -n "gwn_compute_winding_number_batch_bvh_exact" tests/integration_taylor_matrix.cu
```
Expected: Taylor hit exists; exact hit absent.

**Step 5: Commit**

```bash
git add tests/integration_taylor_matrix.cu CMakeLists.txt
git commit -m "test: enforce taylor-only scope for width/index integration matrix"
```

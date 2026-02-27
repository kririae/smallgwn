# Open-Mesh Harnack Angle Tracer Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 `../smallgwn-gradient` 中 Harnack tracer 在非 watertight mesh 上会卡在面穿越处的问题，并补齐当前实现的安全性/测试缺口，使 open mesh 能稳定命中 `w=0.5` 的平滑隐式面。

**Architecture:** 采用“两阶段”策略：先修复现有 face-distance tracer 的 correctness 漏洞（输入校验、错误命中、步长安全），再新增并验证 edge-distance + angle-valued（Algorithm 2）路径。保留 face-distance API（watertight 兼容），新增 angle-trace API 给 open mesh 使用，避免一次性替换导致回归。

**Tech Stack:** CUDA C++20, BVH Taylor moments (Order 0/1/2), GTest/CTest, Harnack inequality (Algorithm 1/2 from `docs/harmonic.tex`), design notes in `docs/harnack.md`.

---

## Options and Recommendation

### Option A: Minimal Patch (Face tracer only)
- 修复当前 `gwn_harnack_trace_ray_impl` 的 critical/important bug，不引入 angle-valued tracing。
- Pros: 改动小、短期风险低。
- Cons: 核心 open-mesh 问题（面穿越卡死）仍未解决。

### Option B: Full Replacement (直接切到 Algorithm 2)
- 直接把现有 batch API 改成 edge-distance + angle-valued。
- Pros: 核心问题直接解决。
- Cons: 兼容风险高，closed mesh 行为变化大，回归半径大。

### Option C (Recommended): Incremental Dual Path
- Phase 1 先修 correctness 漏洞（不改变 API 语义）。
- Phase 2 新增 Algorithm 2 API 并在 open-mesh 测试中切换使用。
- Pros: 兼顾稳定性与问题闭环，便于逐步验证。
- Cons: 短期维护两条路径。

---

### Task 1: Lock Regressions With Failing Tests

**Files:**
- Modify: `../smallgwn-gradient/tests/unit_harnack_trace.cu`
- Modify: `../smallgwn-gradient/tests/integration_harnack_trace.cu`
- Modify: `../smallgwn-gradient/CMakeLists.txt` (only if new test target needed)

**Step 1: Write the failing tests**

新增并先写失败断言：

```cpp
TEST_F(CudaFixture, harnack_batch_rejects_invalid_accessors) {
    // non-empty spans + invalid geometry/bvh/moment must return invalid_argument
}

TEST_F(CudaFixture, harnack_open_mesh_cross_face_requires_angle_mode) {
    // half-octahedron; pick rays known to cross face before smooth w=0.5 level set
    // assert legacy face tracer misses or pins, expected behavior marked for TODO
}

TEST_F(CudaFixture, harnack_normals_oppose_ray_direction) {
    // dot(n, dir) < 0 for valid normals
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cmake -S ../smallgwn-gradient -B ../smallgwn-gradient/build
cmake --build ../smallgwn-gradient/build --target smallgwn_unit_harnack_trace smallgwn_integration_harnack_trace
ctest --test-dir ../smallgwn-gradient/build -R "smallgwn_(unit|integration)_harnack_trace" --output-on-failure
```
Expected: 新增用例至少 1 个 FAIL（当前实现尚未满足）。

**Step 3: Commit**

```bash
git -C ../smallgwn-gradient add tests/unit_harnack_trace.cu tests/integration_harnack_trace.cu CMakeLists.txt
git -C ../smallgwn-gradient commit -m "test: add harnack regressions for validation and open-mesh tracing"
```

---

### Task 2: Fix Public API Validation and Output-Copy Reliability

**Files:**
- Modify: `../smallgwn-gradient/include/gwn/gwn_query.cuh`
- Modify: `../smallgwn-gradient/tests/integration_harnack_trace.cu`

**Step 1: Write the failing test**

在 Task 1 的 `harnack_batch_rejects_invalid_accessors` 基础上，补充错误类型断言（`invalid_argument`）。

**Step 2: Run test to verify it fails**

Run:
```bash
ctest --test-dir ../smallgwn-gradient/build -R smallgwn_unit_harnack_trace --output-on-failure
```
Expected: 失败，提示当前 batch API 未做完整 preflight validation。

**Step 3: Write minimal implementation**

在 `gwn_compute_harnack_trace_batch_bvh_taylor` 中补齐与 winding/gradient API 同级别校验：

```cpp
if (!geometry.is_valid()) return gwn_status::invalid_argument(...);
if (!bvh.is_valid()) return gwn_status::invalid_argument(...);
if (!aabb_tree.is_valid_for(bvh)) return gwn_status::invalid_argument(...);
if (!moment_tree.is_valid_for(bvh)) return gwn_status::invalid_argument(...);
if (!gwn_span_has_storage(...all spans...)) return gwn_status::invalid_argument(...);
```

同时修复 `integration_harnack_trace.cu` 的 `run_trace`：

```cpp
res.ok = ok;  // not unconditional true
if (!ok) ADD_FAILURE() << "device-to-host copy failed";
```

**Step 4: Run test to verify it passes**

Run:
```bash
cmake --build ../smallgwn-gradient/build --target smallgwn_unit_harnack_trace smallgwn_integration_harnack_trace
ctest --test-dir ../smallgwn-gradient/build -R "smallgwn_(unit|integration)_harnack_trace" --output-on-failure
```
Expected: 相关 validation/copy 路径 PASS。

**Step 5: Commit**

```bash
git -C ../smallgwn-gradient add include/gwn/gwn_query.cuh tests/integration_harnack_trace.cu tests/unit_harnack_trace.cu
git -C ../smallgwn-gradient commit -m "fix: harden harnack batch preflight validation and result copy checks"
```

---

### Task 3: Fix Face-Tracer Safety Bugs (No False Hit, No rho>R)

**Files:**
- Modify: `../smallgwn-gradient/include/gwn/detail/gwn_harnack_trace_impl.cuh`
- Modify: `../smallgwn-gradient/tests/unit_harnack_trace.cu`

**Step 1: Write the failing tests**

新增失败用例覆盖：
- `R<=0` 时只有在 residual 满足收敛条件才允许 hit。
- `rho` 最终不得超过 `R`。

```cpp
TEST(HarnackStepSize, step_never_exceeds_radius_after_min_step) { ... }
TEST_F(CudaFixture, harnack_no_false_hit_when_radius_collapses) { ... }
```

**Step 2: Run test to verify it fails**

Run:
```bash
ctest --test-dir ../smallgwn-gradient/build -R smallgwn_unit_harnack_trace --output-on-failure
```
Expected: 至少一个 FAIL。

**Step 3: Write minimal implementation**

关键改动：

```cpp
if (R <= Real(0)) {
    if (residual <= epsilon || (grad_mag > 0 && residual <= epsilon * grad_mag)) {
        fill_result(...);
        return result;
    }
    break; // or continue with robust fallback policy
}

if (rho < min_step) rho = min_step;
if (rho > R) rho = R;  // final clamp after min-step
```

**Step 4: Run test to verify it passes**

Run:
```bash
cmake --build ../smallgwn-gradient/build --target smallgwn_unit_harnack_trace
ctest --test-dir ../smallgwn-gradient/build -R smallgwn_unit_harnack_trace --output-on-failure
```
Expected: 新增 safety 测试 PASS。

**Step 5: Commit**

```bash
git -C ../smallgwn-gradient add include/gwn/detail/gwn_harnack_trace_impl.cuh tests/unit_harnack_trace.cu
git -C ../smallgwn-gradient commit -m "fix: remove false-hit path and enforce harnack step safety"
```

---

### Task 4: Expose Edge-Distance Query as Public Batch API

**Files:**
- Modify: `../smallgwn-gradient/include/gwn/gwn_query.cuh`
- Modify: `../smallgwn-gradient/tests/unit_harnack_trace.cu` (or new `tests/unit_distance_edge.cu`)
- Modify: `../smallgwn-gradient/CMakeLists.txt` (if new test file)

**Step 1: Write the failing test**

```cpp
TEST_F(CudaFixture, edge_distance_matches_cpu_reference) {
    // compare BVH edge distance vs CPU point-segment min on small mesh
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
ctest --test-dir ../smallgwn-gradient/build -R smallgwn_unit_harnack_trace --output-on-failure
```
Expected: fail（API 未暴露或结果不匹配）。

**Step 3: Write minimal implementation**

新增对外接口：

```cpp
gwn_compute_unsigned_edge_distance_batch_bvh(...)
```

内部复用：
`detail::gwn_unsigned_edge_distance_point_bvh_impl`。

**Step 4: Run test to verify it passes**

Run:
```bash
cmake --build ../smallgwn-gradient/build --target smallgwn_unit_harnack_trace
ctest --test-dir ../smallgwn-gradient/build -R smallgwn_unit_harnack_trace --output-on-failure
```

**Step 5: Commit**

```bash
git -C ../smallgwn-gradient add include/gwn/gwn_query.cuh tests/unit_harnack_trace.cu CMakeLists.txt
git -C ../smallgwn-gradient commit -m "feat: add public BVH edge-distance batch query"
```

---

### Task 5: Implement Algorithm 2 (Angle-Valued + Two-Sided Harnack)

**Files:**
- Modify: `../smallgwn-gradient/include/gwn/detail/gwn_harnack_trace_impl.cuh`
- Modify: `../smallgwn-gradient/include/gwn/gwn_query.cuh`

**Step 1: Write the failing tests**

```cpp
TEST_F(CudaFixture, harnack_open_mesh_cross_face_hits_smooth_level_set) {
    // rays for half-octahedron where legacy face-distance pins
    // expect hit with angle mode and t > face intersection t by margin
}

TEST_F(CudaFixture, harnack_angle_mode_converges_to_bracketed_level) {
    // verify wrapped residual to nearest periodic target is small
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
ctest --test-dir ../smallgwn-gradient/build -R "smallgwn_(unit|integration)_harnack_trace" --output-on-failure
```
Expected: angle-mode tests FAIL（实现尚不存在）。

**Step 3: Write minimal implementation**

实现要点（对应 `harmonic.tex` Algorithm 2 + `harnack.md §2.5`）：

```cpp
// wrapped angle value (normalized winding -> solid angle domain)
val = glsl_mod(Real(4*pi) * w, Real(4*pi));

// periodic targets around current value
f0 = (val - phi) / period;      // period = 4*pi, phi = 2*pi for w=0.5
f_minus = period * floor(f0) + phi;
f_plus  = period * ceil (f0) + phi;

// stop if close to either
if (min(val - f_minus, f_plus - val) <= eps * norm(grad_val)) hit;

// R from edge distance, c = -4*pi
rho_minus = harnack_step(val, f_minus, c, R);
rho_plus  = harnack_step(val, f_plus,  c, R);
rho = min(rho_minus, rho_plus);
```

新增 API（不破坏现有 face tracer）：

```cpp
gwn_compute_harnack_trace_angle_batch_bvh_taylor(...)
```

**Step 4: Run test to verify it passes**

Run:
```bash
cmake --build ../smallgwn-gradient/build --target smallgwn_unit_harnack_trace smallgwn_integration_harnack_trace
ctest --test-dir ../smallgwn-gradient/build -R "smallgwn_(unit|integration)_harnack_trace" --output-on-failure
```
Expected: open-mesh regression tests PASS，旧 closed-mesh 用例不回归。

**Step 5: Commit**

```bash
git -C ../smallgwn-gradient add include/gwn/detail/gwn_harnack_trace_impl.cuh include/gwn/gwn_query.cuh tests/unit_harnack_trace.cu tests/integration_harnack_trace.cu
git -C ../smallgwn-gradient commit -m "feat: add edge-distance angle-valued harnack tracer for open meshes"
```

---

### Task 6: Strengthen Test Contracts and Documentation

**Files:**
- Modify: `../smallgwn-gradient/tests/unit_harnack_trace.cu`
- Modify: `../smallgwn-gradient/tests/integration_harnack_trace.cu`
- Modify: `../smallgwn-gradient/tests/unit_winding_gradient.cu`
- Modify: `../smallgwn-gradient/include/gwn/gwn_query.cuh`
- Modify: `/home/krr/Projects/WindingNumber/smallgwn/docs/gwn_issues.md`

**Step 1: Write/adjust failing tests**
- 错误路径断言具体错误类别（而非仅 `!ok`）。
- normal 方向断言 `dot(n, ray_dir) < 0`。
- 增加 closed mesh interior gradient 回归测试（`|∇w|` near 0）。

**Step 2: Run test to verify failures**

Run:
```bash
ctest --test-dir ../smallgwn-gradient/build -R "smallgwn_(unit|integration)_(harnack_trace|winding_gradient)" --output-on-failure
```

**Step 3: Implement updates**
- 修正注释与文档（gradient order 支持 `0/1/2`）。
- `gwn_issues.md` 中把 open-mesh tracer 状态更新为已修复/已切换到 angle mode（注明 commit）。

**Step 4: Run full targeted verification**

Run:
```bash
cmake --build ../smallgwn-gradient/build --target smallgwn_unit_harnack_trace smallgwn_integration_harnack_trace smallgwn_unit_winding_gradient smallgwn_integration_gradient
ctest --test-dir ../smallgwn-gradient/build -R "smallgwn_(unit|integration)_(harnack_trace|winding_gradient|gradient)" --output-on-failure
```

**Step 5: Commit**

```bash
git -C ../smallgwn-gradient add tests/unit_harnack_trace.cu tests/integration_harnack_trace.cu tests/unit_winding_gradient.cu include/gwn/gwn_query.cuh
git -C /home/krr/Projects/WindingNumber/smallgwn add docs/gwn_issues.md
git -C ../smallgwn-gradient commit -m "test: tighten harnack/gradient contracts and normal direction checks"
```

---

## Acceptance Criteria

1. `smallgwn_unit_harnack_trace` 和 `smallgwn_integration_harnack_trace` 全绿。
2. 新增 open-mesh 回归用例在 angle mode 下稳定命中，不再 pin 在面穿越点。
3. Harnack batch API 对 invalid accessor/span-storage 返回 `invalid_argument`。
4. 不再出现 `R<=0` 的假命中与 `rho>R` 安全边界违例。
5. 文档与实现一致（gradient order, face-vs-angle tracer 行为边界）。

## Risks and Mitigations

- Risk: angle wrapping 处数值抖动导致“跳层”。
  - Mitigation: 在 `floor/ceil` 前后加入 tiny bias（与 `epsilon` 同量级）并在测试覆盖临界值。
- Risk: edge-distance 增加单次查询成本。
  - Mitigation: 保持 leaf-first + AABB pruning；先 correctness，再做性能 profiling。
- Risk: API 行为变化影响既有调用。
  - Mitigation: 增量新增 angle API，face API 暂不删除；用文档明确适用范围。


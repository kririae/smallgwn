# Fully-GPU Singular Edge Building Acceleration Plan

## Summary
把当前 upload 阶段的 `CPU unordered_map + sort` 全部替换为 **device-only pipeline（CUB primitives + CUDA kernels）**，目标是：
1. 语义与现有实现一致（保留“oriented incidence sum != 0 即 singular edge”定义）。
2. 不走 host 聚合，不做 hybrid fallback。
3. 对大网格（尤其 1M+ triangles）显著降低构建时间并保持结果稳定可复现（按 canonical edge key 排序输出）。

---

## Scope And Non-Scope
- In scope:
1. singular edge 提取流程 GPU 化
2. upload 路径改为先拷贝三角形到 GPU，再在 GPU 构建 singular edges
3. workspace 复用，避免频繁 cudaMallocAsync
- Out of scope:
1. 查询阶段 `gwn_unsigned_singular_edge_distance_point_impl` 的算法优化
2. Harnack tracking/Newton 路径改动
3. API 语义变更（`GWN` query 接口保持不变）

---

## Public API / Interface Changes
保持外部 `gwn_geometry_accessor` 字段不变，只加“可选 workspace”以支持高吞吐批量 upload：

1. 在 `/home/krr/Projects/WindingNumber/smallgwn/include/gwn/gwn_geometry.cuh` 增加：
- `gwn_singular_edge_build_workspace<Index>`（内部持有临时 device buffers + cub temp storage）
- `upload(..., workspace, stream)` 新重载（不传 workspace 时内部临时创建，行为与现在一致）

2. 保持输出不变：
- `singular_edge_i0`
- `singular_edge_i1`

3. 默认策略：
- 始终走 GPU singular-edge build（不再 host map path）

---

## Implementation Design (Decision Complete)

### 1) Device Pipeline
在新文件 `/home/krr/Projects/WindingNumber/smallgwn/include/gwn/detail/gwn_singular_edge_build_gpu.cuh` 实现以下固定流程：

1. **Expand Oriented Edges kernel**
- 输入：`tri_i0/i1/i2`（device）
- 输出长度 `3*T`：
  - `edge_key[k]` (`uint64_t`) = `(min(i,j) << 32) | max(i,j)`
  - `edge_sign[k]` (`int32_t`) = `+1` if `(i<j)` else `-1`
- `i==j` 的退化边：
  - `edge_sign=0` 并 `valid_flag=0`（后续过滤）

2. **Compact Valid Edges**
- `cub::DeviceSelect::Flagged` 过滤 `valid_flag==1`
- 得到 `M <= 3*T` 条有效有向边

3. **Sort By Edge Key**
- `cub::DeviceRadixSort::SortPairs`
- key: `edge_key`
- value: `edge_sign`
- 排序后同 key 连续

4. **Run-Length Encode Keys**
- `cub::DeviceRunLengthEncode::Encode`
- 得到：
  - `unique_keys[num_runs]`
  - `run_counts[num_runs]`
  - `num_runs`

5. **Build Segment Offsets**
- `cub::DeviceScan::ExclusiveSum(run_counts) -> offsets[num_runs]`
- 构造 `offsets_end[num_runs+1]`（最后一个为 `M`）

6. **Segmented Sum Of Signs**
- `cub::DeviceSegmentedReduce::Sum`
- 对每个 run 求 `sum_sign`（oriented incidence sum）

7. **Filter Non-zero Winding**
- kernel 生成 `is_singular = (sum_sign != 0)`
- `cub::DeviceSelect::Flagged` 从 `unique_keys` 中选出 singular keys

8. **Decode Keys To SoA**
- kernel：
  - `i0 = key >> 32`
  - `i1 = key & 0xffffffff`
- 写入 `singular_edge_i0/i1`

### 2) Memory / Workspace Strategy
- `workspace` 固定持有：
  - expand buffers（keys/signs/flags）
  - sorted buffers
  - RLE outputs（unique keys/run counts/num_runs）
  - offsets / reduced sums
  - select outputs
  - 单一 `cub_temp_storage`（按阶段 max bytes 扩容一次）
- 每次 upload 仅 resize，不重复分配不同 temp 块。
- 全部在调用 stream 上执行，保持当前 stream 语义。

### 3) Integration Points
1. 在 `/home/krr/Projects/WindingNumber/smallgwn/include/gwn/gwn_geometry.cuh`：
- 删除/废弃 `gwn_extract_singular_edges` host map 路径
- `gwn_upload_accessor(...)` 流程改为：
  1. 分配/拷贝 vertex+tri 到 device
  2. 调用 `gwn_build_singular_edges_gpu(...)`
  3. 分配 accessor 的 `singular_edge_*` 并填充

2. 代码风格对齐现有 CUB 用法（参考 `/home/krr/Projects/WindingNumber/smallgwn/include/gwn/detail/gwn_bvh_topology_build_common.cuh`）

---

## Correctness Criteria
与当前语义严格对齐（非性能上的“等价”而是结果集合一致）：

1. canonical undirected edge key（`min,max`）一致
2. oriented incidence sum 规则一致
3. `sum!=0` 保留，`sum==0` 去除
4. 输出按 key 升序稳定（排序后自然满足）

---

## Tests And Scenarios

在 `/home/krr/Projects/WindingNumber/smallgwn/tests` 新增/更新：

1. **Parity test vs CPU reference（仅测试中保留 reference）**
- 随机 triangle soups（含重复面、翻转面、non-manifold）
- 对比 GPU/CPU 提取结果完全相同（edge 集合、顺序）

2. **Degenerate edges**
- 包含 `i==j` 的三角形输入
- 验证这些退化边不会进入 singular 输出

3. **Orientation cancellation**
- 同一 edge 出现 +1/-1 抵消到 0
- 验证被移除

4. **Non-manifold high valence**
- 单 edge 多个 incident faces，净和非 0
- 验证保留并稳定输出

5. **Determinism**
- 同一输入连续运行多次，输出 bitwise 一致

6. **Scale benchmark（非阻塞 CI，可本地性能集）**
- 100k / 1M / 5M triangles 上传耗时
- 记录旧版 CPU 与新版 GPU 构建时延对比

---

## Acceptance Targets
1. 功能：所有 singular-edge 语义测试通过
2. 一致性：与测试 reference 100% 对拍通过
3. 性能：1M triangles 场景，singular-edge build 时间显著优于 CPU 版本（目标 3x+，理想 5x~20x，取决于 GPU）

---

## Risks And Mitigations
1. CUB 临时内存尺寸频繁变化导致抖动
- 方案：workspace 按最大需求保留并复用

2. 大网格下中间数组峰值内存高
- 方案：两段式 select + 分阶段释放/复用同一 buffer 区域

3. 语义回归（尤其 orientation 符号）
- 方案：CPU reference parity tests 覆盖随机和构造案例

---

## Assumptions / Defaults
1. `Index` 默认 `uint32_t`（当前主路径），key 打包用 64-bit。
2. 始终采用 GPU singular-edge 构建，不再保留 host fallback。
3. 不改变现有 query API 与 Harnack 追踪逻辑，仅替换 edge-building 实现。
4. CI 中性能测试可标记为 benchmark 组，不作为每次必跑门禁。

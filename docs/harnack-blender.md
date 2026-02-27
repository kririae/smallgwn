# harnack-blender 实现分析

对 `/tmp/harnack-blender` (Blender Cycles fork) 中 Harnack tracing 实现的详细分析，
对照论文 Gillespie et al. *Ray Tracing Harmonic Functions* (SIGGRAPH 2024)。

源码位于 `intern/cycles/kernel/geom/` 下。

---

## 1  文件结构

| 文件 | 内容 |
|------|------|
| `harnack.ipp` (~2050 行) | 核心算法：solid angle 计算、Harnack 步长、主循环、Newton/bisection 对比方法、gyroid |
| `nonplanar_polygon_intersect.h` (~300 行) | 入口分发：按 scenario/precision/intersection_mode 路由到具体实现 |
| `spherical_harmonics.{h,ipp}` | 球谐函数求值与 Harnack tracing |
| `vector_arithmetic.ipp` | 模板化向量运算（`std::array<T,3>` 上的 dot/cross/fma 等） |
| `../scene/nonplanar_polygon.h` | 场景数据结构 `NonplanarPolygonMesh`，存储所有可调参数 |
| `../../blender/nonplanar_polygon.cpp` | Blender ↔ Cycles 数据同步 |
| `../../../source/blender/modifiers/intern/MOD_harnack.cc` | Blender modifier，将参数写入 mesh attributes |

---

## 2  Solid Angle 计算

实现提供三种 solid angle 公式，通过 `solid_angle_formula` 枚举切换。

### 2.1  Triangulated（默认，对应论文 §4.2.2 Eq.10）

`triangulated_loop_solid_angle<T>()` — 将多边形三角化（连接到顶点均值点 z），
对每个三角形用 van Oosterom-Strackee 公式求 solid angle。

关键实现细节：用**复数累积**代替逐个 atan2 调用。每个三角形的 solid angle
表示为复数 `{real, imag}` 并乘入 `running_angle`，最后取 `2 * arg(running_angle)`。
这利用了论文提到的 atan2 加法恒等式：

```
atan2(a,b) + atan2(c,d) ≡ atan2(ad+bc, bd-ac)  (mod 2π)
```

每 5 个三角形做一次归一化防止溢出。

### 2.2  Prequantum（Chern-Ishida 四元数方法）

`prequantum_loop_solid_angle<T>()` — 用四元数 fiber argument 计算。
沿多边形边依次计算 dihedral 四元数并累乘，最后取 `-2 * fiberArg(q0, qi)`。

### 2.3  Gauss-Bonnet

`gauss_bonnet_loop_solid_angle<T>()` — 用 Gauss-Bonnet 公式：
`Ω = 2π·ρ - Σ(corner_angles)`，其中 ρ 是旋转指标（rotation index），
通过球面立体投影到平面后计算总转角得到。

---

## 3  梯度计算

三种模式，通过 `grad_mode` 切换，在 solid angle 计算的同一循环中累加。

### 3.1  Nicole formula（mode 0，默认）

对应论文 Eq.11 的 Biot-Savart 公式：

```
∇Ω = Σ_i (g_i - g_{i+1}) · (ĝ_i - ĝ_{i+1}) · (g_i × g_{i+1}) / |g_i × g_{i+1}|²
```

实现中拆成每条边的贡献：
```cpp
T n2 = len_squared(n);  // n = cross(g0, g1)
T scale = ((-dot(g0,g1) + dot(g0,g0)) / len(g0)
         + (-dot(g0,g1) + dot(g1,g1)) / len(g1));
grad += n / n2 * scale;
```

### 3.2  Adiels formula 10（mode 1）

```cpp
T scale = (lv + lw) / (lv * lw * (lv * lw + dot(g0, g1)));
grad += n * scale;
```

### 3.3  Adiels formula 8（mode 2）

```cpp
T scale = dot(g0 - g1, normalized(g0) - normalized(g1));
grad += n / n2 * scale;
```

论文 Fig.12 比较了这些公式的数值稳定性，结论是 Biot-Savart (mode 0) 最可靠。

---

## 4  核心 Harnack Tracing 循环

`ray_nonplanar_polygon_intersect_T<T>()` (harnack.ipp:875-1308)

### 4.1  初始化

```cpp
T shift = 4π;           // 对应论文 c(x) = -4π
T lo_bound = 0;
T hi_bound = 4π;
T t = ray_tmin;
T t_overstep = 0;
```

- `shift = 4π` 是固定的下界偏移，对应论文 Appendix C 证明的 c = -4π
- 先调用 `classify_loops()` 将输入 loop 分类为多边形或圆盘（disk）

### 4.2  Angle-Valued 处理

论文 Algorithm 2 用 floor/ceil 找 bracketing level sets。实现做了等价但更特化的处理：

```cpp
T val = glsl_mod(omega - levelset, 4π);  // 归约到 [0, 4π)
```

然后根据 `frequency` 参数在 [0, 4π) 内划分子区间确定 lo_bound/hi_bound。
当 frequency ≤ 0 时退化为单 level set 模式。

### 4.3  R(x) — 到边界的距离

```cpp
T R = distance_to_boundary(pos, &closestPoint, nullptr);
```

对每个 polygon loop 遍历所有线段求最近点距离，对每个 disk 求到圆周的距离，
取全局最小值的平方根。对应论文 §4.2 "R(x) = distance from x to polygonal curve P"。

### 4.4  Harnack 步长

```cpp
auto get_max_step = [](T fx, T R, T lo_bound, T up_bound, T shift) -> T {
    T w = (fx + shift) / (up_bound + shift);
    T v = (fx + shift) / (lo_bound + shift);
    T lo_r = -R/2 * (v + 2 - sqrt(v*v + 8*v));
    T up_r =  R/2 * (w + 2 - sqrt(w*w + 8*w));
    return min(lo_r, up_r);
};
```

对应论文 Eq.7，但同时计算上下两个 level set 的步长取 min（Algorithm 2 的要求）。
注意 `fx + shift` 等价于论文的 `f(x) - c`（因为 shift = -c = 4π）。

### 4.5  停止条件

```cpp
bool close_to_zero(T ang, T lo, T hi, T tol, T3 &grad, bool use_grad) {
    T dis = min(ang - lo, hi - ang);
    T tolScaling = use_grad ? len(grad) : 1;
    return dis < tol * tolScaling;
}
```

当 `use_grad_termination = true` 时，等价于论文 §3.1.2 的 `|f - f*| / ||∇f|| < ε`。
梯度终止是可选的——论文指出它在 GLSL 中通常加速 10-20%，但在 Blender CPU 路径追踪中
有时反而更慢（因为每步多一次梯度求值）。

### 4.6  Overstepping 加速

```cpp
if (r >= t_overstep) {   // commit：overstep 安全
    // ... 检查停止条件 ...
    t += t_overstep + r;
    if (use_overstepping)
        t_overstep = r * 0.75;   // 下一步额外前进 0.75r
} else {                  // reject：回退
    t_overstep = 0;
}
```

对应论文 §3.1.4。实际步长 = r + 0.75r_prev ≈ 1.75r（与论文一致）。
如果 overstep 后的 Harnack 步长 r 不够覆盖 t_overstep，则回退到无 overstep。

---

## 5  论文没有但实现有的扩展

### 5.1  Newton 精化（`use_newton`）

当进入 "loose shell"（接近 level set 但还没到 ε 内）时，切换到 Newton 迭代：

```cpp
T df = dot(ray_D, grad);
T dt = -(val < 2π ? val : val - 4π) / df;
t_newton += dt;
```

最多 8 次迭代。近边界（R < 0.25）时有一个特殊的 "radial approximation" 路径，
利用到边界的距离和切线方向做几何修正。

### 5.2  线性外推（`use_extrapolation`）

用前后两步的值拟合 `val(t) = a·t + b`，预测零点位置并验证。
如果预测点确实接近 level set 则直接返回。

### 5.3  Disk Solid Angle（论文 §4.5 Eq.16）

`disk_solid_angle<T>()` 用椭圆积分计算圆盘的 solid angle：

```cpp
T K_val = elliptic_integral_K(k2);
T Pi_val = elliptic_integral_Pi(alpha2, k2);
```

`classify_loops()` 自动检测近似圆形的 loop（检查共面性和等距性），
将其作为 disk 处理而非多边形，避免对高边数近似圆的逐边求和。

### 5.4  双精度支持

所有核心函数模板化为 `<float>` 或 `<double>`，通过 `precision` 参数切换。
向量类型统一为 `std::array<T, 3>`。

### 5.5  Y 轴裁剪（`clip_y`）

用于截面可视化，将射线裁剪到 y > 0 半空间。

### 5.6  Capture Misses（`capture_misses`）

即使超过 max_iterations 也返回当前位置的 solid angle 值，
用于调试和可视化迭代次数分布。

---

## 6  Gyroid（论文 §4.7）

`ray_gyroid_intersect_T<T>()` (harnack.ipp:1725-) 实现 gyroid 曲面的 Harnack tracing。

关键区别：使用 **4D Harnack 不等式**（论文 Eq.27）：

```cpp
auto getMaxStep4D = [](T fx, T R, T levelset, T shift) -> T {
    T a = (fx + shift) / (levelset + shift);
    T u = pow(3*sqrt(3*a³ + 81*a²) + 27*a, 1./3.);
    return R * abs(u/3 - a/u - 1);
};
```

gyroid 函数 `sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)` 不是 3D 调和函数，
但乘以 `e^{√2·w}` 后是 4D 调和函数。实现中用 `unit_shift = 3` 作为下界
（论文 Appendix E.1 给出 `c = -e^{6√3·R} - e^{2√3·R}`，实现用了更简单的常数）。

---

## 7  对比方法

实现包含论文 §5 中讨论的对比方法，用于实验比较。

### 7.1  Newton 法（`newton_intersect_T`）

纯 Newton 迭代，8 次，步长 clamp 到 [-2, 2]。
不保证 first-hit，对 angle-valued 函数会把跳变误判为根。

### 7.2  Bisection 法（`bisection_intersect_T`）

先倍增找 bracket（最多 10 次），再 100 次二分。
同样不处理 angle-valued 的跳变问题。

---

## 8  球谐函数（论文 §4.1）

`ray_spherical_harmonic_intersect_T<T>()` (harnack.ipp:47-227)

- R(x) = outerRadius - ||x||，其中 outerRadius = 1.25 · radius（论文 §4.1 的 h=1.25）
- shift = unitBound · outerRadius^l（解析最小值，对应论文的 c(h)）
- 梯度用 Quilez 的四面体有限差分（论文 §3.1.3）
- 支持周期性 level set（`frequency > 0` 时用 `findEnclosingLevelsets`）

---

## 9  法线计算

`ray_nonplanar_polygon_normal_T<T>()` (harnack.ipp:1571-1654)

三种模式：
- mode 0: Nicole/Biot-Savart 解析梯度
- mode 1: 有限差分（h = 1e-6）
- mode 2: Adiels 公式

论文 Fig.12 表明有限差分在 angle-valued 函数的跳变处会给出错误法线，
解析公式（mode 0）是首选。

---

## 10  参数汇总

`solid_angle_intersection_params` 结构体中的完整参数列表：

| 参数 | 类型 | 论文对应 | 说明 |
|------|------|----------|------|
| `epsilon` | float | ε | 收敛容差 |
| `levelset` | float | f* | 目标 level set 值（以 4π 为单位存储） |
| `frequency` | float | ω | 角频率，≤0 表示单 level set |
| `max_iterations` | int | — | 最大迭代次数 |
| `solid_angle_formula` | int | — | 0=triangulated, 1=prequantum, 2=gauss-bonnet |
| `use_grad_termination` | bool | §3.1.2 | 梯度归一化停止条件 |
| `use_overstepping` | bool | §3.1.4 | 1.75x 步长加速 |
| `use_newton` | bool | — | loose shell 内 Newton 精化 |
| `use_extrapolation` | bool | — | 线性外推加速 |
| `use_quick_triangulation` | bool | — | 跳过前两个三角形 |
| `clip_y` | bool | — | Y 轴裁剪 |
| `capture_misses` | bool | — | 未命中时返回最近值 |
| `precision` | uint | — | 0=float, 1=double |

---

## 11  与 smallgwn 的关系

harnack-blender 的实现是**逐多边形**的：每次 solid angle 求值遍历所有顶点，
复杂度 O(N)。这对论文中的小多边形（几十个顶点）足够，但对三角形网格（数万面片）
不可行。

smallgwn 的 BVH 加速将 winding number 求值降到 O(log n)，
使 Harnack tracing 可以应用于大规模网格。
具体对应关系见 `docs/harnack.md` §2.4。

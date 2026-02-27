# Harnack Tracing for GWN Implicit Surfaces

Design notes for extending smallgwn with Harnack tracing, BVH-accelerated gradient,
and approximate SDF for generalized winding number (GWN) implicit surfaces.

Primary reference: Gillespie, Yang, Botsch, Crane.
*Ray Tracing Harmonic Functions.* ACM Trans. Graph. 43(4), 2024. (docs/harmonic.tex)

---

## 1  BVH-Accelerated Winding Number Gradient

### 1.1  Existing Taylor Expansion Recap

The winding number of a triangle mesh at query point **x** is:

$$
w(\mathbf{x}) = \frac{1}{4\pi} \sum_{\text{triangles}} \Omega_{\text{tri}}(\mathbf{x})
$$

For a cluster of triangles centred at $\bar{\mathbf{p}}$ (the area-weighted centroid),
smallgwn stores Taylor moments per BVH child slot and approximates the cluster's
contribution via multipole expansion.  Define:

- $\mathbf{r} = \mathbf{x} - \bar{\mathbf{p}}$ — vector from centroid to query
  (code: `qr{x,y,z}`)
- $r = |\mathbf{r}|$, $\hat{\mathbf{r}} = \mathbf{r}/r$
  (code: `qlength_m1 = 1/r`, `qn{x,y,z} = r̂`)
- $K(\mathbf{r}) = 1/|\mathbf{r}|$ — the Green's function for the Laplacian

The expansion is:

$$
\Omega(\mathbf{x}) = \sum_\alpha N_\alpha \,\partial_\alpha K
  - \sum_{\alpha,j} N_{\alpha j} \,\partial_\alpha \partial_j K
  + \tfrac{1}{2}\sum_{\alpha,j,k} N_{\alpha j k} \,\partial_\alpha \partial_j \partial_k K
  + \cdots
$$

where the moments are:

| Moment | Definition | Stored as |
|--------|-----------|-----------|
| $N_\alpha$ | $\sum_i A_i \hat{n}_{i\alpha}$ | `child_n_{x,y,z}` |
| $N_{\alpha\alpha}$ | centred 1st moment | `child_nij_{xx,yy,zz}` |
| $N_{\alpha\beta} + N_{\beta\alpha}$ | symmetrised 1st | `child_n{xy}_n{yx}` etc. |
| $N_{\alpha\beta\gamma}$ | centred 2nd moment | `child_nijk_{xxx,...}` etc. |

### 1.2  Key Insight: Gradient Uses the Same Moments

The gradient $\nabla_\mathbf{x} w = \nabla_\mathbf{r} w$ (since $\mathbf{r} = \mathbf{x} - \bar{\mathbf{p}}$) is
obtained by differentiating the expansion one more time:

$$
\nabla_m \Omega = \sum_\alpha N_\alpha \,\partial_m\partial_\alpha K
  - \sum_{\alpha,j} N_{\alpha j} \,\partial_m\partial_\alpha\partial_j K
  + \cdots
$$

**The moments are identical.  Only the kernel derivatives increase by one order.**
No new BVH data structures, no new refit pass, no new moment types are needed.

### 1.3  Derivatives of $K(\mathbf{r}) = 1/r$

Below, $\delta_{ij}$ is the Kronecker delta and all indices run over $\{x,y,z\}$.

**1st derivative (used by ω₀):**

$$
\partial_\alpha K = -\frac{\hat{r}_\alpha}{r^2}
$$

**2nd derivative (used by ω₁ and ∇ω₀):**

$$
\partial_m\partial_\alpha K = \frac{1}{r^3}\bigl(3\hat{r}_m\hat{r}_\alpha - \delta_{m\alpha}\bigr)
$$

**3rd derivative (used by ω₂, ∇ω₁, and H_w order-0):**

$$
\partial_m\partial_\alpha\partial_j K = \frac{1}{r^4}\bigl[
  -15\hat{r}_m\hat{r}_\alpha\hat{r}_j
  + 3(\delta_{m\alpha}\hat{r}_j + \delta_{mj}\hat{r}_\alpha + \delta_{\alpha j}\hat{r}_m)
\bigr]
$$

**4th derivative (used by ∇ω₂ and H_w order-1):**

$$
\partial_m\partial_\alpha\partial_j\partial_k K = \frac{1}{r^5}\bigl[
  105\,\hat{r}_m\hat{r}_\alpha\hat{r}_j\hat{r}_k
  - 15\,\Sigma_4^{(2)}
  + 3\,\Sigma_4^{(0)}
\bigr]
$$

where:

$$
\Sigma_4^{(2)} = \delta_{m\alpha}\hat{r}_j\hat{r}_k + \delta_{mj}\hat{r}_\alpha\hat{r}_k
  + \delta_{mk}\hat{r}_\alpha\hat{r}_j + \delta_{\alpha j}\hat{r}_m\hat{r}_k
  + \delta_{\alpha k}\hat{r}_m\hat{r}_j + \delta_{jk}\hat{r}_m\hat{r}_\alpha
$$

$$
\Sigma_4^{(0)} = \delta_{m\alpha}\delta_{jk} + \delta_{mj}\delta_{\alpha k} + \delta_{mk}\delta_{\alpha j}
$$

### 1.4  Gradient Formulas by Taylor Order

#### Order-0 gradient (uses `child_n_{x,y,z}` — same as ω₀):

$$
\partial_m\omega_0 = \frac{1}{r^3}\bigl[3\hat{r}_m(\hat{\mathbf{r}}\cdot\mathbf{N}) - N_m\bigr]
$$

Code:

```cpp
// rdotN is already computed for ω₀
Real const rdotN = qnx * Nx + qny * Ny + qnz * Nz;
Real const qlength_m3 = qlength_m2 * qlength_m1;
Real const grad_x = qlength_m3 * (Real(3) * qnx * rdotN - Nx);
Real const grad_y = qlength_m3 * (Real(3) * qny * rdotN - Ny);
Real const grad_z = qlength_m3 * (Real(3) * qnz * rdotN - Nz);
```

This is the **dipole gradient** — identical to the Biot-Savart formula
(Eq. 15 in harmonic.tex) but BVH-accelerated.

#### Order-1 gradient (uses `child_nij_*` — same as ω₁):

Define (reuses quantities from ω₁):

- $S = \sum_{\alpha,j} N_{\alpha j}\hat{r}_\alpha\hat{r}_j$ (scalar, already computed for ω₁)
- $T = \text{tr}(\mathbf{N}) = N_{xx} + N_{yy} + N_{zz}$ (scalar, already computed for ω₁)
- $(\mathbf{v}+\mathbf{u})_m = \sum_j (N_{mj} + N_{jm})\hat{r}_j$ (uses symmetrised moments)

Then:

$$
\partial_m\omega_1 = \frac{1}{r^4}\bigl[(15S - 3T)\hat{r}_m - 3(\mathbf{v}+\mathbf{u})_m\bigr]
$$

Code:

```cpp
// S and T are already computed for ω₁
Real const vu_x = Real(2)*Nij_xx*qnx + Nxy_Nyx*qny + Nzx_Nxz*qnz;
Real const vu_y = Nxy_Nyx*qnx + Real(2)*Nij_yy*qny + Nyz_Nzy*qnz;
Real const vu_z = Nzx_Nxz*qnx + Nyz_Nzy*qny + Real(2)*Nij_zz*qnz;
Real const coeff = Real(15)*S - Real(3)*T;
Real const qlength_m4 = qlength_m2 * qlength_m2;
Real const grad1_x = qlength_m4 * (coeff*qnx - Real(3)*vu_x);
Real const grad1_y = qlength_m4 * (coeff*qny - Real(3)*vu_y);
Real const grad1_z = qlength_m4 * (coeff*qnz - Real(3)*vu_z);
```

#### Order-2 gradient (uses `child_nijk_*` — same as ω₂):

Uses the 4th derivative of K contracted with the same order-2 moments.
The structure parallels the existing ω₂ code
(`gwn_query_winding_impl.cuh` lines 198-233) with an extra $\hat{r}_m$ index.
Full expansion is mechanical but lengthy; see Section 1.5.

### 1.5  Order-2 Gradient: Full Expansion

$$
\partial_m\omega_2 = \frac{1}{2}\sum_{\alpha,j,k} N_{\alpha jk}\,\partial_m\partial_\alpha\partial_j\partial_k K
$$

Using the 4th derivative from Section 1.3 and contracting with the stored moments:

$$
\partial_m\omega_2 = \frac{1}{r^5}\Bigl[
  \tfrac{105}{2}\,\hat{r}_m\,Q_4
  - \tfrac{15}{2}\bigl(\hat{r}_m\,Q_2^{(\alpha jk)} + W_m\bigr)
  + \tfrac{3}{2}\,V_m
\Bigr]
$$

where:

- $Q_4 = \sum_{\alpha,j,k} N_{\alpha jk}\hat{r}_\alpha\hat{r}_j\hat{r}_k$
  — same scalar as computed for ω₂
- $Q_2^{(\alpha jk)} = \sum_{\alpha} N_{\alpha\alpha k}\hat{r}_k + \cdots$
  — contractions with two δ's and one $\hat{r}$ (6 terms)
- $W_m = \sum_{\alpha,j,k} N_{\alpha jk}(\delta_{m\alpha}\hat{r}_j\hat{r}_k
  + \delta_{mj}\hat{r}_\alpha\hat{r}_k + \delta_{mk}\hat{r}_\alpha\hat{r}_j)$
  — three row-extractions from the moment tensor
- $V_m = \sum_\alpha (N_{m\alpha\alpha} + N_{\alpha m\alpha} + N_{\alpha\alpha m})$
  — trace-related terms

These all use the existing `child_nijk_*`, `child_2nxxy_nyxx`, etc. fields.
The algebra is verbose but follows the same pattern as the existing ω₂ code.

### 1.6  Per-Triangle Gradient (Leaf/Brute-Force)

When the BVH traversal reaches a leaf (near-field), we evaluate the gradient
per triangle using the Biot-Savart formula (harmonic.tex Eq. 11):

$$
\nabla\Omega_P(\mathbf{x}) = \sum_{i=1}^{k}
  (\mathbf{g}_i - \mathbf{g}_{i+1}) \cdot
  \left(\frac{\mathbf{g}_i}{|\mathbf{g}_i|} - \frac{\mathbf{g}_{i+1}}{|\mathbf{g}_{i+1}|}\right)
  \frac{\mathbf{g}_i \times \mathbf{g}_{i+1}}{|\mathbf{g}_i \times \mathbf{g}_{i+1}|^2}
$$

where $\mathbf{g}_i = \mathbf{p}_i - \mathbf{x}$ and the sum is over polygon edges.

For a triangle with vertices $\mathbf{a}, \mathbf{b}, \mathbf{c}$, there are
3 edges and the formula yields ~15 lines of vector math, comparable in complexity
to the existing `gwn_signed_solid_angle_triangle_impl`.

### 1.7  Accuracy

Using Order-N moments, both $w(\mathbf{x})$ and $\nabla w(\mathbf{x})$ have the
same relative accuracy $O\bigl((d/r)^{N+1}\bigr)$, where $d$ is the cluster
diameter and $r$ is the query distance.  The far-field criterion
(`qlength2 > child_max_p_dist2 * accuracy_scale2`) is identical.

### 1.8  Summary Table

| Component | Moments needed | Already stored? | Kernel derivative |
|-----------|---------------|-----------------|-------------------|
| ω₀        | $N_\alpha$     | ✅ Order-0      | ∂K  (1st)         |
| ω₁        | $N_{\alpha j}$ | ✅ Order-1      | ∂²K (2nd)         |
| ω₂        | $N_{\alpha jk}$| ✅ Order-2      | ∂³K (3rd)         |
| ∇ω₀       | $N_\alpha$     | ✅ same Order-0 | ∂²K (2nd)         |
| ∇ω₁       | $N_{\alpha j}$ | ✅ same Order-1 | ∂³K (3rd)         |
| ∇ω₂       | $N_{\alpha jk}$| ✅ same Order-2 | ∂⁴K (4th)         |

---

## 2  Harnack Tracing Algorithm

### 2.1  Problem Statement

Given a (possibly non-watertight) triangle mesh, the GWN defines an implicit
surface $S = \{w(\mathbf{x}) = 0.5\}$.  We want to ray trace this surface —
find the first intersection of a ray with $S$.

### 2.2  Why Not Sphere Tracing?

Sphere tracing requires a Lipschitz bound on the implicit function.  The solid
angle / winding number is **not Lipschitz** — it has singularities at mesh
edges where the function jumps through a full period of $4\pi$.  Sphere tracing
with an incorrect Lipschitz bound yields artifacts near these singularities
(harmonic.tex Fig. 26).

### 2.3  The Harnack Tracing Algorithm

Uses the **Harnack inequality** for positive harmonic functions: for
$f > 0$ harmonic on $B_R(\mathbf{x}_0)$, the value at any point
$\mathbf{x} \in B_R$ with $|\mathbf{x} - \mathbf{x}_0| = \rho$ satisfies:

$$
\frac{(R-\rho)R}{(R+\rho)^2}\,f(\mathbf{x}_0) \le f(\mathbf{x})
\le \frac{(R+\rho)R}{(R-\rho)^2}\,f(\mathbf{x}_0)
$$

**Step size formula** (3D): given current value $f_t$, target $f^*$,
shift $c$ such that $f - c > 0$ on $B_R$:

$$
a = \frac{f_t - c}{f^* - c}, \qquad
\rho = \frac{R}{2}\left|a + 2 - \sqrt{a^2 + 8a}\right|
$$

**Algorithm** (harmonic.tex Algorithm 1):

```
HarnackTrace(r₀, v, f*, f, R, c, ε, t_max):
    t ← 0
    do
        rₜ ← r₀ + t·v
        fₜ ← f(rₜ)                  // winding number query — O(log n) via BVH
        if |fₜ - f*| ≤ ε·‖∇f(rₜ)‖:  // gradient-based stopping
            return t
        Rₜ ← R(rₜ)                  // unsigned distance to mesh — O(log n) via BVH
        cₜ ← c(rₜ)                  // lower bound on f within B_Rₜ
        if f* ≤ cₜ:
            ρ ← Rₜ                  // safe max step
        else:
            a ← (fₜ - cₜ) / (f* - cₜ)
            ρ ← (Rₜ/2) · |a + 2 - √(a² + 8a)|
        t ← t + ρ
    while t < t_max
    return -1  // no hit
```

### 2.4  Inputs from smallgwn

| Algorithm input | smallgwn function | Complexity |
|----------------|-------------------|------------|
| $f(\mathbf{x})$ — winding number | `gwn_winding_number_point_bvh_taylor` | O(log n) ✅ |
| $R(\mathbf{x})$ — distance to mesh | `gwn_unsigned_distance_point_bvh` | O(log n) ✅ |
| $\nabla f(\mathbf{x})$ — gradient | **new** (Section 1) | O(log n) ✅ |
| $c(\mathbf{x})$ — lower bound | $-4\pi$ (conservative) | O(1) ✅ |

The conservative choice $c = -4\pi$ is always valid for GWN/solid angle
(harmonic.tex Appendix C).  It yields smaller steps than a tight bound
but is trivial to implement.

### 2.5  Angle-Valued Tracing

The solid angle $\Omega$ is angle-valued with period $4\pi$ (it wraps around
near mesh edges).  Algorithm 2 in harmonic.tex handles this by tracking
bracketing level sets $f_-$ and $f_+$ and reducing $4\pi$ periodicity.
This is needed for GWN tracing of non-watertight meshes.

### 2.6  Over-Stepping Acceleration

From harmonic.tex §3.1.4: at iteration $k$, a step of size
$\delta t = 1.75 \rho_k$ is safe if $\delta t \le \rho_k + \rho_{k+1}$.
This provides a simple 1.5–2× speedup with no additional function evaluations.

### 2.7  Surface Normals

At the intersection point, the surface normal is:

$$
\hat{\mathbf{n}} = \frac{\nabla w}{|\nabla w|}
$$

This uses the BVH-accelerated gradient from Section 1.

---

## 3  Approximate SDF for GWN Implicit Surfaces

### 3.1  The Problem

The GWN defines an implicit surface $S = \{w = 0.5\}$.  We want:

$$
d_S(\mathbf{x}) = \text{signed distance from } \mathbf{x} \text{ to } S
$$

This is **not** the same as the distance to the mesh triangles (which is what
`gwn_signed_distance_point_bvh` computes).  For non-watertight meshes,
$S$ is a smooth "repaired" surface that fills holes, while the triangle
geometry has gaps.

### 3.2  Existing Work

**Feng & Crane (SIGGRAPH 2024):** *A Heat Method for Generalized Signed Distance.*

Solves this problem by diffusing normal vectors (screened Poisson) then
recovering distance (Poisson).  Provides GWN-like robustness with true SDF
properties.  **Requires volumetric discretization** (tet mesh / grid) —
not meshless.

**Other approaches:**
- Xu & Barbič 2014: GWN sign + unsigned distance — discontinuous at surface
- 1-Lipschitz Neural Distance Fields (CGF 2024): neural network, needs training
- Neural Representation of Open Surfaces: semi-signed distance, needs network

**Gap in the literature:** no BVH-accelerated meshless SDF for GWN implicit
surfaces exists.

### 3.3  First-Order Pseudo-SDF

For any smooth implicit function $f$ with level set $\{f = c\}$:

$$
\tilde{d}(\mathbf{x}) = \frac{f(\mathbf{x}) - c}{|\nabla f(\mathbf{x})|}
$$

For GWN:

$$
\boxed{\tilde{d}(\mathbf{x}) = \frac{w(\mathbf{x}) - 0.5}{|\nabla w(\mathbf{x})|}}
$$

**Properties:**

1. $\tilde{d} = 0$ exactly on $S$ ✓
2. $|\nabla\tilde{d}| = 1$ on $S$ ✓ (proof below)
3. $|\nabla\tilde{d}| \approx 1$ near $S$ (first-order accurate)
4. Computable in O(log n) via BVH-accelerated $w$ and $\nabla w$ ✓

**Proof of property 2:**
On $S$ where $w = 0.5$, the directional derivative along
$\hat{\mathbf{n}} = \nabla w / |\nabla w|$ is:

$$
\hat{\mathbf{n}} \cdot \nabla\tilde{d}
= \hat{\mathbf{n}} \cdot \nabla\!\left(\frac{w - 0.5}{|\nabla w|}\right)
= \frac{|\nabla w|}{|\nabla w|} + (w - 0.5) \cdot (\cdots) = 1
$$

The second term vanishes because $w - 0.5 = 0$ on $S$.

### 3.4  Second-Order Correction via Curvature

The first-order SDF deviates from the true SDF away from the surface due to
curvature.  The correction uses the **Hessian** $H_w$ of the winding number:

$$
d_2(\mathbf{x}) \approx \tilde{d}(\mathbf{x}) \cdot
  \left(1 - \frac{\kappa \cdot \tilde{d}(\mathbf{x})}{2}\right)
$$

where $\kappa$ is the mean curvature of the level set:

$$
\kappa = \frac{\nabla w^T \cdot H_w \cdot \nabla w}{|\nabla w|^3}
$$

**Harmonic property:** since $\Delta w = 0$, the Hessian is trace-free:
$\text{tr}(H_w) = H_{xx} + H_{yy} + H_{zz} = 0$.  This means only
5 independent components (not 6), reducing computation.

**BVH acceleration of the Hessian:** $H_w$ is the matrix of second derivatives
$\partial_m\partial_n w$.  For the Taylor expansion:

$$
\partial_m\partial_n\omega_0 = \sum_\alpha N_\alpha\,\partial_m\partial_n\partial_\alpha K
$$

This uses $\partial^3 K$ (3rd derivative) contracted with the **same** order-0
moments $N_\alpha$.  The 3rd derivative is the same kernel used for the order-1
winding number $\omega_1$.  So:

| Quantity | Moments | Kernel order | Already available? |
|----------|---------|-------------|-------------------|
| $w$ (winding number) | Order-0 | ∂K (1st) | ✅ |
| $\nabla w$ (gradient) | Order-0 | ∂²K (2nd) | ✅ (Section 1) |
| $H_w$ (Hessian) | Order-0 | ∂³K (3rd) | ✅ same moments |
| $\kappa$ (curvature) | derived from above | — | ✅ |

**All three quantities** ($w$, $\nabla w$, $H_w$) can be evaluated in a
**single BVH traversal** using the same far-field criterion and the same
precomputed moments.  The only additional work is more arithmetic per node.

### 3.5  Newton Projection (Iterative Refinement)

For points far from the surface where the first-order approximation degrades,
Newton iteration along the gradient projects onto the surface:

```
x_proj ← x
repeat:
    w_k ← w(x_proj)
    ∇w_k ← ∇w(x_proj)
    x_proj ← x_proj - [(w_k - 0.5) / |∇w_k|²] · ∇w_k
until |w_k - 0.5| < ε
d ← |x - x_proj| · sign(w(x) - 0.5)
```

Each iteration requires one $w + \nabla w$ evaluation (O(log n) via BVH).
Typically converges in 3–5 iterations.

**Caveat:** Newton finds *a* root along the gradient direction, not necessarily
the globally closest point on $S$.  Near the medial axis of $S$, the closest
point may be in a different direction.  This is a fundamental limitation shared
by all local projection methods.

### 3.6  Harnack Lower Bound on Distance

The Harnack step size $\rho$ (Section 2.3) is a **conservative lower bound**
on the distance to $S$:

$$
d_S(\mathbf{x}) \ge \rho(\mathbf{x})
$$

This is because the Harnack inequality guarantees the function cannot reach
$f^*$ within distance $\rho$.  Combined with the first-order estimate:

$$
\rho(\mathbf{x}) \le d_S(\mathbf{x}) \lesssim \tilde{d}(\mathbf{x})
$$

The Harnack bound is tighter far from the surface (where $\tilde{d}$ may
overestimate), while $\tilde{d}$ is tighter near the surface.

### 3.7  Comparison with Feng & Crane (SIGGRAPH 2024)

| | Feng & Crane 2024 | Multipole Pseudo-SDF |
|---|---|---|
| Accuracy | Exact (solves PDE) | 1st/2nd order approx |
| Discretisation | Tet mesh / voxel grid | None (meshless) |
| Precomputation | Linear system solve | BVH moments (already have) |
| Point queries | Not supported | O(log n) |
| Offset surfaces | Multiple level sets | Any $f^*$ |
| Far-from-surface | Accurate | Degrades (Newton correctable) |

The two approaches are **complementary**: Feng & Crane is better for global
SDF fields (morphological ops, offsetting); the multipole approach is better
for fast per-point queries (ray tracing, collision detection).

---

## 4  Future Direction: Meshless GSD via Regularised Kernels

### 4.1  Connection to Walk on Spheres

Feng & Crane's GSD heat method solves two PDEs:

1. Screened Poisson: $(I - t\Delta)\mathbf{u} = \boldsymbol{\delta}_S$
   (diffuse normals)
2. Poisson: $\Delta\phi = \nabla\cdot(\mathbf{u}/|\mathbf{u}|)$
   (recover distance)

Walk on Spheres (WoS) and Walk on Stars (WoSt) from Sawhney & Crane (2020,
2023) can solve both equations **without volumetric discretisation**.  However,
nesting step 2 inside step 1 causes variance explosion in the Monte Carlo
estimator.  Boundary Value Caching (Miller et al. 2024) may mitigate this.

### 4.2  Direct Route: Regularised GWN

A potentially simpler approach bypasses PDE solving entirely.

The GWN uses the Laplace kernel: each triangle contributes
$\sim \mathbf{n}\cdot\mathbf{r}/r^3$ (the gradient of $1/r$).

The GSD heat method's step 1 effectively replaces this with a **screened**
(Yukawa) kernel: $\mathbf{n}\cdot\mathbf{r} \cdot e^{-r/\sqrt{t}} / r^3$.

One could define a **regularised winding number** directly:

$$
w_\epsilon(\mathbf{x}) = \frac{1}{4\pi}\sum_{\text{triangles}}
  \Omega_{\text{tri}}^\epsilon(\mathbf{x})
$$

where $\Omega_{\text{tri}}^\epsilon$ uses a screened solid angle kernel.
Then the pseudo-SDF:

$$
d_\epsilon(\mathbf{x}) = \frac{w_\epsilon(\mathbf{x}) - 0.5}{|\nabla w_\epsilon(\mathbf{x})|}
$$

would be smoother than the harmonic version (the singularities at mesh edges
are exponentially damped) and closer to a true SDF.

**BVH compatibility:** the screened kernel $e^{-r/\sqrt{t}}/r^2$ admits
multipole expansions (this is the mathematical basis of the screened FMM
used in molecular dynamics).  The existing BVH tree topology and moment
computation pipeline could be adapted to store screened moments, enabling
O(log n) evaluation.

### 4.3  Research Opportunities

1. **BVH-accelerated ∇w:** Implement the gradient formulas from Section 1.
   This enables Harnack tracing and the pseudo-SDF, with zero new
   precomputation.  (Engineering task, low risk.)

2. **Pseudo-SDF evaluation:** Implement $\tilde{d} = (w - 0.5)/|\nabla w|$
   as a new query type.  Evaluate accuracy on standard test meshes.
   (Engineering + experimental validation.)

3. **Curvature-corrected SDF:** Add Hessian evaluation (same BVH traversal)
   and the second-order correction.  Compare with Feng & Crane.
   (Moderate research contribution.)

4. **Screened GWN:** Replace the Laplace kernel with a screened kernel and
   derive the corresponding multipole expansion.  This would yield a
   better pseudo-SDF and could be a publishable result bridging the gap
   between BVH-accelerated GWN and PDE-based GSD.
   (Research contribution, medium risk.)

---

## 5  Implementation Roadmap

### Phase 1: BVH-Accelerated Gradient (no new precomputation)

1. `gwn_gradient_solid_angle_triangle_impl` — per-triangle Biot-Savart gradient
2. `gwn_winding_gradient_point_bvh_taylor_impl` — BVH traversal returning `vec3`
3. `gwn_compute_winding_gradient_batch_bvh_taylor` — public batch API

### Phase 2: Harnack Tracing

4. Harnack step size function (pure math, ~5 lines)
5. `gwn_harnack_trace_ray` — single-ray tracer using $w$, $\nabla w$, $R$
6. Angle-valued variant (Algorithm 2) for GWN with period $4\pi$

### Phase 3: Pseudo-SDF

7. `gwn_pseudo_sdf_point_bvh_taylor` — returns $(w - 0.5)/|\nabla w|$
8. Newton projection refinement
9. Curvature correction (Hessian evaluation)

### Phase 4: Research Extensions

10. Screened kernel moments and evaluation
11. Comparison with Feng & Crane on standard benchmarks
12. Real-time Harnack tracing demo (CUDA kernel per pixel)

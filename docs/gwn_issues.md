# GWN Library — Known Issues, Limitations, and Open Problems

This document captures known issues, numerical pitfalls, algorithmic limitations,
and research-level open problems observed during development of the `smallgwn` library.
It supplements the design notes in `harnack.md`.

---

## 1  Three Hard Problems for Non-Watertight Meshes

Modern geometry processing identifies three fundamental challenges when the input
mesh is **not watertight** (has holes, self-intersections, or inconsistent normals):

### 1.1  Inside/Outside Classification

**Status:** ✅ Solved by GWN (Jacobson et al. 2013).

The generalized winding number $w(\mathbf{x})$ gives a smooth, real-valued
"inside measure" that is robust to all common mesh pathologies.  Points with
$w > 0.5$ are "inside", $w < 0.5$ are "outside".  The level set
$\{w = 0.5\}$ defines an implicit surface that fills holes via harmonic extension.

### 1.2  Ray Tracing

**Status:** 🔶 Partially solved; ongoing work in this library.

**Sub-problem 2a — watertight meshes:** Trivial; any ray-triangle intersection
test works.

**Sub-problem 2b — non-watertight meshes, approximate level set:**
The face-distance Harnack tracer (committed `fa122d7`) traces the implicit surface
$\{w = 0.5\}$ using the Harnack inequality with face distance as the safe-ball radius.
**Limitation:** when the ray must cross a mesh face to reach the level set (which
happens for open meshes whose level set lies "behind" the mesh), the tracer
converges to the face crossing instead of the smooth level set, because face
distance → 0 at face interiors.

**Sub-problem 2c — non-watertight meshes, smooth level set (Algorithm 2):**
The edge-distance / angle-valued tracer (implemented in `../smallgwn-gradient`
on 2026-02-27) uses:
- **Edge distance** as the safe-ball radius — the harmonic function is smooth
  inside any ball not touching triangle *edges*, even if it crosses face interiors.
- **mod 4π mapping** — the ±4π jump at face crossings is invisible under the
  angle-valued function, making the mapped function continuous across faces.
- **Two-sided Harnack stepping** — bound from both sides simultaneously.

This correctly handles open meshes.  See `harnack.md §2.5` and `plan.md`.

### 1.3  Signed Distance Field (SDF)

**Status:** ❌ Open research problem.  No meshless BVH-accelerated solution exists.

Given the implicit surface $S = \{w = 0.5\}$, we want the true signed distance:
$$d_S(\mathbf{x}) = \text{dist}(\mathbf{x}, S) \cdot \text{sign}(w(\mathbf{x}) - 0.5)$$

This is **not** the same as the distance to the mesh triangles (which has holes
for open meshes) or the naïve GWN pseudo-SDF.

**Known approaches:**

| Method | Accuracy | Meshless? | Per-query O(log n)? |
|--------|----------|-----------|---------------------|
| Feng & Crane SIGGRAPH 2024 (heat method) | Near-exact | ❌ (tet/voxel) | ❌ (precomputed field) |
| $\tilde{d} = (w - 0.5)/|\nabla w|$ | 1st order near S | ✅ | ✅ |
| Curvature-corrected version | 2nd order near S | ✅ | ✅ |
| Newton projection onto S | Accurate, local | ✅ | ✅ (multiple BVH queries) |
| Harnack lower bound $\rho$ | Lower bound only | ✅ | ✅ |
| Screened GWN kernel | Unknown (potential paper) | ✅ | ✅ (needs new moments) |

The first-order pseudo-SDF uses the gradient already implemented:
$$\tilde{d}(\mathbf{x}) = \frac{w(\mathbf{x}) - 0.5}{|\nabla w(\mathbf{x})|}$$

This is exact on $S$ and satisfies $|\nabla \tilde{d}| = 1$ on $S$, but degrades
away from the surface.  See `harnack.md §3`.

---

## 2  Known Implementation Issues

### 2.1  Face-Distance Tracer Fails for Open Meshes

**File:** `include/gwn/detail/gwn_harnack_trace_impl.cuh`  
**Committed:** `fa122d7`

The tracer finds $\{w = 0.5\}$ correctly for watertight meshes.  For open meshes
(e.g., a hemisphere), the $w = 0.5$ level set passes through the interior of the
domain and the ray must cross face interiors.  The face-distance BVH query returns
R → 0 as the query point approaches a face, pinning the tracer.

**Current status:** Use the angle-valued API
`gwn_compute_harnack_trace_angle_batch_bvh_taylor(...)` for open meshes.
The legacy face-distance tracer remains available for watertight cases and
backward compatibility.

**Documentation:** The API docstring in `gwn_query.cuh` already notes this
limitation (see `gwn_harnack_trace_face_batch_bvh_taylor`).

### 2.2  Winding Number at Mesh Vertices is Non-Monotone

**Observed during testing:** For a closed octahedron, the GWN at a *vertex*
is approximately 0.108 (≈ 1/9), not 0 or 1.  This is correct mathematically
(the winding number equals the solid angle subtended divided by 4π, and at a
vertex this is ≈ 1/3 of the sphere solid angle for 3 faces meeting there —
actually ≈ 0.108 for the octahedron's 4-face vertex).

**Implication:** Tests that check the hit point is on the mesh surface ($w ≈ 0.5$)
may fail at vertex vicinities.  Tests should use `EXPECT_NEAR(w, 0.5, tol)` with
`tol ≥ 0.05` when rays might intersect near vertices.

**Not a bug:** This is expected behavior.  The $w = 0.5$ level set is a smooth
closed surface that passes *through* the mesh faces but bulges away from vertices.

### 2.3  False-Positive Risk in Finite-Difference Gradient Tests

**File:** `tests/unit_harnack_gradient.cu` (gradient tests)

Initially, gradient tests were false positives because both the BVH gradient and
the finite-difference reference were near zero at query points far from the mesh.
The relative error $\|g_{\text{bvh}} - g_{\text{fd}}\| / \|g_{\text{fd}}\|$ is
ill-conditioned when $\|g_{\text{fd}}\| \approx 0$.

**Resolution:** Tests now include query points near (but not inside) the mesh,
where the gradient magnitude is significant (> 0.01), and verify absolute error
$< 10^{-3}$ for Order-1 Taylor expansion.  Integration tests use 72 directions
on a sphere to ensure broad coverage.

### 2.4  Unused Variable in Order-2 Gradient

**File:** `include/gwn/detail/gwn_query_gradient_impl.cuh`  
**Found during code review, fixed before Harnack tracing commit.**

The variable `half_15_L` was computed but never used.  Removed in the commit
preceding `fa122d7`.

### 2.5  Interior Edge Double-Evaluation in Edge Distance

**File:** `include/gwn/detail/gwn_query_distance_impl.cuh` (in progress)

For the edge-distance BVH query, interior edges shared by two triangles are
evaluated twice (once per adjacent triangle).  This is harmless (taking the min
over both evaluations gives the correct result) but wastes ~50% of leaf-node work
for closed meshes.

**Not a correctness issue.**  A de-duplication pass would require storing an
edge adjacency table, which adds preprocessing complexity.  Current design accepts
the redundancy.

---

## 3  Algorithmic Limitations

### 3.1  Harnack Tracing Convergence Near Level Set

The Harnack step size goes to zero as the query point approaches the level set
($f_t \to f^*$).  The stopping criterion

$$\min(f_t - f_-, f_+ - f_t) \;/\; \max(|\nabla f_t|, \varepsilon) < \varepsilon$$

prevents infinite loops but means the final reported intersection point has error
of order $\varepsilon \cdot |\nabla f|^{-1}$.  Near the level set, $|\nabla f|$
can be very small (at critical points of the winding number), making the error
large in ray-parameter space.

**Not observed in practice** for non-degenerate query meshes, but be aware of
it for meshes where the level set is nearly tangent to itself (thin features).

### 3.2  The BVH Refit Step Does Not Update Edge Distance

The existing BVH refit (`gwn_bvh_refit`) updates AABB nodes and moment trees
after vertex deformation.  Edge distance is computed on-the-fly from geometry
accessors during traversal — no precomputed edge distance structure exists.
Refit is therefore transparent to edge distance queries.  ✅ No issue.

### 3.3  Taylor Expansion Accuracy Near Mesh

The Taylor expansion error bound is $O((d/r)^{N+1})$ where $d$ is the cluster
diameter and $r$ is the distance from query to cluster centroid.  Very close to
the mesh (r < cluster size), the far-field criterion fails and the traversal
falls back to per-triangle evaluation.

For Harnack tracing, this means the **final BVH query** (closest approach step)
always falls back to brute force over a small number of nearby triangles.  This
is correct but may be a hotspot for very dense meshes.

### 3.4  No Multi-Ray Coherence Exploitation

The batch API processes rays independently.  For sets of rays originating from
the same point (camera rays) or in similar directions, BVH traversal paths are
highly correlated.  A ray-packet traversal (SIMD width 4–8) could give 2–4×
speedup.  This is a known limitation of the current design.

---

## 4  Numerical Precision Notes

### 4.1  The mod 4π Mapping

For the angle-valued tracer, `glsl_mod(x, 4π)` must return a value in `[0, 4π)`.
The standard `fmod(x, m)` has the wrong sign for negative `x`; use:
```cpp
Real glsl_mod(Real x, Real m) { return x - m * floor(x / m); }
```
For `float`, catastrophic cancellation occurs when $|x| \gg 4\pi$.  Since the
solid angle $\Omega \in (-4\pi, 4\pi)$ normally (normalized GWN from a bounded
mesh), this is not an issue in practice.

### 4.2  Harnack Shift Parameter

The two-sided step formula requires a shift $c$ such that `val + c > 0` on the
Harnack ball.  For the angle-valued function `val ∈ [0, 4π)`, the standard
choice is `c = 4π`.  This gives `val + c ∈ [4π, 8π)`, always positive.  ✅

For the face-distance tracer, where `val = w - 0.5 ∈ (-0.5, 0.5)`, the shift
is `c = 0.5` (making `val + c ∈ (0, 1)`).

### 4.3  Gradient Magnitude Near Zero

Near critical points of `w` (where `∇w = 0`), the pseudo-SDF `(w - 0.5)/|∇w|`
diverges.  The first-order approximation breaks down.  In the Newton projection
algorithm, the iteration stalls.

**In practice:** critical points of `w` inside the domain are saddle points
(not minima), so they are transient — the Newton iteration escapes them.
Guard with a minimum `|∇w| ≥ ε` in denominators.

---

## 5  Open Research Problems

### 5.1  True Meshless SDF for GWN Surfaces

As discussed in Section 1.3, no existing method provides an exact, meshless,
per-query-O(log n) SDF for GWN implicit surfaces.

**Promising direction:** Screened (Yukawa) kernel replacement.  Replace the
Laplace kernel $1/r$ in the winding number with $e^{-\kappa r}/r$ (screened
Coulomb).  The resulting "screened winding number":
$$w_\kappa(\mathbf{x}) = \frac{1}{4\pi}\sum_\text{tri}\Omega_\text{tri}^\kappa(\mathbf{x})$$
would have exponentially damped edge singularities, making its level set closer
to a true SDF.  The screened kernel admits multipole expansion (screened FMM),
so the existing BVH tree can be reused with new moment types.

### 5.2  Walk on Spheres for GSD

Keenan Crane's group (Walk on Spheres / Walk on Stars) is likely to combine:
- GWN-defined inside/outside classification
- WoS/WoSt to solve the two PDEs in Feng & Crane's heat method

This would produce a **fully meshless GSD solver** decoupled from any volumetric
discretization.  The key technical hurdle is variance explosion when nesting
two Monte Carlo estimators.  Boundary Value Caching (Miller et al. 2024) may
be the bridge.

The BVH-accelerated gradient from this library is a natural building block:
the WoS random walk needs harmonic measure evaluations, which are essentially
winding number queries.

### 5.3  GPU-Parallel Harnack Tracing

The current Harnack tracing kernel (one CUDA thread per ray) is correct but
does not exploit:
- Warp-level divergence reduction (different rays terminate at different iterations)
- Shared memory caching of BVH nodes along common traversal paths
- Persistent thread blocks for irregular workload

For the GPU rendering use case (one ray per pixel, 1920×1080 frame),
these optimizations are critical.

### 5.4  Multipole for Gradient: Higher Order Hessian

The Hessian $H_w$ (second derivatives of winding number) would enable:
- Curvature-corrected pseudo-SDF (Section 3.4 of `harnack.md`)
- Principal direction estimation on the level set
- Better Harnack convergence (Newton-like step using second-order info)

The Hessian uses the same moments as the gradient (just one more derivative of K).
No new BVH data is required.  Implementation is mechanical but O(9 components)
algebraic expansion.

---

## 6  Implementation Status Summary

| Feature | Status | Notes |
|---------|--------|-------|
| BVH winding number query (Orders 0-2) | ✅ main | Committed |
| BVH unsigned face distance | ✅ main | Committed |
| BVH gradient (Orders 0-2) | ✅ `feat/winding-gradient` | Committed `e6bdb1a` |
| Face-distance Harnack tracer | ✅ `feat/winding-gradient` | Committed `fa122d7` |
| Point-to-segment distance primitive | ✅ `feat/winding-gradient` | Uncommitted |
| BVH edge distance query | ✅ `feat/winding-gradient` | Uncommitted |
| Public API for edge distance | ✅ `../smallgwn-gradient` | Implemented 2026-02-27 |
| Angle-valued Harnack tracer | ✅ `../smallgwn-gradient` | Algorithm 2 (mod 4π + two-sided step) |
| Batch API for angle-valued tracer | ✅ `../smallgwn-gradient` | `gwn_compute_harnack_trace_angle_batch_bvh_taylor` |
| Tests for edge distance + angle tracer | ✅ `../smallgwn-gradient` | Unit + integration coverage added |
| Pseudo-SDF $(w-0.5)/|\nabla w|$ | ⬜ not started | Easy once gradient exists |
| Hessian of winding number | ⬜ not started | Research |
| Screened GWN kernel | ⬜ not started | Research |

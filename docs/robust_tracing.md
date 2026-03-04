# Robust Harnack Tracing for Generalized Winding Number Level Sets

## 1  Background

We consider the problem of intersecting a ray
$\mathbf{r}(t) = \mathbf{r}_0 + t\mathbf{v}$
with the $w = \tfrac12$ level set of the generalized winding number (GWN)
of a triangle mesh $\Sigma$, or equivalently the $\omega = 2\pi$ level set of the
solid angle function $\omega(\mathbf{x}) := 4\pi\, w(\mathbf{x})$.

The Harnack tracing algorithm of Gillespie et al. (2024) provides a
first-hit guarantee for level sets of harmonic functions.  For an
angle-valued function with period $T = 4\pi$, the algorithm (their
Algorithm 2) requires two inputs at every query point $\mathbf{x}$:

1. A radius $R(\mathbf{x}) > 0$ such that $\omega$ (or more precisely, a local
   continuous lift $\tilde\omega$ of $\omega \bmod 4\pi$) is **harmonic** on
   the open ball $B_R(\mathbf{x})$.

2. A lower bound $c(\mathbf{x})$ such that $\tilde\omega(\mathbf{y}) > c$
   for all $\mathbf{y} \in B_R(\mathbf{x})$.

Smaller $R$ and more negative $c$ both reduce the Harnack step size
$\rho$, increasing iteration count.  The key to an efficient tracer is
choosing $R$ and $c$ as large/tight as possible while maintaining
correctness.


## 2  Singular set of the solid angle function

Let $\Sigma$ be an oriented triangle mesh with vertex set $V$, edge set $E$,
and triangle (face) set $F$.  An edge $e \in E$ is *interior* if it is
shared by exactly two triangles with consistent orientation; it is a
*boundary edge* if it belongs to exactly one triangle; and it is
*non-manifold* if it belongs to three or more triangles.

**Definition 2.1.**  The *singular set* of $\omega$ is

$$S := \bigcup_{f \in F} f \;\cup\; \bigcup_{e \in E_\partial} e \;\cup\; \bigcup_{v \in V_\partial} \{v\},$$

where we distinguish the following components:
- Interior faces $f$: $\omega$ has a $4\pi$ jump across $f$.
- Boundary edges $E_\partial$ and their incident vertices $V_\partial$:
  $\omega$ has a genuine singularity (analogous to $\operatorname{atan2}(y,x)$
  at the origin).  The function wraps through a full $4\pi$ period in any
  neighborhood of these edges.

We now show that the angle-valued formulation eliminates all
singularities except those at boundary edges and non-manifold edges.


### 2.1  Interior faces are not singularities in the angle-valued sense

**Proposition 2.2.**  Let $e$ be an interior edge shared by triangles
$f_i$ and $f_j$ with consistent orientation, or let $p$ be an interior
point of a triangle $f_i$.  Then $\omega \bmod 4\pi$ admits a continuous
harmonic lift in a neighborhood of $e$ (resp. $p$).

*Proof.*  Consider a point $p$ in the interior of face $f_i$.  Write

$$\omega(\mathbf{x}) = \Omega_i(\mathbf{x}) + S(\mathbf{x}), \qquad
  S(\mathbf{x}) := \textstyle\sum_{j \neq i} \Omega_j(\mathbf{x}),$$

where $\Omega_k$ is the solid angle of triangle $k$.  The sum $S$ is
smooth and harmonic in a neighborhood of $p$ (since $p$ lies in the
interior of no other triangle).  Write $S_0 := S(p)$.

By the van Oosterom–Strackee formula, $\Omega_i(\mathbf{x})$ has range
$(-2\pi, 2\pi)$, with limits

$$\lim_{\mathbf{x} \to p^+} \Omega_i(\mathbf{x}) = 2\pi, \qquad
  \lim_{\mathbf{x} \to p^-} \Omega_i(\mathbf{x}) = -2\pi,$$

where $p^+$ and $p^-$ denote limits from the positive-normal and
negative-normal sides of $f_i$ respectively.  Hence the total solid angle
has limits

$$\omega \to 2\pi + S_0 \;\text{(positive side)}, \qquad
  \omega \to -2\pi + S_0 \;\text{(negative side)},$$

giving a jump of $\Delta\omega = 4\pi$ across $f_i$.  Define the lift

$$\tilde\omega(\mathbf{x}) := \begin{cases}
  \omega(\mathbf{x}) & \text{on the positive-normal side of } f_i, \\
  \omega(\mathbf{x}) + 4\pi & \text{on the negative-normal side of } f_i.
\end{cases}$$

Then:

- Positive side: $\tilde\omega \to 2\pi + S_0$.
- Negative side: $\tilde\omega = (\Omega_i + S) + 4\pi \to (-2\pi + S_0) + 4\pi = 2\pi + S_0$.

So $\tilde\omega$ is continuous across $f_i$.  Moreover, $\tilde\omega$
equals $\omega + \text{const}$ on each side of $f_i$, and $\omega$ is
harmonic away from the mesh.  By continuity across $f_i$ and harmonicity
on each open half-space, the singularity is removable and $\tilde\omega$
is harmonic across $f_i$.  ∎


### 2.2  Interior edges: singularity cancellation

**Proposition 2.3.**  At an interior edge $e$ shared by two
consistently-oriented triangles $f_i, f_j$, the singular contributions
to $\nabla\omega$ from $\Omega_i$ and $\Omega_j$ cancel, and $\omega$ is
smooth (hence harmonic) in a neighborhood of $e$ away from the triangles
themselves.

*Proof sketch.*  The gradient of the solid angle of a single triangle
is given by the Biot–Savart formula (Eq. 11 of Gillespie et al.):

$$\nabla\Omega_P(\mathbf{x}) = \sum_{k=1}^n (\mathbf{g}_k - \mathbf{g}_{k+1})
  \cdot \left(\frac{\mathbf{g}_k}{\|\mathbf{g}_k\|} - \frac{\mathbf{g}_{k+1}}{\|\mathbf{g}_{k+1}\|}\right)
  \frac{\mathbf{g}_k \times \mathbf{g}_{k+1}}{\|\mathbf{g}_k \times \mathbf{g}_{k+1}\|^2},$$

where $\mathbf{g}_k = \mathbf{p}_k - \mathbf{x}$.  Each edge contributes
one term to this sum for each triangle it belongs to.  For two triangles
$f_i = (A, B, C)$ and $f_j = (B, A, D)$ sharing edge $AB$ with
consistent orientation, the edge $AB$ appears as the directed edge
$A \to B$ in $f_i$ and $B \to A$ in $f_j$.  Their contributions to the
Biot–Savart sum are negatives of each other and cancel.  ∎


### 2.3  Boundary edges: genuine singularities

At a boundary edge $e$, the solid angle of the single incident triangle
has an uncancelled Biot–Savart term that diverges as $\mathbf{x} \to e$.
The function $\omega$ wraps through a full $4\pi$ period as $\mathbf{x}$
circles any point of $e$.  No continuous lift exists in a neighborhood
of $e$; hence $e$ is a true singularity.


### 2.4  Non-manifold and inconsistently-oriented edges

An edge $e$ shared by $k \geq 3$ triangles, or by two triangles with
*inconsistent* orientation (the directed edge appears in the same direction
in both triangles), retains uncancelled singularities.  For safety, such
edges must be classified as singular.


## 3  Correct choice of $R$ for the angle-valued GWN

**Theorem 3.1.**  For the angle-valued solid angle function
$\omega(\mathbf{x}) \bmod 4\pi$ of a triangle mesh $\Sigma$, let $E_\partial$
denote the set of boundary edges (edges shared by $\neq 2$
consistently-oriented triangles).  Then

$$R(\mathbf{x}) := \operatorname{dist}(\mathbf{x},\, E_\partial)$$

is a valid radius for Harnack tracing: the local lift $\tilde\omega$ is
harmonic on $B_R(\mathbf{x})$ for every $\mathbf{x} \notin E_\partial$.

*Proof.*  By construction, $B_R(\mathbf{x})$ contains no boundary edges.
It may contain interior faces and interior edges of $\Sigma$.  By
Propositions 2.2 and 2.3, the lift $\tilde\omega$ is harmonic across
these features.  Hence $\tilde\omega$ is harmonic on $B_R(\mathbf{x})$.  ∎

**Remark 3.2.**  The current implementation uses
$R(\mathbf{x}) = \operatorname{dist}(\mathbf{x}, \Sigma)$
(distance to all triangle faces), which satisfies
$R_{\text{current}} \leq R_{\text{boundary}}$.  This is valid but overly
conservative: near interior faces (far from boundary edges), the current
$R$ approaches zero while the correct $R$ may be large.


### 3.1  Lower bound $c = -4\pi$

For the solid angle of a connected, intersection-free simplicial curve
$P$ on the boundary of a convex domain, Gillespie et al. prove
(Appendix C) that a line segment crosses the $2\pi$ level set at most
once (signed).  This implies that the continuous lift satisfies
$\tilde\omega > -4\pi$ within any ball not containing singular points,
giving $c = -4\pi$.

For triangle meshes used via Section 4.4 of the paper (mesh repair via
GWN), the boundary edges play the role of the curve $P$.  The bound
$c = -4\pi$ is valid when the ball does not contain any boundary edge,
which is guaranteed by $R = \operatorname{dist}(\mathbf{x}, E_\partial)$.


## 4  The closed-mesh degeneracy

For a **closed** (watertight) mesh with no boundary edges,
$E_\partial = \emptyset$, and Theorem 3.1 gives $R = +\infty$.

**Proposition 4.1.**  For a closed, consistently-oriented triangle mesh
$\Sigma$, the solid angle function satisfies $\omega(\mathbf{x}) = 0$ for
all $\mathbf{x}$ in the exterior of $\Sigma$, and $\omega(\mathbf{x}) = 4\pi$
for all $\mathbf{x}$ in the interior.

*Proof.*  This is the discrete Gauss law (Jacobson et al. 2013, §2).  ∎

**Corollary 4.2.**  For a closed mesh, $\omega \bmod 4\pi = 0$
everywhere away from $\Sigma$.  The Harnack tracer (Algorithm 2) sees
$\tilde\omega \equiv 0$ (or $\equiv 4\pi$ modulo the period), so the
distance to the target $2\pi$ is always $2\pi$.  The tracer cannot detect
the $\omega = 2\pi$ level set, which exists only as the $4\pi$ jump
discontinuity on the mesh surface.

**Consequence.**  Using $R = \operatorname{dist}(\mathbf{x}, E_\partial)$
alone is insufficient for finding the $\omega = 2\pi$ level set on
interior faces of a closed (or "closed-like") mesh region.  The level set
at these faces is a *topological* feature (the $4\pi$ jump), not a
smooth crossing of the harmonic lift through $2\pi$.


### 4.1  Nearly-closed meshes

For an open mesh where most edges are interior (e.g., a sphere
with a small hole), the function $\omega$ is nearly constant in the
"closed-like" regions:
$\omega \approx -\varepsilon$ (exterior) and
$\omega \approx 4\pi - \varepsilon$ (interior), where
$\varepsilon = O(A_{\text{hole}} / r^2)$.

The lift $\tilde\omega \approx -\varepsilon$ on both sides of interior
faces (after the $4\pi$ stitching).  With $R = \operatorname{dist}(\mathbf{x},
E_\partial) \gg 1$, the Harnack step $\rho \approx 0.094R$ causes the
tracer to step over interior faces without detecting any crossing.

The current implementation ($R = \operatorname{dist}(\mathbf{x}, \Sigma)$)
avoids this problem: $R \to 0$ near every face, forcing the tracer to
slow down.  When $R$ is very small, the gradient $\|\nabla\omega\|$
becomes large (it diverges as $1/d$ where $d$ is the distance to the
face), and the stopping criterion
$|\omega - 2\pi| / \|\nabla\omega\| < \varepsilon$ triggers.


## 5  Combined approach: Harnack tracing + ray–triangle intersection

To obtain both correctness and efficiency, we propose combining two
independent first-hit queries:

$$t_{\text{hit}} = \min(t_{\text{mesh}},\; t_{\text{fill}}),$$

where:
- $t_{\text{mesh}}$: first ray–triangle intersection via standard BVH
  traversal (Möller–Trumbore test).  Finds the level set at interior
  faces.
- $t_{\text{fill}}$: Harnack trace with
  $R = \operatorname{dist}(\mathbf{x}, E_\partial)$.  Finds the smooth
  component of the $\omega = 2\pi$ level set near boundary edges.


### 5.1  Correctness argument

**Claim 5.1.**  $t_{\text{hit}}$ is the first intersection of the ray
with the $\omega = 2\pi$ level set.

The $\omega = 2\pi$ level set consists of two kinds of points:

1. **Face crossings.**  At each interior face $f$ of $\Sigma$, $\omega$
   jumps from $\approx 0$ to $\approx 4\pi$, crossing $2\pi$ precisely on
   $f$.  These are detected by $t_{\text{mesh}}$ (ray–triangle
   intersection).

2. **Smooth fill.**  Near boundary edges, $\omega$ varies continuously
   through $2\pi$ on a smooth surface spanning the boundary.  These are
   detected by $t_{\text{fill}}$ (Harnack tracing with boundary-edge $R$).

Both parts are subsets of the same connected surface (the $w = \tfrac12$
isosurface of the GWN).  Each detection method provides its own first-hit
guarantee:

- **Ray–triangle intersection** is exact and provides the first
  intersection with the mesh faces.
- **Harnack tracing** converges to the first intersection with the lift's
  target level set (Theorem A.1 of Gillespie et al., with $R =
  \operatorname{dist}(\mathbf{x}, E_\partial)$ satisfying the required
  hypotheses).

Taking the minimum of the two $t$-values gives the overall first hit.


### 5.2  Case analysis

| Case | $t_{\text{mesh}}$ | $t_{\text{fill}}$ | Result |
|------|---------------------|----------------------|--------|
| Closed mesh (no boundary edges) | Face hit ✓ | $R = \infty$, no convergence | $t_{\text{mesh}}$ |
| Open mesh, ray hits interior face | Face hit ✓ | Lift ≈ const, no hit here | $t_{\text{mesh}}$ |
| Open mesh, ray hits smooth fill | No face hit | Harnack finds fill ✓ | $t_{\text{fill}}$ |
| Smooth fill before interior face | Face hit at $t_1$ | Fill hit at $t_0 < t_1$ | $\min = t_{\text{fill}}$ |
| Interior face before smooth fill | Face hit at $t_0$ | Fill hit at $t_1 > t_0$ | $\min = t_{\text{mesh}}$ |
| Non-manifold / inconsistent edges | Conservatively boundary | Both methods work | Correct |


### 5.3  Shading normals

At $t_{\text{mesh}}$ (face crossing), the surface normal is the triangle
normal (or the gradient $\nabla\omega / \|\nabla\omega\|$, which
approximates the triangle normal near the face).

At $t_{\text{fill}}$ (smooth fill), the surface normal is
$\nabla\omega / \|\nabla\omega\|$, which is well-defined and smooth.

At the seam where the smooth fill meets the mesh surface, the normals
are consistent (both approach the triangle normals of the boundary faces).


## 6  Boundary edge identification

### 6.1  Algorithm

For each triangle $(v_0, v_1, v_2)$, extract three directed half-edges:
$(v_0, v_1)$, $(v_1, v_2)$, $(v_2, v_0)$.

For each undirected edge $\{a, b\}$ (keyed as $(\min(a,b), \max(a,b))$),
count the number of incident triangles and check orientation consistency:

- **Count = 1**: boundary edge.
- **Count = 2, consistent orientation** (directed edge $(a,b)$ appears in
  one triangle, $(b,a)$ in the other): interior edge.
- **Count = 2, inconsistent orientation** (directed edge $(a,b)$ appears
  in both triangles): treat as boundary (singularity does not cancel).
- **Count ≥ 3**: non-manifold; treat as boundary.


### 6.2  Robustness guarantee

**Claim 6.1.**  The algorithm has **zero false negatives**: every edge
at which $\omega$ has a genuine singularity is correctly classified as
boundary.

*Proof.*  The algorithm operates exclusively on integer vertex indices.
There are no floating-point comparisons or geometric predicates.  A true
boundary edge has exactly one incident triangle, giving count = 1, which
is always detected.  A non-manifold edge has count ≥ 3, which is always
detected.  An inconsistently-oriented edge has two triangles with the same
directed half-edge, which is detected by the orientation check.

The only source of **false positives** is index duplication: if two
physically-coincident vertices receive different indices (e.g., from mesh
exporters that duplicate vertices at UV seams), a shared edge may be
counted as two separate boundary edges.  This is conservative (safe): it
makes $R$ smaller, not larger.  ∎


### 6.3  Per-triangle encoding

For GPU use, the boundary status is stored as a 3-bit mask per triangle
(`uint8_t boundary_edge_mask`), where bit $i$ indicates that edge $i$
of the triangle is a boundary edge.


## 7  Efficiency analysis

### 7.1  Harnack step size with $R = \operatorname{dist}(\mathbf{x}, E_\partial)$

With $c = -4\pi$ and $\omega \approx 0$ (exterior), the step–radius ratio is

$$\frac{\rho}{R} = \frac{1}{2}\left|\frac{2}{3} + 2 - \sqrt{\left(\frac{2}{3}\right)^2 + \frac{16}{3}}\right| \approx 0.094.$$

For an open mesh with boundary edges at distance $D$ from the query
point, $R \approx D$ and $\rho \approx 0.094D$.  Compare with the current
$R = \operatorname{dist}(\mathbf{x}, \Sigma)$: near an interior face at
distance $d \ll D$, $\rho \approx 0.094d$.  The boundary-edge $R$ gives
a factor $D/d$ improvement in step size.


### 7.2  Ray–triangle intersection cost

Standard BVH ray traversal with Möller–Trumbore tests is well-studied
and typically costs $O(\log N)$ per ray for $N$ triangles.  This is a
one-time cost per ray (finding the first hit), not per Harnack iteration.


### 7.3  Boundary-edge distance query cost

The boundary-edge distance query can reuse the existing triangle BVH,
computing distance only to edges marked as boundary in the per-triangle
mask.  The cost is similar to the current face-distance query (same BVH
structure, same traversal, different primitive test at leaves).


## 8  Implementation plan

1. **CPU preprocessing**: identify boundary edges using the algorithm
   of Section 6.1.  Compute `boundary_edge_mask` per triangle.

2. **GPU boundary-edge distance**: modify the existing edge-distance
   BVH query to skip non-boundary edges using the mask.

3. **GPU ray–triangle intersection**: implement a standard BVH ray
   traversal kernel using the existing AABB tree.

4. **Combined tracer**: execute Harnack trace with boundary-edge $R$
   and ray–triangle intersection in parallel or sequentially; return
   $\min(t_{\text{mesh}}, t_{\text{fill}})$.

5. **Closed-mesh optimization**: when $E_\partial = \emptyset$, skip the
   Harnack trace entirely (it would produce no hits) and rely solely on
   ray–triangle intersection.


## Appendix A: The Harnack step ceiling at $0.75\rho$

The overstep acceleration (Section 3.1.4 of Gillespie et al.) uses a
step of $\delta t = 1.75\rho$ followed by validation: the new Harnack step
$\rho'$ must satisfy $\rho' \geq t_{\text{overstep}} = 0.75\rho$.

For a monotonically-decreasing $R$ (ray approaching the mesh), the
maximum overstep $\delta$ satisfying the validation constraint is

$$\delta_{\max} \approx 0.078R.$$

Since $\rho \approx 0.094R$, this gives
$\delta_{\max} / \rho \approx 0.83$, and the current $0.75$ is
appropriately conservative.  Any attempt to increase the overstep
beyond $\sim 0.83\rho$ will systematically fail validation, causing
backoff cycles that increase total iteration count.  This was confirmed
experimentally: a gradient-based overstep enhancement that attempted
larger oversteps resulted in 2–4× more iterations due to repeated
backoffs.

# Refit Pipeline Maturity Design

## Goal

Make dynamic mesh updates a supported workflow for existing BVH topology.

The first version handles vertex-position changes only. Triangle indices and
primitive count stay fixed. Topology rebuild remains the path for topology
changes.

## Public API

```cpp
gwn_status gwn_update_geometry(
    gwn_geometry_object<Real, Index> &geometry,
    cuda::std::span<Real const> x,
    cuda::std::span<Real const> y,
    cuda::std::span<Real const> z,
    cudaStream_t stream = gwn_default_stream()
) noexcept;

gwn_status gwn_update_geometry(
    gwn_geometry_object<Real, Index> &geometry,
    cudaStream_t stream = gwn_default_stream()
) noexcept;

gwn_status gwn_bvh_refit_aabb_moment(
    gwn_geometry_object<Real, Index> const &geometry,
    gwn_bvh_topology_object<Width, Real, Index> const &topology,
    gwn_bvh_aabb_tree_object<Width, Real, Index> &aabb_tree,
    gwn_bvh_moment_tree_object<Width, Order, Real, Index> &moment_tree,
    cudaStream_t stream = gwn_default_stream()
) noexcept;
```

## Behavior

`gwn_update_geometry(geometry, x, y, z, stream)` enqueues host-to-device vertex
position copies, then recomputes position-derived geometry cache.

`gwn_update_geometry(geometry, stream)` assumes vertex positions were already
modified in place on the device. It enqueues cache recomputation only.

Both overloads preserve triangle indices, boundary-edge masks, and singular-edge
counts. Both recompute vertex normals.

`gwn_bvh_refit_aabb_moment<Order>` refits AABB payloads first, then Taylor
moments. Moment refit reads AABB payloads for `max_p_dist2`, so this order is
part of the public workflow.

## Stream Semantics

The update APIs follow `cudaMemcpyAsync` lifetime rules.

`gwn_status::ok()` means copy and kernel work was enqueued on `stream`. It is not
a stream completion signal. The caller keeps host memory alive until CUDA has
read it, and uses stream sync or events for reuse and destruction.

For device in-place updates, caller-provided writes must precede
`gwn_update_geometry(geometry, stream)` in stream order, or be connected by CUDA
events.

## Checks

Host-position update checks:

- `geometry.accessor().is_valid()`
- `x`, `y`, and `z` have equal length
- input length equals `geometry.vertex_count()`
- non-empty spans have non-null storage

Device in-place update checks:

- `geometry.accessor().is_valid()`
- vertex and normal buffers are present through the accessor invariant

Refit pipeline checks:

- Reuse the existing checks from `gwn_bvh_refit_aabb`
- Reuse the existing checks from `gwn_bvh_refit_moment`
- Add no extra broad validation in the wrapper

## Validation

Unit tests:

- Host update changes positions and refreshes vertex normals.
- Host update rejects size mismatch.
- Device in-place update refreshes vertex normals after a kernel edits vertices.

Pipeline tests:

- Build topology once.
- Update vertices.
- Refit AABB and moment payloads.
- Compare Taylor query results against a fresh rebuild with the updated mesh.

Benchmark:

- Add a dynamic refit stage that updates vertices, refits payloads, and queries.
- Keep topology rebuild out of that stage.

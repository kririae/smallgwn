# smallgwn

A header-only CUDA library implementing generalized winding-number (GWN)-related algorithms on triangle meshes, including [Fast Winding Numbers (FGWN)](https://doi.org/10.1145/3197517.3201337) and [Harnack tracking](https://doi.org/10.1145/3658201), accelerated with wide BVHs via [LBVH](https://research.nvidia.com/publication/2009-03_fast-bvh-construction-gpus) and [H-PLOC](https://doi.org/10.1145/3675377), plus Taylor multipole expansions.

## Requirements

- **CUDA Toolkit** (C++20)

Optional:

- **Eigen3** and **Intel TBB** enable `gwn_eigen_bridge.cuh` (and CMake helper target `gwn::smallgwn_eigen_bridge`).
- **libigl** can be used to load the geometry.
- Core CUDA/GTest test targets configure without these optional dependencies.
- `smallgwn_unit_eigen_bridge` is added only when **Eigen3** and **Intel TBB** are available.
- Tests that use `tests/reference_cpu.cuh` are added only when **Eigen3** and **Intel TBB** are available.
- SDF parity tests are added only when **libigl** is available.

## Building

I recommend using it as a header-only library by including `gwn.cuh` directly in your project, and this project can be added as a submodule.
However, if you want to build the library separately, you can use CMake:

```bash
cmake -B build
cmake --build build
```

For repo-local warning cleanup work, you can opt into a conservative strict-warning profile without
changing default consumer behavior. This only affects repo-defined benchmark/test targets; the
header-only library target remains unchanged for downstream consumers. Reference-heavy suites that
pull in vendored HDK/TBB code keep their default diagnostics so the strict profile stays focused on
the benchmark target plus repo-owned `unit` / `fixtures` sources:

```bash
cmake -S . -B build-strict -DSMALLGWN_ENABLE_STRICT_WARNINGS=ON
cmake --build build-strict
```

## Usage

Include a single header:

```cpp
#include <gwn/gwn.cuh>
```

### Minimal example

```cpp
using Real  = float;
using Index = std::uint32_t;

cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

// Upload geometry to the GPU (SoA layout: separate x/y/z arrays + triangle index arrays)
gwn::gwn_geometry_object<Real, Index> geometry;
gwn::gwn_upload_geometry(
    geometry,
    cuda::std::span<Real const>(vx.data(), vx.size()),
    cuda::std::span<Real const>(vy.data(), vy.size()),
    cuda::std::span<Real const>(vz.data(), vz.size()),
    cuda::std::span<Index const>(i0.data(), i0.size()),
    cuda::std::span<Index const>(i1.data(), i1.size()),
    cuda::std::span<Index const>(i2.data(), i2.size()),
    stream);

// Build the necessary BVH data structures for the geometry
gwn::gwn_bvh4_topology_object<Real, Index> topology;
gwn::gwn_bvh4_aabb_object<Real, Index>     aabb;
gwn::gwn_bvh4_moment_object<1, Real, Index> moments;  // 1 = taylor approximation order
gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<
    1,     // taylor approximation order, 1 would be sufficient
    4,     // BVH width, 4 is a good default
    Real,  // floating-point type
    Index  // index type, in case you are dealing with super large meshes
>(geometry, topology, aabb, moments, stream);

// Batch query example: compute winding numbers at a set of query points
gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_wn;
// ... copy query points to d_qx, d_qy, d_qz and resize d_wn ...
gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
    geometry.accessor(), topology.accessor(), moments.accessor(),
    d_qx.span(), d_qy.span(), d_qz.span(), d_wn.span(),
    2.0f /* accuracy_scale */);

// Point query example: compute signed distance in a kernel
// Pass accessors (not objects) to the GPU, then call __device__ functions
__global__ void compute_signed_distance_kernel(
    gwn::gwn_geometry_accessor<Real, Index> const geom,
    gwn::gwn_bvh_topology_accessor<4, Real, Index> const topo,
    gwn::gwn_bvh_aabb_accessor<4, Real, Index> const aabb,
    gwn::gwn_bvh_moment_tree_accessor<4, 1, Real, Index> const moments,
    Real const *qx, Real const *qy, Real const *qz,
    Real *out_dist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_dist[idx] = gwn::gwn_signed_distance_point_bvh<1, 4, Real, Index>(
            geom, topo, aabb, moments, qx[idx], qy[idx], qz[idx]);
    }
}

// Host-side launch:
// compute_signed_distance_kernel<<<(n + 255) / 256, 256>>>(
//     geometry.accessor(), topology.accessor(), aabb.accessor(), moments.accessor(),
//     d_qx.data(), d_qy.data(), d_qz.data(), d_out.data(), n);
```

### Query API overview

Query operations are available in two forms:
- **Device point APIs** (`__device__` functions callable from user kernels)
- **Batch APIs** (host-callable launchers that process arrays of queries)

Most query families provide both point and batch variants:

| Query Family | Point API | Batch API |
|--------------|-----------|-----------|
| Winding number (exact) | `gwn_winding_number_point_bvh_exact` | `gwn_compute_winding_number_batch_bvh_exact` |
| Winding number (Taylor) | `gwn_winding_number_point_bvh_taylor` | `gwn_compute_winding_number_batch_bvh_taylor` |
| Winding gradient (Taylor) | `gwn_winding_gradient_point_bvh_taylor` | `gwn_compute_winding_gradient_batch_bvh_taylor` |
| Unsigned distance | `gwn_unsigned_distance_point_bvh` | `gwn_compute_unsigned_distance_batch_bvh` |
| Signed distance | `gwn_signed_distance_point_bvh` | `gwn_compute_signed_distance_batch_bvh` |
| Boundary edge distance | `gwn_unsigned_boundary_edge_distance_point_bvh` | `gwn_compute_unsigned_boundary_edge_distance_batch_bvh` |
| Ray first-hit | `gwn_ray_first_hit_bvh` | `gwn_compute_ray_first_hit_batch_bvh` |
| Harnack trace | `gwn_harnack_trace_ray_bvh_taylor` | `gwn_compute_harnack_trace_batch_bvh_taylor` |
| Hybrid trace | `gwn_hybrid_trace_ray_bvh_taylor` | `gwn_compute_hybrid_trace_batch_bvh_taylor` |

Point APIs give you fine-grained control in custom kernels; batch APIs provide convenient host-side launchers.

#### Example: Point vs. Batch API

```cpp
// Batch API: host launches kernel for you
gwn::gwn_compute_winding_number_batch_bvh_exact<4>(
    geometry.accessor(), topology.accessor(),
    d_qx.span(), d_qy.span(), d_qz.span(), d_wn.span(), stream);

// Point API: you write the kernel
__global__ void my_winding_kernel(
    gwn::gwn_geometry_accessor<float> geom,
    gwn::gwn_bvh4_topology_accessor<float> topo,
    float const *qx, float const *qy, float const *qz,
    float *wn, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wn[idx] = gwn::gwn_winding_number_point_bvh_exact<4>(
            geom, topo, qx[idx], qy[idx], qz[idx]);
    }
}
// my_winding_kernel<<<...>>>(geometry.accessor(), topology.accessor(), ...);
```

### Loading from Eigen matrices

To load a mesh from an OBJ file, first parse it into Eigen matrices (or use libigl if available):

```cpp
#include <gwn/gwn_eigen_bridge.cuh>
#include <igl/read_triangle_mesh.h>  // libigl

Eigen::MatrixXf V;  // Nx3 vertices
Eigen::MatrixXi F;  // Mx3 triangles
igl::read_triangle_mesh("mesh.obj", V, F);

gwn::gwn_geometry_object<float> geometry;
gwn::gwn_upload_from_eigen(geometry, V, F, stream);
```

Alternatively, if libigl is not available, you can load OBJ files manually and populate Eigen matrices yourself.

## Testing

Build tests:

```bash
cmake -S . -B build
cmake --build build
```

The suite is organized into `unit`, `integration`, `fixtures`, `perf`, `models`, and `libigl`
slices. For clean-machine correctness checks, the most useful entry points are:

```bash
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L fixtures --output-on-failure
ctest --test-dir build -L perf --output-on-failure
```

The `integration` label is an umbrella that includes both the repo-local `fixtures` suite and the
dataset-driven `models` suites, so `ctest --test-dir build -L integration` may still require
`SMALLGWN_MODEL_DATA_DIR` for the model-backed entries.

The Taylor-matrix integration target still has dedicated light/heavy splits:

```bash
ctest --test-dir build -L light --output-on-failure
ctest --test-dir build -L heavy --output-on-failure
```

Optionally override model inputs for the dataset-driven `models` suites:

```bash
SMALLGWN_MODEL_DATA_DIR=/tmp/common-3d-test-models-subset \
ctest --test-dir build -L models --output-on-failure
```

The `fixtures` slice uses tiny OBJ meshes vendored under `tests/data/`, so it remains available
even when `SMALLGWN_MODEL_DATA_DIR` is unset.

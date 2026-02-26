# smallgwn

A header-only CUDA library implementing [*Fast Winding Numbers for Soups and Clouds*](https://www.dgp.toronto.edu/projects/fast-winding-numbers/) (Barill et al., 2018) for GPU-accelerated winding number and signed distance queries on triangle meshes.
Taylor-series multipole approximations (orders 0–2) with wide BVH acceleration (LBVH or H-PLOC).

## Requirements

- **CUDA Toolkit** (C++20)

Optional (for loading geometry with Eigen matrices):

- **Eigen3** and **Intel TBB** enables `gwn_eigen_bridge.hpp`.
- **libigl** can be used to load the geometry.

## Building

I recommend using it as a header-only library by including `gwn.cuh` directly in your project, and this project can be added as a submodule.
However, if you want to build the library separately, you can use CMake:

```bash
cmake -B build
cmake --build build
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
geometry.upload(
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

### Loading from Eigen matrices

To load a mesh from an OBJ file, first parse it into Eigen matrices (or use libigl if available):

```cpp
#include <gwn/gwn_eigen_bridge.hpp>
#include <igl/read_triangle_mesh.h>  // libigl

Eigen::MatrixXf V;  // Nx3 vertices
Eigen::MatrixXi F;  // Mx3 triangles
igl::read_triangle_mesh("mesh.obj", V, F);

gwn::gwn_geometry_object<float> geometry;
gwn::gwn_upload_from_eigen(geometry, V, F, stream);
```

Alternatively, if libigl is not available, you can load OBJ files manually and populate Eigen matrices yourself.

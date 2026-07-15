# smallgwn

`smallgwn` is a header-only CUDA/C++ library for GPU geometric queries on
triangle meshes.

## Features

- Exact, Taylor-approximate, and Antipodal winding numbers
- Winding-number gradients (Taylor and Antipodal)
- Unsigned point-to-mesh distance
- Ray first-hit queries with primitive ID, barycentrics, and oriented geometric normal

The core API is declared by:

```cpp
#include <gwn/gwn.cuh>
```

## Requirements

- CUDA Toolkit 12 or newer
- A host compiler with C++20 support
- CMake 3.24 or newer
- Eigen3 and Intel TBB when using the Eigen bridge
- libigl when using the mesh-loading code in the example

## Use from CMake

```cmake
include(FetchContent)
set(SMALLGWN_BUILD_EIGEN_BRIDGE ON CACHE BOOL "")
find_package(LIBIGL REQUIRED)
FetchContent_Declare(
    smallgwn
    GIT_REPOSITORY https://github.com/kririae/smallgwn.git
    GIT_TAG main
)
FetchContent_MakeAvailable(smallgwn)
target_link_libraries(my_cuda_target PRIVATE gwn::smallgwn_eigen_bridge)
target_link_libraries(my_cuda_target PRIVATE igl::core)
```

## Usage

This example builds a width-4 BVH with Taylor moments and runs a batch
winding-number query.

```cpp
#include <gwn/gwn.cuh>
#include <gwn/gwn_eigen_bridge.cuh>

#include <cuda/std/span>

#include <cstdint>
#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>
#include <stdexcept>
#include <vector>

using Real = float;
using Index = std::uint32_t;

auto check = [](gwn::gwn_status const &status) {
    if (!status.is_ok())
        throw std::runtime_error(status.message());
};

cudaStream_t const stream = gwn::gwn_default_stream();

Eigen::MatrixXd V;
Eigen::MatrixXi F;
if (!igl::read_triangle_mesh(/* filename */ "mesh.obj", V, F))
    throw std::runtime_error("Could not read mesh.");

gwn::gwn_geometry_object<Real, Index> geometry;
check(gwn::gwn_upload_geometry_from_eigen(geometry, V, F, stream));

gwn::gwn_bvh4_object<Real, Index> bvh;
gwn::gwn_bvh4_moment_object</* Taylor order */ 1, Real, Index> moments;

gwn::gwn_bvh_build_options const build_options{
    .method = gwn::gwn_bvh_build_method::k_hploc,
    // .method = gwn::gwn_bvh_build_method::k_lbvh,
    .hploc_search_radius = 8,
};
check(gwn::gwn_build_bvh(geometry, bvh, build_options, stream));
check(gwn::gwn_refit_bvh_moment</* Taylor order */ 1>(bvh, moments, stream));

/* Bind these spans to non-empty, application-owned device storage. */
gwn::gwn_device_span<Real const> d_qx{device_qx, query_count};
gwn::gwn_device_span<Real const> d_qy{device_qy, query_count};
gwn::gwn_device_span<Real const> d_qz{device_qz, query_count};
gwn::gwn_device_span<Real> d_wn{device_wn, query_count};

check(gwn::gwn_compute_winding_number_taylor_batch</* Taylor order */ 1>(
    bvh,
    moments,
    d_qx,
    d_qy,
    d_qz,
    d_wn,
    /* accuracy scale */ Real(2),
    stream));
```

For dynamic vertex positions, update geometry and refit the BVH and moments:

```cpp
/* x/y/z are application-owned host spans with geometry.vertex_count() elements. */
gwn::gwn_host_span<Real const> x = /* updated host x */;
gwn::gwn_host_span<Real const> y = /* updated host y */;
gwn::gwn_host_span<Real const> z = /* updated host z */;
check(gwn::gwn_update_geometry(geometry, x, y, z, stream));
// Host x/y/z must stay alive until this stream reaches the update.
check(gwn::gwn_refit_bvh(geometry, bvh, stream));
check(gwn::gwn_refit_bvh_moment</* Taylor order */ 1>(bvh, moments, stream));
```

Changing triangle indices or triangle count requires `gwn_build_bvh`. A build or refit invalidates
previously copied BVH accessors. A BVH refit also leaves existing moment objects stale until they
are refit.

### Antipodal winding

Antipodal winding also requires a boundary chain built from the same geometry:

```cpp
gwn::gwn_boundary_chain_object<Index> boundary_chain;
check(gwn::gwn_build_boundary_chain(geometry, boundary_chain, stream));

check(gwn::gwn_compute_winding_number_antipodal_batch(
    geometry, bvh, boundary_chain, d_qx, d_qy, d_qz, d_wn, stream));
```

`gwn_compute_winding_gradient_antipodal_batch` uses the same geometry and boundary chain and does
not require the BVH.

## References

- [The Antipodal Method: Fast, Accurate, and Robust 3D Generalized Winding Numbers](https://doi.org/10.1145/3811323)
- [Fast Winding Numbers for Soups and Clouds](https://doi.org/10.1145/3197517.3201337)
- [Fast BVH Construction on GPUs](https://research.nvidia.com/publication/2009-03_fast-bvh-construction-gpus)
- [H-PLOC](https://doi.org/10.1145/3675377)

## License

MIT. See [LICENSE](LICENSE).

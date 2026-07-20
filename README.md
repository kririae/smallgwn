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

Optional source dependencies are fetched at fixed versions. A parent project can provide the
corresponding standard CMake targets before adding smallgwn to prevent any download.

## Use from CMake

```cmake
include(FetchContent)
set(SMALLGWN_BUILD_EIGEN_BRIDGE ON)
set(SMALLGWN_BUILD_TESTS OFF)
set(SMALLGWN_BUILD_BENCHMARKS OFF)
FetchContent_Declare(
    smallgwn
    GIT_REPOSITORY https://github.com/kririae/smallgwn.git
    GIT_TAG main
)
FetchContent_MakeAvailable(smallgwn)
target_link_libraries(my_cuda_target PRIVATE gwn::smallgwn_eigen_bridge)
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
#include <stdexcept>
#include <vector>

using Real = float;
using Index = std::uint32_t;

auto check = [](gwn::gwn_status const &status) {
    if (!status.is_ok())
        throw std::runtime_error(status.message());
};

cudaStream_t const stream = gwn::gwn_default_stream();

// These functions stand for application-owned mesh loading.
Eigen::MatrixXd V = load_vertex_matrix();
Eigen::MatrixXi F = load_triangle_matrix();

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

// Bind these spans to non-empty, application-owned device storage.
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
// Updated host arrays contain geometry.vertex_count() elements.
gwn::gwn_host_span<Real const> x{updated_host_x, geometry.vertex_count()};
gwn::gwn_host_span<Real const> y{updated_host_y, geometry.vertex_count()};
gwn::gwn_host_span<Real const> z{updated_host_z, geometry.vertex_count()};
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

## Tests and benchmarks

Dataset runners consume directories of indexed PLY triangle meshes. Mesh I/O belongs to the test
and benchmark tooling and uses libigl; it is not part of the header-only library interface.

```bash
cmake -S . -B build -DSMALLGWN_BUILD_TESTS=ON -DSMALLGWN_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
build/tests/smallgwn_e2e --mesh-dir /path/to/ply-directory
build/tests/smallgwn_benchmark --model-dir /path/to/ply-directory --skip-exact
```

Benchmark CSV rows contain raw mesh and run facts, including vertex, triangle, boundary-edge, and
query counts. Backend selection and feature fitting belong to the calling application rather than
the smallgwn library. Use `--winding-query-only` to record only order-1 Taylor and complete
Antipodal query rows for an external comparison.

For heterogeneous datasets whose BVHs exceed the default traversal stack bound, select a supported
capacity explicitly, up to `--stack-capacity 96`. Host validation rejects an insufficient
capacity before launching a traversal query.

## References

- [The Antipodal Method: Fast, Accurate, and Robust 3D Generalized Winding Numbers](https://doi.org/10.1145/3811323)
- [Fast Winding Numbers for Soups and Clouds](https://doi.org/10.1145/3197517.3201337)
- [Fast BVH Construction on GPUs](https://research.nvidia.com/publication/2009-03_fast-bvh-construction-gpus)
- [H-PLOC](https://doi.org/10.1145/3675377)

## License

MIT. See [LICENSE](LICENSE).

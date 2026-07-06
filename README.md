# smallgwn

`smallgwn` is a header-only CUDA/C++ library for geometric queries on triangle
meshes. It works with structure-of-arrays mesh data on the GPU and provides
winding numbers, winding gradients, signed and unsigned distances, boundary-edge
distances, ray first-hit queries, Harnack tracing, and hybrid tracing.

The runtime API is under `include/gwn/`. Most users should include:

```cpp
#include <gwn/gwn.cuh>
```

`gwn_eigen_bridge.cuh` is separate. Include it only when you want to upload
meshes from Eigen matrices.

## Requirements

For the core library:

- CUDA Toolkit with C++20 support.
- CMake 3.24 or newer if you consume the CMake target.

For optional targets and tests:

- `gwn::smallgwn_eigen_bridge` requires Eigen3 and Intel TBB.
- Repository tests require GTest, Eigen3, Intel TBB, and libigl when
  `SMALLGWN_BUILD_TESTS=ON`.
- The benchmark executable is built when `SMALLGWN_BUILD_BENCHMARKS=ON`.

The root CMake project enables tests, benchmarks, and the Eigen bridge target by
default. If you only want the header-only library target, disable those targets
when configuring.

## Use it from CMake

With FetchContent:

```cmake
include(FetchContent)

set(SMALLGWN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(SMALLGWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    smallgwn
    GIT_REPOSITORY https://github.com/kririae/smallgwn.git
    GIT_TAG main
)
FetchContent_MakeAvailable(smallgwn)

target_link_libraries(my_cuda_target PRIVATE gwn::smallgwn)
```

`gwn::smallgwn` adds the include path, links CUDA Runtime, and passes the NVCC
`--expt-relaxed-constexpr` flag required by the public headers.

To configure this repository with only the core target:

```bash
cmake -S . -B build \
    -DSMALLGWN_BUILD_BENCHMARKS=OFF \
    -DSMALLGWN_BUILD_TESTS=OFF \
    -DSMALLGWN_BUILD_EIGEN_BRIDGE_TARGET=OFF
cmake --build build -j
```

The projects under `examples/` are standalone consumers. They fetch `smallgwn`
through CPM and are not part of the main library build.

## Minimal use

This sketch shows the usual flow: upload geometry, build a width-4 BVH with
Taylor moments, then run a batch winding-number query. It omits mesh loading and
result download.

```cpp
#include <gwn/gwn.cuh>

#include <cuda/std/span>

#include <cstdint>
#include <stdexcept>
#include <vector>

using Real = float;
using Index = std::uint32_t;

auto check = [](gwn::gwn_status const &status) {
    if (!status.is_ok()) {
        throw std::runtime_error(status.message());
    }
};

cudaStream_t stream{};
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

std::vector<Real> vx, vy, vz;
std::vector<Index> i0, i1, i2;

gwn::gwn_geometry_object<Real, Index> geometry;
check(gwn::gwn_upload_geometry(
    geometry,
    cuda::std::span<Real const>(vx.data(), vx.size()),
    cuda::std::span<Real const>(vy.data(), vy.size()),
    cuda::std::span<Real const>(vz.data(), vz.size()),
    cuda::std::span<Index const>(i0.data(), i0.size()),
    cuda::std::span<Index const>(i1.data(), i1.size()),
    cuda::std::span<Index const>(i2.data(), i2.size()),
    stream));

gwn::gwn_bvh4_topology_object<Real, Index> topology;
gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
gwn::gwn_bvh4_moment_object<1, Real, Index> moments;

check(gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
    geometry, topology, aabb, moments, stream));

std::vector<Real> qx, qy, qz;
gwn::gwn_device_array<Real> d_qx(stream), d_qy(stream), d_qz(stream), d_wn(stream);

check(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()), stream));
check(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()), stream));
check(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()), stream));
check(d_wn.resize(qx.size(), stream));

check(gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
    geometry.accessor(),
    topology.accessor(),
    moments.accessor(),
    d_qx.span(),
    d_qy.span(),
    d_qz.span(),
    d_wn.span(),
    Real(2),
    stream));
```

Use `gwn_bvh_facade_build_topology_aabb_moment_hploc` instead of the LBVH facade
when you want the H-PLOC topology builder.

## Query APIs

Each query family exposes device point APIs for custom kernels and batch APIs
that launch kernels from the host.

| Query family | Device point API | Batch API |
|--------------|------------------|-----------|
| Winding number, exact | `gwn_winding_number_point_bvh_exact` | `gwn_compute_winding_number_batch_bvh_exact` |
| Winding number, Taylor | `gwn_winding_number_point_bvh_taylor` | `gwn_compute_winding_number_batch_bvh_taylor` |
| Winding gradient, Taylor | `gwn_winding_gradient_point_bvh_taylor` | `gwn_compute_winding_gradient_batch_bvh_taylor` |
| Unsigned distance | `gwn_unsigned_distance_point_bvh` | `gwn_compute_unsigned_distance_batch_bvh` |
| Signed distance | `gwn_signed_distance_point_bvh` | `gwn_compute_signed_distance_batch_bvh` |
| Boundary-edge distance | `gwn_unsigned_boundary_edge_distance_point_bvh` | `gwn_compute_unsigned_boundary_edge_distance_batch_bvh` |
| Ray first-hit | `gwn_ray_first_hit_bvh` | `gwn_compute_ray_first_hit_batch_bvh` |
| Harnack trace | `gwn_harnack_trace_ray_bvh_taylor` | `gwn_compute_harnack_trace_batch_bvh_taylor` |
| Hybrid trace | `gwn_hybrid_trace_ray_bvh_taylor` | `gwn_compute_hybrid_trace_batch_bvh_taylor` |

Taylor winding, gradient, signed-distance, Harnack, and hybrid queries support
moment orders 0, 1, and 2. The public default index type is `std::uint32_t`; the
templates also support `std::uint64_t`.

## Eigen bridge

The Eigen bridge is useful when your mesh loader already produces Eigen
matrices. It is not included by `gwn/gwn.cuh`.

```cpp
#include <gwn/gwn_eigen_bridge.cuh>
#include <igl/read_triangle_mesh.h>

Eigen::MatrixXf V;
Eigen::MatrixXi F;
igl::read_triangle_mesh("mesh.obj", V, F);

gwn::gwn_geometry_object<float> geometry;
gwn::gwn_upload_from_eigen(geometry, V, F, stream);
```

If you build through CMake and want this helper target, keep
`SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET=ON` and provide Eigen3 and Intel TBB.

## Tests

Configure and build the repository tests with the required test dependencies
available:

```bash
cmake -S . -B build
cmake --build build -j
```

The most useful clean-machine checks are the unit tests and repo-local fixture
tests:

```bash
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L fixtures --output-on-failure
```

Some integration tests use external model data. Provide it with
`SMALLGWN_MODEL_DATA_DIR` before running the `models` label:

```bash
SMALLGWN_MODEL_DATA_DIR=/tmp/common-3d-test-models-subset \
ctest --test-dir build -L models --output-on-failure
```

The Taylor matrix integration test is split into light and heavy CTest entries:

```bash
ctest --test-dir build -L light --output-on-failure
ctest --test-dir build -L heavy --output-on-failure
```

## References

- [Fast Winding Numbers for Soups and Clouds](https://doi.org/10.1145/3197517.3201337)
- [Fast BVH Construction on GPUs](https://research.nvidia.com/publication/2009-03_fast-bvh-construction-gpus)
- [H-PLOC](https://doi.org/10.1145/3675377)
- [Harnack tracing](https://doi.org/10.1145/3658201)

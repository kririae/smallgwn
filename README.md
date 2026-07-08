# smallgwn

`smallgwn` is a header-only CUDA/C++ library for GPU geometric queries on
triangle meshes.

## Features

- Exact, Taylor-approximate, and Antipodal winding numbers
- Winding-number gradients (Taylor and Antipodal)
- Signed and unsigned point-to-mesh distances
- Boundary-edge distance
- Ray first-hit queries
- Harnack sphere-march ray tracing and hybrid first-hit tracing

The runtime API is under `include/gwn/`. Most users only need:

```cpp
#include <gwn/gwn.cuh>
```

`gwn_eigen_bridge.cuh` is a separate header for uploading meshes from Eigen
matrices.

## Requirements

- CUDA Toolkit with C++20 support
- CMake 3.24 or newer

## Use from CMake

```cmake
include(FetchContent)
FetchContent_Declare(
    smallgwn
    GIT_REPOSITORY https://github.com/kririae/smallgwn.git
    GIT_TAG main
)
FetchContent_MakeAvailable(smallgwn)
target_link_libraries(my_cuda_target PRIVATE gwn::smallgwn)
```

`gwn::smallgwn` adds the include path, links CUDA Runtime, and passes the
`--expt-relaxed-constexpr` flag required by the public headers.

The examples under `examples/` are standalone consumers that fetch `smallgwn`
through CPM.

## Usage

This example uploads geometry, builds a width-4 BVH with Taylor moments, and
runs a batch winding-number query. It skips mesh loading and result download.

```cpp
#include <gwn/gwn.cuh>

#include <cuda/std/span>

#include <cstdint>
#include <stdexcept>
#include <vector>

using Real = float;
using Index = std::uint32_t;

auto check = [](gwn::gwn_status const &status) {
    if (!status.is_ok())
        throw std::runtime_error(status.message());
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
gwn::gwn_device_array<Real> d_grad_x(stream), d_grad_y(stream), d_grad_z(stream);

check(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()), stream));
check(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()), stream));
check(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()), stream));
check(d_wn.resize(qx.size(), stream));
check(d_grad_x.resize(qx.size(), stream));
check(d_grad_y.resize(qx.size(), stream));
check(d_grad_z.resize(qx.size(), stream));

check(gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
    geometry.accessor(),
    topology.accessor(),
    moments.accessor(),
    d_qx.span(),
    d_qy.span(),
    d_qz.span(),
    d_wn.span(),
    Real(2), /* accuracy_scale */
    stream));
```

Use `gwn_bvh_facade_build_topology_aabb_moment_hploc` instead of the LBVH
facade for the H-PLOC topology builder.

### Antipodal winding

Antipodal winding follows Martens, Trettner, and Bessmeltsev's 2026 method.
Build the boundary-edge chain once, then pass it with BVH topology and AABB
data.

```cpp
gwn::gwn_boundary_chain_object<Index> boundary_edges;
check(gwn::gwn_build_boundary_chain(geometry.accessor(), boundary_edges, stream));

check(gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
    geometry.accessor(), topology.accessor(), aabb.accessor(),
    boundary_edges.accessor(), d_qx.span(), d_qy.span(), d_qz.span(),
    d_wn.span(), stream));

check(gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
    geometry.accessor(), boundary_edges.accessor(),
    d_qx.span(), d_qy.span(), d_qz.span(),
    d_grad_x.span(), d_grad_y.span(), d_grad_z.span(), stream));
```

### Eigen bridge

```cpp
#include <gwn/gwn_eigen_bridge.cuh>

Eigen::MatrixXf V;
Eigen::MatrixXi F;
igl::read_triangle_mesh("mesh.obj", V, F);

gwn::gwn_geometry_object<float> geometry;
gwn::gwn_upload_geometry_from_eigen(geometry, V, F, stream);
```

## Tests

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L fixtures --output-on-failure
```

## References

- [The Antipodal Method: Fast, Accurate, and Robust 3D Generalized Winding Numbers](https://doi.org/10.1145/3811323)
- [Fast Winding Numbers for Soups and Clouds](https://doi.org/10.1145/3197517.3201337)
- [Fast BVH Construction on GPUs](https://research.nvidia.com/publication/2009-03_fast-bvh-construction-gpus)
- [H-PLOC](https://doi.org/10.1145/3675377)
- [Harnack tracing](https://doi.org/10.1145/3658201)

## License

MIT. See [LICENSE](LICENSE).

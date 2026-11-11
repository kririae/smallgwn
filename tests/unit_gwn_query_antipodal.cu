#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_meshes.hpp"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CubeMesh;
using gwn::tests::OpenCubeMesh;

namespace {

class GwnQueryAntipodalTest : public gwn::tests::CudaFixture {};

template <class Mesh> struct AntipodalContext {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_object<Real, Index> bvh;
    gwn::gwn_boundary_chain_object<Index> boundary;
    bool ready{false};
};

template <class Mesh>
void setup_context(
    Mesh const &mesh, AntipodalContext<Mesh> &ctx,
    gwn::gwn_bvh_build_method const method = gwn::gwn_bvh_build_method::k_lbvh
) {
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        ctx.geometry, gwn::gwn_host_span<Real const>(mesh.vx.data(), mesh.vx.size()),
        gwn::gwn_host_span<Real const>(mesh.vy.data(), mesh.vy.size()),
        gwn::gwn_host_span<Real const>(mesh.vz.data(), mesh.vz.size()),
        gwn::gwn_host_span<Index const>(mesh.i0.data(), mesh.i0.size()),
        gwn::gwn_host_span<Index const>(mesh.i1.data(), mesh.i1.size()),
        gwn::gwn_host_span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status =
        gwn::gwn_build_bvh(ctx.geometry, ctx.bvh, gwn::gwn_bvh_build_options{.method = method});
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    status = gwn::gwn_build_boundary_chain(ctx.geometry, ctx.boundary);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ctx.ready = true;
}

std::vector<Real> run_exact(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_out;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_exact_batch(
        bvh, gwn::tests::device_input_span(d_qx.span()), gwn::tests::device_input_span(d_qy.span()),
        gwn::tests::device_input_span(d_qz.span()), gwn::tests::device_span(d_out.span())
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size(), Real(0));
    d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

std::vector<Real> run_antipodal(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_boundary_chain_object<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_out;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());

    gwn::gwn_status const status = gwn::gwn_compute_winding_number_antipodal_batch(
        geometry, bvh, boundary, gwn::tests::device_input_span(d_qx.span()),
        gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
        gwn::tests::device_span(d_out.span())
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size(), Real(0));
    d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

std::array<std::vector<Real>, 3> run_antipodal_gradient(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_object<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_gx;
    gwn::detail::gwn_device_array<Real> d_gy;
    gwn::detail::gwn_device_array<Real> d_gz;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_gx.resize(qx.size());
    d_gy.resize(qx.size());
    d_gz.resize(qx.size());

    gwn::gwn_status const status = gwn::gwn_compute_winding_gradient_antipodal_batch(
        geometry, boundary, gwn::tests::device_input_span(d_qx.span()),
        gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
        gwn::tests::device_span(d_gx.span()), gwn::tests::device_span(d_gy.span()),
        gwn::tests::device_span(d_gz.span())
    );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<std::vector<Real>, 3> out{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    d_gx.copy_to_host(cuda::std::span<Real>(out[0].data(), out[0].size()));
    d_gy.copy_to_host(cuda::std::span<Real>(out[1].data(), out[1].size()));
    d_gz.copy_to_host(cuda::std::span<Real>(out[2].data(), out[2].size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

struct antipodal_gradient_point_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_boundary_chain_accessor<Index> boundary{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_x{};
    cuda::std::span<Real> out_y{};
    cuda::std::span<Real> out_z{};

    __device__ void operator()(std::size_t const query_id) const {
        gwn::gwn_vec3<Real> const gradient = gwn::gwn_winding_gradient_antipodal(
            geometry, boundary, query_x[query_id], query_y[query_id], query_z[query_id]
        );
        out_x[query_id] = gradient.x;
        out_y[query_id] = gradient.y;
        out_z[query_id] = gradient.z;
    }
};

// Counts callback invocations so tests can require exactly one per overflowing query.
struct antipodal_overflow_probe {
    int *count{};

    void __device__ operator()() const noexcept {
        if (count != nullptr)
            atomicAdd(count, 1);
    }
};

template <int StackCapacity> struct antipodal_point_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_bvh4_accessor<Real, Index> bvh{};
    gwn::gwn_boundary_chain_accessor<Index> boundary{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out{};
    cuda::std::span<int> overflow_count{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        out[query_id] = gwn::gwn_winding_number_antipodal<
            4, Real, Index, StackCapacity, antipodal_overflow_probe>(
            geometry, bvh, boundary, query_x[query_id], query_y[query_id], query_z[query_id],
            antipodal_overflow_probe{overflow_count.data() + query_id}
        );
    }
};

template <int StackCapacity = 64>
[[nodiscard]] std::pair<std::vector<Real>, std::vector<int>> run_antipodal_point_kernel(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_boundary_chain_object<Index> const &boundary, std::vector<Real> const &qx,
    std::vector<Real> const &qy, std::vector<Real> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx, d_qy, d_qz, d_out;
    gwn::detail::gwn_device_array<int> d_overflow;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());
    d_overflow.resize(qx.size());
    d_overflow.zero();
    EXPECT_TRUE((gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
                     qx.size(),
                     antipodal_point_functor<StackCapacity>{
                         geometry.accessor(), bvh.accessor(), boundary.accessor(), d_qx.span(),
                         d_qy.span(), d_qz.span(), d_out.span(), d_overflow.span()
                     }
                 ))
                    .is_ok());
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<Real> out(qx.size());
    std::vector<int> overflow(qx.size());
    d_out.copy_to_host(cuda::std::span<Real>(out.data(), out.size()));
    d_overflow.copy_to_host(cuda::std::span<int>(overflow.data(), overflow.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return {out, overflow};
}

std::array<std::vector<Real>, 3> run_antipodal_gradient_point_kernel(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_accessor<Index> const &boundary, std::array<Real, 4> const &qx,
    std::array<Real, 4> const &qy, std::array<Real, 4> const &qz
) {
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_gx;
    gwn::detail::gwn_device_array<Real> d_gy;
    gwn::detail::gwn_device_array<Real> d_gz;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_gx.resize(qx.size());
    d_gy.resize(qx.size());
    d_gz.resize(qx.size());

    gwn::gwn_status const status =
        gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
            qx.size(), antipodal_gradient_point_functor{
                           geometry, boundary, d_qx.span(), d_qy.span(), d_qz.span(), d_gx.span(),
                           d_gy.span(), d_gz.span()
                       }
        );
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::array<std::vector<Real>, 3> out{
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
        std::vector<Real>(qx.size(), Real(0)),
    };
    d_gx.copy_to_host(cuda::std::span<Real>(out[0].data(), out[0].size()));
    d_gy.copy_to_host(cuda::std::span<Real>(out[1].data(), out[1].size()));
    d_gz.copy_to_host(cuda::std::span<Real>(out[2].data(), out[2].size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return out;
}

void expect_near_vector(std::vector<Real> const &actual, std::vector<Real> const &expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], Real(3e-4)) << "query " << i;
}

void expect_near_gradient(
    std::array<std::vector<Real>, 3> const &actual,
    std::array<std::vector<Real>, 3> const &expected, Real const tolerance
) {
    for (int axis = 0; axis < 3; ++axis) {
        ASSERT_EQ(actual[axis].size(), expected[axis].size());
        for (std::size_t i = 0; i < actual[axis].size(); ++i)
            EXPECT_NEAR(actual[axis][i], expected[axis][i], tolerance)
                << "axis " << axis << " query " << i;
    }
}

void expect_finite_gradient(std::array<std::vector<Real>, 3> const &actual) {
    for (int axis = 0; axis < 3; ++axis)
        for (std::size_t i = 0; i < actual[axis].size(); ++i)
            EXPECT_TRUE(std::isfinite(actual[axis][i])) << "axis " << axis << " query " << i;
}

template <class R> struct MeshData {
    std::vector<R> vx, vy, vz;
    std::vector<Index> i0, i1, i2;
};

template <class Mesh> [[nodiscard]] MeshData<Real> to_mesh_data(Mesh const &mesh) {
    return {
        {mesh.vx.begin(), mesh.vx.end()}, {mesh.vy.begin(), mesh.vy.end()},
        {mesh.vz.begin(), mesh.vz.end()}, {mesh.i0.begin(), mesh.i0.end()},
        {mesh.i1.begin(), mesh.i1.end()}, {mesh.i2.begin(), mesh.i2.end()},
    };
}

// Appends one triangle with its own three vertices.
template <class R>
void add_triangle(
    MeshData<R> &mesh, std::array<R, 3> const &a, std::array<R, 3> const &b,
    std::array<R, 3> const &c
) {
    auto const first = static_cast<Index>(mesh.vx.size());
    for (auto const *vertex : {&a, &b, &c}) {
        mesh.vx.push_back((*vertex)[0]);
        mesh.vy.push_back((*vertex)[1]);
        mesh.vz.push_back((*vertex)[2]);
    }
    mesh.i0.push_back(first);
    mesh.i1.push_back(first + 1);
    mesh.i2.push_back(first + 2);
}

// Tilted, off-center, radially jittered torus. Faces with u-index below `open_cut` are removed,
// which leaves two boundary loops. Coordinates straddle every coordinate plane and no edge is axis
// aligned, so query-relative differences exercise every rounding regime.
[[nodiscard]] MeshData<Real>
make_jittered_torus(int const nu, int const nv, int const open_cut, double const scale = 1.0) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> jitter(-1.0, 1.0);
    MeshData<Real> mesh;
    double const ax = 0.37;
    double const ay = -0.61;
    double const pi = 3.141592653589793238462643383279502884;
    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nv; ++j) {
            double const u = 2.0 * pi * i / nu;
            double const v = 2.0 * pi * j / nv;
            double const r = 0.25 * (1.0 + 0.02 * jitter(rng));
            double const px = (0.6 + r * std::cos(v)) * std::cos(u);
            double const py = (0.6 + r * std::cos(v)) * std::sin(u);
            double const pz = r * std::sin(v);
            double const y1 = py * std::cos(ax) - pz * std::sin(ax);
            double const z1 = py * std::sin(ax) + pz * std::cos(ax);
            double const x2 = px * std::cos(ay) + z1 * std::sin(ay);
            double const z2 = -px * std::sin(ay) + z1 * std::cos(ay);
            mesh.vx.push_back(static_cast<Real>((x2 + 0.031) * scale));
            mesh.vy.push_back(static_cast<Real>((y1 - 0.017) * scale));
            mesh.vz.push_back(static_cast<Real>((z2 + 0.011) * scale));
        }
    }
    auto const vertex = [&](int const i, int const j) {
        return static_cast<Index>(((i + nu) % nu) * nv + (j + nv) % nv);
    };
    for (int i = open_cut; i < nu; ++i) {
        for (int j = 0; j < nv; ++j) {
            Index const a = vertex(i, j);
            Index const b = vertex(i + 1, j);
            Index const c = vertex(i + 1, j + 1);
            Index const d = vertex(i, j + 1);
            mesh.i0.insert(mesh.i0.end(), {a, a});
            mesh.i1.insert(mesh.i1.end(), {b, c});
            mesh.i2.insert(mesh.i2.end(), {c, d});
        }
    }
    return mesh;
}

template <class R> struct QuerySet {
    std::vector<R> x, y, z;
};

// Queries whose xy projection lies on a random mesh edge, or within a few ulps of a mesh vertex,
// at heights below, inside, and above the mesh. These +z rays pass through shared edges, shared
// vertices, and boundary vertices, where inconsistent rounded predicates count a triangle twice
// or not at all and the boundary term takes a collinear limit.
[[nodiscard]] QuerySet<Real> make_edge_and_vertex_queries(
    MeshData<Real> const &mesh, std::size_t const count, unsigned const seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> triangle(0, mesh.i0.size() - 1);
    std::uniform_int_distribution<int> corner(0, 2);
    std::uniform_real_distribution<double> lambda(0.02, 0.98);
    std::uniform_real_distribution<double> height(-0.3, 1.3);
    std::uniform_int_distribution<int> ulps(-2, 2);
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    for (Real const value : mesh.vz) {
        lo = std::min<double>(lo, value);
        hi = std::max<double>(hi, value);
    }

    QuerySet<Real> queries;
    queries.x.resize(count);
    queries.y.resize(count);
    queries.z.resize(count);
    for (std::size_t query_id = 0; query_id < count; ++query_id) {
        std::size_t const t = triangle(rng);
        std::array<Index, 3> const vertices{mesh.i0[t], mesh.i1[t], mesh.i2[t]};
        int const k = corner(rng);
        Index const a = vertices[k];
        Index const b = vertices[(k + 1) % 3];
        bool const vertex_mode = (query_id % 4) == 3;
        double const l = lambda(rng);
        auto const place = [&](std::vector<Real> const &coordinate) {
            Real value = vertex_mode
                             ? coordinate[a]
                             : static_cast<Real>(
                                   coordinate[a] + l * (double(coordinate[b]) - coordinate[a])
                               );
            if (vertex_mode)
                for (int u = ulps(rng); u != 0; u += (u > 0 ? -1 : 1))
                    value = std::nextafter(value, u > 0 ? Real(1e30) : Real(-1e30));
            return value;
        };
        queries.x[query_id] = place(mesh.vx);
        queries.y[query_id] = place(mesh.vy);
        queries.z[query_id] = static_cast<Real>(lo + (hi - lo) * height(rng));
    }
    return queries;
}

template <class R> struct AntipodalScene {
    gwn::gwn_geometry_object<R, Index> geometry;
    gwn::gwn_bvh4_object<R, Index> bvh;
    gwn::gwn_boundary_chain_object<Index> boundary;
};

template <class R> void build_scene(MeshData<R> const &mesh, AntipodalScene<R> &scene) {
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        scene.geometry, gwn::gwn_host_span<R const>(mesh.vx.data(), mesh.vx.size()),
        gwn::gwn_host_span<R const>(mesh.vy.data(), mesh.vy.size()),
        gwn::gwn_host_span<R const>(mesh.vz.data(), mesh.vz.size()),
        gwn::gwn_host_span<Index const>(mesh.i0.data(), mesh.i0.size()),
        gwn::gwn_host_span<Index const>(mesh.i1.data(), mesh.i1.size()),
        gwn::gwn_host_span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_build_bvh(scene.geometry, scene.bvh);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    status = gwn::gwn_build_boundary_chain(scene.geometry, scene.boundary);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
}

enum class scene_query { antipodal, exact, distance };

template <class R>
[[nodiscard]] std::vector<R> run_scene_query(
    AntipodalScene<R> const &scene, QuerySet<R> const &queries, scene_query const kind
) {
    gwn::detail::gwn_device_array<R> d_qx, d_qy, d_qz, d_out;
    d_qx.copy_from_host(cuda::std::span<R const>(queries.x.data(), queries.x.size()));
    d_qy.copy_from_host(cuda::std::span<R const>(queries.y.data(), queries.y.size()));
    d_qz.copy_from_host(cuda::std::span<R const>(queries.z.data(), queries.z.size()));
    d_out.resize(queries.x.size());
    auto const qx = gwn::tests::device_input_span(d_qx.span());
    auto const qy = gwn::tests::device_input_span(d_qy.span());
    auto const qz = gwn::tests::device_input_span(d_qz.span());
    auto const out = gwn::tests::device_span(d_out.span());
    gwn::gwn_status status = gwn::gwn_status::ok();
    if (kind == scene_query::antipodal) {
        status = gwn::gwn_compute_winding_number_antipodal_batch(
            scene.geometry, scene.bvh, scene.boundary, qx, qy, qz, out
        );
    } else if (kind == scene_query::exact) {
        status = gwn::gwn_compute_winding_number_exact_batch(scene.bvh, qx, qy, qz, out);
    } else {
        status = gwn::gwn_compute_unsigned_distance_batch(scene.bvh, qx, qy, qz, out);
    }
    EXPECT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    std::vector<R> result(queries.x.size());
    d_out.copy_to_host(cuda::std::span<R>(result.data(), result.size()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return result;
}

// Long-double winding number (Van Oosterom-Strackee per triangle), an independent reference.
template <class R>
[[nodiscard]] long double reference_winding(
    MeshData<R> const &mesh, long double const qx, long double const qy, long double const qz
) {
    long double sum = 0.0L;
    for (std::size_t t = 0; t < mesh.i0.size(); ++t) {
        std::array<Index, 3> const ids{mesh.i0[t], mesh.i1[t], mesh.i2[t]};
        std::array<std::array<long double, 3>, 3> v{};
        std::array<long double, 3> length{};
        for (int k = 0; k < 3; ++k) {
            v[k] = {
                static_cast<long double>(mesh.vx[ids[k]]) - qx,
                static_cast<long double>(mesh.vy[ids[k]]) - qy,
                static_cast<long double>(mesh.vz[ids[k]]) - qz,
            };
            length[k] = std::sqrt(v[k][0] * v[k][0] + v[k][1] * v[k][1] + v[k][2] * v[k][2]);
        }
        auto const dot = [](auto const &a, auto const &b) {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        };
        long double const triple = v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) -
                                   v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0]) +
                                   v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
        long double const denominator = length[0] * length[1] * length[2] +
                                        dot(v[0], v[1]) * length[2] + dot(v[1], v[2]) * length[0] +
                                        dot(v[2], v[0]) * length[1];
        sum += 2.0L * std::atan2(triple, denominator);
    }
    return sum / (4.0L * std::acos(-1.0L));
}

// Central differences of the long-double winding number, with a step relative to `scale`.
template <class R>
[[nodiscard]] std::array<long double, 3> reference_gradient(
    MeshData<R> const &mesh, std::array<long double, 3> const &q, long double const step
) {
    std::array<long double, 3> gradient{};
    for (int axis = 0; axis < 3; ++axis) {
        std::array<long double, 3> plus = q;
        std::array<long double, 3> minus = q;
        plus[axis] += step;
        minus[axis] -= step;
        gradient[axis] = (reference_winding(mesh, plus[0], plus[1], plus[2]) -
                          reference_winding(mesh, minus[0], minus[1], minus[2])) /
                         (2.0L * step);
    }
    return gradient;
}

template <class R, bool ExactFallback> struct projected_edge_sign_functor {
    cuda::std::span<R const> values{};
    cuda::std::span<int> forward_sign{};
    cuda::std::span<int> reverse_sign{};
    cuda::std::span<int> resolved{};

    __device__ void operator()(std::size_t const pair_id) const {
        using gwn::detail::gwn_query_vec2;
        gwn_query_vec2<R> const a{values[4 * pair_id + 0], values[4 * pair_id + 1]};
        gwn_query_vec2<R> const b{values[4 * pair_id + 2], values[4 * pair_id + 3]};
        auto const forward =
            gwn::detail::gwn_antipodal_projected_edge_classify_impl<ExactFallback>(a, b);
        auto const reverse =
            gwn::detail::gwn_antipodal_projected_edge_classify_impl<ExactFallback>(b, a);
        forward_sign[pair_id] = forward.sign;
        reverse_sign[pair_id] = reverse.sign;
        resolved[pair_id] = forward.status == gwn::detail::gwn_antipodal_status::k_done &&
                            reverse.status == gwn::detail::gwn_antipodal_status::k_done;
    }
};

struct EdgeSigns {
    std::vector<int> forward, reverse, resolved;
};

template <class R, bool ExactFallback>
[[nodiscard]] EdgeSigns run_projected_edge_signs(std::vector<R> const &values) {
    std::size_t const count = values.size() / 4;
    gwn::detail::gwn_device_array<R> d_values;
    gwn::detail::gwn_device_array<int> d_forward, d_reverse, d_resolved;
    d_values.copy_from_host(cuda::std::span<R const>(values.data(), values.size()));
    d_forward.resize(count);
    d_reverse.resize(count);
    d_resolved.resize(count);
    EXPECT_TRUE((gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
                     count,
                     projected_edge_sign_functor<R, ExactFallback>{
                         d_values.span(), d_forward.span(), d_reverse.span(), d_resolved.span()
                     }
                 ))
                    .is_ok());
    EdgeSigns signs{std::vector<int>(count), std::vector<int>(count), std::vector<int>(count)};
    d_forward.copy_to_host(cuda::std::span<int>(signs.forward.data(), count));
    d_reverse.copy_to_host(cuda::std::span<int>(signs.reverse.data(), count));
    d_resolved.copy_to_host(cuda::std::span<int>(signs.resolved.data(), count));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    return signs;
}

} // namespace

TEST_F(GwnQueryAntipodalTest, antipodal_closed_mesh_matches_existing_winding) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    expect_near_vector(
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz),
        run_exact(ctx.bvh, qx, qy, qz)
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_open_mesh_matches_existing_winding) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.23), Real(2.2), Real(-0.41), Real(-2.3)};
    std::array<Real, 4> const qy{Real(0.17), Real(0.3), Real(0.26), Real(-0.4)};
    std::array<Real, 4> const qz{Real(0.11), Real(0.2), Real(1.7), Real(0.5)};

    expect_near_vector(
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz),
        run_exact(ctx.bvh, qx, qy, qz)
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_open_mesh_matches_on_collinear_boundary_projections) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(2), Real(2), Real(-2), Real(-2)};
    std::array<Real, 4> const qy{Real(-1), Real(1), Real(-1), Real(1)};
    std::array<Real, 4> const qz{Real(2), Real(2), Real(2), Real(2)};

    expect_near_vector(
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz),
        run_exact(ctx.bvh, qx, qy, qz)
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_matches_long_double_reference) {
    // Generic open-cube points, then a sweep toward the vertical line through a boundary vertex,
    // where the per-edge auxiliary-triangle derivatives carry opposite 1/d poles, then tiny
    // meshes whose squared lengths underflow in float.
    struct gradient_case {
        MeshData<Real> mesh;
        std::array<Real, 3> query;
        long double step;
    };
    std::vector<gradient_case> cases;
    MeshData<Real> const open_cube = to_mesh_data(OpenCubeMesh{});
    for (auto const &query :
         {std::array<Real, 3>{0.21f, 0.18f, 0.15f}, std::array<Real, 3>{1.3f, 0.25f, 0.2f},
          std::array<Real, 3>{-1.4f, -0.33f, 0.42f}, std::array<Real, 3>{0.35f, 1.4f, 1.55f}}) {
        cases.push_back({open_cube, query, 1e-6L});
    }
    MeshData<Real> unit_triangle;
    add_triangle<Real>(unit_triangle, {0, 0, 1}, {1, 0, 1}, {0, 1, 1});
    for (int k = 4; k <= 26; k += 2) {
        Real const offset = std::ldexp(Real(1), -k);
        cases.push_back({unit_triangle, {offset, offset, Real(0)}, 1e-3L * offset});
    }
    for (int exponent : {-40, -110}) {
        Real const s = std::ldexp(Real(1), exponent);
        MeshData<Real> tiny;
        add_triangle<Real>(tiny, {0, 0, s}, {s, 0, s}, {0, s, s});
        cases.push_back({tiny, {s / 4, s / 4, Real(0)}, 1e-6L * s});
    }

    for (std::size_t case_id = 0; case_id < cases.size(); ++case_id) {
        auto const &test_case = cases[case_id];
        AntipodalScene<Real> scene;
        build_scene(test_case.mesh, scene);
        std::array<Real, 4> const qx{test_case.query[0], 0, 0, 0};
        std::array<Real, 4> const qy{test_case.query[1], 0, 0, 0};
        std::array<Real, 4> const qz{test_case.query[2], 0, 0, 0};
        auto const gradient = run_antipodal_gradient(scene.geometry, scene.boundary, qx, qy, qz);
        auto const expected = reference_gradient(
            test_case.mesh, {test_case.query[0], test_case.query[1], test_case.query[2]},
            test_case.step
        );
        long double const magnitude = std::sqrt(
            expected[0] * expected[0] + expected[1] * expected[1] + expected[2] * expected[2]
        );
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(gradient[axis][0], expected[axis], 1e-5L * magnitude)
                << "case " << case_id << " axis " << axis;
        }
    }
}

namespace {

// Long-double Biot-Savart gradient of the winding number, with the cancellation-free denominator
// for nearly antiparallel endpoints. Interior edges cancel between their two triangles.
template <class R>
[[nodiscard]] std::array<long double, 3>
reference_field(MeshData<R> const &mesh, std::array<long double, 3> const &q) {
    auto const edge_field = [](std::array<long double, 3> const &a,
                               std::array<long double, 3> const &b) {
        std::array<long double, 3> const cross{
            a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
        };
        long double const la = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        long double const lb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        long double const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        long double const cross2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        long double const denominator = dot >= 0 ? la * lb + dot : cross2 / (la * lb - dot);
        long double const factor = (la + lb) / (la * lb * denominator);
        return std::array<long double, 3>{cross[0] * factor, cross[1] * factor, cross[2] * factor};
    };
    std::array<long double, 3> field{};
    for (std::size_t t = 0; t < mesh.i0.size(); ++t) {
        std::array<Index, 3> const ids{mesh.i0[t], mesh.i1[t], mesh.i2[t]};
        std::array<std::array<long double, 3>, 3> v{};
        for (int k = 0; k < 3; ++k) {
            v[k] = {
                static_cast<long double>(mesh.vx[ids[k]]) - q[0],
                static_cast<long double>(mesh.vy[ids[k]]) - q[1],
                static_cast<long double>(mesh.vz[ids[k]]) - q[2],
            };
        }
        for (int k = 0; k < 3; ++k) {
            auto const term = edge_field(v[k], v[(k + 1) % 3]);
            for (int axis = 0; axis < 3; ++axis)
                field[axis] += term[axis] / (4.0L * std::acos(-1.0L));
        }
    }
    return field;
}

[[nodiscard]] std::array<Real, 3>
antipodal_gradient_at(MeshData<Real> const &mesh, std::array<Real, 3> const &query) {
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    std::array<Real, 4> const qx{query[0], 0, 0, 0};
    std::array<Real, 4> const qy{query[1], 0, 0, 0};
    std::array<Real, 4> const qz{query[2], 0, 0, 0};
    auto const gradient = run_antipodal_gradient(scene.geometry, scene.boundary, qx, qy, qz);
    return {gradient[0][0], gradient[1][0], gradient[2][0]};
}

} // namespace

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_matches_analytic_field_at_extreme_scales) {
    struct field_case {
        char const *name;
        MeshData<Real> mesh;
        std::array<Real, 3> query{};
    };
    std::vector<field_case> cases;
    auto const single = [](std::array<Real, 3> const &a, std::array<Real, 3> const &b,
                           std::array<Real, 3> const &c) {
        MeshData<Real> mesh;
        add_triangle<Real>(mesh, a, b, c);
        return mesh;
    };
    // Unit-length endpoints passing the query at 2^-80: the squared cross product underflows.
    Real const near = std::ldexp(Real(1), -80);
    cases.push_back({"unit antiparallel", single({1, near, 0}, {-1, near, 0}, {0, 2, 0})});
    // Endpoint norms of 2^-49 pass the range check, but the cross product 2^-158 underflows.
    Real const small = std::ldexp(Real(1), -49);
    Real const offset = std::ldexp(Real(1), -110);
    cases.push_back(
        {"small antiparallel", single({small, offset, 0}, {-small, offset, 0}, {0, small, 0})}
    );
    // An endpoint 2^-140 from the query: its inverse length overflows, the field does not.
    Real const tiny = std::ldexp(Real(1), -140);
    cases.push_back(
        {"overflowing inverse length", single({tiny, 0, 0}, {1, tiny, 0}, {1, 0, tiny})}
    );
    // Near-collinear endpoints whose plain FMA cross product is not antisymmetric, and the same
    // triangle reversed.
    Real const h = std::ldexp(Real(1), -23);
    cases.push_back({"fma cross", single({1 + h, 1, 0}, {-1, h - 1, 0}, {0, 1, 1})});
    cases.push_back({"fma cross reversed", single({-1, h - 1, 0}, {1 + h, 1, 0}, {0, 1, 1})});
    // A query about 4e-6 from a boundary segment of length 2: a plain FMA cross product loses
    // three digits of the field, which a compensated one keeps.
    cases.push_back(
        {"near segment",
         single({0.8244873f, 0.9504502f, 0}, {-0.6331432f, -0.7298631f, 0}, {0.1f, -0.3f, 1})}
    );
    // Endpoints at 2^126 with the edge 1/8 from the query: after rescaling, the near-antiparallel
    // angular factor alone exceeds the float range although the field is about 16.
    Real const huge = std::ldexp(Real(1), 126);
    cases.push_back(
        {"huge antiparallel", single({huge, 0.125f, 0}, {-huge, 0.125f, 0}, {0, huge, 0})}
    );
    // A query within 2^-69 of a vertex: that endpoint's squared length is subnormal while the
    // other endpoints have unit length.
    cases.push_back(
        {"subnormal endpoint length",
         single({0, 0, 0}, {1, 0, 0.5f}, {0, 1, 0.3f}),
         {std::ldexp(1.37f, -70), std::ldexp(2.11f, -70), std::ldexp(-0.73f, -70)}}
    );

    for (auto const &test_case : cases) {
        std::array<Real, 3> const gradient = antipodal_gradient_at(test_case.mesh, test_case.query);
        std::array<long double, 3> const expected = reference_field(
            test_case.mesh, {test_case.query[0], test_case.query[1], test_case.query[2]}
        );
        long double const magnitude = std::sqrt(
            expected[0] * expected[0] + expected[1] * expected[1] + expected[2] * expected[2]
        );
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(gradient[axis], expected[axis], 1e-5L * magnitude)
                << test_case.name << " axis " << axis;
        }
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_is_nan_on_a_boundary_edge) {
    // The gradient is unbounded on the boundary curve, so a query on a boundary edge or at a
    // boundary vertex receives NaN rather than the finite sum of the remaining edges.
    MeshData<Real> mesh;
    add_triangle<Real>(mesh, {-1, 0, 0}, {1, 0, 0}, {0, 1, 0});
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    std::array<Real, 4> const qx{0, 1, Real(0.5), Real(0.25)};
    std::array<Real, 4> const qy{0, 0, Real(0.5), Real(0.25)};
    std::array<Real, 4> const qz{0, 0, 0, Real(0.5)};
    auto const gradient = run_antipodal_gradient(scene.geometry, scene.boundary, qx, qy, qz);
    for (int query_id = 0; query_id < 3; ++query_id)
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_TRUE(std::isnan(gradient[axis][query_id])) << "query " << query_id;
    for (int axis = 0; axis < 3; ++axis)
        EXPECT_TRUE(std::isfinite(gradient[axis][3])) << "axis " << axis;
}

TEST_F(
    GwnQueryAntipodalTest, antipodal_gradient_is_mirror_symmetric_on_collinear_boundary_projections
) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(1), Real(1), Real(1), Real(1)};
    std::array<Real, 4> const qy{Real(0.25), Real(-0.25), Real(0.5), Real(-0.5)};
    std::array<Real, 4> const qz{Real(0.2), Real(0.2), Real(0.4), Real(0.4)};

    std::array<std::vector<Real>, 3> const gradient =
        run_antipodal_gradient(ctx.geometry, ctx.boundary, qx, qy, qz);
    expect_finite_gradient(gradient);

    EXPECT_NEAR(gradient[0][0], gradient[0][1], Real(1e-6));
    EXPECT_NEAR(gradient[1][0], -gradient[1][1], Real(1e-6));
    EXPECT_NEAR(gradient[2][0], gradient[2][1], Real(1e-6));

    EXPECT_NEAR(gradient[0][2], gradient[0][3], Real(1e-6));
    EXPECT_NEAR(gradient[1][2], -gradient[1][3], Real(1e-6));
    EXPECT_NEAR(gradient[2][2], gradient[2][3], Real(1e-6));
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_point_matches_batch) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 4> const qx{Real(0.21), Real(1.3), Real(-1.4), Real(0.35)};
    std::array<Real, 4> const qy{Real(0.18), Real(0.25), Real(-0.33), Real(1.4)};
    std::array<Real, 4> const qz{Real(0.15), Real(0.2), Real(0.42), Real(1.55)};

    std::array<std::vector<Real>, 3> const point = run_antipodal_gradient_point_kernel(
        ctx.geometry.accessor(), ctx.boundary.accessor(), qx, qy, qz
    );
    std::array<std::vector<Real>, 3> const batch =
        run_antipodal_gradient(ctx.geometry, ctx.boundary, qx, qy, qz);
    expect_near_gradient(point, batch, Real(1e-6));
}

TEST_F(GwnQueryAntipodalTest, antipodal_winding_point_matches_batch) {
    OpenCubeMesh mesh;
    AntipodalContext<OpenCubeMesh> ctx;
    setup_context(mesh, ctx);

    std::vector<Real> const qx{Real(0.21), Real(1.3), Real(-1.4), Real(0.35)};
    std::vector<Real> const qy{Real(0.18), Real(0.25), Real(-0.33), Real(1.4)};
    std::vector<Real> const qz{Real(0.15), Real(0.2), Real(0.42), Real(1.55)};
    auto const [point, overflow] =
        run_antipodal_point_kernel(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz);
    std::vector<Real> const batch = run_antipodal(
        ctx.geometry, ctx.bvh, ctx.boundary, {qx[0], qx[1], qx[2], qx[3]},
        {qy[0], qy[1], qy[2], qy[3]}, {qz[0], qz[1], qz[2], qz[3]}
    );
    expect_near_vector(point, batch);
    EXPECT_EQ(overflow, (std::vector<int>{0, 0, 0, 0}));
}

namespace {

// Thirty-two stacked unit triangles at z = 1 ... 32, so a +z ray through them needs every leaf and
// a deep stack. One more triangle has a boundary edge through the vertical line of (0.25, 0.25)
// at heights 2^-110 and 2^-111 above z = 0, where the crossing-height sign cannot be certified.
[[nodiscard]] MeshData<Real> make_triangle_stack() {
    MeshData<Real> mesh;
    for (int triangle_id = 1; triangle_id <= 32; ++triangle_id) {
        Real const z = static_cast<Real>(triangle_id);
        add_triangle<Real>(mesh, {0, 0, z}, {1, 0, z}, {0, 1, z});
    }
    add_triangle<Real>(
        mesh, {-0.75f, 0.25f, std::ldexp(Real(1), -110)}, {1.25f, 0.25f, std::ldexp(Real(1), -111)},
        {0.25f, -0.75f, 1}
    );
    return mesh;
}

} // namespace

TEST_F(GwnQueryAntipodalTest, antipodal_device_query_reports_stack_overflow_once) {
    MeshData<Real> const mesh = make_triangle_stack();
    gwn::gwn_geometry_object<Real, Index> geometry;
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, gwn::gwn_host_span<Real const>(mesh.vx.data(), mesh.vx.size()),
            gwn::gwn_host_span<Real const>(mesh.vy.data(), mesh.vy.size()),
            gwn::gwn_host_span<Real const>(mesh.vz.data(), mesh.vz.size()),
            gwn::gwn_host_span<Index const>(mesh.i0.data(), mesh.i0.size()),
            gwn::gwn_host_span<Index const>(mesh.i1.data(), mesh.i1.size()),
            gwn::gwn_host_span<Index const>(mesh.i2.data(), mesh.i2.size())
        )
            .is_ok()
    );
    gwn::gwn_bvh4_object<Real, Index> bvh;
    ASSERT_TRUE(
        gwn::gwn_build_bvh(
            geometry, bvh, gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_hploc}
        )
            .is_ok()
    );
    gwn::gwn_boundary_chain_object<Index> boundary;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(geometry, boundary).is_ok());
    ASSERT_GT(bvh.accessor().internal_stack_bound, 1u);

    // The first two queries lie below the stack. The third lies exactly on a triangle plane,
    // whose tie stops the rounded pass, so its overflow may first occur in the fallback rerun. The
    // fourth overflows in the rounded pass, and its boundary term would then need the fallback
    // for the uncertified crossing height; it must not rerun the traversal.
    std::vector<Real> const qx{Real(0.25), Real(0.25), Real(0.25), Real(0.25)};
    std::vector<Real> const qy{Real(0.25), Real(0.25), Real(0.25), Real(0.25)};
    std::vector<Real> const qz{Real(-1), Real(-2), Real(5), Real(0)};
    auto const [winding, overflow] =
        run_antipodal_point_kernel<1>(geometry, bvh, boundary, qx, qy, qz);
    for (std::size_t query_id = 0; query_id < winding.size(); ++query_id) {
        EXPECT_TRUE(std::isnan(winding[query_id])) << "query " << query_id;
        EXPECT_EQ(overflow[query_id], 1) << "query " << query_id;
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_returns_nan_for_non_finite_query_relative_coordinates) {
    // A closed octahedron spanning +-2e38 in x: from x = 1.9e38, the vertex at -2e38 lies beyond
    // the float range, and an overflowed coordinate can certify a wrong determinant sign. A NaN
    // query has no winding number either.
    Real const extent = 2e38f;
    MeshData<Real> mesh = to_mesh_data(gwn::tests::OctahedronMesh{});
    for (Real &x : mesh.vx)
        x *= extent;
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    QuerySet<Real> const queries{
        {1.9e38f, std::numeric_limits<Real>::quiet_NaN(), 0.0f},
        {0.0f, 0.0f, 0.1f},
        {0.0f, 0.0f, 0.1f},
    };
    std::vector<Real> const winding = run_scene_query(scene, queries, scene_query::antipodal);
    EXPECT_TRUE(std::isnan(winding[0]));
    EXPECT_TRUE(std::isnan(winding[1]));
    EXPECT_EQ(winding[2], Real(1));
}

TEST_F(GwnQueryAntipodalTest, antipodal_rejects_missing_or_mismatched_boundary) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 1> const qx{Real(0.23)};
    std::array<Real, 1> const qy{Real(0.17)};
    std::array<Real, 1> const qz{Real(0.11)};
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_out;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_out.resize(qx.size());

    gwn::gwn_boundary_chain_object<Index> missing;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_number_antipodal_batch(
            ctx.geometry, ctx.bvh, missing, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_out.span())
        )
            .is_ok()
    );

    gwn::gwn_boundary_chain_object<Index> mismatched;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(ctx.geometry, mismatched).is_ok());
    ++mismatched.accessor().mesh_vertex_count;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_number_antipodal_batch(
            ctx.geometry, ctx.bvh, mismatched, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_out.span())
        )
            .is_ok()
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_batch_rejects_output_overlapping_queries) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    // The fallback pass rereads the queries after the rounded pass wrote its outputs.
    gwn::detail::gwn_device_array<Real> d_storage;
    d_storage.resize(8);
    cuda::std::span<Real> const storage = d_storage.span();
    auto const query = [&](std::size_t const offset) {
        return gwn::tests::device_input_span(storage.subspan(offset, 2));
    };
    for (std::size_t output_offset : {0u, 1u, 2u, 3u, 4u, 5u}) {
        EXPECT_FALSE(
            gwn::gwn_compute_winding_number_antipodal_batch(
                ctx.geometry, ctx.bvh, ctx.boundary, query(0), query(2), query(4),
                gwn::tests::device_span(storage.subspan(output_offset, 2))
            )
                .is_ok()
        ) << "output offset "
          << output_offset;
    }
    EXPECT_TRUE(
        gwn::gwn_compute_winding_number_antipodal_batch(
            ctx.geometry, ctx.bvh, ctx.boundary, query(0), query(2), query(4),
            gwn::tests::device_span(storage.subspan(6, 2))
        )
            .is_ok()
    );
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

TEST_F(GwnQueryAntipodalTest, antipodal_gradient_rejects_bad_inputs) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    std::array<Real, 1> const qx{Real(0.23)};
    std::array<Real, 1> const qy{Real(0.17)};
    std::array<Real, 1> const qz{Real(0.11)};
    gwn::detail::gwn_device_array<Real> d_qx;
    gwn::detail::gwn_device_array<Real> d_qy;
    gwn::detail::gwn_device_array<Real> d_qz;
    gwn::detail::gwn_device_array<Real> d_gx;
    gwn::detail::gwn_device_array<Real> d_gy;
    gwn::detail::gwn_device_array<Real> d_gz;
    d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size()));
    d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size()));
    d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size()));
    d_gx.resize(qx.size());
    d_gy.resize(qx.size());
    d_gz.resize(qx.size());

    gwn::gwn_boundary_chain_object<Index> missing;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            ctx.geometry, missing, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_gx.span()), gwn::tests::device_span(d_gy.span()),
            gwn::tests::device_span(d_gz.span())
        )
            .is_ok()
    );

    gwn::gwn_boundary_chain_object<Index> mismatched;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(ctx.geometry, mismatched).is_ok());
    ++mismatched.accessor().mesh_vertex_count;
    EXPECT_FALSE(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            ctx.geometry, mismatched, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            gwn::tests::device_span(d_gx.span()), gwn::tests::device_span(d_gy.span()),
            gwn::tests::device_span(d_gz.span())
        )
            .is_ok()
    );

    gwn::gwn_device_span<Real> const bad_output{};
    EXPECT_FALSE(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            ctx.geometry, ctx.boundary, gwn::tests::device_input_span(d_qx.span()),
            gwn::tests::device_input_span(d_qy.span()), gwn::tests::device_input_span(d_qz.span()),
            bad_output, gwn::tests::device_span(d_gy.span()), gwn::tests::device_span(d_gz.span())
        )
            .is_ok()
    );
}

TEST_F(GwnQueryAntipodalTest, antipodal_closed_cube_resolves_rays_through_interior_edges) {
    CubeMesh mesh;
    AntipodalContext<CubeMesh> ctx;
    setup_context(mesh, ctx);

    // Points with x == y project onto a face diagonal. The symbolic perturbation assigns each
    // such ray to exactly one of the two adjacent triangles, so the closed cube reports exact
    // integers.
    std::array<Real, 4> const qx{Real(0), Real(0.5), Real(-0.25), Real(2)};
    std::array<Real, 4> const qy{Real(0), Real(0.5), Real(-0.25), Real(2)};
    std::array<Real, 4> const qz{Real(0), Real(0.5), Real(-0.25), Real(2)};
    std::vector<Real> const winding =
        run_antipodal(ctx.geometry, ctx.bvh, ctx.boundary, qx, qy, qz);
    EXPECT_EQ(winding, (std::vector<Real>{Real(1), Real(1), Real(1), Real(0)}));
}

TEST_F(GwnQueryAntipodalTest, antipodal_closed_cube_surface_queries_follow_the_perturbation) {
    CubeMesh const mesh;
    AntipodalScene<Real> scene;
    build_scene(to_mesh_data(mesh), scene);

    // A query on the surface takes the value of the query moved by d3 along +z, then d1 along
    // +x, then d2 along +y, with d3 >> d1 >> d2. Faces, edges, and corners of the closed cube
    // therefore have exact integer values.
    QuerySet<Real> queries;
    std::vector<Real> expected;
    auto const add = [&](Real const x, Real const y, Real const z, Real const value) {
        queries.x.push_back(x);
        queries.y.push_back(y);
        queries.z.push_back(z);
        expected.push_back(value);
    };
    add(0.3f, -0.2f, 1, 0);  // top face: +z leaves the cube
    add(0.3f, -0.2f, -1, 1); // bottom face: +z enters
    add(1, 0.3f, 0.2f, 0);   // x = +1: +z stays on the face, +x leaves
    add(-1, 0.3f, 0.2f, 1);  // x = -1: +x enters
    add(0.3f, 1, 0.2f, 0);   // y = +1: +z and +x stay on the face, +y leaves
    add(0.3f, -1, 0.2f, 1);  // y = -1: +y enters
    add(1, 0.1f, 1, 0);      // top-right edge
    add(-1, 0.1f, -1, 1);    // bottom-left edge: +z moves onto x = -1, then +x enters
    add(1, -1, 0.4f, 0);     // vertical edge at x = +1
    add(-1, -1, 0.4f, 1);    // vertical edge at x = y = -1: +x moves onto y = -1, +y enters
    add(-1, -1, -1, 1);      // lowest corner
    add(1, 1, 1, 0);         // highest corner
    add(-1, 1, -1, 0);       // +y leaves at y = +1
    EXPECT_EQ(run_scene_query(scene, queries, scene_query::antipodal), expected);
}

TEST_F(GwnQueryAntipodalTest, antipodal_plane_side_is_exact_one_ulp_from_a_tilted_face) {
    // Tetrahedron whose face (0,0,0), (1,0,1), (0,1,0) lies in the plane z = x, with the apex
    // below it. Queries one ulp above and below the face are off the surface; only an exact
    // plane-side predicate separates them. The rounded test cannot certify them, so they also
    // exercise the fallback pass of the batch.
    MeshData<Real> mesh;
    std::array<std::array<Real, 3>, 4> const vertex{
        {{0, 0, 0}, {1, 0, 1}, {0, 1, 0}, {0.3f, 0.3f, -1}}
    };
    for (auto const &v : vertex) {
        mesh.vx.push_back(v[0]);
        mesh.vy.push_back(v[1]);
        mesh.vz.push_back(v[2]);
    }
    // Outward orientation: the tilted face's normal (-1, 0, 1) points away from the apex.
    mesh.i0 = {0, 0, 1, 2};
    mesh.i1 = {1, 3, 3, 3};
    mesh.i2 = {2, 1, 2, 0};
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);

    QuerySet<Real> queries;
    std::vector<Real> expected;
    for (auto const &xy :
         {std::array<Real, 2>{0.5f, 0.25f}, std::array<Real, 2>{0.125f, 0.75f},
          std::array<Real, 2>{0.375f, 0.0625f}}) {
        Real const on_face = xy[0];
        for (auto const &[z, value] :
             {std::pair<Real, Real>{std::nextafter(on_face, Real(2)), Real(0)},
              std::pair<Real, Real>{std::nextafter(on_face, Real(-2)), Real(1)},
              std::pair<Real, Real>{on_face, Real(0)}}) {
            queries.x.push_back(xy[0]);
            queries.y.push_back(xy[1]);
            queries.z.push_back(z);
            expected.push_back(value);
        }
    }
    EXPECT_EQ(run_scene_query(scene, queries, scene_query::antipodal), expected);
    auto const [point, overflow] = run_antipodal_point_kernel(
        scene.geometry, scene.bvh, scene.boundary, queries.x, queries.y, queries.z
    );
    EXPECT_EQ(point, expected);
    EXPECT_EQ(overflow, std::vector<int>(expected.size(), 0));
}

TEST_F(GwnQueryAntipodalTest, antipodal_collinear_boundary_edges_take_the_perturbed_limit) {
    // Each case places boundary edges on the query's vertical line or through it, where the
    // atan2 numerator is exactly zero and the boundary term must take its perturbed limit.
    struct collinear_case {
        MeshData<Real> mesh;
        std::array<Real, 3> query;
        std::array<long double, 3> reference_offset;
    };
    std::vector<collinear_case> cases;
    auto const single = [](std::array<Real, 3> const &a, std::array<Real, 3> const &b,
                           std::array<Real, 3> const &c) {
        MeshData<Real> mesh;
        add_triangle<Real>(mesh, a, b, c);
        return mesh;
    };
    std::array<long double, 3> const off_surface{0, 0, 0};
    // Every coordinate ray from the origin hits a boundary vertex; the winding number is 1/8.
    cases.push_back({single({1, 0, 0}, {0, 1, 0}, {0, 0, 1}), {0, 0, 0}, off_surface});
    // An endpoint above the query whose partner has y == 0 on either side of the line.
    cases.push_back({single({0, 0, 1}, {1, 0, 0}, {0.2f, 1, 0.5f}), {0, 0, 0}, off_surface});
    cases.push_back({single({0, 0, 1}, {-1, 0, 0}, {-0.2f, -1, 0.5f}), {0, 0, 0}, off_surface});
    // The same triangles reversed, so the y == 0 partner starts an edge that ends above the query.
    cases.push_back({single({0, 0, 1}, {0.2f, 1, 0.5f}, {1, 0, 0}), {0, 0, 0}, off_surface});
    cases.push_back({single({0, 0, 1}, {-0.2f, -1, 0.5f}, {-1, 0, 0}), {0, 0, 0}, off_surface});
    // An endpoint below the query.
    cases.push_back({single({0, 0, -1}, {1, 0.5f, 0}, {-0.4f, 1, 0.2f}), {0, 0, 0}, off_surface});
    // Boundary edges crossing the vertical line above and below the query.
    cases.push_back({single({-1, 0, 1}, {1, 0, 1}, {0, 1, 1}), {0, 0, 0}, off_surface});
    cases.push_back({single({-1, 0, -1}, {1, 0, -1}, {0, 1, -1}), {0, 0, 0}, off_surface});
    // Three disjoint triangles whose boundary edges pass through the query's lines along +z,
    // +x, and +y.
    MeshData<Real> three = single({-1, 0, 1}, {1, 0, 1}, {0, 1, 1});
    add_triangle<Real>(three, {2, -1, 0}, {2, 1, 0}, {3, 0, 0});
    add_triangle<Real>(three, {0, 2, -1}, {0, 2, 1}, {1, 2, 0});
    cases.push_back({three, {0, 0, 0}, off_surface});
    // Unit-length endpoints within 2^-80 of the vertical line: the projected determinant and the
    // gaps underflow in float although the edge area approaches -pi.
    Real const tiny = std::ldexp(Real(1), -80);
    cases.push_back({single({tiny, 0, 1}, {0, tiny, 1}, {1, 1, 1}), {0, 0, 0}, off_surface});
    // A query on a boundary edge takes the limit from just above it.
    cases.push_back({single({-1, 0, 0}, {1, 0, 0}, {0, 1, 0.5f}), {0, 0, 0}, {0, 0, 1e-9L}});
    cases.push_back({single({-1, 0, 0}, {1, 0, 0}, {0, 1, 0.5f}), {1, 0, 0}, {0, 0, 1e-9L}});

    for (std::size_t case_id = 0; case_id < cases.size(); ++case_id) {
        auto const &test_case = cases[case_id];
        AntipodalScene<Real> scene;
        build_scene(test_case.mesh, scene);
        QuerySet<Real> const query{
            {test_case.query[0]}, {test_case.query[1]}, {test_case.query[2]}
        };
        Real const antipodal = run_scene_query(scene, query, scene_query::antipodal)[0];
        long double const expected = reference_winding(
            test_case.mesh, test_case.query[0] + test_case.reference_offset[0],
            test_case.query[1] + test_case.reference_offset[1],
            test_case.query[2] + test_case.reference_offset[2]
        );
        EXPECT_NEAR(antipodal, expected, 2e-6L) << "case " << case_id;
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_boundary_area_is_accurate_for_subnormal_edge_lengths) {
    // Queries within 2^-70 of a boundary vertex at the origin: that endpoint's squared length is
    // subnormal in float while the other endpoint has unit length, so the atan2 arguments stay
    // large but the subnormal norm keeps only a few bits.
    MeshData<Real> mesh;
    add_triangle<Real>(mesh, {0, 0, 0}, {1, 0, 0.5f}, {0, 1, 0.3f});
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    QuerySet<Real> queries;
    // Non-dyadic directions make the subnormal squared lengths inexact.
    for (auto const &direction :
         {std::array<Real, 3>{1.37f, 2.11f, 0.0f}, std::array<Real, 3>{-3.13f, 1.29f, 2.71f},
          std::array<Real, 3>{2.39f, -1.17f, -3.07f}, std::array<Real, 3>{-1.41f, -2.23f, 1.73f}}) {
        queries.x.push_back(std::ldexp(direction[0], -70));
        queries.y.push_back(std::ldexp(direction[1], -70));
        queries.z.push_back(std::ldexp(direction[2], -70));
    }
    std::vector<Real> const antipodal = run_scene_query(scene, queries, scene_query::antipodal);
    for (std::size_t query_id = 0; query_id < antipodal.size(); ++query_id) {
        long double const expected =
            reference_winding(mesh, queries.x[query_id], queries.y[query_id], queries.z[query_id]);
        EXPECT_NEAR(antipodal[query_id], expected, 2e-6L) << "query " << query_id;
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_projected_edge_signs_are_exact_and_antisymmetric) {
    // Near-collinear float pairs for which the plain FMA determinant gives the same sign for an
    // edge and its reverse. Two triangles sharing such an edge would both count, or both miss, a
    // ray through it at any distance from the surface.
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_int_distribution<int> ulps(-3, 3);
    auto const plain_sign = [](float const ax, float const ay, float const bx, float const by) {
        float const determinant = std::fma(ax, by, -(ay * bx));
        return (determinant > 0.0f) - (determinant < 0.0f);
    };
    std::vector<Real> values;
    std::vector<int> exact_sign;
    while (exact_sign.size() < 256) {
        float const ax = unit(rng);
        float const ay = unit(rng);
        float const lambda = unit(rng) * 3.0f;
        float bx = ax * lambda;
        float by = ay * lambda;
        bx = std::nextafter(bx, ulps(rng) > 0 ? 2.0f : -2.0f);
        double const exact = double(ax) * double(by) - double(ay) * double(bx);
        if (exact == 0.0 || plain_sign(ax, ay, bx, by) != plain_sign(bx, by, ax, ay))
            continue;
        values.insert(values.end(), {ax, ay, bx, by});
        exact_sign.push_back(exact > 0.0 ? 1 : -1);
    }

    EdgeSigns const signs = run_projected_edge_signs<Real, true>(values);
    for (std::size_t pair_id = 0; pair_id < exact_sign.size(); ++pair_id) {
        EXPECT_EQ(signs.forward[pair_id], exact_sign[pair_id]) << "pair " << pair_id;
        EXPECT_EQ(signs.reverse[pair_id], -exact_sign[pair_id]) << "pair " << pair_id;
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_projected_edge_signs_survive_overflow_and_underflow) {
    // Fixed pairs whose products overflow, underflow, or leave subnormal residuals. The rounded
    // classifier must either certify the exact sign or request the fallback, and the fallback must
    // return the exact sign.
    auto const check = [](auto const &values, std::vector<int> const &expected) {
        using R = typename std::decay_t<decltype(values)>::value_type;
        EdgeSigns const rounded = run_projected_edge_signs<R, false>(values);
        EdgeSigns const exact = run_projected_edge_signs<R, true>(values);
        for (std::size_t pair_id = 0; pair_id < expected.size(); ++pair_id) {
            if (rounded.resolved[pair_id] != 0) {
                EXPECT_EQ(rounded.forward[pair_id], expected[pair_id]) << "pair " << pair_id;
                EXPECT_EQ(rounded.reverse[pair_id], -expected[pair_id]) << "pair " << pair_id;
            }
            EXPECT_EQ(exact.resolved[pair_id], 1) << "pair " << pair_id;
            EXPECT_EQ(exact.forward[pair_id], expected[pair_id]) << "pair " << pair_id;
            EXPECT_EQ(exact.reverse[pair_id], -expected[pair_id]) << "pair " << pair_id;
        }
    };

    float const big = std::ldexp(1.0f, 80);
    float const huge = std::ldexp(1.0f, 66);
    float const small = std::ldexp(1.0f, -140);
    check(
        std::vector<float>{
            big,   big,         big,          big / 2,   // right product overflows
            big,   big / 2,     big,          big,       // left product overflows
            huge,  huge * 1.5f, huge * 1.25f, huge * 2,  // both products overflow
            small, small * 2,   small * 3,    small * 5, // subnormal products
            1.0f,  small,       small,        0.0f,      // exact zero factor, underflowing product
        },
        {-1, 1, 1, -1, -1}
    );

    // Normal double products whose exact difference, -2^-1104, lies below the subnormal range.
    double const s = std::ldexp(1.0, -500);
    double const h = std::ldexp(1.0, -52);
    check(std::vector<double>{s * (1 + h), s, s, s * (1 - h)}, {-1});
}

TEST_F(GwnQueryAntipodalTest, antipodal_closed_mesh_counts_shared_edges_and_vertices_once) {
    MeshData<Real> const mesh = make_jittered_torus(48, 24, 0);
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    ASSERT_TRUE(scene.boundary.accessor().empty());

    QuerySet<Real> const queries = make_edge_and_vertex_queries(mesh, 160000, 3);
    std::vector<Real> const antipodal = run_scene_query(scene, queries, scene_query::antipodal);
    std::vector<Real> const exact = run_scene_query(scene, queries, scene_query::exact);
    std::vector<Real> const distance = run_scene_query(scene, queries, scene_query::distance);

    std::size_t checked = 0;
    std::size_t wrong = 0;
    for (std::size_t query_id = 0; query_id < antipodal.size(); ++query_id) {
        // Every query, including one on the surface, receives an integer.
        EXPECT_EQ(antipodal[query_id], std::round(antipodal[query_id])) << "query " << query_id;
        // Rounding-scale distances leave the exact solid-angle reference ambiguous.
        if (!(distance[query_id] > Real(1e-4)))
            continue;
        ++checked;
        Real const expected = std::round(exact[query_id]);
        if (antipodal[query_id] != expected) {
            if (wrong < 4) {
                ADD_FAILURE() << "query (" << queries.x[query_id] << ", " << queries.y[query_id]
                              << ", " << queries.z[query_id] << ") antipodal "
                              << antipodal[query_id] << " exact " << exact[query_id] << " distance "
                              << distance[query_id];
            }
            ++wrong;
        }
    }
    EXPECT_GT(checked, antipodal.size() * 9 / 10);
    EXPECT_EQ(wrong, 0u);
}

TEST_F(GwnQueryAntipodalTest, antipodal_open_mesh_matches_exact_winding_on_edge_and_vertex_rays) {
    MeshData<Real> const mesh = make_jittered_torus(48, 24, 6);
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    ASSERT_FALSE(scene.boundary.accessor().empty());

    // Rays through boundary vertices and edges exercise the collinear boundary limits; no query
    // may come back NaN.
    QuerySet<Real> const queries = make_edge_and_vertex_queries(mesh, 120000, 5);
    std::vector<Real> const antipodal = run_scene_query(scene, queries, scene_query::antipodal);
    std::vector<Real> const exact = run_scene_query(scene, queries, scene_query::exact);
    std::vector<Real> const distance = run_scene_query(scene, queries, scene_query::distance);

    std::size_t checked = 0;
    std::size_t wrong = 0;
    for (std::size_t query_id = 0; query_id < antipodal.size(); ++query_id) {
        EXPECT_FALSE(std::isnan(antipodal[query_id])) << "query " << query_id;
        if (!(distance[query_id] > Real(1e-4)))
            continue;
        ++checked;
        if (std::abs(antipodal[query_id] - exact[query_id]) > Real(2e-3)) {
            if (wrong < 4) {
                ADD_FAILURE() << "query (" << queries.x[query_id] << ", " << queries.y[query_id]
                              << ", " << queries.z[query_id] << ") antipodal "
                              << antipodal[query_id] << " exact " << exact[query_id];
            }
            ++wrong;
        }
    }
    EXPECT_GT(checked, antipodal.size() * 9 / 10);
    EXPECT_EQ(wrong, 0u);
}

TEST_F(GwnQueryAntipodalTest, antipodal_fallback_matches_unscaled_closed_mesh) {
    // Scaling by 2^-110 preserves the winding numbers of queries away from the surface, up to
    // the subnormal rounding of coordinates below 2^-16, but pushes all projected-determinant
    // products below the certified magnitude. Every query therefore reruns with exact fallbacks,
    // which must reproduce the unscaled crossing counts.
    double const scale = std::ldexp(1.0, -110);
    MeshData<Real> const mesh = make_jittered_torus(32, 16, 0);
    MeshData<Real> const tiny_mesh = make_jittered_torus(32, 16, 0, scale);
    AntipodalScene<Real> scene;
    AntipodalScene<Real> tiny_scene;
    build_scene(mesh, scene);
    build_scene(tiny_mesh, tiny_scene);

    QuerySet<Real> const queries = make_edge_and_vertex_queries(mesh, 20000, 9);
    QuerySet<Real> tiny_queries = queries;
    for (auto *values : {&tiny_queries.x, &tiny_queries.y, &tiny_queries.z})
        for (Real &value : *values)
            value = static_cast<Real>(double(value) * scale);

    std::vector<Real> const expected = run_scene_query(scene, queries, scene_query::antipodal);
    std::vector<Real> const tiny =
        run_scene_query(tiny_scene, tiny_queries, scene_query::antipodal);
    std::vector<Real> const distance = run_scene_query(scene, queries, scene_query::distance);
    std::size_t checked = 0;
    for (std::size_t query_id = 0; query_id < expected.size(); ++query_id) {
        if (!(distance[query_id] > Real(1e-4)))
            continue;
        ++checked;
        EXPECT_EQ(tiny[query_id], expected[query_id]) << "query " << query_id;
    }
    EXPECT_GT(checked, expected.size() * 9 / 10);
}

TEST_F(GwnQueryAntipodalTest, antipodal_open_mesh_winding_is_scale_invariant) {
    // At 2^-40, queries within a few ulps of a boundary vertex already underflow their squared
    // lengths; at 2^-70 and 2^-110 every squared length and projected determinant underflows. Both
    // terms must route such queries through the fallback.
    MeshData<Real> const mesh = make_jittered_torus(32, 16, 4);
    AntipodalScene<Real> scene;
    build_scene(mesh, scene);
    QuerySet<Real> const queries = make_edge_and_vertex_queries(mesh, 4000, 13);
    std::vector<Real> const expected = run_scene_query(scene, queries, scene_query::antipodal);
    std::vector<Real> const distance = run_scene_query(scene, queries, scene_query::distance);

    // At 2^66, offsets exceed the overflow ceiling of the rounded boundary loop.
    for (int exponent : {-40, -70, -110, 66}) {
        double const scale = std::ldexp(1.0, exponent);
        AntipodalScene<Real> tiny_scene;
        build_scene(make_jittered_torus(32, 16, 4, scale), tiny_scene);
        QuerySet<Real> tiny_queries = queries;
        for (auto *values : {&tiny_queries.x, &tiny_queries.y, &tiny_queries.z})
            for (Real &value : *values)
                value = static_cast<Real>(double(value) * scale);
        std::vector<Real> const tiny =
            run_scene_query(tiny_scene, tiny_queries, scene_query::antipodal);
        // Coordinates scaled by 2^-110 round onto the subnormal grid, which moves the mesh.
        Real const tolerance = exponent == -110 ? Real(2e-3) : Real(2e-5);
        std::size_t wrong = 0;
        for (std::size_t query_id = 0; query_id < expected.size(); ++query_id) {
            if (!(distance[query_id] > Real(1e-3)))
                continue;
            if (!(std::abs(tiny[query_id] - expected[query_id]) <= tolerance)) {
                if (wrong < 4) {
                    ADD_FAILURE() << "scale 2^" << exponent << " query " << query_id << ": "
                                  << tiny[query_id] << " vs " << expected[query_id];
                }
                ++wrong;
            }
        }
        EXPECT_EQ(wrong, 0u) << "scale 2^" << exponent;
    }
}

TEST_F(GwnQueryAntipodalTest, antipodal_double_matches_float_and_its_own_scaled_copy) {
    MeshData<Real> const mesh = make_jittered_torus(32, 16, 4);
    QuerySet<Real> const queries = make_edge_and_vertex_queries(mesh, 4000, 17);
    AntipodalScene<Real> float_scene;
    build_scene(mesh, float_scene);
    std::vector<Real> const float_winding =
        run_scene_query(float_scene, queries, scene_query::antipodal);
    std::vector<Real> const distance = run_scene_query(float_scene, queries, scene_query::distance);

    // The double mesh holds the same coordinates, and these scalings are exact while every vertex
    // keeps its coordinates within the documented exponent spread. At 2^-600 every predicate
    // underflows into the rescaled expansions and the boundary range fallback; at 2^500 the
    // offsets exceed the overflow ceiling of the rounded boundary loop.
    for (int exponent : {0, -400, -600, 500}) {
        double const scale = std::ldexp(1.0, exponent);
        MeshData<double> wide;
        for (std::size_t vertex_id = 0; vertex_id < mesh.vx.size(); ++vertex_id) {
            wide.vx.push_back(double(mesh.vx[vertex_id]) * scale);
            wide.vy.push_back(double(mesh.vy[vertex_id]) * scale);
            wide.vz.push_back(double(mesh.vz[vertex_id]) * scale);
        }
        wide.i0 = mesh.i0;
        wide.i1 = mesh.i1;
        wide.i2 = mesh.i2;
        QuerySet<double> wide_queries;
        for (std::size_t query_id = 0; query_id < queries.x.size(); ++query_id) {
            wide_queries.x.push_back(double(queries.x[query_id]) * scale);
            wide_queries.y.push_back(double(queries.y[query_id]) * scale);
            wide_queries.z.push_back(double(queries.z[query_id]) * scale);
        }
        AntipodalScene<double> wide_scene;
        build_scene(wide, wide_scene);
        std::vector<double> const wide_winding =
            run_scene_query(wide_scene, wide_queries, scene_query::antipodal);
        std::size_t wrong = 0;
        for (std::size_t query_id = 0; query_id < queries.x.size(); ++query_id) {
            if (!(distance[query_id] > Real(1e-3)))
                continue;
            if (!(std::abs(wide_winding[query_id] - double(float_winding[query_id])) <= 2e-5)) {
                if (wrong < 4) {
                    ADD_FAILURE() << "scale 2^" << exponent << " query " << query_id << ": "
                                  << wide_winding[query_id] << " vs " << float_winding[query_id];
                }
                ++wrong;
            }
        }
        EXPECT_EQ(wrong, 0u) << "scale 2^" << exponent;
    }
}

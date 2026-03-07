#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_utils.hpp"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using HostMesh = gwn::tests::HostMesh;

enum class topology_builder {
    k_lbvh,
    k_hploc,
};

constexpr Real k_accuracy_scale = Real(2);
constexpr Real k_builder_epsilon = Real(2e-3);

[[nodiscard]] gwn::gwn_status build_facade_for_builder(
    topology_builder const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &topology,
    gwn::gwn_bvh4_aabb_object<Real, Index> &aabb,
    gwn::gwn_bvh4_moment_object<1, Real, Index> &moment
) {
    if (builder == topology_builder::k_hploc) {
        return gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<1, 4, Real, Index>(
            geometry, topology, aabb, moment
        );
    }
    return gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
        geometry, topology, aabb, moment
    );
}

[[nodiscard]] std::array<std::vector<Real>, 3> make_fixture_query_soa(HostMesh const &mesh) {
    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();
    Real max_x = std::numeric_limits<Real>::lowest();
    Real max_y = std::numeric_limits<Real>::lowest();
    Real max_z = std::numeric_limits<Real>::lowest();

    for (std::size_t i = 0; i < mesh.vertex_x.size(); ++i) {
        min_x = std::min(min_x, mesh.vertex_x[i]);
        min_y = std::min(min_y, mesh.vertex_y[i]);
        min_z = std::min(min_z, mesh.vertex_z[i]);
        max_x = std::max(max_x, mesh.vertex_x[i]);
        max_y = std::max(max_y, mesh.vertex_y[i]);
        max_z = std::max(max_z, mesh.vertex_z[i]);
    }

    Real const center_x = (min_x + max_x) * Real(0.5);
    Real const center_y = (min_y + max_y) * Real(0.5);
    Real const center_z = (min_z + max_z) * Real(0.5);
    Real const extent_x = std::max(max_x - min_x, Real(1e-2));
    Real const extent_y = std::max(max_y - min_y, Real(1e-2));
    Real const extent_z = std::max(max_z - min_z, Real(1e-2));

    std::array<std::array<Real, 3>, 9> const queries = {{
        {{center_x, center_y, center_z}},
        {{center_x + Real(1.3) * extent_x, center_y, center_z}},
        {{center_x - Real(1.3) * extent_x, center_y, center_z}},
        {{center_x, center_y + Real(1.3) * extent_y, center_z}},
        {{center_x, center_y - Real(1.3) * extent_y, center_z}},
        {{center_x, center_y, center_z + Real(1.3) * extent_z}},
        {{center_x, center_y, center_z - Real(1.3) * extent_z}},
        {{center_x + Real(0.31) * extent_x, center_y - Real(0.27) * extent_y,
          center_z + Real(0.23) * extent_z}},
        {{center_x - Real(0.19) * extent_x, center_y + Real(0.37) * extent_y,
          center_z - Real(0.29) * extent_z}},
    }};

    std::array<std::vector<Real>, 3> soa{};
    soa[0].reserve(queries.size());
    soa[1].reserve(queries.size());
    soa[2].reserve(queries.size());
    for (auto const &query : queries) {
        soa[0].push_back(query[0]);
        soa[1].push_back(query[1]);
        soa[2].push_back(query[2]);
    }
    return soa;
}

[[nodiscard]] gwn::gwn_status compute_builder_winding_order1(
    topology_builder const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    std::array<std::vector<Real>, 3> const &query_soa, std::vector<Real> &output
) {
    std::size_t const query_count = query_soa[0].size();
    output.assign(query_count, Real(0));

    gwn::gwn_bvh4_topology_object<Real, Index> topology;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment;
    GWN_RETURN_ON_ERROR(build_facade_for_builder(builder, geometry, topology, aabb, moment));

    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_out;
    GWN_RETURN_ON_ERROR(d_qx.resize(query_count));
    GWN_RETURN_ON_ERROR(d_qy.resize(query_count));
    GWN_RETURN_ON_ERROR(d_qz.resize(query_count));
    GWN_RETURN_ON_ERROR(d_out.resize(query_count));
    GWN_RETURN_ON_ERROR(
        d_qx.copy_from_host(cuda::std::span<Real const>(query_soa[0].data(), query_count))
    );
    GWN_RETURN_ON_ERROR(
        d_qy.copy_from_host(cuda::std::span<Real const>(query_soa[1].data(), query_count))
    );
    GWN_RETURN_ON_ERROR(
        d_qz.copy_from_host(cuda::std::span<Real const>(query_soa[2].data(), query_count))
    );
    GWN_RETURN_ON_ERROR((gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
        geometry.accessor(), topology.accessor(), moment.accessor(), d_qx.span(), d_qy.span(),
        d_qz.span(), d_out.span(), k_accuracy_scale
    )));

    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()));
    GWN_RETURN_ON_ERROR(d_out.copy_to_host(cuda::std::span<Real>(output.data(), output.size())));
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()));
    return gwn::gwn_status::ok();
}

} // namespace

TEST(smallgwn_integration_builtin_models, topology_builders_agree_on_repo_fixtures) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    std::vector<std::filesystem::path> const model_paths =
        gwn::tests::collect_builtin_fixture_paths();
    ASSERT_EQ(model_paths.size(), 3u);

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());

        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        ASSERT_TRUE(maybe_mesh.has_value()) << "Failed to load built-in fixture " << model_path;
        HostMesh const &mesh = *maybe_mesh;

        auto const query_soa = make_fixture_query_soa(mesh);
        std::size_t const query_count = query_soa[0].size();
        ASSERT_EQ(query_soa[1].size(), query_count);
        ASSERT_EQ(query_soa[2].size(), query_count);
        ASSERT_GT(query_count, 0u);

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
            geometry, cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);

        std::vector<Real> winding_lbvh;
        gwn::gwn_status const lbvh_status = compute_builder_winding_order1(
            topology_builder::k_lbvh, geometry, query_soa, winding_lbvh
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(lbvh_status);
        ASSERT_TRUE(lbvh_status.is_ok()) << gwn::tests::status_to_debug_string(lbvh_status);

        std::vector<Real> winding_hploc;
        gwn::gwn_status const hploc_status = compute_builder_winding_order1(
            topology_builder::k_hploc, geometry, query_soa, winding_hploc
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(hploc_status);
        ASSERT_TRUE(hploc_status.is_ok()) << gwn::tests::status_to_debug_string(hploc_status);

        ASSERT_EQ(winding_lbvh.size(), query_count);
        ASSERT_EQ(winding_hploc.size(), query_count);
        ++tested_model_count;

        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            EXPECT_NEAR(winding_lbvh[query_index], winding_hploc[query_index], k_builder_epsilon)
                << "query index: " << query_index;
        }
    }

    ASSERT_EQ(tested_model_count, model_paths.size());
}

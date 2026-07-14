#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_utils.cuh"

namespace {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using HostMesh = gwn::tests::HostMesh;
constexpr int k_stack_capacity = gwn::tests::k_test_stack_capacity;

struct dynamic_query_soa {
    std::array<std::vector<Real>, 3> origin{};
    std::array<std::vector<Real>, 3> ray_origin{};
    std::array<std::vector<Real>, 3> direction{};

    [[nodiscard]] std::size_t size() const noexcept { return origin[0].size(); }
};

struct dynamic_query_results {
    std::vector<Real> distance{};
    std::vector<Real> winding{};
    std::array<std::vector<Real>, 3> gradient{};
    std::vector<Real> antipodal_winding{};
    std::array<std::vector<Real>, 3> antipodal_gradient{};
    std::vector<gwn::gwn_ray_first_hit_result<Real, Index>> hit{};
};

[[nodiscard]] char const *builder_name(gwn::gwn_bvh_build_method const method) noexcept {
    switch (method) {
    case gwn::gwn_bvh_build_method::k_lbvh: return "lbvh";
    case gwn::gwn_bvh_build_method::k_hploc: return "hploc";
    }
    return "unknown";
}

[[nodiscard]] gwn::gwn_status
upload_mesh(HostMesh const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry) noexcept {
    return gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vertex_x),
        cuda::std::span<Real const>(mesh.vertex_y), cuda::std::span<Real const>(mesh.vertex_z),
        cuda::std::span<Index const>(mesh.tri_i0), cuda::std::span<Index const>(mesh.tri_i1),
        cuda::std::span<Index const>(mesh.tri_i2)
    );
}

[[nodiscard]] HostMesh deform_mesh(HostMesh mesh) {
    for (std::size_t vertex_id = 0; vertex_id < mesh.vertex_x.size(); ++vertex_id) {
        Real const x = mesh.vertex_x[vertex_id];
        Real const y = mesh.vertex_y[vertex_id];
        Real const z = mesh.vertex_z[vertex_id];
        mesh.vertex_x[vertex_id] = Real(1.15) * x + Real(0.10) * y + Real(0.25);
        mesh.vertex_y[vertex_id] = Real(0.85) * y - Real(0.15) * z - Real(0.20);
        mesh.vertex_z[vertex_id] = Real(1.05) * z + Real(0.12) * x + Real(0.30);
    }
    return mesh;
}

[[nodiscard]] dynamic_query_soa make_dynamic_queries(HostMesh const &mesh) {
    dynamic_query_soa queries{};
    std::size_t const query_count = std::min<std::size_t>(mesh.tri_i0.size(), 12);
    for (int axis = 0; axis < 3; ++axis) {
        queries.origin[axis].reserve(query_count);
        queries.ray_origin[axis].reserve(query_count);
        queries.direction[axis].reserve(query_count);
    }

    Real min_x = *std::min_element(mesh.vertex_x.begin(), mesh.vertex_x.end());
    Real max_x = *std::max_element(mesh.vertex_x.begin(), mesh.vertex_x.end());
    Real min_y = *std::min_element(mesh.vertex_y.begin(), mesh.vertex_y.end());
    Real max_y = *std::max_element(mesh.vertex_y.begin(), mesh.vertex_y.end());
    Real min_z = *std::min_element(mesh.vertex_z.begin(), mesh.vertex_z.end());
    Real max_z = *std::max_element(mesh.vertex_z.begin(), mesh.vertex_z.end());
    Real const diagonal = std::sqrt(
        (max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y) +
        (max_z - min_z) * (max_z - min_z)
    );

    // Centroid-normal queries avoid shared edges. Point queries use a mesh-scale offset, while ray
    // origins use a triangle-scale offset to keep the ray-triangle solve local to each primitive.
    for (std::size_t triangle_id = 0; triangle_id < mesh.tri_i0.size(); ++triangle_id) {
        std::array<Index, 3> const index{
            mesh.tri_i0[triangle_id], mesh.tri_i1[triangle_id], mesh.tri_i2[triangle_id]
        };
        std::array<std::array<Real, 3>, 3> vertex{};
        for (int corner = 0; corner < 3; ++corner) {
            std::size_t const vertex_id = static_cast<std::size_t>(index[corner]);
            vertex[corner] = {
                mesh.vertex_x[vertex_id], mesh.vertex_y[vertex_id], mesh.vertex_z[vertex_id]
            };
        }
        std::array<Real, 3> const edge1{
            vertex[1][0] - vertex[0][0], vertex[1][1] - vertex[0][1], vertex[1][2] - vertex[0][2]
        };
        std::array<Real, 3> const edge2{
            vertex[2][0] - vertex[0][0], vertex[2][1] - vertex[0][1], vertex[2][2] - vertex[0][2]
        };
        std::array<Real, 3> normal{
            edge1[1] * edge2[2] - edge1[2] * edge2[1], edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        };
        Real const normal_length =
            std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (!(normal_length > Real(0)))
            continue;
        Real const edge1_length2 = edge1[0] * edge1[0] + edge1[1] * edge1[1] + edge1[2] * edge1[2];
        Real const edge2_length2 = edge2[0] * edge2[0] + edge2[1] * edge2[1] + edge2[2] * edge2[2];
        Real const ray_offset = std::sqrt(std::max(edge1_length2, edge2_length2));

        for (int axis = 0; axis < 3; ++axis) {
            Real const centroid = (vertex[0][axis] + vertex[1][axis] + vertex[2][axis]) / Real(3);
            normal[axis] /= normal_length;
            queries.origin[axis].push_back(centroid + diagonal * normal[axis]);
            queries.ray_origin[axis].push_back(centroid + ray_offset * normal[axis]);
            queries.direction[axis].push_back(-normal[axis]);
        }
        if (queries.size() == query_count)
            break;
    }
    return queries;
}

[[nodiscard]] gwn::gwn_status run_dynamic_queries(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_bvh4_moment_object<2, Real, Index> const &moment,
    gwn::gwn_boundary_chain_object<Index> const &boundary, dynamic_query_soa const &query,
    dynamic_query_results &result
) {
    std::size_t const count = query.size();
    std::array<gwn::detail::gwn_device_array<Real>, 3> device_origin{};
    std::array<gwn::detail::gwn_device_array<Real>, 3> device_ray_origin{};
    std::array<gwn::detail::gwn_device_array<Real>, 3> device_direction{};
    std::array<gwn::detail::gwn_device_array<Real>, 3> device_gradient{};
    std::array<gwn::detail::gwn_device_array<Real>, 3> device_antipodal_gradient{};
    gwn::detail::gwn_device_array<Real> device_distance{};
    gwn::detail::gwn_device_array<Real> device_winding{};
    gwn::detail::gwn_device_array<Real> device_antipodal_winding{};
    gwn::detail::gwn_device_array<gwn::gwn_ray_first_hit_result<Real, Index>> device_hit{};

    for (int axis = 0; axis < 3; ++axis) {
        device_origin[axis].resize(count);
        device_ray_origin[axis].resize(count);
        device_direction[axis].resize(count);
        device_gradient[axis].resize(count);
        device_antipodal_gradient[axis].resize(count);
        device_origin[axis].copy_from_host(cuda::std::span<Real const>(query.origin[axis]));
        device_ray_origin[axis].copy_from_host(cuda::std::span<Real const>(query.ray_origin[axis]));
        device_direction[axis].copy_from_host(cuda::std::span<Real const>(query.direction[axis]));
    }
    device_distance.resize(count);
    device_winding.resize(count);
    device_antipodal_winding.resize(count);
    device_hit.resize(count);

    GWN_RETURN_ON_ERROR((gwn::gwn_compute_unsigned_distance_batch<4, Real, Index, k_stack_capacity>(
        bvh, device_origin[0].span(), device_origin[1].span(), device_origin[2].span(),
        device_distance.span()
    )));
    GWN_RETURN_ON_ERROR(
        (gwn::gwn_compute_winding_number_taylor_batch<2, 4, Real, Index, k_stack_capacity>(
            bvh, moment, device_origin[0].span(), device_origin[1].span(), device_origin[2].span(),
            device_winding.span()
        ))
    );
    GWN_RETURN_ON_ERROR(
        (gwn::gwn_compute_winding_gradient_taylor_batch<2, 4, Real, Index, k_stack_capacity>(
            bvh, moment, device_origin[0].span(), device_origin[1].span(), device_origin[2].span(),
            device_gradient[0].span(), device_gradient[1].span(), device_gradient[2].span()
        ))
    );
    GWN_RETURN_ON_ERROR(
        (gwn::gwn_compute_winding_number_antipodal_batch<4, Real, Index, k_stack_capacity>(
            geometry, bvh, boundary, device_origin[0].span(), device_origin[1].span(),
            device_origin[2].span(), device_antipodal_winding.span()
        ))
    );
    GWN_RETURN_ON_ERROR(
        gwn::gwn_compute_winding_gradient_antipodal_batch(
            geometry, boundary, device_origin[0].span(), device_origin[1].span(),
            device_origin[2].span(), device_antipodal_gradient[0].span(),
            device_antipodal_gradient[1].span(), device_antipodal_gradient[2].span()
        )
    );
    GWN_RETURN_ON_ERROR((gwn::gwn_compute_ray_first_hit_batch<4, Real, Index, k_stack_capacity>(
        bvh, device_ray_origin[0].span(), device_ray_origin[1].span(), device_ray_origin[2].span(),
        device_direction[0].span(), device_direction[1].span(), device_direction[2].span(),
        device_hit.span()
    )));

    result.distance.resize(count);
    result.winding.resize(count);
    result.antipodal_winding.resize(count);
    result.hit.resize(count);
    for (int axis = 0; axis < 3; ++axis) {
        result.gradient[axis].resize(count);
        result.antipodal_gradient[axis].resize(count);
    }
    // The queries and copies share the default stream. One final synchronization makes every
    // host result complete before the temporary device arrays release their storage.
    device_distance.copy_to_host(cuda::std::span<Real>(result.distance));
    device_winding.copy_to_host(cuda::std::span<Real>(result.winding));
    device_antipodal_winding.copy_to_host(cuda::std::span<Real>(result.antipodal_winding));
    device_hit.copy_to_host(
        cuda::std::span<gwn::gwn_ray_first_hit_result<Real, Index>>(result.hit)
    );
    for (int axis = 0; axis < 3; ++axis) {
        device_gradient[axis].copy_to_host(cuda::std::span<Real>(result.gradient[axis]));
        device_antipodal_gradient[axis].copy_to_host(
            cuda::std::span<Real>(result.antipodal_gradient[axis])
        );
    }
    return gwn::gwn_cuda_to_status(cudaDeviceSynchronize());
}

class GwnDynamicWorkflowTest : public ::testing::TestWithParam<gwn::gwn_bvh_build_method> {};

TEST_P(GwnDynamicWorkflowTest, refit_matches_fresh_build_for_every_query_family) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    gwn::gwn_bvh_build_method const method = GetParam();
    SCOPED_TRACE(builder_name(method));
    std::vector<std::filesystem::path> const mesh_paths = gwn::tests::collect_mesh_paths();
    ASSERT_FALSE(mesh_paths.empty());

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &mesh_path : mesh_paths) {
        SCOPED_TRACE(mesh_path.string());
        std::optional<HostMesh> const loaded = gwn::tests::load_obj_mesh(mesh_path);
        if (!loaded.has_value()) {
            std::cerr << "smallgwn input skip: failed to parse " << mesh_path << '\n';
            continue;
        }
        HostMesh const initial_mesh = *loaded;
        HostMesh const mutated_mesh = deform_mesh(initial_mesh);
        dynamic_query_soa const query = make_dynamic_queries(mutated_mesh);
        ASSERT_GT(query.size(), 0u) << "No non-degenerate triangle in " << mesh_path;
        ++tested_model_count;

        gwn::gwn_geometry_object<Real, Index> dynamic_geometry{};
        gwn::gwn_status status = upload_mesh(initial_mesh, dynamic_geometry);
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
        gwn::gwn_bvh4_object<Real, Index> dynamic_bvh{};
        ASSERT_TRUE(
            gwn::gwn_build_bvh(
                dynamic_geometry, dynamic_bvh, gwn::gwn_bvh_build_options{.method = method}
            )
                .is_ok()
        );
        gwn::gwn_bvh4_moment_object<2, Real, Index> dynamic_moment{};
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(dynamic_bvh, dynamic_moment).is_ok());
        gwn::gwn_boundary_chain_object<Index> dynamic_boundary{};
        ASSERT_TRUE(gwn::gwn_build_boundary_chain(dynamic_geometry, dynamic_boundary).is_ok());

        dynamic_query_results initial{};
        status = run_dynamic_queries(
            dynamic_geometry, dynamic_bvh, dynamic_moment, dynamic_boundary, query, initial
        );
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

        status = gwn::gwn_update_geometry(
            dynamic_geometry, cuda::std::span<Real const>(mutated_mesh.vertex_x),
            cuda::std::span<Real const>(mutated_mesh.vertex_y),
            cuda::std::span<Real const>(mutated_mesh.vertex_z)
        );
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
        ASSERT_TRUE(gwn::gwn_refit_bvh(dynamic_geometry, dynamic_bvh).is_ok());
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(dynamic_bvh, dynamic_moment).is_ok());

        gwn::gwn_geometry_object<Real, Index> fresh_geometry{};
        status = upload_mesh(mutated_mesh, fresh_geometry);
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
        gwn::gwn_bvh4_object<Real, Index> fresh_bvh{};
        ASSERT_TRUE(
            gwn::gwn_build_bvh(
                fresh_geometry, fresh_bvh, gwn::gwn_bvh_build_options{.method = method}
            )
                .is_ok()
        );
        gwn::gwn_bvh4_moment_object<2, Real, Index> fresh_moment{};
        ASSERT_TRUE(gwn::gwn_refit_bvh_moment<2>(fresh_bvh, fresh_moment).is_ok());
        gwn::gwn_boundary_chain_object<Index> fresh_boundary{};
        ASSERT_TRUE(gwn::gwn_build_boundary_chain(fresh_geometry, fresh_boundary).is_ok());

        dynamic_query_results refit{};
        dynamic_query_results fresh{};
        status = run_dynamic_queries(
            dynamic_geometry, dynamic_bvh, dynamic_moment, dynamic_boundary, query, refit
        );
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
        status = run_dynamic_queries(
            fresh_geometry, fresh_bvh, fresh_moment, fresh_boundary, query, fresh
        );
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

        auto const expect_hit_matches_mesh = [&](auto const &hit, char const *const label) {
            SCOPED_TRACE(label);
            std::size_t const primitive_id = static_cast<std::size_t>(hit.primitive_id);
            ASSERT_LT(primitive_id, mutated_mesh.tri_i0.size());

            std::array<Index, 3> const index{
                mutated_mesh.tri_i0[primitive_id],
                mutated_mesh.tri_i1[primitive_id],
                mutated_mesh.tri_i2[primitive_id],
            };
            std::array<std::array<Real, 3>, 3> vertex{};
            for (int corner = 0; corner < 3; ++corner) {
                std::size_t const vertex_id = static_cast<std::size_t>(index[corner]);
                ASSERT_LT(vertex_id, mutated_mesh.vertex_x.size());
                vertex[corner] = {
                    mutated_mesh.vertex_x[vertex_id],
                    mutated_mesh.vertex_y[vertex_id],
                    mutated_mesh.vertex_z[vertex_id],
                };
            }

            std::array<Real, 3> const edge1{
                vertex[1][0] - vertex[0][0],
                vertex[1][1] - vertex[0][1],
                vertex[1][2] - vertex[0][2],
            };
            std::array<Real, 3> const edge2{
                vertex[2][0] - vertex[0][0],
                vertex[2][1] - vertex[0][1],
                vertex[2][2] - vertex[0][2],
            };
            std::array<Real, 3> const expected_normal{
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            };
            std::array<Real, 3> const actual_normal{
                hit.geometric_normal_x,
                hit.geometric_normal_y,
                hit.geometric_normal_z,
            };

            // A first-hit tie may select a different overlapping primitive after a fresh build.
            // Validate each record against the mesh instead of freezing traversal tie order.
            Real normal_alignment = Real(0);
            for (int axis = 0; axis < 3; ++axis)
                normal_alignment += actual_normal[axis] * expected_normal[axis];
            EXPECT_GT(normal_alignment, Real(0));
        };

        bool observed_geometry_change = false;
        for (std::size_t query_id = 0; query_id < query.size(); ++query_id) {
            SCOPED_TRACE(::testing::Message() << "query " << query_id);
            observed_geometry_change =
                observed_geometry_change ||
                std::abs(refit.distance[query_id] - initial.distance[query_id]) > Real(1e-4);
            EXPECT_NEAR(refit.distance[query_id], fresh.distance[query_id], Real(2e-5));
            EXPECT_NEAR(refit.winding[query_id], fresh.winding[query_id], Real(2e-3));
            EXPECT_NEAR(
                refit.antipodal_winding[query_id], fresh.antipodal_winding[query_id], Real(2e-5)
            );
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_NEAR(
                    refit.gradient[axis][query_id], fresh.gradient[axis][query_id], Real(2e-3)
                );
                EXPECT_NEAR(
                    refit.antipodal_gradient[axis][query_id],
                    fresh.antipodal_gradient[axis][query_id], Real(2e-5)
                );
            }

            auto const &refit_hit = refit.hit[query_id];
            auto const &fresh_hit = fresh.hit[query_id];
            ASSERT_EQ(refit_hit.status, fresh_hit.status);
            ASSERT_TRUE(refit_hit.hit());
            EXPECT_NEAR(refit_hit.t, fresh_hit.t, Real(2e-5));
            if (refit_hit.primitive_id == fresh_hit.primitive_id) {
                EXPECT_NEAR(refit_hit.u, fresh_hit.u, Real(2e-5));
                EXPECT_NEAR(refit_hit.v, fresh_hit.v, Real(2e-5));
                EXPECT_NEAR(refit_hit.geometric_normal_x, fresh_hit.geometric_normal_x, Real(2e-5));
                EXPECT_NEAR(refit_hit.geometric_normal_y, fresh_hit.geometric_normal_y, Real(2e-5));
                EXPECT_NEAR(refit_hit.geometric_normal_z, fresh_hit.geometric_normal_z, Real(2e-5));
            }
            expect_hit_matches_mesh(refit_hit, "refit hit");
            expect_hit_matches_mesh(fresh_hit, "fresh hit");
        }
        EXPECT_TRUE(observed_geometry_change) << "The deformation did not affect any query.";
    }
    EXPECT_GT(tested_model_count, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Builders, GwnDynamicWorkflowTest,
    ::testing::Values(gwn::gwn_bvh_build_method::k_lbvh, gwn::gwn_bvh_build_method::k_hploc),
    [](testing::TestParamInfo<gwn::gwn_bvh_build_method> const &info) {
        return std::string(builder_name(info.param));
    }
);

} // namespace

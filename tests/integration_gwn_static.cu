#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
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
constexpr gwn::gwn_query_batch_config k_query_batch_config{
    .block_size = gwn::k_gwn_default_query_batch_block_size,
    .stack_capacity = k_stack_capacity,
};

struct fixture_ray_soa {
    std::array<std::vector<Real>, 3> origin{};
    std::array<std::vector<Real>, 3> direction{};
};

struct reference_ray_first_hit_result {
    double t{-1.0};
    double second_t{std::numeric_limits<double>::infinity()};
    Index primitive_id{gwn::gwn_invalid_index<Index>()};

    [[nodiscard]] bool hit() const noexcept { return t >= 0.0; }
};

[[nodiscard]] fixture_ray_soa make_fixture_ray_soa(HostMesh const &mesh) {
    std::array<double, 3> lower{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
    };
    std::array<double, 3> upper{
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
    };
    for (std::size_t vertex_id = 0; vertex_id < mesh.vertex_x.size(); ++vertex_id) {
        std::array<double, 3> const vertex{
            static_cast<double>(mesh.vertex_x[vertex_id]),
            static_cast<double>(mesh.vertex_y[vertex_id]),
            static_cast<double>(mesh.vertex_z[vertex_id]),
        };
        for (int axis = 0; axis < 3; ++axis) {
            lower[axis] = std::min(lower[axis], vertex[axis]);
            upper[axis] = std::max(upper[axis], vertex[axis]);
        }
    }

    std::array<double, 3> center{};
    std::array<double, 3> extent{};
    double diagonal2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = 0.5 * (lower[axis] + upper[axis]);
        extent[axis] = std::max(upper[axis] - lower[axis], 1e-3);
        diagonal2 += extent[axis] * extent[axis];
    }
    double const diagonal = std::sqrt(diagonal2);

    fixture_ray_soa rays{};
    constexpr std::size_t k_triangle_ray_count = 8;
    constexpr std::size_t k_global_ray_count = 32;
    std::size_t const sampled_triangle_count = std::min(k_triangle_ray_count, mesh.tri_i0.size());
    std::size_t const capacity = 2 * sampled_triangle_count + k_global_ray_count;
    for (int axis = 0; axis < 3; ++axis) {
        rays.origin[axis].reserve(capacity);
        rays.direction[axis].reserve(capacity);
    }

    auto append_ray = [&](std::array<double, 3> const &origin,
                          std::array<double, 3> const &direction) {
        double const length = std::sqrt(
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]
        );
        if (!(length > 0.0))
            return;
        for (int axis = 0; axis < 3; ++axis) {
            rays.origin[axis].push_back(static_cast<Real>(origin[axis]));
            rays.direction[axis].push_back(static_cast<Real>(direction[axis] / length));
        }
    };

    // A bounded primitive sample keeps the independent O(rays * triangles) reference practical on
    // large meshes. Two centroid rays exercise both triangle sides without shared-edge ambiguity.
    for (std::size_t sample_id = 0; sample_id < sampled_triangle_count; ++sample_id) {
        std::size_t const triangle_id = sample_id * mesh.tri_i0.size() / sampled_triangle_count;
        std::array<std::array<double, 3>, 3> vertex{};
        std::array<Index, 3> const index{
            mesh.tri_i0[triangle_id], mesh.tri_i1[triangle_id], mesh.tri_i2[triangle_id]
        };
        for (int corner = 0; corner < 3; ++corner) {
            std::size_t const vertex_id = static_cast<std::size_t>(index[corner]);
            vertex[corner] = {
                static_cast<double>(mesh.vertex_x[vertex_id]),
                static_cast<double>(mesh.vertex_y[vertex_id]),
                static_cast<double>(mesh.vertex_z[vertex_id]),
            };
        }
        std::array<double, 3> const edge1{
            vertex[1][0] - vertex[0][0],
            vertex[1][1] - vertex[0][1],
            vertex[1][2] - vertex[0][2],
        };
        std::array<double, 3> const edge2{
            vertex[2][0] - vertex[0][0],
            vertex[2][1] - vertex[0][1],
            vertex[2][2] - vertex[0][2],
        };
        std::array<double, 3> const normal{
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        };
        double const normal_length =
            std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (!(normal_length > 0.0))
            continue;

        std::array<double, 3> centroid{};
        for (int axis = 0; axis < 3; ++axis)
            centroid[axis] = (vertex[0][axis] + vertex[1][axis] + vertex[2][axis]) / 3.0;
        for (double const side : {-1.0, 1.0}) {
            std::array<double, 3> origin{};
            std::array<double, 3> direction{};
            for (int axis = 0; axis < 3; ++axis) {
                direction[axis] = -side * normal[axis] / normal_length;
                origin[axis] = centroid[axis] - direction[axis] * (0.75 * diagonal);
            }
            append_ray(origin, direction);
        }
    }

    // A Fibonacci sphere gives deterministic traversal-heavy rays. One quarter point away from
    // the mesh so the same batch strongly constrains both hit and miss behavior.
    constexpr double k_golden_angle = 2.39996322972865332;
    for (std::size_t ray_id = 0; ray_id < k_global_ray_count; ++ray_id) {
        double const sphere_z =
            1.0 - 2.0 * (static_cast<double>(ray_id) + 0.5) / k_global_ray_count;
        double const sphere_r = std::sqrt(std::max(0.0, 1.0 - sphere_z * sphere_z));
        double const angle = k_golden_angle * static_cast<double>(ray_id);
        std::array<double, 3> const radial{
            sphere_r * std::cos(angle), sphere_r * std::sin(angle), sphere_z
        };
        std::array<double, 3> origin{};
        std::array<double, 3> target{};
        for (int axis = 0; axis < 3; ++axis) {
            origin[axis] = center[axis] + 2.5 * diagonal * radial[axis];
            double const phase = angle * static_cast<double>(axis + 1) + 0.37 * axis;
            target[axis] = center[axis] + 0.31 * extent[axis] * std::sin(phase);
        }
        std::array<double, 3> direction{};
        for (int axis = 0; axis < 3; ++axis)
            direction[axis] = ray_id % 4 == 3 ? radial[axis] : target[axis] - origin[axis];
        append_ray(origin, direction);
    }
    return rays;
}

[[nodiscard]] reference_ray_first_hit_result reference_ray_first_hit(
    HostMesh const &mesh, std::array<Real, 3> const &origin, std::array<Real, 3> const &direction
) noexcept {
    reference_ray_first_hit_result result{};
    double best_t = std::numeric_limits<double>::infinity();
    double second_t = std::numeric_limits<double>::infinity();

    // This scalar double-precision Moller-Trumbore path is intentionally independent of the GPU
    // Pluecker intersector and of BVH traversal.
    for (std::size_t triangle_id = 0; triangle_id < mesh.tri_i0.size(); ++triangle_id) {
        std::array<std::array<double, 3>, 3> vertex{};
        std::array<Index, 3> const index{
            mesh.tri_i0[triangle_id], mesh.tri_i1[triangle_id], mesh.tri_i2[triangle_id]
        };
        for (int corner = 0; corner < 3; ++corner) {
            std::size_t const vertex_id = static_cast<std::size_t>(index[corner]);
            vertex[corner] = {
                static_cast<double>(mesh.vertex_x[vertex_id]),
                static_cast<double>(mesh.vertex_y[vertex_id]),
                static_cast<double>(mesh.vertex_z[vertex_id]),
            };
        }

        std::array<double, 3> edge1{};
        std::array<double, 3> edge2{};
        std::array<double, 3> ray_direction{};
        std::array<double, 3> ray_origin{};
        for (int axis = 0; axis < 3; ++axis) {
            edge1[axis] = vertex[1][axis] - vertex[0][axis];
            edge2[axis] = vertex[2][axis] - vertex[0][axis];
            ray_direction[axis] = static_cast<double>(direction[axis]);
            ray_origin[axis] = static_cast<double>(origin[axis]);
        }
        std::array<double, 3> const p{
            ray_direction[1] * edge2[2] - ray_direction[2] * edge2[1],
            ray_direction[2] * edge2[0] - ray_direction[0] * edge2[2],
            ray_direction[0] * edge2[1] - ray_direction[1] * edge2[0],
        };
        double const determinant = edge1[0] * p[0] + edge1[1] * p[1] + edge1[2] * p[2];
        if (std::abs(determinant) <= 1e-12)
            continue;

        double const inverse_determinant = 1.0 / determinant;
        std::array<double, 3> const translated{
            ray_origin[0] - vertex[0][0],
            ray_origin[1] - vertex[0][1],
            ray_origin[2] - vertex[0][2],
        };
        double const u = (translated[0] * p[0] + translated[1] * p[1] + translated[2] * p[2]) *
                         inverse_determinant;
        if (u < -1e-10 || u > 1.0 + 1e-10)
            continue;

        std::array<double, 3> const q{
            translated[1] * edge1[2] - translated[2] * edge1[1],
            translated[2] * edge1[0] - translated[0] * edge1[2],
            translated[0] * edge1[1] - translated[1] * edge1[0],
        };
        double const v =
            (ray_direction[0] * q[0] + ray_direction[1] * q[1] + ray_direction[2] * q[2]) *
            inverse_determinant;
        if (v < -1e-10 || u + v > 1.0 + 1e-10)
            continue;

        double const t =
            (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]) * inverse_determinant;
        if (t < 0.0)
            continue;
        if (t < best_t) {
            second_t = best_t;
            best_t = t;
            result.primitive_id = static_cast<Index>(triangle_id);
        } else if (t < second_t) {
            second_t = t;
        }
    }

    if (best_t < std::numeric_limits<double>::infinity())
        result.t = best_t;
    result.second_t = second_t;
    return result;
}

} // namespace

TEST(gwn_static_workflow, ray_first_hit_matches_cpu_reference) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    std::vector<std::filesystem::path> const model_paths = gwn::tests::collect_mesh_paths();
    ASSERT_FALSE(model_paths.empty());

    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value()) {
            std::cerr << "smallgwn input skip: failed to parse " << model_path << '\n';
            continue;
        }
        ++tested_model_count;
        HostMesh const &mesh = *maybe_mesh;
        fixture_ray_soa const rays = make_fixture_ray_soa(mesh);
        std::size_t const ray_count = rays.origin[0].size();
        ASSERT_GT(ray_count, 24u);

        std::vector<reference_ray_first_hit_result> reference(ray_count);
        std::size_t reference_hit_count = 0;
        for (std::size_t ray_id = 0; ray_id < ray_count; ++ray_id) {
            reference[ray_id] = reference_ray_first_hit(
                mesh, {rays.origin[0][ray_id], rays.origin[1][ray_id], rays.origin[2][ray_id]},
                {rays.direction[0][ray_id], rays.direction[1][ray_id], rays.direction[2][ray_id]}
            );
            reference_hit_count += reference[ray_id].hit() ? 1u : 0u;
        }
        ASSERT_GT(reference_hit_count, ray_count / 4);
        ASSERT_GT(ray_count - reference_hit_count, ray_count / 8);

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
            geometry, gwn::gwn_host_span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            gwn::gwn_host_span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            gwn::gwn_host_span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        );
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(upload_status);
        ASSERT_TRUE(upload_status.is_ok()) << gwn::tests::status_to_debug_string(upload_status);

        std::array<gwn::detail::gwn_device_array<Real>, 3> device_origin{};
        std::array<gwn::detail::gwn_device_array<Real>, 3> device_direction{};
        gwn::detail::gwn_device_array<gwn::gwn_ray_first_hit_result<Real, Index>> device_hits{};
        for (int axis = 0; axis < 3; ++axis) {
            device_origin[axis].resize(ray_count);
            device_direction[axis].resize(ray_count);
            device_origin[axis].copy_from_host(cuda::std::span<Real const>(rays.origin[axis]));
            device_direction[axis].copy_from_host(
                cuda::std::span<Real const>(rays.direction[axis])
            );
        }
        device_hits.resize(ray_count);

        for (gwn::gwn_bvh_build_method const method :
             {gwn::gwn_bvh_build_method::k_lbvh, gwn::gwn_bvh_build_method::k_hploc}) {
            SCOPED_TRACE(method == gwn::gwn_bvh_build_method::k_lbvh ? "LBVH" : "H-PLOC");
            gwn::gwn_bvh4_object<Real, Index> bvh;
            ASSERT_TRUE(
                gwn::gwn_build_bvh(geometry, bvh, gwn::gwn_bvh_build_options{.method = method})
                    .is_ok()
            );
            gwn::gwn_status const query_status =
                gwn::gwn_compute_ray_first_hit_batch<k_query_batch_config, 4, Real, Index>(
                    bvh, gwn::tests::device_input_span(device_origin[0].span()),
                    gwn::tests::device_input_span(device_origin[1].span()),
                    gwn::tests::device_input_span(device_origin[2].span()),
                    gwn::tests::device_input_span(device_direction[0].span()),
                    gwn::tests::device_input_span(device_direction[1].span()),
                    gwn::tests::device_input_span(device_direction[2].span()),
                    gwn::tests::device_span(device_hits.span())
                );
            ASSERT_TRUE(query_status.is_ok()) << gwn::tests::status_to_debug_string(query_status);

            std::vector<gwn::gwn_ray_first_hit_result<Real, Index>> actual(ray_count);
            device_hits.copy_to_host(cuda::std::span(actual));
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

            for (std::size_t ray_id = 0; ray_id < ray_count; ++ray_id) {
                ASSERT_EQ(actual[ray_id].hit(), reference[ray_id].hit()) << "ray " << ray_id;
                if (!actual[ray_id].hit())
                    continue;

                double const tolerance = 5e-5 * std::max(1.0, std::abs(reference[ray_id].t));
                EXPECT_NEAR(static_cast<double>(actual[ray_id].t), reference[ray_id].t, tolerance)
                    << "ray " << ray_id;
                // Primitive identity is not stable for coincident or singular first hits. The
                // hit state and nearest-hit parameter are the stable contract for dataset meshes.
            }
        }
    }
    EXPECT_GT(tested_model_count, 0u);
}

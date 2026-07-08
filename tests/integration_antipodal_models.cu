#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
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

[[nodiscard]] Real get_env_real(char const *name, Real const default_value) {
    char const *value = std::getenv(name);
    if (value == nullptr)
        return default_value;
    char *end = nullptr;
    double const parsed = std::strtod(value, &end);
    if (end == value || parsed != parsed || parsed <= -std::numeric_limits<double>::infinity() ||
        parsed >= std::numeric_limits<double>::infinity()) {
        return std::numeric_limits<Real>::quiet_NaN();
    }
    return static_cast<Real>(parsed);
}

[[nodiscard]] bool get_env_enabled(char const *name) {
    char const *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return false;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' ||
           value[0] == 'Y';
}

[[nodiscard]] std::size_t
max_count_for_fraction(std::size_t const sample_count, Real const fraction) {
    double const scaled = static_cast<double>(sample_count) * static_cast<double>(fraction);
    return static_cast<std::size_t>(std::ceil(scaled));
}

struct AntipodalErrorStats {
    std::size_t sample_count{0};
    std::size_t nonfinite_count{0};
    std::size_t over_tolerance_count{0};
    std::size_t integer_scale_count{0};
    std::size_t max_error_query{0};
    double max_abs_error{0.0};
    double mean_abs_error{0.0};
    double rms_abs_error{0.0};
};

[[nodiscard]] AntipodalErrorStats compute_error_stats(
    std::vector<Real> const &exact_output, std::vector<Real> const &antipodal_output,
    Real const tolerance
) {
    AntipodalErrorStats stats{};
    stats.sample_count = exact_output.size();
    double abs_sum = 0.0;
    double square_sum = 0.0;
    for (std::size_t query_id = 0; query_id < exact_output.size(); ++query_id) {
        Real const exact = exact_output[query_id];
        Real const antipodal = antipodal_output[query_id];
        if (!std::isfinite(exact) || !std::isfinite(antipodal)) {
            ++stats.nonfinite_count;
            continue;
        }

        double const abs_error = std::abs(static_cast<double>(antipodal) - exact);
        if (abs_error > stats.max_abs_error) {
            stats.max_abs_error = abs_error;
            stats.max_error_query = query_id;
        }
        if (abs_error > static_cast<double>(tolerance))
            ++stats.over_tolerance_count;
        if (abs_error > 0.5)
            ++stats.integer_scale_count;
        abs_sum += abs_error;
        square_sum += abs_error * abs_error;
    }

    if (stats.sample_count > stats.nonfinite_count) {
        double const finite_count = static_cast<double>(stats.sample_count - stats.nonfinite_count);
        stats.mean_abs_error = abs_sum / finite_count;
        stats.rms_abs_error = std::sqrt(square_sum / finite_count);
    }
    return stats;
}

[[nodiscard]] std::array<std::vector<Real>, 3>
make_antipodal_query_soa(HostMesh const &mesh, std::size_t const query_count) {
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

    std::array<std::array<Real, 3>, 9> const anchor_queries = {{
        {{center_x, center_y, center_z}},
        {{center_x + Real(1.37) * extent_x, center_y + Real(0.11) * extent_y, center_z}},
        {{center_x - Real(1.29) * extent_x, center_y - Real(0.07) * extent_y, center_z}},
        {{center_x + Real(0.13) * extent_x, center_y + Real(1.31) * extent_y, center_z}},
        {{center_x - Real(0.17) * extent_x, center_y - Real(1.23) * extent_y, center_z}},
        {{center_x, center_y + Real(0.19) * extent_y, center_z + Real(1.41) * extent_z}},
        {{center_x, center_y - Real(0.23) * extent_y, center_z - Real(1.33) * extent_z}},
        {{center_x + Real(0.31) * extent_x, center_y - Real(0.27) * extent_y,
          center_z + Real(0.23) * extent_z}},
        {{center_x - Real(0.19) * extent_x, center_y + Real(0.37) * extent_y,
          center_z - Real(0.29) * extent_z}},
    }};

    std::array<std::vector<Real>, 3> soa{};
    soa[0].reserve(query_count);
    soa[1].reserve(query_count);
    soa[2].reserve(query_count);
    for (auto const &query : anchor_queries) {
        if (soa[0].size() >= query_count)
            break;
        soa[0].push_back(query[0]);
        soa[1].push_back(query[1]);
        soa[2].push_back(query[2]);
    }
    for (std::size_t query_id = soa[0].size(); query_id < query_count; ++query_id) {
        std::uint32_t const hx = static_cast<std::uint32_t>((query_id * 73856093u) ^ 0x9e3779b9u);
        std::uint32_t const hy = static_cast<std::uint32_t>((query_id * 19349663u) ^ 0x85ebca6bu);
        std::uint32_t const hz = static_cast<std::uint32_t>((query_id * 83492791u) ^ 0xc2b2ae35u);
        auto const unit = [](std::uint32_t const value) {
            return Real(value & 0xffffu) / Real(0xffffu);
        };

        bool const near_field = (query_id % 10) < 7;
        Real const scale = near_field ? Real(1.7) : Real(5.0);
        soa[0].push_back(center_x + (unit(hx) - Real(0.5)) * scale * extent_x);
        soa[1].push_back(center_y + (unit(hy) - Real(0.5)) * scale * extent_y);
        soa[2].push_back(center_z + (unit(hz) - Real(0.5)) * scale * extent_z);
    }
    return soa;
}

[[nodiscard]] gwn::gwn_status compute_exact_and_antipodal(
    HostMesh const &mesh, std::array<std::vector<Real>, 3> const &query_soa,
    std::vector<Real> &exact_output, std::vector<Real> &antipodal_output
) {
    std::size_t const query_count = query_soa[0].size();
    exact_output.assign(query_count, Real(0));
    antipodal_output.assign(query_count, Real(0));

    gwn::gwn_geometry_object<Real, Index> geometry;
    GWN_RETURN_ON_ERROR(
        gwn::gwn_upload_geometry(
            geometry, cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        )
    );

    gwn::gwn_bvh4_topology_object<Real, Index> bvh;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    GWN_RETURN_ON_ERROR(
        (gwn::gwn_bvh_facade_build_topology_aabb_lbvh<4, Real, Index>(geometry, bvh, aabb))
    );

    gwn::gwn_boundary_chain_object<Index> boundary;
    GWN_RETURN_ON_ERROR(gwn::gwn_build_boundary_chain(geometry.accessor(), boundary));

    gwn::gwn_device_array<Real> d_qx;
    gwn::gwn_device_array<Real> d_qy;
    gwn::gwn_device_array<Real> d_qz;
    gwn::gwn_device_array<Real> d_out;
    GWN_RETURN_ON_ERROR(
        d_qx.copy_from_host(cuda::std::span<Real const>(query_soa[0].data(), query_count))
    );
    GWN_RETURN_ON_ERROR(
        d_qy.copy_from_host(cuda::std::span<Real const>(query_soa[1].data(), query_count))
    );
    GWN_RETURN_ON_ERROR(
        d_qz.copy_from_host(cuda::std::span<Real const>(query_soa[2].data(), query_count))
    );
    GWN_RETURN_ON_ERROR(d_out.resize(query_count));

    GWN_RETURN_ON_ERROR((gwn::gwn_compute_winding_number_batch_bvh_exact<Real, Index>(
        geometry.accessor(), bvh.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span()
    )));
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()));
    GWN_RETURN_ON_ERROR(
        d_out.copy_to_host(cuda::std::span<Real>(exact_output.data(), exact_output.size()))
    );
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()));

    GWN_RETURN_ON_ERROR((gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), boundary.accessor(), d_qx.span(),
        d_qy.span(), d_qz.span(), d_out.span()
    )));
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()));
    GWN_RETURN_ON_ERROR(
        d_out.copy_to_host(cuda::std::span<Real>(antipodal_output.data(), antipodal_output.size()))
    );
    GWN_RETURN_ON_ERROR(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()));
    return gwn::gwn_status::ok();
}

} // namespace

TEST(smallgwn_integration_antipodal_models, sampled_models_report_exact_winding_error) {
    SMALLGWN_SKIP_IF_NO_CUDA();

    std::vector<std::filesystem::path> model_paths = gwn::tests::collect_model_paths();
    if (model_paths.empty())
        GTEST_SKIP() << "No model input found. Set SMALLGWN_MODEL_DATA_DIR or SMALLGWN_MODEL_PATH.";

    std::size_t const model_limit =
        gwn::tests::get_env_positive_size_t("SMALLGWN_ANTIPODAL_MODEL_LIMIT", 4);
    std::size_t const query_count =
        gwn::tests::get_env_positive_size_t("SMALLGWN_ANTIPODAL_QUERY_COUNT", 1024);
    Real const tolerance = get_env_real("SMALLGWN_ANTIPODAL_TOLERANCE", Real(5e-4));
    Real const over_tolerance_fraction =
        get_env_real("SMALLGWN_ANTIPODAL_MAX_OVER_TOLERANCE_FRACTION", Real(1e-2));
    Real const integer_scale_fraction =
        get_env_real("SMALLGWN_ANTIPODAL_MAX_INTEGER_SCALE_FRACTION", Real(1e-3));
    bool const strict = get_env_enabled("SMALLGWN_ANTIPODAL_STRICT");
    if (model_paths.size() > model_limit)
        model_paths.resize(model_limit);

    std::size_t attempted_model_count = 0;
    std::size_t unreadable_model_count = 0;
    std::size_t tested_model_count = 0;
    for (std::filesystem::path const &model_path : model_paths) {
        SCOPED_TRACE(model_path.string());
        ++attempted_model_count;

        std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
        if (!maybe_mesh.has_value()) {
            ++unreadable_model_count;
            ADD_FAILURE() << "Unreadable OBJ model: " << model_path.string();
            continue;
        }

        HostMesh const &mesh = *maybe_mesh;
        auto const query_soa = make_antipodal_query_soa(mesh, query_count);
        std::vector<Real> exact_output;
        std::vector<Real> antipodal_output;
        gwn::gwn_status const status =
            compute_exact_and_antipodal(mesh, query_soa, exact_output, antipodal_output);
        SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
        ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

        ++tested_model_count;
        std::cout << "[gwn-antipodal] model=" << model_path.filename().string()
                  << " vertices=" << mesh.vertex_x.size() << " triangles=" << mesh.tri_i0.size()
                  << " queries=" << query_soa[0].size() << "\n";
        ASSERT_EQ(antipodal_output.size(), exact_output.size());
        AntipodalErrorStats const stats =
            compute_error_stats(exact_output, antipodal_output, tolerance);
        std::cout << "[gwn-antipodal] error_stats"
                  << " tolerance=" << tolerance << " max_abs=" << stats.max_abs_error
                  << " max_query=" << stats.max_error_query << " mean_abs=" << stats.mean_abs_error
                  << " rms_abs=" << stats.rms_abs_error
                  << " over_tolerance=" << stats.over_tolerance_count
                  << " integer_scale=" << stats.integer_scale_count
                  << " nonfinite=" << stats.nonfinite_count << "\n";

        std::size_t const over_tolerance_budget =
            max_count_for_fraction(stats.sample_count, over_tolerance_fraction);
        std::size_t const integer_scale_budget =
            max_count_for_fraction(stats.sample_count, integer_scale_fraction);

        EXPECT_EQ(stats.nonfinite_count, 0u);
        if (strict) {
            EXPECT_EQ(stats.over_tolerance_count, 0u)
                << "max error query: " << stats.max_error_query;
            EXPECT_EQ(stats.integer_scale_count, 0u)
                << "max error query: " << stats.max_error_query;
        } else {
            EXPECT_LE(stats.over_tolerance_count, over_tolerance_budget)
                << "max error query: " << stats.max_error_query;
            EXPECT_LE(stats.integer_scale_count, integer_scale_budget)
                << "max error query: " << stats.max_error_query;
        }
    }

    std::cout << "[gwn-antipodal] model_summary attempted=" << attempted_model_count
              << " tested=" << tested_model_count << " unreadable=" << unreadable_model_count
              << "\n";
    ASSERT_GT(tested_model_count, 0u) << "No valid OBJ models were exercised.";
}

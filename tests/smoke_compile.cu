#include <cstdint>

#include <gwn/gwn.cuh>

int main() {
    using real_type = float;
    using index_type = std::int64_t;

    gwn::gwn_geometry_accessor<real_type, index_type> accessor{};
    gwn::gwn_bvh_accessor<real_type, index_type> bvh{};
    gwn::gwn_bvh_data4_accessor<real_type, index_type> bvh_data{};

    real_type const one_value[1] = {0.0f};
    cuda::std::span<real_type const> const one_query(one_value, 1);
    cuda::std::span<real_type const> const query_x{};
    cuda::std::span<real_type const> const query_y{};
    cuda::std::span<real_type const> const query_z{};
    cuda::std::span<real_type> const output{};

    gwn::gwn_status const invalid_result =
        gwn::gwn_compute_winding_number_batch<real_type, index_type>(
            accessor, one_query, query_y, query_z, output
        );
    if (invalid_result.is_ok())
        return 1;

    gwn::gwn_status const valid_result =
        gwn::gwn_compute_winding_number_batch<real_type, index_type>(
            accessor, query_x, query_y, query_z, output
        );
    if (!valid_result.is_ok())
        return 1;

    gwn::gwn_status const bvh_missing_result =
        gwn::gwn_compute_winding_number_batch_bvh_exact<real_type, index_type>(
            accessor, bvh, query_x, query_y, query_z, output
        );
    if (bvh_missing_result.is_ok())
        return 1;

    gwn::gwn_status const taylor_build_result =
        gwn::gwn_build_bvh4_lbvh_taylor<0, real_type, index_type>(accessor, bvh, bvh_data);
    if (!taylor_build_result.is_ok())
        return 1;

    gwn::gwn_status const taylor_query_missing_result =
        gwn::gwn_compute_winding_number_batch_bvh_taylor<0, real_type, index_type>(
            accessor, bvh, bvh_data, query_x, query_y, query_z, output
        );
    return taylor_query_missing_result.is_ok() ? 1 : 0;
}

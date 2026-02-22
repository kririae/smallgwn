#include <gwn/gwn.cuh>

#include <cstdint>

int main() {
  using real_type = float;
  using index_type = std::int64_t;

  gwn::gwn_geometry_accessor<real_type, index_type> accessor{};
  gwn::gwn_bvh_accessor<real_type, index_type> bvh{};

  const real_type one_value[1] = {0.0f};
  const cuda::std::span<const real_type> one_query(one_value, 1);
  const cuda::std::span<const real_type> query_x{};
  const cuda::std::span<const real_type> query_y{};
  const cuda::std::span<const real_type> query_z{};
  const cuda::std::span<real_type> output{};

  const gwn::gwn_status invalid_result =
      gwn::gwn_compute_winding_number_batch<real_type, index_type>(
          accessor, one_query, query_y, query_z, output);
  if (invalid_result.is_ok()) {
    return 1;
  }

  const gwn::gwn_status valid_result =
      gwn::gwn_compute_winding_number_batch<real_type, index_type>(
          accessor, query_x, query_y, query_z, output);
  if (!valid_result.is_ok()) {
    return 1;
  }

  const gwn::gwn_status bvh_missing_result =
      gwn::gwn_compute_winding_number_batch_bvh_exact<real_type, index_type>(
          accessor, bvh, query_x, query_y, query_z, output);
  return bvh_missing_result.is_ok() ? 1 : 0;
}

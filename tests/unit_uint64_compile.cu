#include <cstdint>

#include <gtest/gtest.h>

#include <gwn/gwn_bvh.cuh>
#include <gwn/gwn_bvh_topology_build.cuh>
#include <gwn/gwn_geometry.cuh>

template gwn::gwn_status gwn::gwn_bvh_topology_build_lbvh<4, float, std::uint64_t, std::uint64_t>(
    gwn_geometry_object<float, std::uint64_t> const &,
    gwn_bvh_topology_object<4, float, std::uint64_t> &, cudaStream_t
) noexcept;

template gwn::gwn_status gwn::gwn_bvh_topology_build_hploc<4, float, std::uint64_t, std::uint64_t>(
    gwn_geometry_object<float, std::uint64_t> const &,
    gwn_bvh_topology_object<4, float, std::uint64_t> &, cudaStream_t
) noexcept;

TEST(smallgwn_unit_uint64_compile, topology_templates_instantiate) { SUCCEED(); }

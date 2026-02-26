#pragma once

#if defined(__CUDACC__) && defined(__NVCC__) && !defined(__CUDACC_RELAXED_CONSTEXPR__)
#error                                                                                             \
    "smallgwn requires NVCC flag --expt-relaxed-constexpr. Link gwn::smallgwn in CMake or add the flag manually."
#endif

#include "gwn_assert.cuh"
#include "gwn_bvh.cuh"
#include "gwn_bvh_facade.cuh"
#include "gwn_bvh_refit.cuh"
#include "gwn_bvh_topology_build.cuh"
#include "gwn_geometry.cuh"
#include "gwn_kernel_utils.cuh"
#include "gwn_query.cuh"
#include "gwn_utils.cuh"

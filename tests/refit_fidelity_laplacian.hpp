/// \file refit_fidelity_laplacian.hpp
/// \brief libigl frame generation for refit fidelity validation.

#pragma once

#include <cstddef>
#include <vector>

namespace gwn::tests {

struct gwn_refit_fidelity_frame {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

/// \brief Generate Laplacian-smoothed vertex frames while keeping triangle
///        indices unchanged.
///
/// \param vx,vy,vz Vertex positions.
/// \param fi0,fi1,fi2 Triangle indices.
/// \param vertex_count Number of vertices.
/// \param tri_count Number of triangles.
/// \param frame_count Number of output frames.
/// \param smoothing_iterations Iterations applied before each output frame.
/// \param smoothing_lambda Blend weight for each explicit smoothing step.
/// \return Smoothed frames in temporal order.
std::vector<gwn_refit_fidelity_frame> gwn_generate_laplacian_smoothing_frames(
    float const *vx, float const *vy, float const *vz, std::size_t vertex_count, int const *fi0,
    int const *fi1, int const *fi2, std::size_t tri_count, std::size_t frame_count,
    int smoothing_iterations, float smoothing_lambda
);

} // namespace gwn::tests

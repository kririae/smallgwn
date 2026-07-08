/// \file refit_fidelity_laplacian.cpp
/// \brief libigl Laplacian smoothing helper for refit fidelity validation.

#include "refit_fidelity_laplacian.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <igl/adjacency_matrix.h>

namespace gwn::tests {

namespace {

using MatV = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatF = Eigen::MatrixXi;
using SparseI = Eigen::SparseMatrix<int, Eigen::RowMajor>;

MatV make_vertices(
    float const *vx, float const *vy, float const *vz, std::size_t const vertex_count
) {
    MatV V(static_cast<int>(vertex_count), 3);
    for (int row = 0; row < V.rows(); ++row) {
        V(row, 0) = vx[row];
        V(row, 1) = vy[row];
        V(row, 2) = vz[row];
    }
    return V;
}

MatF make_faces(int const *fi0, int const *fi1, int const *fi2, std::size_t const tri_count) {
    MatF F(static_cast<int>(tri_count), 3);
    for (int row = 0; row < F.rows(); ++row) {
        F(row, 0) = fi0[row];
        F(row, 1) = fi1[row];
        F(row, 2) = fi2[row];
    }
    return F;
}

gwn_refit_fidelity_frame make_frame(MatV const &V) {
    gwn_refit_fidelity_frame frame;
    frame.x.resize(static_cast<std::size_t>(V.rows()));
    frame.y.resize(static_cast<std::size_t>(V.rows()));
    frame.z.resize(static_cast<std::size_t>(V.rows()));
    for (int row = 0; row < V.rows(); ++row) {
        frame.x[static_cast<std::size_t>(row)] = V(row, 0);
        frame.y[static_cast<std::size_t>(row)] = V(row, 1);
        frame.z[static_cast<std::size_t>(row)] = V(row, 2);
    }
    return frame;
}

void smooth_once(SparseI const &A, float const lambda, MatV &V) {
    MatV next = V;
    for (int row = 0; row < A.rows(); ++row) {
        Eigen::RowVector3f sum = Eigen::RowVector3f::Zero();
        int degree = 0;
        for (SparseI::InnerIterator it(A, row); it; ++it) {
            sum += V.row(it.col());
            ++degree;
        }
        if (degree > 0) {
            Eigen::RowVector3f const average = sum / static_cast<float>(degree);
            next.row(row) = (1.0f - lambda) * V.row(row) + lambda * average;
        }
    }
    V = next;
}

} // namespace

std::vector<gwn_refit_fidelity_frame> gwn_generate_laplacian_smoothing_frames(
    float const *vx, float const *vy, float const *vz, std::size_t const vertex_count,
    int const *fi0, int const *fi1, int const *fi2, std::size_t const tri_count,
    std::size_t const frame_count, int const smoothing_iterations, float const smoothing_lambda
) {
    MatV V = make_vertices(vx, vy, vz, vertex_count);
    MatF const F = make_faces(fi0, fi1, fi2, tri_count);

    Eigen::SparseMatrix<int> adjacency;
    igl::adjacency_matrix(F, adjacency);
    SparseI const A = adjacency;

    int const iterations = std::max(smoothing_iterations, 1);
    float const lambda = std::clamp(smoothing_lambda, 0.0f, 1.0f);

    std::vector<gwn_refit_fidelity_frame> frames;
    frames.reserve(frame_count);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        for (int iter = 0; iter < iterations; ++iter)
            smooth_once(A, lambda, V);
        frames.push_back(make_frame(V));
    }
    return frames;
}

} // namespace gwn::tests

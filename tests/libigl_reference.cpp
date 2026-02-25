/// \file libigl_reference.cpp
/// \brief libigl-based reference implementations for SDF parity tests.
///
/// Compiled by the host C++ compiler (not nvcc) to avoid SIMD/intrinsic
/// incompatibilities in libigl's fast winding number code.

#include "libigl_reference.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>

namespace gwn::tests {

namespace {

using MatV = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatF = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using VecS = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using VecI = Eigen::Matrix<int, Eigen::Dynamic, 1>;

// Build Eigen V/F matrices from SoA vertex + index arrays.
void build_eigen_mesh(
    float const *vx, float const *vy, float const *vz, std::size_t vertex_count, int const *fi0,
    int const *fi1, int const *fi2, std::size_t tri_count, MatV &V, MatF &F
) {
    int const nv = static_cast<int>(vertex_count);
    int const nf = static_cast<int>(tri_count);
    V.resize(nv, 3);
    for (int r = 0; r < nv; ++r) {
        V(r, 0) = vx[r];
        V(r, 1) = vy[r];
        V(r, 2) = vz[r];
    }
    F.resize(nf, 3);
    for (int r = 0; r < nf; ++r) {
        F(r, 0) = fi0[r];
        F(r, 1) = fi1[r];
        F(r, 2) = fi2[r];
    }
}

Eigen::MatrixXf
build_query_matrix(float const *qx, float const *qy, float const *qz, std::size_t n) {
    Eigen::MatrixXf P(static_cast<int>(n), 3);
    for (std::size_t i = 0; i < n; ++i) {
        int const r = static_cast<int>(i);
        P(r, 0) = qx[i];
        P(r, 1) = qy[i];
        P(r, 2) = qz[i];
    }
    return P;
}

} // namespace

std::vector<float> libigl_unsigned_distance(
    float const *vx, float const *vy, float const *vz, std::size_t vertex_count, int const *fi0,
    int const *fi1, int const *fi2, std::size_t tri_count, float const *qx, float const *qy,
    float const *qz, std::size_t n
) {
    MatV V;
    MatF F;
    build_eigen_mesh(vx, vy, vz, vertex_count, fi0, fi1, fi2, tri_count, V, F);

    Eigen::MatrixXf P = build_query_matrix(qx, qy, qz, n);

    VecS sqrD;
    VecI I;
    Eigen::MatrixXf C;
    igl::point_mesh_squared_distance(P, V, F, sqrD, I, C);

    std::vector<float> result(n);
    for (std::size_t i = 0; i < n; ++i)
        result[i] = std::sqrt(sqrD(static_cast<int>(i)));
    return result;
}

std::vector<float> libigl_signed_distance(
    float const *vx, float const *vy, float const *vz, std::size_t vertex_count, int const *fi0,
    int const *fi1, int const *fi2, std::size_t tri_count, float const *qx, float const *qy,
    float const *qz, std::size_t n
) {
    MatV V;
    MatF F;
    build_eigen_mesh(vx, vy, vz, vertex_count, fi0, fi1, fi2, tri_count, V, F);

    Eigen::MatrixXf P = build_query_matrix(qx, qy, qz, n);

    VecS S;
    VecI I;
    Eigen::MatrixXf C;
    Eigen::MatrixXf N;
    igl::signed_distance(P, V, F, igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S, I, C, N);

    std::vector<float> result(n);
    for (std::size_t i = 0; i < n; ++i)
        result[i] = S(static_cast<int>(i));
    return result;
}

} // namespace gwn::tests

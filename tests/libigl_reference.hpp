/// \file libigl_reference.hpp
/// \brief libigl-based reference computations for SDF parity tests.
///
/// These declarations are implemented in libigl_reference.cpp (compiled
/// by the host C++ compiler) so that libigl headers — which use SIMD
/// intrinsics incompatible with nvcc — are kept out of CUDA translation
/// units.

#pragma once

#include <cstddef>
#include <vector>

namespace gwn::tests {

/// Compute unsigned (Euclidean) distance from each query to the mesh
/// using libigl's \c point_mesh_squared_distance and taking the sqrt.
///
/// \param vx,vy,vz  Vertex positions (size = #vertices).
/// \param fi0,fi1,fi2  Triangle indices (size = #triangles).
/// \param qx,qy,qz  Query points (size = \p n).
/// \param n  Number of queries.
/// \return  Vector of unsigned distances (size = \p n).
std::vector<float> libigl_unsigned_distance(
    float const *vx, float const *vy, float const *vz, std::size_t vertex_count, int const *fi0,
    int const *fi1, int const *fi2, std::size_t tri_count, float const *qx, float const *qy,
    float const *qz, std::size_t n
);

/// Compute signed distance from each query to the mesh using libigl's
/// \c signed_distance with \c SIGNED_DISTANCE_TYPE_WINDING_NUMBER.
///
/// \param vx,vy,vz  Vertex positions (size = #vertices).
/// \param fi0,fi1,fi2  Triangle indices (size = #triangles).
/// \param qx,qy,qz  Query points (size = \p n).
/// \param n  Number of queries.
/// \return  Vector of signed distances (size = \p n).
std::vector<float> libigl_signed_distance(
    float const *vx, float const *vy, float const *vz, std::size_t vertex_count, int const *fi0,
    int const *fi1, int const *fi2, std::size_t tri_count, float const *qx, float const *qy,
    float const *qz, std::size_t n
);

} // namespace gwn::tests

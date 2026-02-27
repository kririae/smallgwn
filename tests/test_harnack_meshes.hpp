#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "test_utils.hpp"

namespace gwn::tests {

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;

// ---------------------------------------------------------------------------
// Mesh definitions shared across unit / integration / behavior-match tests.
// ---------------------------------------------------------------------------

struct OctahedronMesh {
    static constexpr std::size_t Nv = 6;
    static constexpr std::size_t Nt = 8;
    std::array<Real, 6> vx{1, -1, 0, 0, 0, 0};
    std::array<Real, 6> vy{0, 0, 1, -1, 0, 0};
    std::array<Real, 6> vz{0, 0, 0, 0, 1, -1};
    std::array<Index, 8> i0{0, 2, 1, 3, 2, 1, 3, 0};
    std::array<Index, 8> i1{2, 1, 3, 0, 0, 2, 1, 3};
    std::array<Index, 8> i2{4, 4, 4, 4, 5, 5, 5, 5};
};

// Half-octahedron: 4 upper triangles (z > 0), non-watertight.
struct HalfOctahedronMesh {
    static constexpr std::size_t Nv = 5;
    static constexpr std::size_t Nt = 4;
    std::array<Real, 5> vx{1, -1, 0, 0, 0};
    std::array<Real, 5> vy{0, 0, 1, -1, 0};
    std::array<Real, 5> vz{0, 0, 0, 0, 1};
    std::array<Index, 4> i0{0, 2, 1, 3};
    std::array<Index, 4> i1{2, 1, 3, 0};
    std::array<Index, 4> i2{4, 4, 4, 4};
};

// Closed cube: 8 vertices at (±1,±1,±1), 12 triangles, 0 singular edges.
// Vertex layout:
//   0=(-1,-1,-1) 1=(1,-1,-1) 2=(1,1,-1) 3=(-1,1,-1)
//   4=(-1,-1, 1) 5=(1,-1, 1) 6=(1,1, 1) 7=(-1,1, 1)
struct CubeMesh {
    static constexpr std::size_t Nv = 8;
    static constexpr std::size_t Nt = 12;
    std::array<Real, 8> vx{-1, 1, 1, -1, -1, 1, 1, -1};
    std::array<Real, 8> vy{-1, -1, 1, 1, -1, -1, 1, 1};
    std::array<Real, 8> vz{-1, -1, -1, -1, 1, 1, 1, 1};
    // z=-1: 0,2,1 + 0,3,2   z=+1: 4,5,6 + 4,6,7
    // x=-1: 0,4,7 + 0,7,3   x=+1: 1,2,6 + 1,6,5
    // y=-1: 0,1,5 + 0,5,4   y=+1: 3,7,6 + 3,6,2
    std::array<Index, 12> i0{0,0, 4,4, 0,0, 1,1, 0,0, 3,3};
    std::array<Index, 12> i1{2,3, 5,6, 4,7, 2,6, 1,5, 7,6};
    std::array<Index, 12> i2{1,2, 6,7, 7,3, 6,5, 5,4, 6,2};
};

// Open cube: z=+1 face removed → 10 triangles, 4 singular edges.
struct OpenCubeMesh {
    static constexpr std::size_t Nv = 8;
    static constexpr std::size_t Nt = 10;
    std::array<Real, 8> vx{-1, 1, 1, -1, -1, 1, 1, -1};
    std::array<Real, 8> vy{-1, -1, 1, 1, -1, -1, 1, 1};
    std::array<Real, 8> vz{-1, -1, -1, -1, 1, 1, 1, 1};
    // Same as CubeMesh but without z=+1 face (triangles 4,5,6 and 4,6,7).
    std::array<Index, 10> i0{0,0, 0,0, 1,1, 0,0, 3,3};
    std::array<Index, 10> i1{2,3, 4,7, 2,6, 1,5, 7,6};
    std::array<Index, 10> i2{1,2, 7,3, 6,5, 5,4, 6,2};
};

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

inline Real half_octahedron_face_z(Real const x, Real const y) {
    return Real(1) - std::abs(x) - std::abs(y);
}

inline void generate_sphere_rays(
    Real const radius, int const n_lat, int const n_lon,
    std::vector<Real> &ox, std::vector<Real> &oy, std::vector<Real> &oz,
    std::vector<Real> &dx, std::vector<Real> &dy, std::vector<Real> &dz
) {
    constexpr Real pi = Real(3.14159265358979323846);
    for (int la = 1; la < n_lat; ++la) {
        Real const theta = pi * Real(la) / Real(n_lat);
        Real const st = std::sin(theta);
        Real const ct = std::cos(theta);
        for (int lo = 0; lo < n_lon; ++lo) {
            Real const phi = Real(2) * pi * Real(lo) / Real(n_lon);
            Real const x = radius * st * std::cos(phi);
            Real const y = radius * st * std::sin(phi);
            Real const z = radius * ct;
            ox.push_back(x);
            oy.push_back(y);
            oz.push_back(z);
            Real const inv_r = Real(1) / radius;
            dx.push_back(-x * inv_r);
            dy.push_back(-y * inv_r);
            dz.push_back(-z * inv_r);
        }
    }
}

inline Real wrapped_angle_residual(
    Real const winding, Real const target_winding = Real(0.5)
) {
    constexpr Real k_pi = Real(3.14159265358979323846);
    Real const period = Real(4) * k_pi;
    Real const omega = period * winding;
    Real const target_omega = period * target_winding;
    Real const val = omega - target_omega - period * std::floor((omega - target_omega) / period);
    return std::min(val, period - val);
}

} // namespace gwn::tests

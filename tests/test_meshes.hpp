#pragma once

#include <array>
#include <cstdint>

namespace gwn::tests {

using Real = float;
using Index = std::uint32_t;

// Small analytic meshes shared by local query and boundary-chain tests.

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

// Closed cube: 8 vertices at (±1,±1,±1), 12 triangles, and an empty boundary chain.
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
    std::array<Index, 12> i0{0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3};
    std::array<Index, 12> i1{2, 3, 5, 6, 4, 7, 2, 6, 1, 5, 7, 6};
    std::array<Index, 12> i2{1, 2, 6, 7, 7, 3, 6, 5, 5, 4, 6, 2};
};

// Open cube: the z=+1 face is removed, leaving four boundary-chain edges.
struct OpenCubeMesh {
    static constexpr std::size_t Nv = 8;
    static constexpr std::size_t Nt = 10;
    std::array<Real, 8> vx{-1, 1, 1, -1, -1, 1, 1, -1};
    std::array<Real, 8> vy{-1, -1, 1, 1, -1, -1, 1, 1};
    std::array<Real, 8> vz{-1, -1, -1, -1, 1, 1, 1, 1};
    // Same as CubeMesh but without z=+1 face (triangles 4,5,6 and 4,6,7).
    std::array<Index, 10> i0{0, 0, 0, 0, 1, 1, 0, 0, 3, 3};
    std::array<Index, 10> i1{2, 3, 4, 7, 2, 6, 1, 5, 7, 6};
    std::array<Index, 10> i2{1, 2, 7, 3, 6, 5, 5, 4, 6, 2};
};

} // namespace gwn::tests

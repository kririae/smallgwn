#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace gwn::tests {

using Real = float;
using Index = std::uint32_t;

struct HostMesh {
    std::vector<Real> vertex_x;
    std::vector<Real> vertex_y;
    std::vector<Real> vertex_z;
    std::vector<Index> tri_i0;
    std::vector<Index> tri_i1;
    std::vector<Index> tri_i2;
};

[[nodiscard]] std::optional<HostMesh> load_ply_mesh(std::filesystem::path const &path);

[[nodiscard]] std::vector<std::filesystem::path>
collect_ply_model_paths(std::filesystem::path const &model_dir);

void set_mesh_directory(std::filesystem::path directory);

[[nodiscard]] std::vector<std::filesystem::path> collect_mesh_paths();

} // namespace gwn::tests

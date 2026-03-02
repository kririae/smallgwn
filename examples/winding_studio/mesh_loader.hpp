#pragma once

#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace winding_studio {

struct LoadedMesh {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
};

[[nodiscard]] inline bool normalize_mesh(LoadedMesh &mesh, std::string &error) {
    if (mesh.positions.empty() || mesh.indices.empty()) {
        error = "mesh data is empty";
        return false;
    }
    if (mesh.positions.size() % 3 != 0) {
        error = "vertex array must be xyz-interleaved";
        return false;
    }

    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < mesh.positions.size(); i += 3) {
        float const x = mesh.positions[i + 0];
        float const y = mesh.positions[i + 1];
        float const z = mesh.positions[i + 2];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
    }

    float const cx = 0.5f * (min_x + max_x);
    float const cy = 0.5f * (min_y + max_y);
    float const cz = 0.5f * (min_z + max_z);
    float const extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    if (!(extent > 0.0f)) {
        error = "mesh bounding box has zero extent";
        return false;
    }
    float const scale = 2.0f / extent;
    for (std::size_t i = 0; i < mesh.positions.size(); i += 3) {
        mesh.positions[i + 0] = (mesh.positions[i + 0] - cx) * scale;
        mesh.positions[i + 1] = (mesh.positions[i + 1] - cy) * scale;
        mesh.positions[i + 2] = (mesh.positions[i + 2] - cz) * scale;
    }
    return true;
}

[[nodiscard]] inline bool
load_mesh_from_file(std::string const &mesh_path, LoadedMesh &mesh, std::string &error) {
    mesh = LoadedMesh{};
    std::filesystem::path const path(mesh_path);
    if (!std::filesystem::is_regular_file(path)) {
        error = "mesh file not found: " + path.string();
        return false;
    }

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    if (!igl::read_triangle_mesh(path.string(), vertices, faces)) {
        error = "libigl failed to parse mesh: " + path.string();
        return false;
    }
    if (vertices.cols() != 3 || faces.cols() != 3 || vertices.rows() == 0 || faces.rows() == 0) {
        error = "mesh has invalid shape (expected Nx3, Mx3): " + path.string();
        return false;
    }

    mesh.positions.resize(static_cast<std::size_t>(vertices.rows()) * 3u);
    for (Eigen::Index i = 0; i < vertices.rows(); ++i) {
        mesh.positions[static_cast<std::size_t>(i) * 3u + 0u] = static_cast<float>(vertices(i, 0));
        mesh.positions[static_cast<std::size_t>(i) * 3u + 1u] = static_cast<float>(vertices(i, 1));
        mesh.positions[static_cast<std::size_t>(i) * 3u + 2u] = static_cast<float>(vertices(i, 2));
    }

    mesh.indices.resize(static_cast<std::size_t>(faces.rows()) * 3u);
    for (Eigen::Index i = 0; i < faces.rows(); ++i) {
        for (int k = 0; k < 3; ++k) {
            Eigen::Index const value = faces(i, k);
            if (value < 0 ||
                value > static_cast<Eigen::Index>(std::numeric_limits<std::uint32_t>::max()) ||
                value >= vertices.rows()) {
                error = "mesh has invalid face indices: " + path.string();
                return false;
            }
            mesh.indices[static_cast<std::size_t>(i) * 3u + static_cast<std::size_t>(k)] =
                static_cast<std::uint32_t>(value);
        }
    }
    return normalize_mesh(mesh, error);
}

} // namespace winding_studio

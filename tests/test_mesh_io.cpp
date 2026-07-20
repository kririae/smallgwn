#include "test_mesh_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>

namespace {

std::filesystem::path g_mesh_directory;

} // namespace

namespace gwn::tests {

std::vector<std::filesystem::path> collect_ply_model_paths(std::filesystem::path const &model_dir) {
    std::vector<std::filesystem::path> model_paths;
    for (auto const &entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file())
            continue;
        std::filesystem::path const path = entry.path();
        if (path.extension() == ".ply")
            model_paths.push_back(path);
    }
    std::sort(model_paths.begin(), model_paths.end());
    return model_paths;
}

void set_mesh_directory(std::filesystem::path directory) {
    g_mesh_directory = std::move(directory);
}

std::vector<std::filesystem::path> collect_mesh_paths() {
    if (g_mesh_directory.empty())
        return {};
    return collect_ply_model_paths(g_mesh_directory);
}

std::optional<HostMesh> load_ply_mesh(std::filesystem::path const &path) {
    if (path.extension() != ".ply")
        return std::nullopt;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> vertices;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> triangles;
    if (!igl::read_triangle_mesh(path.string(), vertices, triangles) || vertices.rows() == 0 ||
        triangles.rows() == 0) {
        return std::nullopt;
    }

    auto const vertex_count = static_cast<std::size_t>(vertices.rows());
    if (vertex_count > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
        return std::nullopt;

    HostMesh mesh;
    mesh.vertex_x.reserve(vertex_count);
    mesh.vertex_y.reserve(vertex_count);
    mesh.vertex_z.reserve(vertex_count);
    for (Eigen::Index row = 0; row < vertices.rows(); ++row) {
        double const x = vertices(row, 0);
        double const y = vertices(row, 1);
        double const z = vertices(row, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            return std::nullopt;
        mesh.vertex_x.push_back(static_cast<Real>(x));
        mesh.vertex_y.push_back(static_cast<Real>(y));
        mesh.vertex_z.push_back(static_cast<Real>(z));
    }

    auto const triangle_count = static_cast<std::size_t>(triangles.rows());
    mesh.tri_i0.reserve(triangle_count);
    mesh.tri_i1.reserve(triangle_count);
    mesh.tri_i2.reserve(triangle_count);
    for (Eigen::Index row = 0; row < triangles.rows(); ++row) {
        int const i0 = triangles(row, 0);
        int const i1 = triangles(row, 1);
        int const i2 = triangles(row, 2);
        if (i0 < 0 || i1 < 0 || i2 < 0 || static_cast<std::size_t>(i0) >= vertex_count ||
            static_cast<std::size_t>(i1) >= vertex_count ||
            static_cast<std::size_t>(i2) >= vertex_count) {
            return std::nullopt;
        }
        mesh.tri_i0.push_back(static_cast<Index>(i0));
        mesh.tri_i1.push_back(static_cast<Index>(i1));
        mesh.tri_i2.push_back(static_cast<Index>(i2));
    }

    return mesh;
}

} // namespace gwn::tests

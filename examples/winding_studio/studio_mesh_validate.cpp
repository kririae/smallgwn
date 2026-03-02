#include "studio_mesh_validate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace winding_studio::app {

bool validate_mesh_data(MeshData const &mesh, std::string &error) noexcept {
    error.clear();

    if (mesh.positions.empty()) {
        error = "mesh positions are empty";
        return false;
    }
    if (mesh.indices.empty()) {
        error = "mesh indices are empty";
        return false;
    }
    if ((mesh.positions.size() % 3u) != 0u) {
        error = "mesh positions must be xyz-interleaved";
        return false;
    }
    if ((mesh.indices.size() % 3u) != 0u) {
        error = "mesh indices must be triangle-aligned";
        return false;
    }

    std::size_t const vertex_count = mesh.positions.size() / 3u;
    auto const max_it = std::max_element(mesh.indices.begin(), mesh.indices.end());
    if (max_it != mesh.indices.end() && static_cast<std::size_t>(*max_it) >= vertex_count) {
        error = "mesh indices reference out-of-range vertices";
        return false;
    }

    return true;
}

} // namespace winding_studio::app

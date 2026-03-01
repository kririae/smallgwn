#include "ws_core.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <tbb/parallel_for.h>

#include "ws_math.hpp"

namespace winding_studio::app {

[[nodiscard]] winding_studio::HostMeshSoA to_host_mesh_soa(MeshData const &mesh) {
    winding_studio::HostMeshSoA out{};
    std::size_t const vertex_count = mesh.positions.size() / 3u;
    std::size_t const tri_count = mesh.indices.size() / 3u;
    out.vx.resize(vertex_count);
    out.vy.resize(vertex_count);
    out.vz.resize(vertex_count);
    out.i0.resize(tri_count);
    out.i1.resize(tri_count);
    out.i2.resize(tri_count);

    tbb::parallel_for(std::size_t{0}, vertex_count, [&](std::size_t const i) {
        out.vx[i] = mesh.positions[i * 3u + 0u];
        out.vy[i] = mesh.positions[i * 3u + 1u];
        out.vz[i] = mesh.positions[i * 3u + 2u];
    });
    tbb::parallel_for(std::size_t{0}, tri_count, [&](std::size_t const i) {
        out.i0[i] = mesh.indices[i * 3u + 0u];
        out.i1[i] = mesh.indices[i * 3u + 1u];
        out.i2[i] = mesh.indices[i * 3u + 2u];
    });
    return out;
}

[[nodiscard]] MeshData to_mesh_data(winding_studio::LoadedMesh const &mesh) {
    MeshData out{};
    out.positions = mesh.positions;
    out.indices = mesh.indices;
    return out;
}

[[nodiscard]] std::size_t triangle_count(MeshData const &mesh) { return mesh.indices.size() / 3u; }

[[nodiscard]] MeshData build_default_mesh() {
    MeshData mesh{};
    mesh.positions = {
        1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    mesh.indices = {
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
    };
    return mesh;
}

[[nodiscard]] MeshData build_closed_octa_mesh() {
    MeshData mesh{};
    mesh.positions = {
        1.0f, 0.0f,  0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f, -1.0f,
    };
    mesh.indices = {
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
    };
    return mesh;
}

[[nodiscard]] char const *view_mode_name(ViewMode const view_mode) {
    switch (view_mode) {
    case ViewMode::k_split: return "Split";
    case ViewMode::k_raster: return "Raster";
    case ViewMode::k_harnack: return "Harnack";
    case ViewMode::k_voxel: return "Voxel";
    }
    return "Unknown";
}

[[nodiscard]] bool has_valid_mesh_index(AppState const &state, int const index) {
    return index >= 0 && static_cast<std::size_t>(index) < state.mesh_library.size();
}

[[nodiscard]] bool has_active_mesh(AppState const &state) {
    return has_valid_mesh_index(state, state.active_mesh_index);
}

[[nodiscard]] std::string mesh_list_label(MeshLibraryEntry const &entry, bool const is_active) {
    std::ostringstream oss;
    if (is_active)
        oss << "> ";
    oss << entry.name << " (" << entry.triangle_count << " tris)";
    if (entry.is_builtin)
        oss << "  [Built-in]";
    return oss.str();
}

[[nodiscard]] winding_studio::CameraFrame
make_harnack_camera_frame(AppState const &state, int const width, int const height) {
    CameraBasis const basis = build_camera_basis(state);
    winding_studio::CameraFrame camera{};
    camera.origin_x = basis.eye.x;
    camera.origin_y = basis.eye.y;
    camera.origin_z = basis.eye.z;
    camera.forward_x = basis.forward.x;
    camera.forward_y = basis.forward.y;
    camera.forward_z = basis.forward.z;
    camera.right_x = basis.right.x;
    camera.right_y = basis.right.y;
    camera.right_z = basis.right.z;
    camera.up_x = basis.ortho_up.x;
    camera.up_y = basis.ortho_up.y;
    camera.up_z = basis.ortho_up.z;
    camera.tan_half_fov = std::tan(0.5f * 45.0f * (k_pi / 180.0f));
    camera.aspect = static_cast<float>(width) / static_cast<float>(std::max(height, 1));
    camera.width = width;
    camera.height = height;
    return camera;
}

} // namespace winding_studio::app

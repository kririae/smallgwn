#pragma once

#include "harnack_tracer.hpp"
#include "mesh_loader.hpp"
#include "ws_types.hpp"

#include <string>

namespace winding_studio::app {

[[nodiscard]] winding_studio::HostMeshSoA to_host_mesh_soa(MeshData const &mesh);
[[nodiscard]] MeshData to_mesh_data(winding_studio::LoadedMesh const &mesh);
[[nodiscard]] std::size_t triangle_count(MeshData const &mesh);

[[nodiscard]] MeshData build_default_mesh();
[[nodiscard]] MeshData build_closed_octa_mesh();

[[nodiscard]] char const *view_mode_name(ViewMode view_mode);
[[nodiscard]] bool has_valid_mesh_index(AppState const &state, int index);
[[nodiscard]] bool has_active_mesh(AppState const &state);
[[nodiscard]] std::string mesh_list_label(MeshLibraryEntry const &entry, bool is_active);

[[nodiscard]] winding_studio::CameraFrame
make_harnack_camera_frame(AppState const &state, int width, int height);

} // namespace winding_studio::app

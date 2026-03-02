#pragma once

#include "studio_state.hpp"

#include <string>

namespace winding_studio {
struct HostMeshSoA;
struct LoadedMesh;
} // namespace winding_studio

namespace winding_studio::app {

/**
 * @brief Convert local MeshData to smallgwn-oriented SoA host buffers.
 */
[[nodiscard]] winding_studio::HostMeshSoA to_host_mesh_soa(MeshData const &mesh);

/**
 * @brief Convert loader output into local MeshData.
 */
[[nodiscard]] MeshData to_mesh_data(winding_studio::LoadedMesh const &mesh);

/**
 * @brief Return triangle count (indices.size()/3).
 */
[[nodiscard]] std::size_t triangle_count(MeshData const &mesh);

/**
 * @brief Built-in half-octahedron starter mesh.
 */
[[nodiscard]] MeshData build_default_mesh();

/**
 * @brief Built-in closed octahedron mesh.
 */
[[nodiscard]] MeshData build_closed_octa_mesh();

/**
 * @brief Human-readable view mode name.
 */
[[nodiscard]] char const *view_mode_name(ViewMode view_mode);

/**
 * @brief Check whether mesh index is valid in current mesh library.
 */
[[nodiscard]] bool has_valid_mesh_index(AppState const &state, int index);

/**
 * @brief Check whether current state has an active mesh.
 */
[[nodiscard]] bool has_active_mesh(AppState const &state);

/**
 * @brief Build listbox label for a mesh entry.
 */
[[nodiscard]] std::string mesh_list_label(MeshLibraryEntry const &entry, bool is_active);

/**
 * @brief Add mesh to library with automatic unique naming.
 */
[[nodiscard]] int add_mesh_to_library(
    AppState &state, MeshData mesh, std::string name, bool is_builtin
);

/**
 * @brief Remove mesh entry and keep selected/active indices consistent.
 */
void remove_mesh_from_library(AppState &state, int index);

/**
 * @brief Reset state fields that require an active mesh.
 */
void clear_active_mesh_state(AppState &state);

} // namespace winding_studio::app

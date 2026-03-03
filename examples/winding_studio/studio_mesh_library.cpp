#include "studio_mesh_library.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <tbb/parallel_for.h>

#include "harnack_tracer.hpp"
#include "mesh_loader.hpp"

namespace winding_studio::app {

winding_studio::HostMeshSoA to_host_mesh_soa(MeshData const &mesh) {
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

MeshData to_mesh_data(winding_studio::LoadedMesh const &mesh) {
    MeshData out{};
    out.positions = mesh.positions;
    out.indices = mesh.indices;
    return out;
}

std::size_t triangle_count(MeshData const &mesh) { return mesh.indices.size() / 3u; }

MeshData build_default_mesh() {
    MeshData mesh{};
    mesh.positions = {
        1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    mesh.indices = {
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
    };
    return mesh;
}

MeshData build_closed_octa_mesh() {
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

char const *view_mode_name(ViewMode const view_mode) {
    switch (view_mode) {
    case ViewMode::k_split: return "Split";
    case ViewMode::k_raster: return "Raster";
    case ViewMode::k_harnack: return "Harnack";
    case ViewMode::k_voxel: return "Voxel";
    }
    return "Unknown";
}

bool has_valid_mesh_index(AppState const &state, int const index) {
    return index >= 0 && static_cast<std::size_t>(index) < state.mesh_library.size();
}

bool has_active_mesh(AppState const &state) {
    return has_valid_mesh_index(state, state.active_mesh_index);
}

std::string mesh_list_label(MeshLibraryEntry const &entry, bool const is_active) {
    std::ostringstream oss;
    if (is_active)
        oss << "> ";
    oss << entry.name << " (" << entry.triangle_count << " tris)";
    if (entry.is_builtin)
        oss << "  [Built-in]";
    return oss.str();
}

static std::string make_unique_mesh_name(AppState const &state, std::string name) {
    if (name.empty())
        name = "Imported Mesh";
    auto const name_exists = [&](std::string const &candidate) {
        for (MeshLibraryEntry const &entry : state.mesh_library)
            if (entry.name == candidate)
                return true;
        return false;
    };
    if (!name_exists(name))
        return name;

    std::string const base = name;
    int suffix = 2;
    while (true) {
        std::ostringstream oss;
        oss << base << " (" << suffix << ")";
        std::string const candidate = oss.str();
        if (!name_exists(candidate))
            return candidate;
        ++suffix;
    }
}

int add_mesh_to_library(AppState &state, MeshData mesh, std::string name, bool const is_builtin) {
    MeshLibraryEntry entry{};
    entry.name = make_unique_mesh_name(state, std::move(name));
    entry.triangle_count = triangle_count(mesh);
    entry.mesh = std::move(mesh);
    entry.is_builtin = is_builtin;
    state.mesh_library.push_back(std::move(entry));
    return static_cast<int>(state.mesh_library.size() - 1u);
}

void clear_active_mesh_state(AppState &state) {
    state.active_mesh_index = -1;
    state.active_mesh_name = "None";
    state.triangle_count = 0;
    state.force_harnack_refresh = false;
    state.force_voxel_refresh = false;
    state.harnack_hit_count = 0;
    state.harnack_pixel_count = 0;
    state.last_harnack_ms = 0.0f;
    state.voxel_occupied_count = 0;
    state.voxel_requested_count = 0;
    state.voxel_truncated = false;
    state.voxel_grid_total = 0;
    state.voxel_grid_nx = 1;
    state.voxel_grid_ny = 1;
    state.voxel_grid_nz = 1;
    state.last_voxel_ms = 0.0f;
    state.mesh_bounds = winding_studio::voxel::MeshBounds{};
}

void remove_mesh_from_library(AppState &state, int const index) {
    if (!has_valid_mesh_index(state, index))
        return;

    MeshLibraryEntry removed = std::move(state.mesh_library[static_cast<std::size_t>(index)]);
    bool const removed_active = (index == state.active_mesh_index);
    state.mesh_library.erase(state.mesh_library.begin() + index);

    auto const adjust_index_after_erase = [&](int &value) {
        if (value > index)
            --value;
        else if (value == index)
            value = -1;
    };
    adjust_index_after_erase(state.selected_mesh_index);
    adjust_index_after_erase(state.active_mesh_index);

    if (state.mesh_library.empty()) {
        clear_active_mesh_state(state);
        state.selected_mesh_index = -1;
        state.status_line = "Removed mesh: " + removed.name + ". Library is empty.";
        return;
    }

    if (removed_active) {
        clear_active_mesh_state(state);
        state.selected_mesh_index =
            std::clamp(index, 0, static_cast<int>(state.mesh_library.size()) - 1);
        state.status_line = "Removed mesh: " + removed.name + ". No active mesh.";
        return;
    }

    if (!has_valid_mesh_index(state, state.selected_mesh_index)) {
        state.selected_mesh_index =
            std::clamp(index, 0, static_cast<int>(state.mesh_library.size()) - 1);
    }
    state.status_line = "Removed mesh: " + removed.name;
}

} // namespace winding_studio::app

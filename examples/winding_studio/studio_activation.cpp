#include "studio_activation.hpp"

#include "studio_mesh_validate.hpp"

#include <exception>
#include <string>

namespace winding_studio::app {

namespace {

[[nodiscard]] bool restore_previous_uploads(
    AppState const &state, int const previous_active_index, MeshRenderer &renderer,
    winding_studio::HarnackTracer &harnack_tracer, winding_studio::Voxelizer &voxelizer,
    std::string &restore_error
) {
    restore_error.clear();
    if (!has_valid_mesh_index(state, previous_active_index))
        return true;

    MeshLibraryEntry const &previous_entry =
        state.mesh_library[static_cast<std::size_t>(previous_active_index)];

    std::string validate_error;
    if (!validate_mesh_data(previous_entry.mesh, validate_error)) {
        restore_error = "restore validation failed: " + validate_error;
        return false;
    }

    winding_studio::HostMeshSoA const previous_host = to_host_mesh_soa(previous_entry.mesh);
    std::string backend_error;
    if (!harnack_tracer.upload_mesh(previous_host, backend_error)) {
        restore_error = "restore harnack upload failed: " + backend_error;
        return false;
    }
    backend_error.clear();
    if (!voxelizer.upload_mesh(previous_host, backend_error)) {
        restore_error = "restore voxel upload failed: " + backend_error;
        return false;
    }

    try {
        renderer.upload_mesh(previous_entry.mesh);
    } catch (std::exception const &e) {
        restore_error = std::string("restore raster upload failed: ") + e.what();
        return false;
    }

    return true;
}

} // namespace

bool activate_mesh_by_index(
    AppState &state, int const index, std::string const &status_prefix, MeshRenderer &renderer,
    winding_studio::HarnackTracer &harnack_tracer, winding_studio::Voxelizer &voxelizer,
    std::string &error
) {
    error.clear();
    if (!has_valid_mesh_index(state, index)) {
        error = "mesh index is out of range";
        return false;
    }

    int const previous_active_index = state.active_mesh_index;
    MeshLibraryEntry const &entry = state.mesh_library[static_cast<std::size_t>(index)];

    std::string validate_error;
    if (!validate_mesh_data(entry.mesh, validate_error)) {
        error = "mesh validation failed: " + validate_error;
        return false;
    }

    winding_studio::HostMeshSoA const host_mesh = to_host_mesh_soa(entry.mesh);
    std::string tracer_error;
    if (!harnack_tracer.upload_mesh(host_mesh, tracer_error)) {
        std::string restore_error;
        if (!restore_previous_uploads(
                state, previous_active_index, renderer, harnack_tracer, voxelizer, restore_error
            )) {
            clear_active_mesh_state(state);
            error = "harnack upload failed: " + tracer_error + "; " + restore_error;
            return false;
        }
        error = "harnack upload failed: " + tracer_error;
        return false;
    }

    std::string voxel_error;
    if (!voxelizer.upload_mesh(host_mesh, voxel_error)) {
        std::string restore_error;
        if (!restore_previous_uploads(
                state, previous_active_index, renderer, harnack_tracer, voxelizer, restore_error
            )) {
            clear_active_mesh_state(state);
            error = "voxel upload failed: " + voxel_error + "; " + restore_error;
            return false;
        }
        error = "voxel upload failed: " + voxel_error;
        return false;
    }

    try {
        renderer.upload_mesh(entry.mesh);
    } catch (std::exception const &e) {
        std::string restore_error;
        if (!restore_previous_uploads(
                state, previous_active_index, renderer, harnack_tracer, voxelizer, restore_error
            )) {
            clear_active_mesh_state(state);
            error = std::string("raster upload failed: ") + e.what() + "; " + restore_error;
            return false;
        }
        error = std::string("raster upload failed: ") + e.what();
        return false;
    }

    state.active_mesh_index = index;
    state.selected_mesh_index = index;
    state.active_mesh_name = entry.name;
    state.triangle_count = entry.triangle_count;
    state.force_harnack_refresh = true;
    state.force_voxel_refresh = true;
    state.harnack_hit_count = 0;
    state.harnack_pixel_count = 0;
    state.voxel_occupied_count = 0;
    state.voxel_requested_count = 0;
    state.voxel_truncated = false;
    state.voxel_grid_total = 0;
    state.voxel_grid_nx = 1;
    state.voxel_grid_ny = 1;
    state.voxel_grid_nz = 1;
    state.mesh_bounds = winding_studio::voxel::compute_mesh_bounds(
        entry.mesh.positions.data(), entry.mesh.positions.size() / 3u
    );

    if (!status_prefix.empty())
        state.status_line = status_prefix + entry.name;
    return true;
}

} // namespace winding_studio::app

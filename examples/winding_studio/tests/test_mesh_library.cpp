#include <cassert>
#include <string>

#include "studio_mesh_library.hpp"

using namespace winding_studio::app;

int main() {
    AppState state{};

    MeshData mesh_a{};
    mesh_a.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    mesh_a.indices = {0u, 1u, 2u};

    MeshData mesh_b = mesh_a;

    int const a0 = add_mesh_to_library(state, mesh_a, "Mesh", false);
    int const a1 = add_mesh_to_library(state, mesh_b, "Mesh", false);
    assert(a0 == 0);
    assert(a1 == 1);
    assert(state.mesh_library[0].name == "Mesh");
    assert(state.mesh_library[1].name == "Mesh (2)");

    state.active_mesh_index = 1;
    state.selected_mesh_index = 1;
    state.active_mesh_name = state.mesh_library[1].name;
    state.triangle_count = state.mesh_library[1].triangle_count;

    remove_mesh_from_library(state, 1);
    assert(state.active_mesh_index == -1);
    assert(state.selected_mesh_index == 0);
    assert(state.active_mesh_name == "None");

    state.voxel_occupied_count = 10;
    state.voxel_requested_count = 15;
    state.voxel_truncated = true;
    clear_active_mesh_state(state);
    assert(state.voxel_occupied_count == 0);
    assert(state.voxel_requested_count == 0);
    assert(!state.voxel_truncated);

    return 0;
}

#include <cassert>
#include <string>

#include "studio_mesh_validate.hpp"

using namespace winding_studio::app;

int main() {
    {
        MeshData mesh{};
        mesh.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        mesh.indices = {0u, 1u, 2u};
        std::string error;
        assert(validate_mesh_data(mesh, error));
        assert(error.empty());
    }

    {
        MeshData mesh{};
        mesh.positions = {0.0f, 0.0f};
        mesh.indices = {0u, 1u, 2u};
        std::string error;
        assert(!validate_mesh_data(mesh, error));
        assert(!error.empty());
    }

    {
        MeshData mesh{};
        mesh.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        mesh.indices = {0u, 1u};
        std::string error;
        assert(!validate_mesh_data(mesh, error));
        assert(!error.empty());
    }

    {
        MeshData mesh{};
        mesh.positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        mesh.indices = {0u, 1u, 5u};
        std::string error;
        assert(!validate_mesh_data(mesh, error));
        assert(!error.empty());
    }

    return 0;
}

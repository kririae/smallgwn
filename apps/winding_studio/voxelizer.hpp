#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "harnack_tracer.hpp"
#include "voxel_grid_spec.hpp"

namespace winding_studio {

struct VoxelizeConfig {
    float target_winding{0.5f};
    float accuracy_scale{2.0f};
};

struct VoxelizeStats {
    std::size_t occupied_count{0};
    std::size_t total_voxels{0};
};

class Voxelizer final {
public:
    Voxelizer();
    ~Voxelizer();

    Voxelizer(Voxelizer &&other) noexcept;
    Voxelizer &operator=(Voxelizer &&other) noexcept;

    Voxelizer(Voxelizer const &) = delete;
    Voxelizer &operator=(Voxelizer const &) = delete;

    [[nodiscard]] bool has_mesh() const noexcept;

    [[nodiscard]] bool upload_mesh(HostMeshSoA const &mesh, std::string &error);
    [[nodiscard]] bool voxelize(
        voxel::VoxelGridSpec const &grid, VoxelizeConfig const &config, unsigned int gl_buffer_id,
        std::size_t gl_buffer_capacity, VoxelizeStats &out, std::string &error
    );

private:
    class Impl;
    Impl *impl_{nullptr};
};

} // namespace winding_studio

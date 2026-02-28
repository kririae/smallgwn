#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace winding_studio::voxel {

struct MeshBounds {
    float min_x{0.0f};
    float min_y{0.0f};
    float min_z{0.0f};
    float max_x{0.0f};
    float max_y{0.0f};
    float max_z{0.0f};
};

struct VoxelGridSpec {
    std::uint32_t nx{1};
    std::uint32_t ny{1};
    std::uint32_t nz{1};
    float origin_x{0.0f};
    float origin_y{0.0f};
    float origin_z{0.0f};
    float step_x{1.0f};
    float step_y{1.0f};
    float step_z{1.0f};
    float actual_dx{1.0f};
    std::uint64_t total_voxels{1};
};

[[nodiscard]] inline MeshBounds compute_mesh_bounds(
    float const *positions, std::size_t const vertex_count
) noexcept {
    MeshBounds bounds{};
    if (positions == nullptr || vertex_count == 0)
        return bounds;

    bounds.min_x = bounds.max_x = positions[0];
    bounds.min_y = bounds.max_y = positions[1];
    bounds.min_z = bounds.max_z = positions[2];

    for (std::size_t i = 1; i < vertex_count; ++i) {
        float const x = positions[i * 3u + 0u];
        float const y = positions[i * 3u + 1u];
        float const z = positions[i * 3u + 2u];
        bounds.min_x = std::min(bounds.min_x, x);
        bounds.max_x = std::max(bounds.max_x, x);
        bounds.min_y = std::min(bounds.min_y, y);
        bounds.max_y = std::max(bounds.max_y, y);
        bounds.min_z = std::min(bounds.min_z, z);
        bounds.max_z = std::max(bounds.max_z, z);
    }
    return bounds;
}

[[nodiscard]] inline std::uint64_t safe_mul_u64(std::uint64_t a, std::uint64_t b) noexcept {
    if (a == 0 || b == 0)
        return 0;
    if (a > (std::numeric_limits<std::uint64_t>::max() / b))
        return std::numeric_limits<std::uint64_t>::max();
    return a * b;
}

[[nodiscard]] inline std::uint32_t dim_from_extent(float const extent, float const dx) noexcept {
    float const clamped_extent = std::max(extent, 0.0f);
    if (!(dx > 0.0f))
        return 1u;
    long long const n = static_cast<long long>(std::ceil(clamped_extent / dx));
    return static_cast<std::uint32_t>(std::max<long long>(1, n));
}

[[nodiscard]] inline VoxelGridSpec make_voxel_grid_from_dx(
    MeshBounds const &bounds, float requested_dx, std::uint64_t max_voxels
) noexcept {
    constexpr float k_min_extent = 1e-6f;
    constexpr float k_min_dx = 1e-6f;

    VoxelGridSpec grid{};
    max_voxels = std::max<std::uint64_t>(1u, max_voxels);

    float const extent_x = std::max(bounds.max_x - bounds.min_x, k_min_extent);
    float const extent_y = std::max(bounds.max_y - bounds.min_y, k_min_extent);
    float const extent_z = std::max(bounds.max_z - bounds.min_z, k_min_extent);

    if (!(requested_dx > 0.0f))
        requested_dx = std::max({extent_x, extent_y, extent_z}) / 64.0f;
    requested_dx = std::max(requested_dx, k_min_dx);

    float actual_dx = requested_dx;

    auto recompute_dims = [&](float const dx) {
        grid.nx = dim_from_extent(extent_x, dx);
        grid.ny = dim_from_extent(extent_y, dx);
        grid.nz = dim_from_extent(extent_z, dx);
        grid.total_voxels = safe_mul_u64(safe_mul_u64(grid.nx, grid.ny), grid.nz);
    };
    recompute_dims(actual_dx);

    for (int i = 0; i < 24 && grid.total_voxels > max_voxels; ++i) {
        double const ratio =
            static_cast<double>(grid.total_voxels) / static_cast<double>(max_voxels);
        double const scale = std::cbrt(std::max(ratio, 1.0));
        actual_dx = std::max(actual_dx * static_cast<float>(scale * 1.0001), k_min_dx);
        recompute_dims(actual_dx);
    }

    while (grid.total_voxels > max_voxels) {
        float const step_x = extent_x / static_cast<float>(grid.nx);
        float const step_y = extent_y / static_cast<float>(grid.ny);
        float const step_z = extent_z / static_cast<float>(grid.nz);
        if (step_x >= step_y && step_x >= step_z && grid.nx > 1u) {
            --grid.nx;
        } else if (step_y >= step_x && step_y >= step_z && grid.ny > 1u) {
            --grid.ny;
        } else if (grid.nz > 1u) {
            --grid.nz;
        } else {
            break;
        }
        grid.total_voxels = safe_mul_u64(safe_mul_u64(grid.nx, grid.ny), grid.nz);
    }

    grid.origin_x = bounds.min_x;
    grid.origin_y = bounds.min_y;
    grid.origin_z = bounds.min_z;
    grid.step_x = extent_x / static_cast<float>(grid.nx);
    grid.step_y = extent_y / static_cast<float>(grid.ny);
    grid.step_z = extent_z / static_cast<float>(grid.nz);
    grid.actual_dx = std::max({grid.step_x, grid.step_y, grid.step_z});
    return grid;
}

} // namespace winding_studio::voxel

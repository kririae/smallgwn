#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <GL/glew.h>
#include <cub/device/device_scan.cuh>

#include <gwn/gwn.cuh>

#include "voxelizer.hpp"

namespace winding_studio {

namespace {

using Real = float;
using Index = std::uint32_t;

constexpr std::uint64_t k_chunk_voxel_count = 1u << 20; // 1,048,576 voxels per chunk

struct voxel_kernel_grid {
    std::uint32_t nx{};
    std::uint32_t ny{};
    std::uint32_t nz{};
    Real origin_x{};
    Real origin_y{};
    Real origin_z{};
    Real step_x{};
    Real step_y{};
    Real step_z{};
};

[[nodiscard]] std::uint32_t grid_dim_for_count(std::uint64_t const count, int const block_size) {
    std::uint64_t const blocks = (count + static_cast<std::uint64_t>(block_size) - 1u) /
                                 static_cast<std::uint64_t>(block_size);
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(blocks, std::numeric_limits<std::uint32_t>::max())
    );
}

__global__ void fill_query_chunk_kernel(
    voxel_kernel_grid const grid, std::uint64_t const begin_linear_index,
    std::uint64_t const total_voxels, Real *query_x, Real *query_y, Real *query_z
) {
    std::uint64_t const local_idx =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint64_t const linear_idx = begin_linear_index + local_idx;
    if (linear_idx >= total_voxels)
        return;

    std::uint64_t const xy = static_cast<std::uint64_t>(grid.nx) * grid.ny;
    std::uint32_t const iz = static_cast<std::uint32_t>(linear_idx / xy);
    std::uint64_t const rem = linear_idx - static_cast<std::uint64_t>(iz) * xy;
    std::uint32_t const iy = static_cast<std::uint32_t>(rem / grid.nx);
    std::uint32_t const ix =
        static_cast<std::uint32_t>(rem - static_cast<std::uint64_t>(iy) * grid.nx);

    query_x[local_idx] = grid.origin_x + (static_cast<Real>(ix) + Real(0.5)) * grid.step_x;
    query_y[local_idx] = grid.origin_y + (static_cast<Real>(iy) + Real(0.5)) * grid.step_y;
    query_z[local_idx] = grid.origin_z + (static_cast<Real>(iz) + Real(0.5)) * grid.step_z;
}

__global__ void compute_mask_kernel(
    Real const *winding, std::uint64_t const count, Real const target_winding, std::uint32_t *mask
) {
    std::uint64_t const idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    Real const w = winding[idx];
    mask[idx] = (w >= target_winding) ? 1u : 0u;
}

__global__ void emit_instances_compact_kernel(
    Real const *query_x, Real const *query_y, Real const *query_z, std::uint32_t const *mask,
    std::uint32_t const *prefix, std::uint64_t const count, std::uint64_t const base_output_index,
    float4 *instance_positions, std::uint32_t const instance_capacity
) {
    std::uint64_t const idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    if (mask[idx] == 0u)
        return;

    std::uint64_t const out_idx = base_output_index + static_cast<std::uint64_t>(prefix[idx]);
    if (out_idx >= static_cast<std::uint64_t>(instance_capacity))
        return;

    instance_positions[out_idx] = make_float4(query_x[idx], query_y[idx], query_z[idx], Real(1));
}

void throw_if_error(gwn::gwn_status const &status, std::string_view const context) {
    if (status.is_ok())
        return;
    throw std::runtime_error(std::string(context) + ": " + status.message());
}

void throw_if_cuda(cudaError_t const err, std::string_view const context) {
    if (err == cudaSuccess)
        return;
    throw std::runtime_error(std::string(context) + ": " + std::string(cudaGetErrorString(err)));
}

} // namespace

class Voxelizer::Impl final {
public:
    [[nodiscard]] bool has_mesh() const noexcept { return mesh_ready_; }

    [[nodiscard]] bool upload_mesh(HostMeshSoA const &mesh, std::string &error) noexcept {
        try {
            if (mesh.vx.size() != mesh.vy.size() || mesh.vx.size() != mesh.vz.size()) {
                error = "vertex SoA buffers must have the same size";
                return false;
            }
            if (mesh.i0.size() != mesh.i1.size() || mesh.i0.size() != mesh.i2.size()) {
                error = "triangle index buffers must have the same size";
                return false;
            }
            if (mesh.vx.empty() || mesh.i0.empty()) {
                error = "mesh is empty";
                return false;
            }

            throw_if_error(
                geometry_.upload(
                    cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
                    cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
                    cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
                    cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
                    cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
                    cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
                ),
                "geometry.upload"
            );

            throw_if_error(
                gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                    geometry_, bvh_, aabb_, moments_
                ),
                "gwn_bvh_facade_build_topology_aabb_moment_lbvh"
            );
            throw_if_error(
                gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "cudaDeviceSynchronize"
            );

            mesh_ready_ = true;
            return true;
        } catch (std::exception const &e) {
            error = e.what();
            mesh_ready_ = false;
            return false;
        }
    }

    [[nodiscard]] bool voxelize(
        voxel::VoxelGridSpec const &grid, VoxelizeConfig const &config, unsigned int gl_buffer_id,
        std::size_t gl_buffer_capacity, VoxelizeStats &out, std::string &error
    ) noexcept {
        try {
            if (!mesh_ready_) {
                error = "no mesh uploaded for voxelization";
                return false;
            }
            if (gl_buffer_id == 0u) {
                error = "GL instance buffer is invalid";
                return false;
            }
            if (gl_buffer_capacity == 0u) {
                error = "GL instance buffer capacity is zero";
                return false;
            }
            if (grid.total_voxels == 0u) {
                out = VoxelizeStats{};
                return true;
            }

            std::uint32_t const instance_capacity = static_cast<std::uint32_t>(
                std::min<std::size_t>(gl_buffer_capacity, std::numeric_limits<std::uint32_t>::max())
            );
            if (instance_capacity == 0u) {
                error = "instance buffer capacity is zero after clamping";
                return false;
            }

            throw_if_error(d_instances_.resize(instance_capacity), "resize d_instances");

            std::size_t const chunk_capacity = static_cast<std::size_t>(
                std::min<std::uint64_t>(grid.total_voxels, k_chunk_voxel_count)
            );
            throw_if_error(d_query_x_.resize(chunk_capacity), "resize d_query_x");
            throw_if_error(d_query_y_.resize(chunk_capacity), "resize d_query_y");
            throw_if_error(d_query_z_.resize(chunk_capacity), "resize d_query_z");
            throw_if_error(d_winding_.resize(chunk_capacity), "resize d_winding");
            throw_if_error(d_mask_.resize(chunk_capacity), "resize d_mask");
            throw_if_error(d_prefix_.resize(chunk_capacity), "resize d_prefix");

            std::size_t scan_temp_bytes = 0;
            throw_if_cuda(
                cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_temp_bytes, d_mask_.data(), d_prefix_.data(), chunk_capacity
                ),
                "cub::DeviceScan::ExclusiveSum(temp_storage_bytes)"
            );
            if (scan_temp_bytes > 0u)
                throw_if_error(d_scan_temp_.resize(scan_temp_bytes), "resize d_scan_temp");
            else
                throw_if_error(d_scan_temp_.clear(), "clear d_scan_temp");

            int constexpr k_block_size = 256;
            voxel_kernel_grid const kernel_grid{
                grid.nx,       grid.ny,     grid.nz,     grid.origin_x, grid.origin_y,
                grid.origin_z, grid.step_x, grid.step_y, grid.step_z,
            };
            Real const target_w = config.target_winding;

            std::uint64_t selected_total = 0u;
            for (std::uint64_t begin = 0u; begin < grid.total_voxels; begin += chunk_capacity) {
                std::uint64_t const count = std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(chunk_capacity), grid.total_voxels - begin
                );
                std::uint32_t const launch_grid = grid_dim_for_count(count, k_block_size);

                fill_query_chunk_kernel<<<launch_grid, k_block_size>>>(
                    kernel_grid, begin, grid.total_voxels, d_query_x_.data(), d_query_y_.data(),
                    d_query_z_.data()
                );
                throw_if_error(
                    gwn::gwn_cuda_to_status(cudaGetLastError()), "fill_query_chunk_kernel launch"
                );

                auto const query_x = cuda::std::span<Real const>(d_query_x_.data(), count);
                auto const query_y = cuda::std::span<Real const>(d_query_y_.data(), count);
                auto const query_z = cuda::std::span<Real const>(d_query_z_.data(), count);
                auto const winding = cuda::std::span<Real>(d_winding_.data(), count);
                throw_if_error(
                    gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                        geometry_.accessor(), bvh_.accessor(), moments_.accessor(), query_x,
                        query_y, query_z, winding, config.accuracy_scale
                    ),
                    "gwn_compute_winding_number_batch_bvh_taylor"
                );

                compute_mask_kernel<<<launch_grid, k_block_size>>>(
                    d_winding_.data(), count, target_w, d_mask_.data()
                );
                throw_if_error(
                    gwn::gwn_cuda_to_status(cudaGetLastError()), "compute_mask_kernel launch"
                );

                throw_if_cuda(
                    cub::DeviceScan::ExclusiveSum(
                        d_scan_temp_.data(), scan_temp_bytes, d_mask_.data(), d_prefix_.data(),
                        count
                    ),
                    "cub::DeviceScan::ExclusiveSum"
                );

                std::uint32_t tail_prefix = 0u;
                std::uint32_t tail_mask = 0u;
                throw_if_cuda(
                    cudaMemcpy(
                        &tail_prefix, d_prefix_.data() + (count - 1u), sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost
                    ),
                    "copy tail prefix"
                );
                throw_if_cuda(
                    cudaMemcpy(
                        &tail_mask, d_mask_.data() + (count - 1u), sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost
                    ),
                    "copy tail mask"
                );

                std::uint64_t const chunk_selected =
                    static_cast<std::uint64_t>(tail_prefix) + static_cast<std::uint64_t>(tail_mask);
                if (chunk_selected == 0u)
                    continue;

                emit_instances_compact_kernel<<<launch_grid, k_block_size>>>(
                    d_query_x_.data(), d_query_y_.data(), d_query_z_.data(), d_mask_.data(),
                    d_prefix_.data(), count, selected_total, d_instances_.data(), instance_capacity
                );
                throw_if_error(
                    gwn::gwn_cuda_to_status(cudaGetLastError()),
                    "emit_instances_compact_kernel launch"
                );

                selected_total += chunk_selected;
            }

            throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "voxelize sync");

            std::size_t const occupied_count =
                std::min<std::size_t>(static_cast<std::size_t>(selected_total), instance_capacity);
            h_instances_.resize(occupied_count);
            if (occupied_count > 0u) {
                throw_if_cuda(
                    cudaMemcpy(
                        h_instances_.data(), d_instances_.data(), occupied_count * sizeof(float4),
                        cudaMemcpyDeviceToHost
                    ),
                    "copy d_instances"
                );
            }

            glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_id);
            if (occupied_count > 0u) {
                glBufferSubData(
                    GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(occupied_count * sizeof(float4)),
                    h_instances_.data()
                );
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            out.total_voxels = static_cast<std::size_t>(grid.total_voxels);
            out.occupied_count = occupied_count;
            return true;
        } catch (std::exception const &e) {
            error = e.what();
            return false;
        }
    }

private:
    bool mesh_ready_{false};
    gwn::gwn_geometry_object<Real, Index> geometry_{};
    gwn::gwn_bvh4_topology_object<Real, Index> bvh_{};
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb_{};
    gwn::gwn_bvh4_moment_object<1, Real, Index> moments_{};

    gwn::gwn_device_array<Real> d_query_x_{};
    gwn::gwn_device_array<Real> d_query_y_{};
    gwn::gwn_device_array<Real> d_query_z_{};
    gwn::gwn_device_array<Real> d_winding_{};
    gwn::gwn_device_array<std::uint32_t> d_mask_{};
    gwn::gwn_device_array<std::uint32_t> d_prefix_{};
    gwn::gwn_device_array<std::uint8_t> d_scan_temp_{};
    gwn::gwn_device_array<float4> d_instances_{};

    std::vector<float4> h_instances_{};
};

Voxelizer::Voxelizer() : impl_(new Impl()) {}

Voxelizer::~Voxelizer() {
    delete impl_;
    impl_ = nullptr;
}

Voxelizer::Voxelizer(Voxelizer &&other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }

Voxelizer &Voxelizer::operator=(Voxelizer &&other) noexcept {
    if (this == &other)
        return *this;
    delete impl_;
    impl_ = other.impl_;
    other.impl_ = nullptr;
    return *this;
}

bool Voxelizer::has_mesh() const noexcept { return impl_ != nullptr && impl_->has_mesh(); }

bool Voxelizer::upload_mesh(HostMeshSoA const &mesh, std::string &error) {
    if (impl_ == nullptr) {
        error = "internal voxelizer state missing";
        return false;
    }
    return impl_->upload_mesh(mesh, error);
}

bool Voxelizer::voxelize(
    voxel::VoxelGridSpec const &grid, VoxelizeConfig const &config, unsigned int const gl_buffer_id,
    std::size_t const gl_buffer_capacity, VoxelizeStats &out, std::string &error
) {
    if (impl_ == nullptr) {
        error = "internal voxelizer state missing";
        return false;
    }
    return impl_->voxelize(grid, config, gl_buffer_id, gl_buffer_capacity, out, error);
}

} // namespace winding_studio

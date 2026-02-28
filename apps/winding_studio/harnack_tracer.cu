#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

#include <gwn/gwn.cuh>

#include "harnack_tracer.hpp"

namespace winding_studio {

namespace {

using Real = float;
using Index = std::uint32_t;

struct shading_vec3 {
    Real x{};
    Real y{};
    Real z{};
};

__host__ __device__ [[nodiscard]] shading_vec3
operator+(shading_vec3 const &a, shading_vec3 const &b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ [[nodiscard]] shading_vec3
operator*(shading_vec3 const &a, Real const s) noexcept {
    return {a.x * s, a.y * s, a.z * s};
}

__host__ __device__ [[nodiscard]] shading_vec3
operator*(shading_vec3 const &a, shading_vec3 const &b) noexcept {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ [[nodiscard]] Real
shading_dot(shading_vec3 const &a, shading_vec3 const &b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ [[nodiscard]] shading_vec3 shading_normalize(shading_vec3 const &v) noexcept {
    Real const m2 = shading_dot(v, v);
    if (!(m2 > Real(0)))
        return {Real(0), Real(0), Real(1)};
    Real const inv = rsqrtf(m2);
    return {v.x * inv, v.y * inv, v.z * inv};
}

__host__ __device__ [[nodiscard]] Real shading_clamp01(Real const v) noexcept {
    return v < Real(0) ? Real(0) : (v > Real(1) ? Real(1) : v);
}

__host__ __device__ [[nodiscard]] std::uint8_t shading_to_u8(Real const v) noexcept {
    return static_cast<std::uint8_t>(shading_clamp01(v) * Real(255) + Real(0.5));
}

__host__ __device__ [[nodiscard]] shading_vec3
compute_three_light_shading(shading_vec3 const &normal) noexcept {
    shading_vec3 const n = shading_normalize(normal);

    shading_vec3 const key_dir = shading_normalize({Real(0.5), Real(0.7), Real(0.5)});
    shading_vec3 const fill_dir = shading_normalize({Real(-0.6), Real(0.3), Real(0.4)});

    Real const key_dot = shading_dot(n, key_dir);
    Real const key = key_dot > Real(0) ? key_dot : Real(0);
    Real const fill_dot = shading_dot(n, fill_dir);
    Real const fill = fill_dot > Real(0) ? fill_dot : Real(0);

    Real const nz_abs = n.z > Real(0) ? n.z : -n.z;
    Real const rim_base = Real(1) - nz_abs;
    Real const rim_c = shading_clamp01(rim_base);
    Real const rim = rim_c * rim_c * rim_c;

    shading_vec3 const base{Real(0.80), Real(0.80), Real(0.82)};
    shading_vec3 const key_col{Real(1.00), Real(0.98), Real(0.92)};
    shading_vec3 const fill_col{Real(0.50), Real(0.60), Real(0.78)};
    shading_vec3 const rim_col{Real(0.85), Real(0.88), Real(0.95)};

    shading_vec3 const lit = base * Real(0.18) + base * key_col * (key * Real(0.62)) +
                             base * fill_col * (fill * Real(0.35)) + rim_col * (rim * Real(0.22));

    return {shading_clamp01(lit.x), shading_clamp01(lit.y), shading_clamp01(lit.z)};
}

struct harnack_gpu_camera {
    Real origin_x, origin_y, origin_z;
    Real forward_x, forward_y, forward_z;
    Real right_x, right_y, right_z;
    Real up_x, up_y, up_z;
    Real tan_half_fov, aspect;
    int width, height;
};

struct harnack_render_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry;
    gwn::gwn_bvh_topology_accessor<4, Real, Index> bvh;
    gwn::gwn_bvh_aabb_accessor<4, Real, Index> aabb_tree;
    gwn::gwn_bvh_moment_tree_accessor<4, 1, Real, Index> moment_tree;

    harnack_gpu_camera camera;
    Real target_winding;
    Real epsilon;
    int max_iterations;
    Real t_max;
    Real accuracy_scale;

    cuda::std::span<std::uint8_t> rgba;
    cuda::std::span<int> hit_counter;

    __device__ void operator()(std::size_t const pixel_id) const {
        int const px = static_cast<int>(pixel_id) % camera.width;
        int const py = static_cast<int>(pixel_id) / camera.width;

        Real const sx =
            ((static_cast<Real>(px) + Real(0.5)) / static_cast<Real>(camera.width)) * Real(2) -
            Real(1);
        Real const sy =
            Real(1) -
            ((static_cast<Real>(py) + Real(0.5)) / static_cast<Real>(camera.height)) * Real(2);

        Real dx = camera.forward_x + camera.right_x * (sx * camera.aspect * camera.tan_half_fov) +
                  camera.up_x * (sy * camera.tan_half_fov);
        Real dy = camera.forward_y + camera.right_y * (sx * camera.aspect * camera.tan_half_fov) +
                  camera.up_y * (sy * camera.tan_half_fov);
        Real dz = camera.forward_z + camera.right_z * (sx * camera.aspect * camera.tan_half_fov) +
                  camera.up_z * (sy * camera.tan_half_fov);
        Real const inv_len = rsqrtf(dx * dx + dy * dy + dz * dz);
        dx *= inv_len;
        dy *= inv_len;
        dz *= inv_len;

        auto const res = gwn::detail::gwn_harnack_trace_ray_impl<1, 4, Real, Index, 64>(
            geometry, bvh, aabb_tree, moment_tree, camera.origin_x, camera.origin_y,
            camera.origin_z, dx, dy, dz, target_winding, epsilon, max_iterations, t_max,
            accuracy_scale
        );

        std::size_t const o = pixel_id * 4u;
        if (!res.hit()) {
            rgba[o + 0] = std::uint8_t(21);
            rgba[o + 1] = std::uint8_t(22);
            rgba[o + 2] = std::uint8_t(27);
            rgba[o + 3] = std::uint8_t(255);
            return;
        }

        atomicAdd(hit_counter.data(), 1);

        shading_vec3 const color =
            compute_three_light_shading({res.normal_x, res.normal_y, res.normal_z});
        rgba[o + 0] = shading_to_u8(color.x);
        rgba[o + 1] = shading_to_u8(color.y);
        rgba[o + 2] = shading_to_u8(color.z);
        rgba[o + 3] = std::uint8_t(255);
    }
};

void throw_if_error(gwn::gwn_status const &status, std::string_view const context) {
    if (status.is_ok())
        return;
    throw std::runtime_error(std::string(context) + ": " + status.message());
}

} // namespace

class HarnackTracer::Impl final {
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

    [[nodiscard]] bool trace(
        CameraFrame const &camera, HarnackTraceConfig const &config, HarnackTraceImages &out,
        std::string &error
    ) noexcept {
        try {
            if (!mesh_ready_) {
                error = "no mesh uploaded for tracing";
                return false;
            }
            if (camera.width <= 0 || camera.height <= 0) {
                error = "camera width/height must be positive";
                return false;
            }

            std::size_t const pixel_count =
                static_cast<std::size_t>(camera.width) * static_cast<std::size_t>(camera.height);

            throw_if_error(d_rgba_.resize(pixel_count * 4u), "resize d_rgba");
            throw_if_error(d_hit_counter_.resize(1), "resize d_hit_counter");
            throw_if_error(d_hit_counter_.zero(), "zero d_hit_counter");

            harnack_gpu_camera const gpu_cam{
                camera.origin_x,     camera.origin_y,  camera.origin_z, camera.forward_x,
                camera.forward_y,    camera.forward_z, camera.right_x,  camera.right_y,
                camera.right_z,      camera.up_x,      camera.up_y,     camera.up_z,
                camera.tan_half_fov, camera.aspect,    camera.width,    camera.height,
            };

            harnack_render_functor const functor{
                geometry_.accessor(),
                bvh_.accessor(),
                aabb_.accessor(),
                moments_.accessor(),
                gpu_cam,
                config.target_winding,
                config.epsilon,
                config.max_iterations,
                config.t_max,
                config.accuracy_scale,
                d_rgba_.span(),
                d_hit_counter_.span(),
            };
            throw_if_error(
                gwn::detail::gwn_launch_linear_kernel<128>(pixel_count, functor),
                "harnack_render_kernel"
            );

            out.width = camera.width;
            out.height = camera.height;
            out.harnack_rgba.resize(pixel_count * 4u);
            throw_if_error(
                d_rgba_.copy_to_host(
                    cuda::std::span<std::uint8_t>(out.harnack_rgba.data(), out.harnack_rgba.size())
                ),
                "copy d_rgba"
            );

            int host_hits = 0;
            throw_if_error(
                d_hit_counter_.copy_to_host(cuda::std::span<int>(&host_hits, 1)), "copy hit_counter"
            );
            throw_if_error(
                gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "cudaDeviceSynchronize"
            );
            out.hit_count = static_cast<std::size_t>(host_hits);

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

    gwn::gwn_device_array<std::uint8_t> d_rgba_{};
    gwn::gwn_device_array<int> d_hit_counter_{};
};

HarnackTracer::HarnackTracer() : impl_(new Impl()) {}

HarnackTracer::~HarnackTracer() {
    delete impl_;
    impl_ = nullptr;
}

HarnackTracer::HarnackTracer(HarnackTracer &&other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

HarnackTracer &HarnackTracer::operator=(HarnackTracer &&other) noexcept {
    if (this == &other)
        return *this;
    delete impl_;
    impl_ = other.impl_;
    other.impl_ = nullptr;
    return *this;
}

bool HarnackTracer::has_mesh() const noexcept { return impl_ != nullptr && impl_->has_mesh(); }

bool HarnackTracer::upload_mesh(HostMeshSoA const &mesh, std::string &error) {
    if (impl_ == nullptr) {
        error = "internal tracer state missing";
        return false;
    }
    return impl_->upload_mesh(mesh, error);
}

bool HarnackTracer::trace(
    CameraFrame const &camera, HarnackTraceConfig const &config, HarnackTraceImages &out,
    std::string &error
) {
    if (impl_ == nullptr) {
        error = "internal tracer state missing";
        return false;
    }
    return impl_->trace(camera, config, out, error);
}

} // namespace winding_studio

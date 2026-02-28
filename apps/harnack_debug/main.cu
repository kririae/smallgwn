#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gwn/gwn.cuh>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

using Real = float;
using Index = std::uint32_t;

struct Vec3 {
    Real x{Real(0)};
    Real y{Real(0)};
    Real z{Real(0)};
};

struct MeshData {
    std::vector<Real> vx;
    std::vector<Real> vy;
    std::vector<Real> vz;
    std::vector<Index> i0;
    std::vector<Index> i1;
    std::vector<Index> i2;
};

struct Options {
    int width{640};
    int height{480};
    Real fov_degrees{Real(40)};
    std::string mesh{"half"};
    std::string normal_out{"normal.png"};
    std::string depth_out{"depth.png"};
    Real epsilon{Real(1e-3)};
    int max_iterations{2048};
    Real t_max{Real(100)};
    Real accuracy_scale{Real(2)};
};

[[nodiscard]] Vec3 operator+(Vec3 const &a, Vec3 const &b) noexcept {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 operator-(Vec3 const &a, Vec3 const &b) noexcept {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 operator*(Vec3 const &a, Real const s) noexcept {
    return Vec3{a.x * s, a.y * s, a.z * s};
}

[[nodiscard]] Real dot(Vec3 const &a, Vec3 const &b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(Vec3 const &a, Vec3 const &b) noexcept {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] Vec3 normalize(Vec3 const &v) noexcept {
    Real const m2 = dot(v, v);
    if (!(m2 > Real(0)))
        return Vec3{Real(0), Real(0), Real(1)};
    Real const inv_m = Real(1) / std::sqrt(m2);
    return Vec3{v.x * inv_m, v.y * inv_m, v.z * inv_m};
}

[[nodiscard]] MeshData make_octahedron_mesh() {
    MeshData mesh;
    mesh.vx = {Real(1), Real(-1), Real(0), Real(0), Real(0), Real(0)};
    mesh.vy = {Real(0), Real(0), Real(1), Real(-1), Real(0), Real(0)};
    mesh.vz = {Real(0), Real(0), Real(0), Real(0), Real(1), Real(-1)};
    mesh.i0 = {0, 2, 1, 3, 2, 1, 3, 0};
    mesh.i1 = {2, 1, 3, 0, 0, 2, 1, 3};
    mesh.i2 = {4, 4, 4, 4, 5, 5, 5, 5};
    return mesh;
}

[[nodiscard]] MeshData make_half_octahedron_mesh() {
    MeshData mesh;
    mesh.vx = {Real(1), Real(-1), Real(0), Real(0), Real(0)};
    mesh.vy = {Real(0), Real(0), Real(1), Real(-1), Real(0)};
    mesh.vz = {Real(0), Real(0), Real(0), Real(0), Real(1)};
    mesh.i0 = {0, 2, 1, 3};
    mesh.i1 = {2, 1, 3, 0};
    mesh.i2 = {4, 4, 4, 4};
    return mesh;
}

[[nodiscard]] bool parse_int(std::string const &s, int &v) {
    try {
        std::size_t consumed = 0;
        int const parsed = std::stoi(s, &consumed);
        if (consumed != s.size())
            return false;
        v = parsed;
        return true;
    } catch (...) { return false; }
}

[[nodiscard]] bool parse_real(std::string const &s, Real &v) {
    try {
        std::size_t consumed = 0;
        float const parsed = std::stof(s, &consumed);
        if (consumed != s.size())
            return false;
        v = parsed;
        return true;
    } catch (...) { return false; }
}

void print_help(char const *argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "Options:\n"
              << "  --width <int>         Image width (default: 640)\n"
              << "  --height <int>        Image height (default: 480)\n"
              << "  --fov <float>         Vertical FOV in degrees (default: 40)\n"
              << "  --mesh <octa|half>    Built-in mesh (default: half)\n"
              << "  --normal-out <path>   Normal PNG output path (default: normal.png)\n"
              << "  --depth-out <path>    Depth PNG output path (default: depth.png)\n"
              << "  --epsilon <float>     Harnack epsilon (default: 1e-3)\n"
              << "  --max-iters <int>     Max iterations (default: 2048)\n"
              << "  --tmax <float>        Ray t_max (default: 100)\n"
              << "  --accuracy <float>    Taylor accuracy scale (default: 2)\n"
              << "  --help                Show this message\n";
}

[[nodiscard]] bool parse_options(int argc, char **argv, Options &opt) {
    for (int i = 1; i < argc; ++i) {
        std::string const key(argv[i]);
        auto read_value = [&](std::string &out) -> bool {
            if (i + 1 >= argc)
                return false;
            out = argv[++i];
            return true;
        };

        if (key == "--help") {
            print_help(argv[0]);
            return false;
        }
        if (key == "--width") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.width))
                return false;
            continue;
        }
        if (key == "--height") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.height))
                return false;
            continue;
        }
        if (key == "--fov") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.fov_degrees))
                return false;
            continue;
        }
        if (key == "--mesh") {
            std::string value;
            if (!read_value(value))
                return false;
            opt.mesh = value;
            continue;
        }
        if (key == "--normal-out") {
            if (!read_value(opt.normal_out))
                return false;
            continue;
        }
        if (key == "--depth-out") {
            if (!read_value(opt.depth_out))
                return false;
            continue;
        }
        if (key == "--epsilon") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.epsilon))
                return false;
            continue;
        }
        if (key == "--max-iters") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.max_iterations))
                return false;
            continue;
        }
        if (key == "--tmax") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.t_max))
                return false;
            continue;
        }
        if (key == "--accuracy") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.accuracy_scale))
                return false;
            continue;
        }
        return false;
    }
    return true;
}

void throw_if_error(gwn::gwn_status const &status, std::string_view const context) {
    if (status.is_ok())
        return;
    throw std::runtime_error(std::string(context) + ": " + status.message());
}

[[nodiscard]] bool write_png_rgb(
    std::string const &path, int const width, int const height, std::vector<std::uint8_t> const &rgb
) {
    int const stride = width * 3;
    return stbi_write_png(path.c_str(), width, height, 3, rgb.data(), stride) != 0;
}

std::uint8_t to_u8(Real const v) {
    Real c = std::clamp(v, Real(0), Real(1));
    return static_cast<std::uint8_t>(std::lround(c * Real(255)));
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt;
        if (!parse_options(argc, argv, opt)) {
            std::cerr << "Invalid options.\n";
            print_help(argv[0]);
            return 1;
        }
        if (opt.width <= 0 || opt.height <= 0) {
            std::cerr << "width/height must be positive.\n";
            return 1;
        }
        if (opt.mesh != "octa" && opt.mesh != "half") {
            std::cerr << "mesh must be 'octa' or 'half'.\n";
            return 1;
        }

        cudaError_t const probe = cudaFree(nullptr);
        if (probe != cudaSuccess) {
            std::cerr << "CUDA runtime unavailable: " << cudaGetErrorString(probe) << "\n";
            return 2;
        }

        MeshData const mesh =
            (opt.mesh == "half") ? make_half_octahedron_mesh() : make_octahedron_mesh();

        gwn::gwn_geometry_object<Real, Index> geometry;
        throw_if_error(
            geometry.upload(
                cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
                cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
                cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
                cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
                cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
                cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
            ),
            "geometry.upload"
        );

        gwn::gwn_bvh4_topology_object<Real, Index> bvh;
        gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
        gwn::gwn_bvh4_moment_object<1, Real, Index> moments;
        throw_if_error(
            gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, 4, Real, Index>(
                geometry, bvh, aabb, moments
            ),
            "gwn_bvh_facade_build_topology_aabb_moment_lbvh"
        );

        std::size_t const pixel_count =
            static_cast<std::size_t>(opt.width) * static_cast<std::size_t>(opt.height);
        std::vector<Real> ox(pixel_count), oy(pixel_count), oz(pixel_count);
        std::vector<Real> dx(pixel_count), dy(pixel_count), dz(pixel_count);

        Vec3 cam_origin{};
        Vec3 cam_target{};
        Vec3 cam_up{};
        if (opt.mesh == "half") {
            // Open mesh is the primary harnack-debug case for the
            // always-edge-distance tracer.
            cam_origin = Vec3{Real(0), Real(0), Real(2.3)};
            cam_target = Vec3{Real(0), Real(0), Real(0.25)};
            cam_up = Vec3{Real(0), Real(1), Real(0)};
        } else {
            cam_origin = Vec3{Real(2.6), Real(2.3), Real(2.2)};
            cam_target = Vec3{Real(0), Real(0), Real(0)};
            cam_up = Vec3{Real(0), Real(0), Real(1)};
        }
        Vec3 const forward = normalize(cam_target - cam_origin);
        Vec3 const right = normalize(cross(forward, cam_up));
        Vec3 const up = cross(right, forward);
        Real const tan_half_fov =
            std::tan((opt.fov_degrees * Real(3.14159265358979323846) / Real(180)) * Real(0.5));
        Real const aspect = static_cast<Real>(opt.width) / static_cast<Real>(opt.height);

        for (int py = 0; py < opt.height; ++py) {
            for (int px = 0; px < opt.width; ++px) {
                std::size_t const idx = static_cast<std::size_t>(py) * opt.width + px;

                Real const sx =
                    ((static_cast<Real>(px) + Real(0.5)) / static_cast<Real>(opt.width)) * Real(2) -
                    Real(1);
                Real const sy =
                    Real(1) -
                    ((static_cast<Real>(py) + Real(0.5)) / static_cast<Real>(opt.height)) * Real(2);
                Vec3 const rd = normalize(
                    forward + right * (sx * aspect * tan_half_fov) + up * (sy * tan_half_fov)
                );

                ox[idx] = cam_origin.x;
                oy[idx] = cam_origin.y;
                oz[idx] = cam_origin.z;
                dx[idx] = rd.x;
                dy[idx] = rd.y;
                dz[idx] = rd.z;
            }
        }

        gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
        gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;

        throw_if_error(
            d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), ox.size())), "copy d_ox"
        );
        throw_if_error(
            d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), oy.size())), "copy d_oy"
        );
        throw_if_error(
            d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), oz.size())), "copy d_oz"
        );
        throw_if_error(
            d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), dx.size())), "copy d_dx"
        );
        throw_if_error(
            d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), dy.size())), "copy d_dy"
        );
        throw_if_error(
            d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), dz.size())), "copy d_dz"
        );
        throw_if_error(d_t.resize(pixel_count), "resize d_t");
        throw_if_error(d_nx.resize(pixel_count), "resize d_nx");
        throw_if_error(d_ny.resize(pixel_count), "resize d_ny");
        throw_if_error(d_nz.resize(pixel_count), "resize d_nz");

        throw_if_error(
            gwn::gwn_compute_harnack_trace_batch_bvh_taylor<1, Real, Index>(
                geometry.accessor(), bvh.accessor(), aabb.accessor(), moments.accessor(),
                d_ox.span(), d_oy.span(), d_oz.span(), d_dx.span(), d_dy.span(), d_dz.span(),
                d_t.span(), d_nx.span(), d_ny.span(), d_nz.span(), Real(0.5), opt.epsilon,
                opt.max_iterations, opt.t_max, opt.accuracy_scale
            ),
            "gwn_compute_harnack_trace_batch_bvh_taylor"
        );

        throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "cudaDeviceSynchronize");

        std::vector<Real> host_t(pixel_count), host_nx(pixel_count), host_ny(pixel_count),
            host_nz(pixel_count);
        throw_if_error(
            d_t.copy_to_host(cuda::std::span<Real>(host_t.data(), host_t.size())), "copy host_t"
        );
        throw_if_error(
            d_nx.copy_to_host(cuda::std::span<Real>(host_nx.data(), host_nx.size())), "copy host_nx"
        );
        throw_if_error(
            d_ny.copy_to_host(cuda::std::span<Real>(host_ny.data(), host_ny.size())), "copy host_ny"
        );
        throw_if_error(
            d_nz.copy_to_host(cuda::std::span<Real>(host_nz.data(), host_nz.size())), "copy host_nz"
        );
        throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "cudaDeviceSynchronize");

        std::size_t hit_count = 0;
        Real t_min = std::numeric_limits<Real>::infinity();
        Real t_max = Real(0);
        for (Real const t : host_t) {
            if (t >= Real(0)) {
                ++hit_count;
                t_min = std::min(t_min, t);
                t_max = std::max(t_max, t);
            }
        }
        if (hit_count == 0)
            std::cerr << "No hits found. Try changing camera or mesh.\n";
        Real const hit_ratio = static_cast<Real>(hit_count) / static_cast<Real>(pixel_count);
        if (hit_ratio < Real(0.02)) {
            std::cerr << "warning: low hit ratio (" << hit_ratio
                      << "). For debugging this tracer, prefer --mesh half.\n";
        }

        std::vector<std::uint8_t> normal_rgb(pixel_count * 3, std::uint8_t(0));
        std::vector<std::uint8_t> depth_rgb(pixel_count * 3, std::uint8_t(0));
        Real const t_range = (t_max > t_min) ? (t_max - t_min) : Real(1);

        for (std::size_t i = 0; i < pixel_count; ++i) {
            std::size_t const o = i * 3;
            if (host_t[i] < Real(0))
                continue;

            normal_rgb[o + 0] = to_u8(host_nx[i] * Real(0.5) + Real(0.5));
            normal_rgb[o + 1] = to_u8(host_ny[i] * Real(0.5) + Real(0.5));
            normal_rgb[o + 2] = to_u8(host_nz[i] * Real(0.5) + Real(0.5));

            Real const depth_n = (host_t[i] - t_min) / t_range;
            std::uint8_t const d = to_u8(Real(1) - depth_n);
            depth_rgb[o + 0] = d;
            depth_rgb[o + 1] = d;
            depth_rgb[o + 2] = d;
        }

        if (!write_png_rgb(opt.normal_out, opt.width, opt.height, normal_rgb))
            throw std::runtime_error("failed to write normal PNG");
        if (!write_png_rgb(opt.depth_out, opt.width, opt.height, depth_rgb))
            throw std::runtime_error("failed to write depth PNG");

        std::cout << "Rendered " << opt.width << "x" << opt.height << " pixels. hits=" << hit_count
                  << "/" << pixel_count << "\n";
        std::cout << "Wrote normal: " << opt.normal_out << "\n";
        std::cout << "Wrote depth : " << opt.depth_out << "\n";
        return 0;
    } catch (std::exception const &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
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

constexpr Real k_pi = Real(3.14159265358979323846);
constexpr Real k_four_pi = Real(4) * k_pi;
constexpr int k_bvh_width = 4;
constexpr int k_stack_capacity = gwn::k_gwn_default_traversal_stack_capacity;

struct Vec3 {
    Real x{Real(0)};
    Real y{Real(0)};
    Real z{Real(0)};
};

struct CameraBasis {
    Vec3 eye{};
    Vec3 target{};
    Vec3 forward{};
    Vec3 right{};
    Vec3 up{};
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
    int width{915};
    int height{666};
    Real fov_degrees{Real(45)};
    std::string mesh{"half"};
    std::string mesh_file{};
    std::string normal_out{"normal.png"};
    std::string depth_out{"depth.png"};

    Real target_winding{Real(0.5)};
    Real epsilon{Real(1e-3)};
    int max_iterations{2048};
    Real t_max{Real(100)};
    Real accuracy_scale{Real(2)};

    Real yaw{Real(0)};
    Real pitch{Real(-0.35)};
    Real camera_distance{Real(2.7)};
    Real target_x{Real(0)};
    Real target_y{Real(0)};
    Real target_z{Real(0)};

    bool profile_hole{false};
    int profile_px{-1};
    int profile_py{-1};
    int profile_samples{8192};
    std::string profile_out{};
};

struct PointProbe {
    Real winding{Real(0)};
    Real grad_x{Real(0)};
    Real grad_y{Real(0)};
    Real grad_z{Real(0)};
    Real r_face{Real(0)};
    Real r_edge{Real(0)};
};

struct HoleInfo {
    bool found{false};
    std::size_t area{0};
    int centroid_x{-1};
    int centroid_y{-1};
    int sample_x{-1};
    int sample_y{-1};
    int min_x{-1};
    int min_y{-1};
    int max_x{-1};
    int max_y{-1};
};

struct IterRecord {
    int iter{0};
    Real t_eval{Real(0)};
    Real w{Real(0)};
    Real wrapped{Real(0)};
    Real dist{Real(0)};
    Real grad_omega_mag{Real(0)};
    Real r_face{Real(0)};
    Real r_edge{Real(0)};
    Real rho{Real(0)};
    Real t_overstep_before{Real(0)};
    bool accepted{false};
    bool hit_by_grad{false};
    bool hit_by_radius{false};
    bool invalid_radius{false};
    bool invalid_rho{false};
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
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --width <int>            Image width (default: 915)\n"
        << "  --height <int>           Image height (default: 666)\n"
        << "  --fov <float>            Vertical FOV in degrees (default: 45)\n"
        << "  --mesh <octa|half>       Built-in mesh (default: half)\n"
        << "  --mesh-file <path>       External OBJ mesh (normalized to [-1,1])\n"
        << "  --normal-out <path>      Normal PNG output path (default: normal.png)\n"
        << "  --depth-out <path>       Depth PNG output path (default: depth.png)\n"
        << "  --target-w <float>       Target winding iso-value (default: 0.5)\n"
        << "  --epsilon <float>        Harnack epsilon (default: 1e-3)\n"
        << "  --max-iters <int>        Max iterations (default: 2048)\n"
        << "  --tmax <float>           Ray t_max (default: 100)\n"
        << "  --accuracy <float>       Taylor accuracy scale (default: 2)\n"
        << "  --yaw <float>            Orbit yaw in radians (default: 0)\n"
        << "  --pitch <float>          Orbit pitch in radians (default: -0.35)\n"
        << "  --camera-distance <f>    Orbit camera distance (default: 2.7)\n"
        << "  --target-x <f>           Camera target x (default: 0)\n"
        << "  --target-y <f>           Camera target y (default: 0)\n"
        << "  --target-z <f>           Camera target z (default: 0)\n"
        << "  --profile-hole           Auto-profile largest internal background hole\n"
        << "  --profile-px <int>       Profile ray pixel x\n"
        << "  --profile-py <int>       Profile ray pixel y\n"
        << "  --profile-samples <int>  Dense W(t) sample count (default: 8192)\n"
        << "  --profile-out <path>     Optional CSV output for Harnack iteration log\n"
        << "  --help                   Show this message\n";
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

        if (key == "--help")
            return false;

        if (key == "--profile-hole") {
            opt.profile_hole = true;
            continue;
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
        if (key == "--mesh-file") {
            if (!read_value(opt.mesh_file))
                return false;
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
        if (key == "--target-w") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.target_winding))
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
        if (key == "--yaw") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.yaw))
                return false;
            continue;
        }
        if (key == "--pitch") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.pitch))
                return false;
            continue;
        }
        if (key == "--camera-distance") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.camera_distance))
                return false;
            continue;
        }
        if (key == "--target-x") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.target_x))
                return false;
            continue;
        }
        if (key == "--target-y") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.target_y))
                return false;
            continue;
        }
        if (key == "--target-z") {
            std::string value;
            if (!read_value(value) || !parse_real(value, opt.target_z))
                return false;
            continue;
        }
        if (key == "--profile-px") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.profile_px))
                return false;
            continue;
        }
        if (key == "--profile-py") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.profile_py))
                return false;
            continue;
        }
        if (key == "--profile-samples") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.profile_samples))
                return false;
            continue;
        }
        if (key == "--profile-out") {
            if (!read_value(opt.profile_out))
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

[[nodiscard]] bool normalize_mesh(MeshData &mesh, std::string &error) {
    if (mesh.vx.empty() || mesh.i0.empty()) {
        error = "mesh is empty";
        return false;
    }

    Real min_x = std::numeric_limits<Real>::infinity();
    Real min_y = std::numeric_limits<Real>::infinity();
    Real min_z = std::numeric_limits<Real>::infinity();
    Real max_x = -std::numeric_limits<Real>::infinity();
    Real max_y = -std::numeric_limits<Real>::infinity();
    Real max_z = -std::numeric_limits<Real>::infinity();

    for (std::size_t i = 0; i < mesh.vx.size(); ++i) {
        min_x = std::min(min_x, mesh.vx[i]);
        min_y = std::min(min_y, mesh.vy[i]);
        min_z = std::min(min_z, mesh.vz[i]);
        max_x = std::max(max_x, mesh.vx[i]);
        max_y = std::max(max_y, mesh.vy[i]);
        max_z = std::max(max_z, mesh.vz[i]);
    }

    Real const cx = Real(0.5) * (min_x + max_x);
    Real const cy = Real(0.5) * (min_y + max_y);
    Real const cz = Real(0.5) * (min_z + max_z);
    Real const extent = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    if (!(extent > Real(0))) {
        error = "mesh bounding box has zero extent";
        return false;
    }

    Real const scale = Real(2) / extent;
    for (std::size_t i = 0; i < mesh.vx.size(); ++i) {
        mesh.vx[i] = (mesh.vx[i] - cx) * scale;
        mesh.vy[i] = (mesh.vy[i] - cy) * scale;
        mesh.vz[i] = (mesh.vz[i] - cz) * scale;
    }

    return true;
}

[[nodiscard]] bool parse_obj_face_index(
    std::string const &token, int const vertex_count, int &out_index
) {
    std::string index_token = token;
    std::size_t const slash = index_token.find('/');
    if (slash != std::string::npos)
        index_token = index_token.substr(0, slash);
    if (index_token.empty())
        return false;

    int parsed = 0;
    if (!parse_int(index_token, parsed) || parsed == 0)
        return false;

    int idx = -1;
    if (parsed > 0)
        idx = parsed - 1;
    else
        idx = vertex_count + parsed;

    if (idx < 0 || idx >= vertex_count)
        return false;

    out_index = idx;
    return true;
}

[[nodiscard]] bool load_obj_mesh(std::string const &path, MeshData &mesh, std::string &error) {
    std::ifstream in(path);
    if (!in) {
        error = "failed to open OBJ: " + path;
        return false;
    }

    std::vector<Vec3> vertices;
    std::vector<Index> i0;
    std::vector<Index> i1;
    std::vector<Index> i2;

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#')
            continue;

        if (line.size() >= 2 && line[0] == 'v' && std::isspace(static_cast<unsigned char>(line[1]))) {
            std::istringstream iss(line.substr(1));
            Real x = Real(0), y = Real(0), z = Real(0);
            if (!(iss >> x >> y >> z)) {
                error = "invalid vertex at line " + std::to_string(line_no);
                return false;
            }
            vertices.push_back(Vec3{x, y, z});
            continue;
        }

        if (line.size() >= 2 && line[0] == 'f' && std::isspace(static_cast<unsigned char>(line[1]))) {
            std::istringstream iss(line.substr(1));
            std::vector<int> face;
            std::string token;
            while (iss >> token) {
                int idx = -1;
                if (!parse_obj_face_index(token, static_cast<int>(vertices.size()), idx)) {
                    error = "invalid face index at line " + std::to_string(line_no);
                    return false;
                }
                face.push_back(idx);
            }
            if (face.size() < 3u) {
                error = "face has fewer than 3 vertices at line " + std::to_string(line_no);
                return false;
            }
            for (std::size_t k = 1; k + 1 < face.size(); ++k) {
                i0.push_back(static_cast<Index>(face[0]));
                i1.push_back(static_cast<Index>(face[k]));
                i2.push_back(static_cast<Index>(face[k + 1]));
            }
            continue;
        }
    }

    if (vertices.empty() || i0.empty()) {
        error = "OBJ contains no triangles: " + path;
        return false;
    }

    mesh = MeshData{};
    mesh.vx.resize(vertices.size());
    mesh.vy.resize(vertices.size());
    mesh.vz.resize(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        mesh.vx[i] = vertices[i].x;
        mesh.vy[i] = vertices[i].y;
        mesh.vz[i] = vertices[i].z;
    }
    mesh.i0 = std::move(i0);
    mesh.i1 = std::move(i1);
    mesh.i2 = std::move(i2);

    return normalize_mesh(mesh, error);
}

[[nodiscard]] CameraBasis build_camera_basis(Options const &opt) {
    CameraBasis basis{};
    basis.target = Vec3{opt.target_x, opt.target_y, opt.target_z};
    basis.eye = Vec3{
        basis.target.x + std::sin(opt.yaw) * opt.camera_distance,
        basis.target.y + std::sin(opt.pitch) * opt.camera_distance + Real(0.3),
        basis.target.z + std::cos(opt.yaw) * opt.camera_distance,
    };
    Vec3 const up_hint{Real(0), Real(1), Real(0)};
    basis.forward = normalize(basis.target - basis.eye);
    basis.right = normalize(cross(basis.forward, up_hint));
    basis.up = cross(basis.right, basis.forward);
    return basis;
}

[[nodiscard]] Vec3 ray_direction_for_pixel(
    int const px, int const py, Options const &opt, CameraBasis const &camera
) {
    Real const tan_half_fov = std::tan((opt.fov_degrees * k_pi / Real(180)) * Real(0.5));
    Real const aspect = static_cast<Real>(opt.width) / static_cast<Real>(std::max(opt.height, 1));

    Real const sx =
        ((static_cast<Real>(px) + Real(0.5)) / static_cast<Real>(opt.width)) * Real(2) - Real(1);
    Real const sy =
        Real(1) - ((static_cast<Real>(py) + Real(0.5)) / static_cast<Real>(opt.height)) * Real(2);

    return normalize(
        camera.forward + camera.right * (sx * aspect * tan_half_fov) + camera.up * (sy * tan_half_fov)
    );
}

[[nodiscard]] HoleInfo find_largest_internal_hole(
    std::vector<Real> const &hit_t, int const width, int const height
) {
    HoleInfo out{};
    if (width <= 0 || height <= 0)
        return out;

    std::size_t const n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (hit_t.size() != n)
        return out;

    std::vector<std::uint8_t> is_bg(n, std::uint8_t(0));
    for (std::size_t i = 0; i < n; ++i)
        is_bg[i] = (hit_t[i] < Real(0)) ? std::uint8_t(1) : std::uint8_t(0);

    std::vector<std::uint8_t> visited(n, std::uint8_t(0));
    std::vector<int> queue;
    queue.reserve(n);

    auto enqueue = [&](int const x, int const y) {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return;
        std::size_t const idx = static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x);
        if (visited[idx] || !is_bg[idx])
            return;
        visited[idx] = std::uint8_t(1);
        queue.push_back(static_cast<int>(idx));
    };

    for (int x = 0; x < width; ++x) {
        enqueue(x, 0);
        enqueue(x, height - 1);
    }
    for (int y = 0; y < height; ++y) {
        enqueue(0, y);
        enqueue(width - 1, y);
    }

    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        int const idx = queue[qi];
        int const x = idx % width;
        int const y = idx / width;
        enqueue(x - 1, y);
        enqueue(x + 1, y);
        enqueue(x, y - 1);
        enqueue(x, y + 1);
    }

    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            std::size_t const seed = static_cast<std::size_t>(y0) * width + static_cast<std::size_t>(x0);
            if (visited[seed] || !is_bg[seed])
                continue;

            std::size_t area = 0;
            long long sum_x = 0;
            long long sum_y = 0;
            int min_x = x0;
            int min_y = y0;
            int max_x = x0;
            int max_y = y0;

            queue.clear();
            visited[seed] = std::uint8_t(1);
            queue.push_back(static_cast<int>(seed));
            for (std::size_t qi = 0; qi < queue.size(); ++qi) {
                int const idx = queue[qi];
                int const x = idx % width;
                int const y = idx / width;
                ++area;
                sum_x += x;
                sum_y += y;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);

                auto try_push = [&](int const nx, int const ny) {
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                        return;
                    std::size_t const ni =
                        static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx);
                    if (visited[ni] || !is_bg[ni])
                        return;
                    visited[ni] = std::uint8_t(1);
                    queue.push_back(static_cast<int>(ni));
                };
                try_push(x - 1, y);
                try_push(x + 1, y);
                try_push(x, y - 1);
                try_push(x, y + 1);
            }

            if (area > out.area) {
                out.found = true;
                out.area = area;
                out.centroid_x = static_cast<int>(std::lround(static_cast<double>(sum_x) / area));
                out.centroid_y = static_cast<int>(std::lround(static_cast<double>(sum_y) / area));
                out.sample_x = x0;
                out.sample_y = y0;
                out.min_x = min_x;
                out.min_y = min_y;
                out.max_x = max_x;
                out.max_y = max_y;
            }
        }
    }

    return out;
}

__global__ void probe_point_kernel(
    gwn::gwn_geometry_accessor<Real, Index> geometry,
    gwn::gwn_bvh4_topology_accessor<Real, Index> bvh,
    gwn::gwn_bvh4_aabb_accessor<Real, Index> aabb,
    gwn::gwn_bvh4_moment_accessor<1, Real, Index> moments, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale, PointProbe *out
) {
    auto const wg = gwn::detail::gwn_winding_and_gradient_point_bvh_taylor_impl<
        1, k_bvh_width, Real, Index, k_stack_capacity>(geometry, bvh, moments, qx, qy, qz, accuracy_scale);

    Real const r_face = gwn::detail::gwn_unsigned_distance_point_bvh_impl<
        k_bvh_width, Real, Index, k_stack_capacity>(
        geometry, bvh, aabb, qx, qy, qz, std::numeric_limits<Real>::infinity()
    );
    Real const r_edge = gwn::detail::gwn_unsigned_edge_distance_point_bvh_impl<
        k_bvh_width, Real, Index, k_stack_capacity>(
        geometry, bvh, aabb, qx, qy, qz, std::numeric_limits<Real>::infinity()
    );

    out[0] = PointProbe{
        wg.winding,
        wg.gradient.x,
        wg.gradient.y,
        wg.gradient.z,
        r_face,
        r_edge,
    };
}

[[nodiscard]] PointProbe probe_point(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> const &bvh,
    gwn::gwn_bvh4_aabb_object<Real, Index> const &aabb,
    gwn::gwn_bvh4_moment_object<1, Real, Index> const &moments, Real const qx, Real const qy,
    Real const qz, Real const accuracy_scale, gwn::gwn_device_array<PointProbe> &d_probe
) {
    probe_point_kernel<<<1, 1>>>(
        geometry.accessor(), bvh.accessor(), aabb.accessor(), moments.accessor(), qx, qy, qz,
        accuracy_scale, d_probe.data()
    );
    throw_if_error(gwn::gwn_cuda_to_status(cudaGetLastError()), "probe_point_kernel launch");

    PointProbe host{};
    throw_if_error(d_probe.copy_to_host(cuda::std::span<PointProbe>(&host, 1)), "copy probe");
    throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "probe sync");
    return host;
}

void write_profile_csv(std::string const &path, std::vector<IterRecord> const &records) {
    std::filesystem::path p(path);
    if (!p.parent_path().empty())
        std::filesystem::create_directories(p.parent_path());

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open profile CSV: " + path);

    out << "iter,t_eval,w,wrapped,dist,grad_omega_mag,r_face,r_edge,rho,t_overstep_before,accepted,hit_by_grad,hit_by_radius,invalid_radius,invalid_rho\n";
    for (IterRecord const &r : records) {
        out << r.iter << ',' << r.t_eval << ',' << r.w << ',' << r.wrapped << ',' << r.dist << ','
            << r.grad_omega_mag << ',' << r.r_face << ',' << r.r_edge << ',' << r.rho << ','
            << r.t_overstep_before << ',' << (r.accepted ? 1 : 0) << ','
            << (r.hit_by_grad ? 1 : 0) << ',' << (r.hit_by_radius ? 1 : 0) << ','
            << (r.invalid_radius ? 1 : 0) << ',' << (r.invalid_rho ? 1 : 0) << '\n';
    }
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
        if (!(opt.fov_degrees > Real(0)) || !(opt.camera_distance > Real(0)) ||
            opt.max_iterations < 0 || !(opt.t_max > Real(0)) || opt.profile_samples < 2) {
            std::cerr << "invalid numeric options.\n";
            return 1;
        }
        if (!opt.mesh_file.empty() && !std::filesystem::exists(opt.mesh_file)) {
            std::cerr << "mesh file does not exist: " << opt.mesh_file << "\n";
            return 1;
        }
        if (opt.mesh_file.empty() && opt.mesh != "octa" && opt.mesh != "half") {
            std::cerr << "mesh must be 'octa' or 'half' unless --mesh-file is provided.\n";
            return 1;
        }

        cudaError_t const probe = cudaFree(nullptr);
        if (probe != cudaSuccess) {
            std::cerr << "CUDA runtime unavailable: " << cudaGetErrorString(probe) << "\n";
            return 2;
        }

        MeshData mesh{};
        if (!opt.mesh_file.empty()) {
            std::string load_error;
            if (!load_obj_mesh(opt.mesh_file, mesh, load_error)) {
                std::cerr << "mesh load failed: " << load_error << "\n";
                return 1;
            }
        } else {
            mesh = (opt.mesh == "half") ? make_half_octahedron_mesh() : make_octahedron_mesh();
        }

        gwn::gwn_geometry_object<Real, Index> geometry;
        throw_if_error(
            gwn::gwn_upload_geometry(
                geometry,
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

        CameraBasis const camera = build_camera_basis(opt);

        std::vector<Real> ox(pixel_count), oy(pixel_count), oz(pixel_count);
        std::vector<Real> dx(pixel_count), dy(pixel_count), dz(pixel_count);

        for (int py = 0; py < opt.height; ++py) {
            for (int px = 0; px < opt.width; ++px) {
                std::size_t const idx = static_cast<std::size_t>(py) * opt.width + px;
                Vec3 const rd = ray_direction_for_pixel(px, py, opt, camera);
                ox[idx] = camera.eye.x;
                oy[idx] = camera.eye.y;
                oz[idx] = camera.eye.z;
                dx[idx] = rd.x;
                dy[idx] = rd.y;
                dz[idx] = rd.z;
            }
        }

        gwn::gwn_device_array<Real> d_ox, d_oy, d_oz, d_dx, d_dy, d_dz;
        gwn::gwn_device_array<Real> d_t, d_nx, d_ny, d_nz;

        throw_if_error(d_ox.copy_from_host(cuda::std::span<Real const>(ox.data(), ox.size())), "copy d_ox");
        throw_if_error(d_oy.copy_from_host(cuda::std::span<Real const>(oy.data(), oy.size())), "copy d_oy");
        throw_if_error(d_oz.copy_from_host(cuda::std::span<Real const>(oz.data(), oz.size())), "copy d_oz");
        throw_if_error(d_dx.copy_from_host(cuda::std::span<Real const>(dx.data(), dx.size())), "copy d_dx");
        throw_if_error(d_dy.copy_from_host(cuda::std::span<Real const>(dy.data(), dy.size())), "copy d_dy");
        throw_if_error(d_dz.copy_from_host(cuda::std::span<Real const>(dz.data(), dz.size())), "copy d_dz");
        throw_if_error(d_t.resize(pixel_count), "resize d_t");
        throw_if_error(d_nx.resize(pixel_count), "resize d_nx");
        throw_if_error(d_ny.resize(pixel_count), "resize d_ny");
        throw_if_error(d_nz.resize(pixel_count), "resize d_nz");

        throw_if_error(
            gwn::gwn_compute_harnack_trace_batch_bvh_taylor<1, Real, Index>(
                geometry.accessor(), bvh.accessor(), aabb.accessor(), moments.accessor(),
                d_ox.span(), d_oy.span(), d_oz.span(), d_dx.span(), d_dy.span(), d_dz.span(),
                d_t.span(), d_nx.span(), d_ny.span(), d_nz.span(), opt.target_winding, opt.epsilon,
                opt.max_iterations, opt.t_max, opt.accuracy_scale
            ),
            "gwn_compute_harnack_trace_batch_bvh_taylor"
        );
        throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "trace sync");

        std::vector<Real> host_t(pixel_count), host_nx(pixel_count), host_ny(pixel_count),
            host_nz(pixel_count);
        throw_if_error(d_t.copy_to_host(cuda::std::span<Real>(host_t.data(), host_t.size())), "copy host_t");
        throw_if_error(d_nx.copy_to_host(cuda::std::span<Real>(host_nx.data(), host_nx.size())), "copy host_nx");
        throw_if_error(d_ny.copy_to_host(cuda::std::span<Real>(host_ny.data(), host_ny.size())), "copy host_ny");
        throw_if_error(d_nz.copy_to_host(cuda::std::span<Real>(host_nz.data(), host_nz.size())), "copy host_nz");
        throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "copy sync");

        std::size_t hit_count = 0;
        Real t_min = std::numeric_limits<Real>::infinity();
        Real t_max_hit = Real(0);
        for (Real const t : host_t) {
            if (t >= Real(0)) {
                ++hit_count;
                t_min = std::min(t_min, t);
                t_max_hit = std::max(t_max_hit, t);
            }
        }

        if (hit_count == 0)
            std::cerr << "No hits found.\n";

        std::vector<std::uint8_t> normal_rgb(pixel_count * 3, std::uint8_t(0));
        std::vector<std::uint8_t> depth_rgb(pixel_count * 3, std::uint8_t(0));
        Real const t_range = (t_max_hit > t_min) ? (t_max_hit - t_min) : Real(1);

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
                  << "/" << pixel_count << " targetW=" << opt.target_winding << "\n";
        std::cout << "Wrote normal: " << opt.normal_out << "\n";
        std::cout << "Wrote depth : " << opt.depth_out << "\n";

        int profile_px = opt.profile_px;
        int profile_py = opt.profile_py;
        if (opt.profile_hole) {
            HoleInfo const hole = find_largest_internal_hole(host_t, opt.width, opt.height);
            if (hole.found) {
                profile_px = hole.sample_x;
                profile_py = hole.sample_y;
                std::cout << "Detected largest internal hole: area=" << hole.area << " bbox=["
                          << hole.min_x << ',' << hole.min_y << " -> " << hole.max_x << ','
                          << hole.max_y << "] centroid=(" << hole.centroid_x << ',' << hole.centroid_y
                          << ") sample=(" << hole.sample_x << ',' << hole.sample_y << ")\n";
            } else {
                std::cout << "No internal hole region detected in this render.\n";
            }
        }

        bool const has_profile_pixel =
            (profile_px >= 0 && profile_py >= 0 && profile_px < opt.width && profile_py < opt.height);
        if (has_profile_pixel) {
            std::cout << "\n=== Ray Profile @ pixel (" << profile_px << ", " << profile_py << ") ===\n";

            Vec3 const ray_dir = ray_direction_for_pixel(profile_px, profile_py, opt, camera);
            std::size_t const pixel_idx =
                static_cast<std::size_t>(profile_py) * opt.width + static_cast<std::size_t>(profile_px);
            bool const tracer_hit = host_t[pixel_idx] >= Real(0);
            std::cout << "Trace image result: hit=" << (tracer_hit ? "yes" : "no")
                      << " t=" << host_t[pixel_idx] << "\n";

            int const dense_n = std::max(opt.profile_samples, 2);
            std::vector<Real> sample_t(static_cast<std::size_t>(dense_n));
            std::vector<Real> qx(static_cast<std::size_t>(dense_n));
            std::vector<Real> qy(static_cast<std::size_t>(dense_n));
            std::vector<Real> qz(static_cast<std::size_t>(dense_n));
            for (int i = 0; i < dense_n; ++i) {
                Real const alpha = static_cast<Real>(i) / static_cast<Real>(dense_n - 1);
                Real const t = alpha * opt.t_max;
                sample_t[static_cast<std::size_t>(i)] = t;
                qx[static_cast<std::size_t>(i)] = camera.eye.x + t * ray_dir.x;
                qy[static_cast<std::size_t>(i)] = camera.eye.y + t * ray_dir.y;
                qz[static_cast<std::size_t>(i)] = camera.eye.z + t * ray_dir.z;
            }

            gwn::gwn_device_array<Real> d_qx, d_qy, d_qz, d_w;
            throw_if_error(d_qx.copy_from_host(cuda::std::span<Real const>(qx.data(), qx.size())), "copy d_qx");
            throw_if_error(d_qy.copy_from_host(cuda::std::span<Real const>(qy.data(), qy.size())), "copy d_qy");
            throw_if_error(d_qz.copy_from_host(cuda::std::span<Real const>(qz.data(), qz.size())), "copy d_qz");
            throw_if_error(d_w.resize(qx.size()), "resize d_w");

            throw_if_error(
                gwn::gwn_compute_winding_number_batch_bvh_taylor<1, Real, Index>(
                    geometry.accessor(), bvh.accessor(), moments.accessor(), d_qx.span(), d_qy.span(),
                    d_qz.span(), d_w.span(), opt.accuracy_scale
                ),
                "dense winding query"
            );
            throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "dense winding sync");

            std::vector<Real> sample_w(sample_t.size(), Real(0));
            throw_if_error(d_w.copy_to_host(cuda::std::span<Real>(sample_w.data(), sample_w.size())), "copy sample_w");
            throw_if_error(gwn::gwn_cuda_to_status(cudaDeviceSynchronize()), "copy sample_w sync");

            Real min_w = std::numeric_limits<Real>::infinity();
            Real max_w = -std::numeric_limits<Real>::infinity();
            Real min_abs_delta = std::numeric_limits<Real>::infinity();
            Real near_t = Real(0);
            int sign_crossings = 0;

            for (int i = 0; i < dense_n; ++i) {
                Real const w = sample_w[static_cast<std::size_t>(i)];
                min_w = std::min(min_w, w);
                max_w = std::max(max_w, w);
                Real const abs_delta = std::abs(w - opt.target_winding);
                if (abs_delta < min_abs_delta) {
                    min_abs_delta = abs_delta;
                    near_t = sample_t[static_cast<std::size_t>(i)];
                }
                if (i > 0) {
                    Real const a = sample_w[static_cast<std::size_t>(i - 1)] - opt.target_winding;
                    Real const b = sample_w[static_cast<std::size_t>(i)] - opt.target_winding;
                    if ((a <= Real(0) && b >= Real(0)) || (a >= Real(0) && b <= Real(0))) {
                        if (a != b)
                            ++sign_crossings;
                    }
                }
            }

            std::cout << "Dense ray samples: N=" << dense_n << " W_min=" << min_w
                      << " W_max=" << max_w << " crossings=" << sign_crossings
                      << " nearest |W-target|=" << min_abs_delta << " at t=" << near_t << "\n";

            gwn::gwn_device_array<PointProbe> d_probe;
            throw_if_error(d_probe.resize(1), "resize d_probe");

            std::vector<IterRecord> records;
            records.reserve(static_cast<std::size_t>(opt.max_iterations) + 2u);

            Real const dir_len = std::sqrt(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z);
            Real const target_omega = k_four_pi * opt.target_winding;

            Real t = Real(0);
            Real t_overstep = Real(0);
            int commits = 0;
            int backoffs = 0;
            bool loop_hit = false;
            Real loop_hit_t = Real(-1);
            std::string terminal_reason = "t_max_exceeded";

            for (int iter = 0; iter <= opt.max_iterations && t < opt.t_max; ++iter) {
                Real const t_eval = t + t_overstep;
                Real const px = camera.eye.x + t_eval * ray_dir.x;
                Real const py = camera.eye.y + t_eval * ray_dir.y;
                Real const pz = camera.eye.z + t_eval * ray_dir.z;

                PointProbe const pp = probe_point(
                    geometry, bvh, aabb, moments, px, py, pz, opt.accuracy_scale, d_probe
                );

                Real const grad_mag =
                    std::sqrt(pp.grad_x * pp.grad_x + pp.grad_y * pp.grad_y + pp.grad_z * pp.grad_z);
                Real const grad_omega_mag = k_four_pi * grad_mag;
                Real const omega = k_four_pi * pp.winding;
                Real const wrapped = gwn::detail::gwn_glsl_mod<Real>(omega - target_omega, k_four_pi);
                Real const dist_lo = wrapped;
                Real const dist_hi = k_four_pi - wrapped;
                Real const dist = (dist_lo < dist_hi) ? dist_lo : dist_hi;

                IterRecord rec{};
                rec.iter = iter;
                rec.t_eval = t_eval;
                rec.w = pp.winding;
                rec.wrapped = wrapped;
                rec.dist = dist;
                rec.grad_omega_mag = grad_omega_mag;
                rec.r_face = pp.r_face;
                rec.r_edge = pp.r_edge;
                rec.t_overstep_before = t_overstep;

                if (!(pp.r_face >= Real(0))) {
                    rec.invalid_radius = true;
                    terminal_reason = "invalid_radius";
                    records.push_back(rec);
                    break;
                }

                Real const rho =
                    gwn::detail::gwn_harnack_constrained_two_sided_step(
                        wrapped, Real(0), k_four_pi, -k_four_pi, pp.r_face
                    ) /
                    dir_len;
                rec.rho = rho;

                if (!(rho >= Real(0))) {
                    rec.invalid_rho = true;
                    terminal_reason = "invalid_rho";
                    records.push_back(rec);
                    break;
                }

                if (rho >= t_overstep) {
                    rec.accepted = true;
                    bool const hit_by_grad = dist < opt.epsilon * grad_omega_mag;
                    bool const hit_by_radius = pp.r_face < opt.epsilon;
                    rec.hit_by_grad = hit_by_grad;
                    rec.hit_by_radius = hit_by_radius;
                    records.push_back(rec);

                    if (hit_by_grad || hit_by_radius) {
                        loop_hit = true;
                        loop_hit_t = t_eval;
                        terminal_reason = hit_by_grad ? "grad_stop" : "radius_stop";
                        break;
                    }

                    t += t_overstep + rho;
                    t_overstep = rho * Real(0.75);
                    ++commits;
                } else {
                    rec.accepted = false;
                    records.push_back(rec);
                    t_overstep = Real(0);
                    ++backoffs;
                }
            }

            if (!loop_hit && terminal_reason == "t_max_exceeded" && t < opt.t_max &&
                static_cast<int>(records.size()) > opt.max_iterations) {
                terminal_reason = "max_iterations_reached";
            }

            std::cout << "Harnack loop result: hit=" << (loop_hit ? "yes" : "no")
                      << " t=" << loop_hit_t << " commits=" << commits << " backoffs=" << backoffs
                      << " steps=" << records.size() << " reason=" << terminal_reason << "\n";

            std::cout << "Iter log (first 40 rows):\n";
            std::cout << "iter t_eval w dist gradOm Rface Redge rho overstep accept hitG hitR\n";
            int const show = std::min<int>(static_cast<int>(records.size()), 40);
            for (int i = 0; i < show; ++i) {
                IterRecord const &r = records[static_cast<std::size_t>(i)];
                std::cout << r.iter << ' ' << r.t_eval << ' ' << r.w << ' ' << r.dist << ' '
                          << r.grad_omega_mag << ' ' << r.r_face << ' ' << r.r_edge << ' ' << r.rho
                          << ' ' << r.t_overstep_before << ' ' << (r.accepted ? 1 : 0) << ' '
                          << (r.hit_by_grad ? 1 : 0) << ' ' << (r.hit_by_radius ? 1 : 0) << '\n';
            }
            if (static_cast<int>(records.size()) > show)
                std::cout << "... (" << (records.size() - static_cast<std::size_t>(show))
                          << " more rows)\n";

            if (!opt.profile_out.empty()) {
                write_profile_csv(opt.profile_out, records);
                std::cout << "Wrote profile CSV: " << opt.profile_out << "\n";
            }
        }

        return 0;
    } catch (std::exception const &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

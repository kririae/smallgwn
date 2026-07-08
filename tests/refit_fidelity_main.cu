#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gwn/gwn.cuh>

#include "refit_fidelity_laplacian.hpp"
#include "test_utils.hpp"

namespace {

using Real = float;
using Index = std::uint32_t;
using HostMesh = gwn::tests::HostMesh;
using Frame = gwn::tests::gwn_refit_fidelity_frame;

constexpr int k_width = 4;
constexpr Real k_accuracy_scale = Real(2);

enum class builder_kind {
    k_lbvh,
    k_hploc,
};

struct options {
    std::optional<std::filesystem::path> model_path{};
    std::optional<std::filesystem::path> model_dir{};
    std::optional<std::filesystem::path> csv_path{};
    std::size_t model_limit{0};
    std::size_t frame_count{4};
    std::size_t query_count{512};
    int smoothing_iterations{1};
    Real smoothing_lambda{Real(0.25)};
    Real value_epsilon{Real(2e-5)};
    Real gradient_epsilon{Real(2e-4)};
};

struct bounds {
    Real min_x{};
    Real min_y{};
    Real min_z{};
    Real max_x{};
    Real max_y{};
    Real max_z{};
};

struct query_set {
    std::vector<Real> x;
    std::vector<Real> y;
    std::vector<Real> z;
    gwn::gwn_device_array<Real> dx;
    gwn::gwn_device_array<Real> dy;
    gwn::gwn_device_array<Real> dz;
};

struct diff_stats {
    double max_abs{0.0};
    double p99_abs{0.0};
    double mean_abs{0.0};
};

struct pipeline_state {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_bvh4_aabb_object<Real, Index> aabb;
    gwn::gwn_boundary_chain_object<Index> boundary;
    gwn::gwn_bvh4_moment_object<0, Real, Index> moment_o0;
    gwn::gwn_bvh4_moment_object<1, Real, Index> moment_o1;
    gwn::gwn_bvh4_moment_object<2, Real, Index> moment_o2;
};

struct csv_writer {
    std::ofstream stream{};

    explicit csv_writer(std::optional<std::filesystem::path> const &path) {
        if (path.has_value()) {
            stream.open(*path, std::ios::out | std::ios::trunc);
            stream << "model,builder,frame,stage,order,triangles,queries,max_abs,p99_abs,mean_abs,"
                      "epsilon,success\n";
        }
    }

    void append(
        std::filesystem::path const &model, std::string_view const builder, std::size_t const frame,
        std::string_view const stage, int const order, std::size_t const triangles,
        std::size_t const queries, diff_stats const &stats, double const epsilon, bool const success
    ) {
        if (!stream.is_open())
            return;
        stream << model.filename().string() << "," << builder << "," << frame << "," << stage << ","
               << order << "," << triangles << "," << queries << "," << stats.max_abs << ","
               << stats.p99_abs << "," << stats.mean_abs << "," << epsilon << ","
               << (success ? 1 : 0) << "\n";
    }
};

[[nodiscard]] std::string_view to_string(builder_kind const builder) noexcept {
    switch (builder) {
    case builder_kind::k_lbvh: return "lbvh";
    case builder_kind::k_hploc: return "hploc";
    }
    return "unknown";
}

template <class T> [[nodiscard]] bool parse_positive(std::string_view const text, T &out) {
    T value{};
    auto const *first = text.data();
    auto const *last = text.data() + text.size();
    auto const [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc() || ptr != last || value <= T(0))
        return false;
    out = value;
    return true;
}

[[nodiscard]] bool parse_float(std::string_view const text, Real &out) {
    try {
        std::size_t parsed = 0;
        float const value = std::stof(std::string(text), &parsed);
        if (parsed != text.size() || !(value > 0.0f))
            return false;
        out = value;
        return true;
    } catch (...) { return false; }
}

[[nodiscard]] std::optional<std::string_view>
option_value(int &i, int const argc, char const *const *argv) {
    if (i + 1 >= argc)
        return std::nullopt;
    ++i;
    return std::string_view(argv[i]);
}

[[nodiscard]] bool parse_cli(int const argc, char const *const *argv, options &out) {
    for (int i = 1; i < argc; ++i) {
        std::string_view const arg(argv[i]);
        if (arg == "--model") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value())
                return false;
            out.model_path = std::filesystem::path(std::string(*value));
        } else if (arg == "--model-dir") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value())
                return false;
            out.model_dir = std::filesystem::path(std::string(*value));
        } else if (arg == "--csv") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value())
                return false;
            out.csv_path = std::filesystem::path(std::string(*value));
        } else if (arg == "--model-limit") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_positive(*value, out.model_limit))
                return false;
        } else if (arg == "--frames") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_positive(*value, out.frame_count))
                return false;
        } else if (arg == "--queries") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_positive(*value, out.query_count))
                return false;
        } else if (arg == "--smooth-iters") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_positive(*value, out.smoothing_iterations))
                return false;
        } else if (arg == "--smooth-lambda") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_float(*value, out.smoothing_lambda))
                return false;
        } else if (arg == "--value-epsilon") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_float(*value, out.value_epsilon))
                return false;
        } else if (arg == "--gradient-epsilon") {
            auto const value = option_value(i, argc, argv);
            if (!value.has_value() || !parse_float(*value, out.gradient_epsilon))
                return false;
        } else if (arg == "--help") {
            return false;
        } else {
            return false;
        }
    }
    return out.model_path.has_value() != out.model_dir.has_value();
}

void print_usage(char const *argv0) {
    std::cerr << "usage:\n"
              << "  " << argv0 << " --model <mesh.obj> [options]\n"
              << "  " << argv0 << " --model-dir <dir> [options]\n\n"
              << "options:\n"
              << "  --frames <N>              smoothed frames per model (default: 4)\n"
              << "  --queries <N>             query count (default: 512)\n"
              << "  --smooth-iters <N>        smoothing iterations per frame (default: 1)\n"
              << "  --smooth-lambda <x>       smoothing blend weight (default: 0.25)\n"
              << "  --model-limit <N>         max models from --model-dir\n"
              << "  --value-epsilon <x>       winding diff gate (default: 2e-5)\n"
              << "  --gradient-epsilon <x>    gradient diff gate (default: 2e-4)\n"
              << "  --csv <path>              write CSV report\n";
}

[[nodiscard]] bounds bounds_of(HostMesh const &mesh) {
    bounds b{mesh.vertex_x[0], mesh.vertex_y[0], mesh.vertex_z[0],
             mesh.vertex_x[0], mesh.vertex_y[0], mesh.vertex_z[0]};
    for (std::size_t i = 1; i < mesh.vertex_x.size(); ++i) {
        b.min_x = std::min(b.min_x, mesh.vertex_x[i]);
        b.min_y = std::min(b.min_y, mesh.vertex_y[i]);
        b.min_z = std::min(b.min_z, mesh.vertex_z[i]);
        b.max_x = std::max(b.max_x, mesh.vertex_x[i]);
        b.max_y = std::max(b.max_y, mesh.vertex_y[i]);
        b.max_z = std::max(b.max_z, mesh.vertex_z[i]);
    }
    return b;
}

[[nodiscard]] HostMesh mesh_with_frame(HostMesh const &mesh, Frame const &frame) {
    HostMesh out = mesh;
    out.vertex_x = frame.x;
    out.vertex_y = frame.y;
    out.vertex_z = frame.z;
    return out;
}

[[nodiscard]] bool status_ok(gwn::gwn_status const status, char const *label) {
    if (status.is_ok())
        return true;
    std::cerr << label << ": " << gwn::tests::status_to_debug_string(status) << "\n";
    return false;
}

[[nodiscard]] bool upload_geometry(HostMesh const &mesh, gwn::gwn_geometry_object<Real, Index> &g) {
    return status_ok(
        gwn::gwn_upload_geometry(
            g, cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size())
        ),
        "upload geometry"
    );
}

[[nodiscard]] bool update_geometry(Frame const &frame, gwn::gwn_geometry_object<Real, Index> &g) {
    return status_ok(
        gwn::gwn_update_geometry(
            g, cuda::std::span<Real const>(frame.x.data(), frame.x.size()),
            cuda::std::span<Real const>(frame.y.data(), frame.y.size()),
            cuda::std::span<Real const>(frame.z.data(), frame.z.size())
        ),
        "update geometry"
    );
}

[[nodiscard]] bool build_topology(
    builder_kind const builder, gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_topology_object<Real, Index> &topology
) {
    if (builder == builder_kind::k_hploc) {
        return status_ok(
            gwn::gwn_bvh_topology_build_hploc<k_width, Real, Index>(geometry, topology),
            "build hploc topology"
        );
    }
    return status_ok(
        gwn::gwn_bvh_topology_build_lbvh<k_width, Real, Index>(geometry, topology),
        "build lbvh topology"
    );
}

[[nodiscard]] bool refit_aabb_and_boundary(
    gwn::gwn_bvh4_topology_object<Real, Index> const &topology, pipeline_state &state
) {
    if (!status_ok(
            gwn::gwn_bvh_refit_aabb<k_width, Real, Index>(state.geometry, topology, state.aabb),
            "refit aabb"
        ))
        return false;
    return status_ok(
        gwn::gwn_build_boundary_chain(state.geometry.accessor(), state.boundary),
        "build boundary chain"
    );
}

template <int Order>
[[nodiscard]] gwn::gwn_bvh4_moment_object<Order, Real, Index> &moment_of(pipeline_state &state);

template <> gwn::gwn_bvh4_moment_object<0, Real, Index> &moment_of<0>(pipeline_state &state) {
    return state.moment_o0;
}

template <> gwn::gwn_bvh4_moment_object<1, Real, Index> &moment_of<1>(pipeline_state &state) {
    return state.moment_o1;
}

template <> gwn::gwn_bvh4_moment_object<2, Real, Index> &moment_of<2>(pipeline_state &state) {
    return state.moment_o2;
}

template <int Order>
[[nodiscard]] bool
refit_moment(gwn::gwn_bvh4_topology_object<Real, Index> const &topology, pipeline_state &state) {
    return status_ok(
        gwn::gwn_bvh_refit_moment<Order, k_width, Real, Index>(
            state.geometry, topology, state.aabb, moment_of<Order>(state)
        ),
        "refit moment"
    );
}

[[nodiscard]] std::vector<int> checked_int_indices(std::vector<Index> const &indices) {
    std::vector<int> out;
    out.reserve(indices.size());
    for (Index const value : indices) {
        if (value > static_cast<Index>(std::numeric_limits<int>::max()))
            return {};
        out.push_back(static_cast<int>(value));
    }
    return out;
}

[[nodiscard]] std::vector<Frame> make_frames(HostMesh const &mesh, options const &opts) {
    std::vector<int> const i0 = checked_int_indices(mesh.tri_i0);
    std::vector<int> const i1 = checked_int_indices(mesh.tri_i1);
    std::vector<int> const i2 = checked_int_indices(mesh.tri_i2);
    if (i0.empty() || i1.empty() || i2.empty())
        return {};
    return gwn::tests::gwn_generate_laplacian_smoothing_frames(
        mesh.vertex_x.data(), mesh.vertex_y.data(), mesh.vertex_z.data(), mesh.vertex_x.size(),
        i0.data(), i1.data(), i2.data(), mesh.tri_i0.size(), opts.frame_count,
        opts.smoothing_iterations, opts.smoothing_lambda
    );
}

[[nodiscard]] bool make_queries(HostMesh const &mesh, std::size_t const count, query_set &queries) {
    bounds const b = bounds_of(mesh);
    Real const cx = (b.min_x + b.max_x) * Real(0.5);
    Real const cy = (b.min_y + b.max_y) * Real(0.5);
    Real const cz = (b.min_z + b.max_z) * Real(0.5);
    Real const ex = std::max(b.max_x - b.min_x, Real(1e-3));
    Real const ey = std::max(b.max_y - b.min_y, Real(1e-3));
    Real const ez = std::max(b.max_z - b.min_z, Real(1e-3));

    queries.x.resize(count);
    queries.y.resize(count);
    queries.z.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        Real const u = Real((i * 37u + 11u) % 101u) / Real(100);
        Real const v = Real((i * 53u + 7u) % 103u) / Real(102);
        Real const w = Real((i * 71u + 19u) % 107u) / Real(106);
        queries.x[i] = cx + (u - Real(0.5)) * ex * Real(2.7);
        queries.y[i] = cy + (v - Real(0.5)) * ey * Real(2.7);
        queries.z[i] = cz + (w - Real(0.5)) * ez * Real(2.7);
    }

    return status_ok(
               queries.dx.copy_from_host(cuda::std::span<Real const>(queries.x.data(), count)),
               "copy query x"
           ) &&
           status_ok(
               queries.dy.copy_from_host(cuda::std::span<Real const>(queries.y.data(), count)),
               "copy query y"
           ) &&
           status_ok(
               queries.dz.copy_from_host(cuda::std::span<Real const>(queries.z.data(), count)),
               "copy query z"
           );
}

[[nodiscard]] bool copy_output(gwn::gwn_device_array<Real> &device, std::vector<Real> &host) {
    host.resize(device.size());
    if (!status_ok(
            device.copy_to_host(cuda::std::span<Real>(host.data(), host.size())), "copy output"
        ))
        return false;
    cudaError_t const sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        std::cerr << "cuda sync: " << cudaGetErrorString(sync) << "\n";
        return false;
    }
    return true;
}

template <int Order>
[[nodiscard]] bool query_taylor_winding(
    pipeline_state &state, gwn::gwn_bvh4_topology_object<Real, Index> const &topology,
    query_set &queries, std::vector<Real> &out
) {
    gwn::gwn_device_array<Real> d_out;
    if (!status_ok(d_out.resize(queries.x.size()), "resize winding output"))
        return false;
    if (!status_ok(
            gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, Real, Index>(
                state.geometry.accessor(), topology.accessor(), moment_of<Order>(state).accessor(),
                queries.dx.span(), queries.dy.span(), queries.dz.span(), d_out.span(),
                k_accuracy_scale
            ),
            "query taylor winding"
        ))
        return false;
    return copy_output(d_out, out);
}

template <int Order>
[[nodiscard]] bool query_taylor_gradient(
    pipeline_state &state, gwn::gwn_bvh4_topology_object<Real, Index> const &topology,
    query_set &queries, std::vector<Real> &out_x, std::vector<Real> &out_y, std::vector<Real> &out_z
) {
    gwn::gwn_device_array<Real> d_x;
    gwn::gwn_device_array<Real> d_y;
    gwn::gwn_device_array<Real> d_z;
    if (!status_ok(d_x.resize(queries.x.size()), "resize gradient x") ||
        !status_ok(d_y.resize(queries.x.size()), "resize gradient y") ||
        !status_ok(d_z.resize(queries.x.size()), "resize gradient z"))
        return false;
    if (!status_ok(
            gwn::gwn_compute_winding_gradient_batch_bvh_taylor<Order, Real, Index>(
                state.geometry.accessor(), topology.accessor(), moment_of<Order>(state).accessor(),
                queries.dx.span(), queries.dy.span(), queries.dz.span(), d_x.span(), d_y.span(),
                d_z.span(), k_accuracy_scale
            ),
            "query taylor gradient"
        ))
        return false;
    return copy_output(d_x, out_x) && copy_output(d_y, out_y) && copy_output(d_z, out_z);
}

[[nodiscard]] bool query_antipodal_winding(
    pipeline_state &state, gwn::gwn_bvh4_topology_object<Real, Index> const &topology,
    query_set &queries, std::vector<Real> &out
) {
    gwn::gwn_device_array<Real> d_out;
    if (!status_ok(d_out.resize(queries.x.size()), "resize antipodal output"))
        return false;
    if (!status_ok(
            gwn::gwn_compute_winding_number_batch_bvh_antipodal<Real, Index>(
                state.geometry.accessor(), topology.accessor(), state.aabb.accessor(),
                state.boundary.accessor(), queries.dx.span(), queries.dy.span(), queries.dz.span(),
                d_out.span()
            ),
            "query antipodal winding"
        ))
        return false;
    return copy_output(d_out, out);
}

[[nodiscard]] bool query_antipodal_gradient(
    pipeline_state &state, query_set &queries, std::vector<Real> &out_x, std::vector<Real> &out_y,
    std::vector<Real> &out_z
) {
    gwn::gwn_device_array<Real> d_x;
    gwn::gwn_device_array<Real> d_y;
    gwn::gwn_device_array<Real> d_z;
    if (!status_ok(d_x.resize(queries.x.size()), "resize antipodal gradient x") ||
        !status_ok(d_y.resize(queries.x.size()), "resize antipodal gradient y") ||
        !status_ok(d_z.resize(queries.x.size()), "resize antipodal gradient z"))
        return false;
    if (!status_ok(
            gwn::gwn_compute_winding_gradient_batch_antipodal<Real, Index>(
                state.geometry.accessor(), state.boundary.accessor(), queries.dx.span(),
                queries.dy.span(), queries.dz.span(), d_x.span(), d_y.span(), d_z.span()
            ),
            "query antipodal gradient"
        ))
        return false;
    return copy_output(d_x, out_x) && copy_output(d_y, out_y) && copy_output(d_z, out_z);
}

[[nodiscard]] diff_stats scalar_diff(std::vector<Real> const &a, std::vector<Real> const &b) {
    std::vector<double> diffs;
    diffs.reserve(a.size());
    double sum = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return {
                std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            };
        double const d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        diffs.push_back(d);
        sum += d;
        max_abs = std::max(max_abs, d);
    }
    std::sort(diffs.begin(), diffs.end());
    std::size_t const p99_index =
        diffs.empty() ? 0 : std::min(diffs.size() - 1, (diffs.size() * 99) / 100);
    return {
        max_abs, diffs.empty() ? 0.0 : diffs[p99_index],
        diffs.empty() ? 0.0 : sum / static_cast<double>(diffs.size())
    };
}

[[nodiscard]] diff_stats vector_diff(
    std::vector<Real> const &ax, std::vector<Real> const &ay, std::vector<Real> const &az,
    std::vector<Real> const &bx, std::vector<Real> const &by, std::vector<Real> const &bz
) {
    std::vector<double> diffs;
    diffs.reserve(ax.size());
    double sum = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < ax.size(); ++i) {
        if (!std::isfinite(ax[i]) || !std::isfinite(ay[i]) || !std::isfinite(az[i]) ||
            !std::isfinite(bx[i]) || !std::isfinite(by[i]) || !std::isfinite(bz[i])) {
            return {
                std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            };
        }
        double const dx = static_cast<double>(ax[i]) - static_cast<double>(bx[i]);
        double const dy = static_cast<double>(ay[i]) - static_cast<double>(by[i]);
        double const dz = static_cast<double>(az[i]) - static_cast<double>(bz[i]);
        double const d = std::sqrt(dx * dx + dy * dy + dz * dz);
        diffs.push_back(d);
        sum += d;
        max_abs = std::max(max_abs, d);
    }
    std::sort(diffs.begin(), diffs.end());
    std::size_t const p99_index =
        diffs.empty() ? 0 : std::min(diffs.size() - 1, (diffs.size() * 99) / 100);
    return {
        max_abs, diffs.empty() ? 0.0 : diffs[p99_index],
        diffs.empty() ? 0.0 : sum / static_cast<double>(diffs.size())
    };
}

[[nodiscard]] bool record(
    csv_writer &csv, std::filesystem::path const &model, builder_kind const builder,
    std::size_t const frame, std::string_view const stage, int const order,
    std::size_t const triangles, std::size_t const queries, diff_stats const &stats,
    double const epsilon
) {
    bool const success = stats.max_abs <= epsilon;
    std::cout << "[refit-fidelity] model=" << model.filename().string()
              << " builder=" << to_string(builder) << " frame=" << frame << " stage=" << stage
              << " order=" << order << " max=" << stats.max_abs << " p99=" << stats.p99_abs
              << " mean=" << stats.mean_abs << " eps=" << epsilon
              << " status=" << (success ? "ok" : "fail") << "\n";
    csv.append(
        model, to_string(builder), frame, stage, order, triangles, queries, stats, epsilon, success
    );
    return success;
}

template <int Order>
[[nodiscard]] bool validate_taylor_order(
    std::filesystem::path const &model_path, builder_kind const builder, std::size_t const frame_id,
    gwn::gwn_bvh4_topology_object<Real, Index> const &topology, pipeline_state &dynamic_state,
    pipeline_state &fresh_state, query_set &queries, options const &opts, csv_writer &csv
) {
    if (!refit_moment<Order>(topology, dynamic_state) ||
        !refit_moment<Order>(topology, fresh_state))
        return false;

    std::vector<Real> dynamic_winding;
    std::vector<Real> fresh_winding;
    if (!query_taylor_winding<Order>(dynamic_state, topology, queries, dynamic_winding) ||
        !query_taylor_winding<Order>(fresh_state, topology, queries, fresh_winding))
        return false;
    bool ok = record(
        csv, model_path, builder, frame_id, "taylor_winding", Order,
        dynamic_state.geometry.triangle_count(), queries.x.size(),
        scalar_diff(dynamic_winding, fresh_winding), opts.value_epsilon
    );

    std::vector<Real> dynamic_gx;
    std::vector<Real> dynamic_gy;
    std::vector<Real> dynamic_gz;
    std::vector<Real> fresh_gx;
    std::vector<Real> fresh_gy;
    std::vector<Real> fresh_gz;
    if (!query_taylor_gradient<Order>(
            dynamic_state, topology, queries, dynamic_gx, dynamic_gy, dynamic_gz
        ) ||
        !query_taylor_gradient<Order>(fresh_state, topology, queries, fresh_gx, fresh_gy, fresh_gz))
        return false;
    ok = record(
             csv, model_path, builder, frame_id, "taylor_gradient", Order,
             dynamic_state.geometry.triangle_count(), queries.x.size(),
             vector_diff(dynamic_gx, dynamic_gy, dynamic_gz, fresh_gx, fresh_gy, fresh_gz),
             opts.gradient_epsilon
         ) &&
         ok;
    return ok;
}

[[nodiscard]] bool validate_antipodal(
    std::filesystem::path const &model_path, builder_kind const builder, std::size_t const frame_id,
    gwn::gwn_bvh4_topology_object<Real, Index> const &topology, pipeline_state &dynamic_state,
    pipeline_state &fresh_state, query_set &queries, options const &opts, csv_writer &csv
) {
    std::vector<Real> dynamic_winding;
    std::vector<Real> fresh_winding;
    if (!query_antipodal_winding(dynamic_state, topology, queries, dynamic_winding) ||
        !query_antipodal_winding(fresh_state, topology, queries, fresh_winding))
        return false;
    bool ok = record(
        csv, model_path, builder, frame_id, "antipodal_winding", -1,
        dynamic_state.geometry.triangle_count(), queries.x.size(),
        scalar_diff(dynamic_winding, fresh_winding), opts.value_epsilon
    );

    std::vector<Real> dynamic_gx;
    std::vector<Real> dynamic_gy;
    std::vector<Real> dynamic_gz;
    std::vector<Real> fresh_gx;
    std::vector<Real> fresh_gy;
    std::vector<Real> fresh_gz;
    if (!query_antipodal_gradient(dynamic_state, queries, dynamic_gx, dynamic_gy, dynamic_gz) ||
        !query_antipodal_gradient(fresh_state, queries, fresh_gx, fresh_gy, fresh_gz))
        return false;
    ok = record(
             csv, model_path, builder, frame_id, "antipodal_gradient", -1,
             dynamic_state.geometry.triangle_count(), queries.x.size(),
             vector_diff(dynamic_gx, dynamic_gy, dynamic_gz, fresh_gx, fresh_gy, fresh_gz),
             opts.gradient_epsilon
         ) &&
         ok;
    return ok;
}

[[nodiscard]] bool validate_builder(
    std::filesystem::path const &model_path, HostMesh const &mesh, std::vector<Frame> const &frames,
    builder_kind const builder, query_set &queries, options const &opts, csv_writer &csv
) {
    pipeline_state dynamic_state;
    if (!upload_geometry(mesh, dynamic_state.geometry))
        return false;

    gwn::gwn_bvh4_topology_object<Real, Index> topology;
    if (!build_topology(builder, dynamic_state.geometry, topology))
        return false;

    bool ok = true;
    for (std::size_t frame_id = 0; frame_id < frames.size(); ++frame_id) {
        Frame const &frame = frames[frame_id];
        if (!update_geometry(frame, dynamic_state.geometry))
            return false;
        if (!refit_aabb_and_boundary(topology, dynamic_state))
            return false;

        pipeline_state fresh_state;
        HostMesh const frame_mesh = mesh_with_frame(mesh, frame);
        if (!upload_geometry(frame_mesh, fresh_state.geometry))
            return false;
        if (!refit_aabb_and_boundary(topology, fresh_state))
            return false;

        ok = validate_taylor_order<0>(
                 model_path, builder, frame_id, topology, dynamic_state, fresh_state, queries, opts,
                 csv
             ) &&
             ok;
        ok = validate_taylor_order<1>(
                 model_path, builder, frame_id, topology, dynamic_state, fresh_state, queries, opts,
                 csv
             ) &&
             ok;
        ok = validate_taylor_order<2>(
                 model_path, builder, frame_id, topology, dynamic_state, fresh_state, queries, opts,
                 csv
             ) &&
             ok;
        ok = validate_antipodal(
                 model_path, builder, frame_id, topology, dynamic_state, fresh_state, queries, opts,
                 csv
             ) &&
             ok;
    }
    return ok;
}

[[nodiscard]] bool
validate_model(std::filesystem::path const &model_path, options const &opts, csv_writer &csv) {
    std::optional<HostMesh> const maybe_mesh = gwn::tests::load_obj_mesh(model_path);
    if (!maybe_mesh.has_value()) {
        std::cerr << "load failed: " << model_path << "\n";
        return false;
    }
    HostMesh const &mesh = *maybe_mesh;
    std::vector<Frame> const frames = make_frames(mesh, opts);
    if (frames.empty()) {
        std::cerr << "frame generation failed: " << model_path << "\n";
        return false;
    }

    query_set queries;
    if (!make_queries(mesh, opts.query_count, queries))
        return false;

    std::cout << "[refit-fidelity] model=" << model_path.filename().string()
              << " vertices=" << mesh.vertex_x.size() << " triangles=" << mesh.tri_i0.size()
              << " frames=" << frames.size() << " queries=" << queries.x.size() << "\n";

    bool ok = true;
    ok = validate_builder(model_path, mesh, frames, builder_kind::k_lbvh, queries, opts, csv) && ok;
    ok =
        validate_builder(model_path, mesh, frames, builder_kind::k_hploc, queries, opts, csv) && ok;
    return ok;
}

[[nodiscard]] std::vector<std::filesystem::path> collect_models(options const &opts) {
    std::vector<std::filesystem::path> paths;
    if (opts.model_path.has_value())
        paths.push_back(*opts.model_path);
    else if (opts.model_dir.has_value())
        paths = gwn::tests::collect_obj_model_paths(*opts.model_dir);
    if (opts.model_limit > 0 && paths.size() > opts.model_limit)
        paths.resize(opts.model_limit);
    return paths;
}

} // namespace

int main(int argc, char **argv) try {
    options opts;
    if (!parse_cli(argc, argv, opts)) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::filesystem::path> const models = collect_models(opts);
    if (models.empty()) {
        std::cerr << "no OBJ models found\n";
        return 2;
    }

    csv_writer csv(opts.csv_path);
    if (opts.csv_path.has_value() && !csv.stream.good()) {
        std::cerr << "failed to open CSV: " << opts.csv_path->string() << "\n";
        return 2;
    }

    bool ok = true;
    for (std::filesystem::path const &model : models)
        ok = validate_model(model, opts, csv) && ok;
    return ok ? 0 : 1;
} catch (std::exception const &e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
} catch (...) {
    std::cerr << "fatal: unknown exception\n";
    return 1;
}

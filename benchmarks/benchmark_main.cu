#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gwn/gwn.cuh>

#include "benchmark_csv.hpp"
#include "benchmark_utils.hpp"
#include "test_utils.hpp"

namespace {

using Real = gwn::bench::Real;
using Index = gwn::bench::Index;
using HostMesh = gwn::bench::HostMesh;

constexpr int k_bvh_width = 4;
constexpr int k_default_warmup = 5;
constexpr int k_default_iters = 20;
constexpr std::size_t k_default_query_count = 1'000'000;
constexpr float k_default_accuracy_scale = 2.0f;
constexpr int k_default_stack_capacity = 32;
constexpr bool k_default_stream_sync_per_iter = true;
constexpr std::uint64_t k_default_seed = 0xB3A9E5D4ULL;

enum class benchmark_topology_builder {
    k_lbvh,
    k_hploc,
};

[[nodiscard]] constexpr std::string_view
to_string(benchmark_topology_builder const builder) noexcept {
    switch (builder) {
    case benchmark_topology_builder::k_lbvh: return "lbvh";
    case benchmark_topology_builder::k_hploc: return "hploc";
    }
    return "unknown";
}

struct benchmark_cli_options {
    std::optional<std::filesystem::path> model_path{};
    std::optional<std::filesystem::path> model_dir{};
    std::size_t query_count{k_default_query_count};
    int warmup_iters{k_default_warmup};
    int measure_iters{k_default_iters};
    float accuracy_scale{k_default_accuracy_scale};
    int stack_capacity{k_default_stack_capacity};
    bool stream_sync_per_iter{k_default_stream_sync_per_iter};
    std::filesystem::path csv_path{"smallgwn_bench.csv"};
    std::uint64_t seed{k_default_seed};
    benchmark_topology_builder topology_builder{benchmark_topology_builder::k_lbvh};
};

void print_usage(char const *argv0) {
    std::cout << "Usage:\n"
              << "  " << argv0 << " --model <mesh.obj> [options]\n"
              << "  " << argv0 << " --model-dir <dir> [options]\n\n"
              << "Options:\n"
              << "  --queries <N>                 Query count (default: " << k_default_query_count
              << ")\n"
              << "  --warmup <N>                  Warmup iterations (default: " << k_default_warmup
              << ")\n"
              << "  --iters <N>                   Measured iterations (default: " << k_default_iters
              << ")\n"
              << "  --accuracy-scale <x>          Taylor accuracy scale (default: "
              << k_default_accuracy_scale << ")\n"
              << "  --stack-capacity <N>          Traversal stack capacity dispatch (default: "
              << k_default_stack_capacity << ")\n"
              << "  --stream-sync-per-iter <0|1>  Sync stream every iteration (default: "
              << (k_default_stream_sync_per_iter ? 1 : 0) << ")\n"
              << "  --builder <lbvh|hploc>        Topology builder (default: lbvh)\n"
              << "  --csv <path>                  CSV output path (default: smallgwn_bench.csv)\n"
              << "  --seed <u64>                  RNG seed (default: " << k_default_seed << ")\n"
              << "  --help                        Show this message\n";
}

template <class T> [[nodiscard]] bool parse_positive_integral(std::string_view const text, T &out) {
    static_assert(std::is_integral_v<T>);
    if (text.empty())
        return false;
    T parsed{};
    char const *begin = text.data();
    char const *end = text.data() + text.size();
    auto const [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc() || ptr != end || parsed <= 0)
        return false;
    out = parsed;
    return true;
}

template <class T>
[[nodiscard]] bool parse_non_negative_integral(std::string_view const text, T &out) {
    static_assert(std::is_integral_v<T>);
    if (text.empty())
        return false;
    T parsed{};
    char const *begin = text.data();
    char const *end = text.data() + text.size();
    auto const [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc() || ptr != end || parsed < 0)
        return false;
    out = parsed;
    return true;
}

[[nodiscard]] bool parse_float_value(std::string_view const text, float &out) {
    if (text.empty())
        return false;
    try {
        std::size_t offset = 0;
        float const parsed = std::stof(std::string(text), &offset);
        if (offset != text.size() || !(parsed > 0.0f))
            return false;
        out = parsed;
        return true;
    } catch (...) { return false; }
}

enum class parse_cli_result {
    k_ok,
    k_help,
    k_error,
};

[[nodiscard]] parse_cli_result
parse_cli(int const argc, char const *const *argv, benchmark_cli_options &options) {
    for (int i = 1; i < argc; ++i) {
        std::string_view const arg(argv[i]);
        auto require_value = [&](std::string_view const name) -> std::optional<std::string_view> {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << ".\n";
                return std::nullopt;
            }
            return std::string_view(argv[++i]);
        };

        if (arg == "--help") {
            print_usage(argv[0]);
            return parse_cli_result::k_help;
        }
        if (arg == "--model") {
            auto const value = require_value(arg);
            if (!value.has_value())
                return parse_cli_result::k_error;
            options.model_path = std::filesystem::path(*value);
            continue;
        }
        if (arg == "--model-dir") {
            auto const value = require_value(arg);
            if (!value.has_value())
                return parse_cli_result::k_error;
            options.model_dir = std::filesystem::path(*value);
            continue;
        }
        if (arg == "--queries") {
            auto const value = require_value(arg);
            if (!value.has_value() ||
                !parse_positive_integral<std::size_t>(*value, options.query_count)) {
                std::cerr << "Invalid --queries value.\n";
                return parse_cli_result::k_error;
            }
            continue;
        }
        if (arg == "--warmup") {
            auto const value = require_value(arg);
            if (!value.has_value() ||
                !parse_non_negative_integral<int>(*value, options.warmup_iters)) {
                std::cerr << "Invalid --warmup value.\n";
                return parse_cli_result::k_error;
            }
            continue;
        }
        if (arg == "--iters") {
            auto const value = require_value(arg);
            if (!value.has_value() ||
                !parse_positive_integral<int>(*value, options.measure_iters)) {
                std::cerr << "Invalid --iters value.\n";
                return parse_cli_result::k_error;
            }
            continue;
        }
        if (arg == "--accuracy-scale") {
            auto const value = require_value(arg);
            if (!value.has_value() || !parse_float_value(*value, options.accuracy_scale)) {
                std::cerr << "Invalid --accuracy-scale value.\n";
                return parse_cli_result::k_error;
            }
            continue;
        }
        if (arg == "--stack-capacity") {
            auto const value = require_value(arg);
            if (!value.has_value() ||
                !parse_positive_integral<int>(*value, options.stack_capacity)) {
                std::cerr << "Invalid --stack-capacity value.\n";
                return parse_cli_result::k_error;
            }
            continue;
        }
        if (arg == "--stream-sync-per-iter") {
            auto const value = require_value(arg);
            if (!value.has_value() || (*value != "0" && *value != "1")) {
                std::cerr << "Invalid --stream-sync-per-iter value (must be 0 or 1).\n";
                return parse_cli_result::k_error;
            }
            options.stream_sync_per_iter = (*value == "1");
            continue;
        }
        if (arg == "--builder") {
            auto const value = require_value(arg);
            if (!value.has_value()) {
                std::cerr << "Invalid --builder value.\n";
                return parse_cli_result::k_error;
            }
            if (*value == "lbvh")
                options.topology_builder = benchmark_topology_builder::k_lbvh;
            else if (*value == "hploc")
                options.topology_builder = benchmark_topology_builder::k_hploc;
            else {
                std::cerr << "Invalid --builder value (must be lbvh or hploc).\n";
                return parse_cli_result::k_error;
            }
            continue;
        }
        if (arg == "--csv") {
            auto const value = require_value(arg);
            if (!value.has_value()) {
                std::cerr << "Invalid --csv value.\n";
                return parse_cli_result::k_error;
            }
            options.csv_path = std::filesystem::path(*value);
            continue;
        }
        if (arg == "--seed") {
            auto const value = require_value(arg);
            if (!value.has_value() ||
                !parse_positive_integral<std::uint64_t>(*value, options.seed)) {
                std::cerr << "Invalid --seed value.\n";
                return parse_cli_result::k_error;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return parse_cli_result::k_error;
    }

    if (options.model_path.has_value() == options.model_dir.has_value()) {
        std::cerr << "Exactly one of --model or --model-dir must be provided.\n";
        return parse_cli_result::k_error;
    }
    return parse_cli_result::k_ok;
}

[[nodiscard]] std::vector<std::filesystem::path>
collect_models(benchmark_cli_options const &options) {
    std::vector<std::filesystem::path> models{};
    if (options.model_path.has_value()) {
        if (!std::filesystem::is_regular_file(*options.model_path)) {
            std::cerr << "Model path is not a regular file: " << options.model_path->string()
                      << "\n";
            return {};
        }
        models.push_back(*options.model_path);
        return models;
    }

    if (!std::filesystem::is_directory(*options.model_dir)) {
        std::cerr << "Model directory does not exist: " << options.model_dir->string() << "\n";
        return {};
    }
    models = gwn::tests::collect_obj_model_paths(*options.model_dir);
    if (models.empty())
        std::cerr << "No OBJ files found under: " << options.model_dir->string() << "\n";
    return models;
}

template <int Order>
gwn::gwn_status run_taylor_query_with_stack_capacity(
    int const stack_capacity, gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_bvh_accessor<Real, Index> const &topology,
    gwn::gwn_bvh_moment4_accessor<Real, Index> const &moment,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    float const accuracy_scale, cudaStream_t const stream
) noexcept {
    switch (stack_capacity) {
    case 16:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 16>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 24:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 24>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 32:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 32>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 48:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 48>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 64:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 64>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 96:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 96>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 128:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 128>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 192:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 192>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    case 256:
        return gwn::gwn_compute_winding_number_batch_bvh_taylor<Order, 4, Real, Index, 256>(
            geometry, topology, moment, query_x, query_y, query_z, output, accuracy_scale, stream
        );
    default:
        return gwn::gwn_status::invalid_argument(
            "Unsupported --stack-capacity. Supported: 16,24,32,48,64,96,128,192,256."
        );
    }
}

template <class SetupFn, class StageFn>
gwn::bench::gwn_benchmark_stage_result run_stage(
    std::string const &stage_name, std::string const &unit,
    std::size_t const throughput_denominator, benchmark_cli_options const &options,
    std::size_t const vertex_count, std::size_t const triangle_count, std::size_t const query_count,
    cudaStream_t const stream, SetupFn &&setup_fn, StageFn &&stage_fn
) {
    gwn::bench::gwn_benchmark_stage_result result{};
    result.stage = stage_name;
    result.unit = unit;
    result.warmup_iters = options.warmup_iters;
    result.measure_iters = options.measure_iters;
    result.vertex_count = vertex_count;
    result.triangle_count = triangle_count;
    result.query_count = query_count;

    gwn::gwn_status const setup_status = setup_fn();
    if (!setup_status.is_ok()) {
        result.success = false;
        result.error_message = gwn::bench::gwn_status_to_string(setup_status);
        return result;
    }

    std::vector<float> latencies_ms{};
    gwn::gwn_status const measure_status = gwn::bench::gwn_measure_stage_latency_ms(
        options.warmup_iters, options.measure_iters, stream, options.stream_sync_per_iter, stage_fn,
        latencies_ms
    );
    if (!measure_status.is_ok()) {
        result.success = false;
        result.error_message = gwn::bench::gwn_status_to_string(measure_status);
        return result;
    }

    auto const stats = gwn::bench::summarize_stage_ms(latencies_ms);
    result.success = true;
    result.latency_mean_ms = stats.mean_ms;
    result.latency_p50_ms = stats.p50_ms;
    result.latency_p95_ms = stats.p95_ms;

    if (stats.mean_ms > 0.0) {
        result.throughput_per_s =
            static_cast<double>(throughput_denominator) / (stats.mean_ms / 1000.0);
    }
    return result;
}

void print_stage_result(
    std::filesystem::path const &model_path, gwn::bench::gwn_benchmark_stage_result const &result
) {
    std::cout << "  [" << model_path.filename().string() << "] " << std::left << std::setw(24)
              << result.stage;
    if (!result.success) {
        std::cout << " FAILED: " << result.error_message << "\n";
        return;
    }

    std::cout << " mean=" << std::setw(9) << std::fixed << std::setprecision(3)
              << result.latency_mean_ms << " ms"
              << " p50=" << std::setw(9) << result.latency_p50_ms << " ms"
              << " p95=" << std::setw(9) << result.latency_p95_ms << " ms"
              << " thr=" << std::setprecision(2) << std::scientific << result.throughput_per_s
              << " " << result.unit << std::defaultfloat << "\n";
}

} // namespace

int main(int argc, char **argv) try {
    benchmark_cli_options options{};
    parse_cli_result const parse_result = parse_cli(argc, argv, options);
    if (parse_result == parse_cli_result::k_help)
        return 0;
    if (parse_result != parse_cli_result::k_ok)
        return 1;

    std::vector<std::filesystem::path> const models = collect_models(options);
    if (models.empty())
        return 1;

    int device_count = 0;
    cudaError_t const device_count_result = cudaGetDeviceCount(&device_count);
    if (device_count_result != cudaSuccess || device_count <= 0) {
        std::cerr << "CUDA device unavailable: " << cudaGetErrorName(device_count_result) << " ("
                  << cudaGetErrorString(device_count_result) << ")\n";
        return 1;
    }

    int runtime_version = 0;
    cudaDeviceProp device_prop{};
    int current_device = 0;
    (void)cudaRuntimeGetVersion(&runtime_version);
    (void)cudaGetDevice(&current_device);
    (void)cudaGetDeviceProperties(&device_prop, current_device);

    gwn::bench::gwn_benchmark_csv_writer csv_writer(options.csv_path.string());
    if (!csv_writer.good()) {
        std::cerr << "Failed to open CSV output: " << options.csv_path.string() << "\n";
        return 1;
    }
    csv_writer.write_header();

    gwn::bench::gwn_benchmark_csv_context csv_context{};
    csv_context.timestamp = gwn::bench::gwn_now_timestamp_string();
    csv_context.gpu_name = device_prop.name;
    csv_context.cuda_runtime_version = runtime_version;
    csv_context.topology_builder = std::string(to_string(options.topology_builder));
    csv_context.accuracy_scale = options.accuracy_scale;
    csv_context.stack_capacity = options.stack_capacity;
    csv_context.seed = options.seed;

    std::cout << "smallgwn benchmark\n";
    std::cout << "  GPU: " << csv_context.gpu_name << "\n";
    std::cout << "  CUDA runtime version: " << runtime_version << "\n";
    std::cout << "  Models: " << models.size() << ", Queries/model: " << options.query_count
              << ", Warmup: " << options.warmup_iters << ", Iters: " << options.measure_iters
              << "\n";
    std::cout << "  Topology builder: " << to_string(options.topology_builder) << "\n";
    std::cout << "  CSV: " << options.csv_path.string() << "\n\n";

    std::size_t successful_models = 0;
    for (std::filesystem::path const &model_path : models) {
        std::optional<HostMesh> const loaded = gwn::tests::load_obj_mesh(model_path);
        if (!loaded.has_value()) {
            std::cerr << "Skipping unreadable model: " << model_path.string() << "\n";
            continue;
        }
        HostMesh const &mesh = *loaded;

        cudaStream_t stream{};
        gwn::gwn_status const stream_create_status =
            gwn::gwn_cuda_to_status(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (!stream_create_status.is_ok()) {
            std::cerr << "Failed to create CUDA stream for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(stream_create_status) << "\n";
            continue;
        }
        auto destroy_stream =
            gwn::gwn_make_scope_exit([&]() noexcept { (void)cudaStreamDestroy(stream); });

        gwn::gwn_geometry_object<Real, Index> geometry;
        gwn::gwn_status const upload_status = geometry.upload(
            cuda::std::span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            cuda::std::span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            cuda::std::span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            cuda::std::span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            cuda::std::span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            cuda::std::span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()), stream
        );
        if (!upload_status.is_ok()) {
            std::cerr << "Geometry upload failed for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(upload_status) << "\n";
            continue;
        }
        gwn::gwn_status const upload_sync_status =
            gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream));
        if (!upload_sync_status.is_ok()) {
            std::cerr << "Geometry upload sync failed for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(upload_sync_status) << "\n";
            continue;
        }

        auto const query_host =
            gwn::bench::gwn_make_mixed_query_soa(mesh, options.query_count, options.seed);
        gwn::gwn_device_array<Real> d_qx(stream);
        gwn::gwn_device_array<Real> d_qy(stream);
        gwn::gwn_device_array<Real> d_qz(stream);
        gwn::gwn_device_array<Real> d_out(stream);
        gwn::gwn_status const qx_status = d_qx.copy_from_host(
            cuda::std::span<Real const>(query_host[0].data(), query_host[0].size()), stream
        );
        gwn::gwn_status const qy_status = d_qy.copy_from_host(
            cuda::std::span<Real const>(query_host[1].data(), query_host[1].size()), stream
        );
        gwn::gwn_status const qz_status = d_qz.copy_from_host(
            cuda::std::span<Real const>(query_host[2].data(), query_host[2].size()), stream
        );
        gwn::gwn_status const out_status = d_out.resize(options.query_count, stream);
        if (!qx_status.is_ok() || !qy_status.is_ok() || !qz_status.is_ok() || !out_status.is_ok()) {
            std::cerr << "Query buffer setup failed for model " << model_path.string() << "\n";
            continue;
        }
        gwn::gwn_status const query_sync_status =
            gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream));
        if (!query_sync_status.is_ok()) {
            std::cerr << "Query buffer sync failed for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(query_sync_status) << "\n";
            continue;
        }

        gwn::gwn_bvh_object<Real, Index> topology;
        gwn::gwn_bvh_aabb_object<Real, Index> aabb_tree;
        gwn::gwn_bvh_moment_object<Real, Index> moment_tree;
        std::string const builder_name{to_string(options.topology_builder)};

        auto build_topology = [&]() noexcept -> gwn::gwn_status {
            if (options.topology_builder == benchmark_topology_builder::k_hploc) {
                return gwn::gwn_bvh_topology_build_hploc<k_bvh_width, Real, Index>(
                    geometry, topology, stream
                );
            }
            return gwn::gwn_bvh_topology_build_lbvh<k_bvh_width, Real, Index>(
                geometry, topology, stream
            );
        };

        auto build_facade_o0 = [&]() noexcept -> gwn::gwn_status {
            if (options.topology_builder == benchmark_topology_builder::k_hploc) {
                return gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<
                    0, k_bvh_width, Real, Index>(
                    geometry, topology, aabb_tree, moment_tree, stream
                );
            }
            return gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<0, k_bvh_width, Real, Index>(
                geometry, topology, aabb_tree, moment_tree, stream
            );
        };

        auto build_facade_o1 = [&]() noexcept -> gwn::gwn_status {
            if (options.topology_builder == benchmark_topology_builder::k_hploc) {
                return gwn::gwn_bvh_facade_build_topology_aabb_moment_hploc<
                    1, k_bvh_width, Real, Index>(
                    geometry, topology, aabb_tree, moment_tree, stream
                );
            }
            return gwn::gwn_bvh_facade_build_topology_aabb_moment_lbvh<1, k_bvh_width, Real, Index>(
                geometry, topology, aabb_tree, moment_tree, stream
            );
        };

        std::cout << "Model: " << model_path.string() << " (V=" << mesh.vertex_x.size()
                  << ", F=" << mesh.tri_i0.size() << ", Q=" << options.query_count << ")\n";

        auto emit_result = [&](gwn::bench::gwn_benchmark_stage_result const &result) {
            print_stage_result(model_path, result);
            csv_writer.append_row(csv_context, model_path.string(), result);
        };

        auto const topology_stage = run_stage(
            std::string("topology_build_") + builder_name, "triangles/s", mesh.tri_i0.size(),
            options, mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_topology
        );
        emit_result(topology_stage);

        auto const refit_aabb_stage = run_stage(
            "refit_aabb", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_topology, [&]() noexcept {
            return gwn::gwn_bvh_refit_aabb<k_bvh_width, Real, Index>(
                geometry, topology, aabb_tree, stream
            );
        }
        );
        emit_result(refit_aabb_stage);

        auto setup_topology_and_aabb = [&]() noexcept {
            gwn::gwn_status const topology_status = build_topology();
            if (!topology_status.is_ok())
                return topology_status;
            return gwn::gwn_bvh_refit_aabb<k_bvh_width, Real, Index>(
                geometry, topology, aabb_tree, stream
            );
        };

        auto const refit_moment_o0_stage = run_stage(
            "refit_moment_o0", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, setup_topology_and_aabb,
            [&]() noexcept {
            return gwn::gwn_bvh_refit_moment<0, k_bvh_width, Real, Index>(
                geometry, topology, aabb_tree, moment_tree, stream
            );
        }
        );
        emit_result(refit_moment_o0_stage);

        auto const refit_moment_o1_stage = run_stage(
            "refit_moment_o1", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, setup_topology_and_aabb,
            [&]() noexcept {
            return gwn::gwn_bvh_refit_moment<1, k_bvh_width, Real, Index>(
                geometry, topology, aabb_tree, moment_tree, stream
            );
        }
        );
        emit_result(refit_moment_o1_stage);

        auto const facade_o0_stage = run_stage(
            "facade_o0", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_facade_o0
        );
        emit_result(facade_o0_stage);

        auto const facade_o1_stage = run_stage(
            "facade_o1", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_facade_o1
        );
        emit_result(facade_o1_stage);

        auto const query_o0_stage = run_stage(
            "query_taylor_o0", "queries/s", options.query_count, options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_facade_o0, [&]() noexcept {
            return run_taylor_query_with_stack_capacity<0>(
                options.stack_capacity, geometry.accessor(), topology.accessor(),
                moment_tree.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(),
                options.accuracy_scale, stream
            );
        }
        );
        emit_result(query_o0_stage);

        auto const query_o1_stage = run_stage(
            "query_taylor_o1", "queries/s", options.query_count, options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_facade_o1, [&]() noexcept {
            return run_taylor_query_with_stack_capacity<1>(
                options.stack_capacity, geometry.accessor(), topology.accessor(),
                moment_tree.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(),
                options.accuracy_scale, stream
            );
        }
        );
        emit_result(query_o1_stage);

        std::cout << "\n";
        ++successful_models;
    }

    if (successful_models == 0) {
        std::cerr << "No model benchmark completed successfully.\n";
        return 1;
    }

    std::cout << "Benchmark complete. Successful models: " << successful_models << "/"
              << models.size() << "\n";
    std::cout << "CSV written to: " << options.csv_path.string() << "\n";
    return 0;
} catch (std::exception const &e) {
    std::cerr << "Unhandled std::exception: " << e.what() << "\n";
    return 1;
} catch (...) {
    std::cerr << "Unhandled unknown exception.\n";
    return 1;
}

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "benchmark_csv.cuh"
#include "benchmark_utils.cuh"
#include "test_utils.cuh"

namespace {

using Real = gwn::bench::Real;
using Index = gwn::bench::Index;
using HostMesh = gwn::bench::HostMesh;

constexpr int k_bvh_width = 4;
constexpr int k_default_warmup = 5;
constexpr int k_default_iters = 20;
constexpr std::size_t k_default_query_count = 1'000'000;
constexpr float k_default_accuracy_scale = 2.0f;
constexpr int k_default_stack_capacity = 96;
constexpr bool k_default_stream_sync_per_iter = true;
constexpr std::uint64_t k_default_seed = 0xB3A9E5D4ULL;

enum class benchmark_bvh_builder {
    k_lbvh,
    k_hploc,
};

[[nodiscard]] constexpr std::string_view to_string(benchmark_bvh_builder const builder) noexcept {
    switch (builder) {
    case benchmark_bvh_builder::k_lbvh: return "lbvh";
    case benchmark_bvh_builder::k_hploc: return "hploc";
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
    benchmark_bvh_builder builder{benchmark_bvh_builder::k_hploc};
    bool skip_exact{false};
    bool winding_query_only{false};
};

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
        std::cerr << "Model directory missing: " << options.model_dir->string() << "\n";
        return {};
    }
    models = gwn::tests::collect_ply_model_paths(*options.model_dir);
    if (models.empty())
        std::cerr << "No PLY files found under: " << options.model_dir->string() << "\n";
    return models;
}

template <int StackCapacity, int Order>
[[nodiscard]] gwn::gwn_status run_taylor_query(
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_bvh4_moment_object<Order, Real, Index> const &moment,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    float const accuracy_scale, cudaStream_t const stream
) noexcept {
    constexpr gwn::gwn_query_batch_config config{
        .block_size = gwn::k_gwn_default_query_batch_block_size,
        .stack_capacity = StackCapacity,
    };
    return gwn::gwn_compute_winding_number_taylor_batch<Order, config, 4, Real, Index>(
        bvh, moment, gwn::tests::device_span(query_x), gwn::tests::device_span(query_y),
        gwn::tests::device_span(query_z), gwn::tests::device_span(output), accuracy_scale, stream
    );
}

[[nodiscard]] gwn::gwn_status run_exact_query(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output, cudaStream_t const stream
) noexcept {
    return gwn::gwn_compute_winding_number_exact_batch(
        bvh, gwn::tests::device_span(query_x), gwn::tests::device_span(query_y),
        gwn::tests::device_span(query_z), gwn::tests::device_span(output), stream
    );
}

template <int StackCapacity> struct antipodal_crossing_functor {
    gwn::gwn_bvh4_accessor<Real, Index> bvh{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};

    __device__ void operator()(std::size_t const query_id) const {
        gwn::detail::gwn_query_vec3<Real> const query(
            query_x[query_id], query_y[query_id], query_z[query_id]
        );
        Real value = std::numeric_limits<Real>::quiet_NaN();
        for (int retry_id = 0; retry_id < 3; ++retry_id) {
            auto const result =
                gwn::detail::gwn_signed_ray_crossing_count_impl<4, Real, Index, StackCapacity>(
                    bvh, query, gwn::detail::gwn_antipodal_ray_axis_for_retry_impl(retry_id)
                );
            if (result.status == gwn::detail::gwn_antipodal_axis_result::k_singular)
                continue;
            value = result.value;
            break;
        }
        output[query_id] = value;
    }
};

template <int StackCapacity>
gwn::gwn_status run_antipodal_crossing_query(
    gwn::gwn_bvh4_accessor<Real, Index> const &bvh, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output, cudaStream_t const stream
) noexcept {
    return gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
        output.size(),
        antipodal_crossing_functor<StackCapacity>{
            bvh,
            query_x,
            query_y,
            query_z,
            output,
        },
        stream
    );
}

template <int StackCapacity>
gwn::gwn_status run_antipodal_crossing(
    gwn::gwn_bvh4_accessor<Real, Index> const &bvh, cuda::std::span<Real const> const query_x,
    cuda::std::span<Real const> const query_y, cuda::std::span<Real const> const query_z,
    cuda::std::span<Real> const output, cudaStream_t const stream
) noexcept {
    return run_antipodal_crossing_query<StackCapacity>(
        bvh, query_x, query_y, query_z, output, stream
    );
}

struct antipodal_boundary_functor {
    gwn::gwn_geometry_accessor<Real, Index> geometry{};
    gwn::gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> output{};

    __device__ void operator()(std::size_t const query_id) const {
        gwn::detail::gwn_query_vec3<Real> const query(
            query_x[query_id], query_y[query_id], query_z[query_id]
        );
        Real value = std::numeric_limits<Real>::quiet_NaN();
        for (int retry_id = 0; retry_id < 3; ++retry_id) {
            auto const result = gwn::detail::gwn_antipodal_boundary_contribution_impl(
                geometry, boundary_chain, query,
                gwn::detail::gwn_antipodal_ray_axis_for_retry_impl(retry_id)
            );
            if (result.status == gwn::detail::gwn_antipodal_axis_result::k_singular)
                continue;
            value = result.value;
            break;
        }
        output[query_id] = value;
    }
};

gwn::gwn_status run_antipodal_boundary_query(
    gwn::gwn_geometry_accessor<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_accessor<Index> const &boundary_chain,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    cudaStream_t const stream
) noexcept {
    return gwn::detail::gwn_launch_linear_kernel<gwn::detail::k_gwn_default_block_size>(
        output.size(),
        antipodal_boundary_functor{
            geometry,
            boundary_chain,
            query_x,
            query_y,
            query_z,
            output,
        },
        stream
    );
}

gwn::gwn_status run_antipodal_gradient_query(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_boundary_chain_object<Index> const &boundary_chain,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output_x,
    cuda::std::span<Real> const output_y, cuda::std::span<Real> const output_z,
    cudaStream_t const stream
) noexcept {
    return gwn::gwn_compute_winding_gradient_antipodal_batch(
        geometry, boundary_chain, gwn::tests::device_span(query_x),
        gwn::tests::device_span(query_y), gwn::tests::device_span(query_z),
        gwn::tests::device_span(output_x), gwn::tests::device_span(output_y),
        gwn::tests::device_span(output_z), stream
    );
}

template <int StackCapacity>
gwn::gwn_status run_antipodal_query(
    gwn::gwn_geometry_object<Real, Index> const &geometry,
    gwn::gwn_bvh4_object<Real, Index> const &bvh,
    gwn::gwn_boundary_chain_object<Index> const &boundary_chain,
    cuda::std::span<Real const> const query_x, cuda::std::span<Real const> const query_y,
    cuda::std::span<Real const> const query_z, cuda::std::span<Real> const output,
    cudaStream_t const stream
) noexcept {
    constexpr gwn::gwn_query_batch_config config{
        .block_size = gwn::k_gwn_default_query_batch_block_size,
        .stack_capacity = StackCapacity,
    };
    return gwn::gwn_compute_winding_number_antipodal_batch<config, 4, Real, Index>(
        geometry, bvh, boundary_chain, gwn::tests::device_span(query_x),
        gwn::tests::device_span(query_y), gwn::tests::device_span(query_z),
        gwn::tests::device_span(output), stream
    );
}

template <int StackCapacity>
gwn::gwn_status run_ray_first_hit(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, cuda::std::span<Real const> const ray_origin_x,
    cuda::std::span<Real const> const ray_origin_y, cuda::std::span<Real const> const ray_origin_z,
    cuda::std::span<Real const> const ray_dir_x, cuda::std::span<Real const> const ray_dir_y,
    cuda::std::span<Real const> const ray_dir_z,
    cuda::std::span<gwn::gwn_ray_first_hit_result<Real, Index>> const output,
    cudaStream_t const stream
) noexcept {
    constexpr gwn::gwn_query_batch_config config{
        .block_size = gwn::k_gwn_default_query_batch_block_size,
        .stack_capacity = StackCapacity,
    };
    return gwn::gwn_compute_ray_first_hit_batch<config, 4, Real, Index>(
        bvh, gwn::tests::device_span(ray_origin_x), gwn::tests::device_span(ray_origin_y),
        gwn::tests::device_span(ray_origin_z), gwn::tests::device_span(ray_dir_x),
        gwn::tests::device_span(ray_dir_y), gwn::tests::device_span(ray_dir_z),
        gwn::tests::device_span(output), Real(0), std::numeric_limits<Real>::infinity(), stream
    );
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

template <int StackCapacity>
int run_benchmark(
    benchmark_cli_options const &options, std::vector<std::filesystem::path> const &models
) {
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
    csv_context.builder = std::string(to_string(options.builder));
    csv_context.accuracy_scale = options.accuracy_scale;
    csv_context.stack_capacity = StackCapacity;
    csv_context.stream_sync_per_iter = options.stream_sync_per_iter;
    csv_context.seed = options.seed;

    std::cout << "smallgwn benchmark\n";
    std::cout << "  GPU: " << csv_context.gpu_name << "\n";
    std::cout << "  CUDA runtime version: " << runtime_version << "\n";
    std::cout << "  Models: " << models.size() << ", Queries/model: " << options.query_count
              << ", Warmup: " << options.warmup_iters << ", Iters: " << options.measure_iters
              << "\n";
    std::cout << "  BVH builder: " << to_string(options.builder) << "\n";
    std::cout << "  CSV: " << options.csv_path.string() << "\n\n";

    std::size_t successful_models = 0;
    for (std::filesystem::path const &model_path : models) {
        std::optional<HostMesh> const loaded = gwn::tests::load_ply_mesh(model_path);
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
        gwn::gwn_boundary_chain_object<Index> boundary_chain;
        gwn::gwn_status const upload_status = gwn::gwn_upload_geometry(
            geometry, gwn::gwn_host_span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
            gwn::gwn_host_span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
            gwn::gwn_host_span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
            gwn::gwn_host_span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()), stream
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

        gwn::gwn_status const boundary_status =
            gwn::gwn_build_boundary_chain(geometry, boundary_chain, stream);
        if (!boundary_status.is_ok()) {
            std::cerr << "Boundary build failed for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(boundary_status) << "\n";
            continue;
        }
        gwn::gwn_status const boundary_sync_status =
            gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream));
        if (!boundary_sync_status.is_ok()) {
            std::cerr << "Boundary build sync failed for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(boundary_sync_status) << "\n";
            continue;
        }
        csv_context.boundary_edge_count = boundary_chain.accessor().edge_count();

        std::vector<Real> dynamic_x = mesh.vertex_x;
        std::vector<Real> dynamic_y = mesh.vertex_y;
        std::vector<Real> dynamic_z = mesh.vertex_z;
        for (std::size_t i = 0; i < dynamic_z.size(); ++i) {
            int const phase = static_cast<int>(i % 17u) - 8;
            Real const wave = Real(0.001) * static_cast<Real>(phase);
            dynamic_z[i] += wave;
        }

        auto const query_host =
            gwn::bench::gwn_make_mixed_query_soa(mesh, options.query_count, options.seed);
        auto const ray_host =
            gwn::bench::gwn_make_mixed_ray_soa(mesh, options.query_count, options.seed);

        gwn::detail::gwn_device_array<Real> d_qx(stream);
        gwn::detail::gwn_device_array<Real> d_qy(stream);
        gwn::detail::gwn_device_array<Real> d_qz(stream);
        gwn::detail::gwn_device_array<Real> d_out(stream);
        gwn::detail::gwn_device_array<Real> d_grad_x(stream);
        gwn::detail::gwn_device_array<Real> d_grad_y(stream);
        gwn::detail::gwn_device_array<Real> d_grad_z(stream);
        gwn::detail::gwn_device_array<Real> d_ray_ox(stream);
        gwn::detail::gwn_device_array<Real> d_ray_oy(stream);
        gwn::detail::gwn_device_array<Real> d_ray_oz(stream);
        gwn::detail::gwn_device_array<Real> d_ray_dx(stream);
        gwn::detail::gwn_device_array<Real> d_ray_dy(stream);
        gwn::detail::gwn_device_array<Real> d_ray_dz(stream);
        gwn::detail::gwn_device_array<gwn::gwn_ray_first_hit_result<Real, Index>> d_ray_hit(stream);

        d_qx.copy_from_host(
            cuda::std::span<Real const>(query_host[0].data(), query_host[0].size()), stream
        );
        d_qy.copy_from_host(
            cuda::std::span<Real const>(query_host[1].data(), query_host[1].size()), stream
        );
        d_qz.copy_from_host(
            cuda::std::span<Real const>(query_host[2].data(), query_host[2].size()), stream
        );
        d_out.resize(options.query_count, stream);
        d_grad_x.resize(options.query_count, stream);
        d_grad_y.resize(options.query_count, stream);
        d_grad_z.resize(options.query_count, stream);
        d_ray_ox.copy_from_host(
            cuda::std::span<Real const>(ray_host.origin[0].data(), ray_host.origin[0].size()),
            stream
        );
        d_ray_oy.copy_from_host(
            cuda::std::span<Real const>(ray_host.origin[1].data(), ray_host.origin[1].size()),
            stream
        );
        d_ray_oz.copy_from_host(
            cuda::std::span<Real const>(ray_host.origin[2].data(), ray_host.origin[2].size()),
            stream
        );
        d_ray_dx.copy_from_host(
            cuda::std::span<Real const>(ray_host.direction[0].data(), ray_host.direction[0].size()),
            stream
        );
        d_ray_dy.copy_from_host(
            cuda::std::span<Real const>(ray_host.direction[1].data(), ray_host.direction[1].size()),
            stream
        );
        d_ray_dz.copy_from_host(
            cuda::std::span<Real const>(ray_host.direction[2].data(), ray_host.direction[2].size()),
            stream
        );
        d_ray_hit.resize(options.query_count, stream);
        gwn::gwn_status const query_sync_status =
            gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream));
        if (!query_sync_status.is_ok()) {
            std::cerr << "Query buffer sync failed for model " << model_path.string() << ": "
                      << gwn::bench::gwn_status_to_string(query_sync_status) << "\n";
            continue;
        }

        gwn::gwn_bvh4_object<Real, Index> bvh;
        gwn::gwn_bvh4_moment_object<0, Real, Index> moment_tree_o0;
        gwn::gwn_bvh4_moment_object<1, Real, Index> moment_tree_o1;
        gwn::gwn_bvh4_moment_object<2, Real, Index> moment_tree_o2;
        std::string const builder_name{to_string(options.builder)};

        auto build_bvh = [&]() noexcept -> gwn::gwn_status {
            gwn::gwn_bvh_build_method const method =
                options.builder == benchmark_bvh_builder::k_hploc
                    ? gwn::gwn_bvh_build_method::k_hploc
                    : gwn::gwn_bvh_build_method::k_lbvh;
            return gwn::gwn_build_bvh(
                geometry, bvh, gwn::gwn_bvh_build_options{.method = method}, stream
            );
        };

        auto build_bvh_moment_o0 = [&]() noexcept -> gwn::gwn_status {
            GWN_RETURN_ON_ERROR(build_bvh());
            return gwn::gwn_refit_bvh_moment<0>(bvh, moment_tree_o0, stream);
        };

        auto build_bvh_moment_o1 = [&]() noexcept -> gwn::gwn_status {
            GWN_RETURN_ON_ERROR(build_bvh());
            return gwn::gwn_refit_bvh_moment<1>(bvh, moment_tree_o1, stream);
        };

        auto build_bvh_moment_o2 = [&]() noexcept -> gwn::gwn_status {
            GWN_RETURN_ON_ERROR(build_bvh());
            return gwn::gwn_refit_bvh_moment<2>(bvh, moment_tree_o2, stream);
        };

        std::cout << "Model: " << model_path.string() << " (V=" << mesh.vertex_x.size()
                  << ", F=" << mesh.tri_i0.size() << ", BE=" << csv_context.boundary_edge_count
                  << ", Q=" << options.query_count << ")\n";

        auto emit_result = [&](gwn::bench::gwn_benchmark_stage_result const &result) {
            print_stage_result(model_path, result);
            csv_writer.append_row(csv_context, model_path.string(), result);
        };

        auto setup_bvh_and_boundary = [&]() noexcept {
            gwn::gwn_status const setup_status = build_bvh();
            if (!setup_status.is_ok())
                return setup_status;
            return gwn::gwn_build_boundary_chain(geometry, boundary_chain, stream);
        };

        auto measure_query_o1 = [&]() {
            return run_stage(
                "query_taylor_o1", "queries/s", options.query_count, options, mesh.vertex_x.size(),
                mesh.tri_i0.size(), options.query_count, stream, build_bvh_moment_o1,
                [&]() noexcept {
                return run_taylor_query<StackCapacity, 1>(
                    bvh, moment_tree_o1, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(),
                    options.accuracy_scale, stream
                );
            }
            );
        };
        auto measure_query_antipodal = [&]() {
            return run_stage(
                "query_antipodal", "queries/s", options.query_count, options, mesh.vertex_x.size(),
                mesh.tri_i0.size(), options.query_count, stream, setup_bvh_and_boundary,
                [&]() noexcept {
                return run_antipodal_query<StackCapacity>(
                    geometry, bvh, boundary_chain, d_qx.span(), d_qy.span(), d_qz.span(),
                    d_out.span(), stream
                );
            }
            );
        };

        if (options.winding_query_only) {
            auto const query_o1_stage = measure_query_o1();
            emit_result(query_o1_stage);
            auto const query_antipodal_stage = measure_query_antipodal();
            emit_result(query_antipodal_stage);
            if (query_o1_stage.success && query_antipodal_stage.success) {
                double const speedup =
                    query_o1_stage.latency_mean_ms / query_antipodal_stage.latency_mean_ms;
                std::cout << "    taylor_o1_vs_antipodal_speedup=" << std::fixed
                          << std::setprecision(3) << speedup << std::defaultfloat << "\n";
                ++successful_models;
            }
            std::cout << "\n";
            continue;
        }

        auto const bvh_stage = run_stage(
            std::string("bvh_build_") + builder_name, "triangles/s", mesh.tri_i0.size(), options,
            mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_bvh
        );
        emit_result(bvh_stage);

        auto const refit_bvh_stage = run_stage(
            "refit_bvh", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_bvh,
            [&]() noexcept { return gwn::gwn_refit_bvh(geometry, bvh, stream); }
        );
        emit_result(refit_bvh_stage);

        auto setup_boundary = [&]() noexcept {
            return gwn::gwn_build_boundary_chain(geometry, boundary_chain, stream);
        };

        double ray_mix_hit_ratio = 0.0;
        auto setup_bvh_and_validate_rays = [&]() -> gwn::gwn_status {
            gwn::gwn_status const setup_status = build_bvh();
            if (!setup_status.is_ok())
                return setup_status;

            gwn::gwn_status const ray_status = run_ray_first_hit<StackCapacity>(
                bvh, d_ray_ox.span(), d_ray_oy.span(), d_ray_oz.span(), d_ray_dx.span(),
                d_ray_dy.span(), d_ray_dz.span(), d_ray_hit.span(), stream
            );
            if (!ray_status.is_ok())
                return ray_status;

            std::vector<gwn::gwn_ray_first_hit_result<Real, Index>> ray_hit_host(
                options.query_count
            );
            d_ray_hit.copy_to_host(
                cuda::std::span<gwn::gwn_ray_first_hit_result<Real, Index>>(ray_hit_host), stream
            );
            gwn::gwn_status const sync_status =
                gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream));
            if (!sync_status.is_ok())
                return sync_status;

            std::size_t const hit_count = static_cast<std::size_t>(std::count_if(
                ray_hit_host.begin(), ray_hit_host.end(),
                [](gwn::gwn_ray_first_hit_result<Real, Index> const &hit) { return hit.hit(); }
            ));
            ray_mix_hit_ratio =
                static_cast<double>(hit_count) / static_cast<double>(options.query_count);
            if (!(ray_mix_hit_ratio > 0.05 && ray_mix_hit_ratio < 0.95)) {
                return gwn::gwn_status::invalid_argument(
                    "Ray benchmark mix hit ratio is outside (0.05, 0.95)."
                );
            }

            return gwn::gwn_status::ok();
        };

        auto const refit_moment_o0_stage = run_stage(
            "refit_moment_o0", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_bvh,
            [&]() noexcept { return gwn::gwn_refit_bvh_moment<0>(bvh, moment_tree_o0, stream); }
        );
        emit_result(refit_moment_o0_stage);

        auto const refit_moment_o1_stage = run_stage(
            "refit_moment_o1", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_bvh,
            [&]() noexcept { return gwn::gwn_refit_bvh_moment<1>(bvh, moment_tree_o1, stream); }
        );
        emit_result(refit_moment_o1_stage);

        auto const refit_moment_o2_stage = run_stage(
            "refit_moment_o2", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_bvh,
            [&]() noexcept { return gwn::gwn_refit_bvh_moment<2>(bvh, moment_tree_o2, stream); }
        );
        emit_result(refit_moment_o2_stage);

        auto const boundary_chain_stage = run_stage(
            "build_boundary_chain", "triangles/s", mesh.tri_i0.size(), options,
            mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, [&]() noexcept {
            return gwn::gwn_build_boundary_chain(geometry, boundary_chain, stream);
        }
        );
        emit_result(boundary_chain_stage);
        if (boundary_chain_stage.success)
            std::cout << "    boundary_edges=" << boundary_chain.accessor().edge_count() << "\n";

        auto const build_bvh_moment_o0_stage = run_stage(
            "build_bvh_moment_o0", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_bvh_moment_o0
        );
        emit_result(build_bvh_moment_o0_stage);

        auto const build_bvh_moment_o1_stage = run_stage(
            "build_bvh_moment_o1", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_bvh_moment_o1
        );
        emit_result(build_bvh_moment_o1_stage);

        auto const build_bvh_moment_o2_stage = run_stage(
            "build_bvh_moment_o2", "triangles/s", mesh.tri_i0.size(), options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream,
            [&]() noexcept { return gwn::gwn_status::ok(); }, build_bvh_moment_o2
        );
        emit_result(build_bvh_moment_o2_stage);

        auto const query_o0_stage = run_stage(
            "query_taylor_o0", "queries/s", options.query_count, options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_bvh_moment_o0, [&]() noexcept {
            return run_taylor_query<StackCapacity, 0>(
                bvh, moment_tree_o0, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(),
                options.accuracy_scale, stream
            );
        }
        );
        emit_result(query_o0_stage);

        auto const query_o1_stage = measure_query_o1();
        emit_result(query_o1_stage);

        auto const query_o2_stage = run_stage(
            "query_taylor_o2", "queries/s", options.query_count, options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, build_bvh_moment_o2, [&]() noexcept {
            return run_taylor_query<StackCapacity, 2>(
                bvh, moment_tree_o2, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(),
                options.accuracy_scale, stream
            );
        }
        );
        emit_result(query_o2_stage);

        std::optional<gwn::bench::gwn_benchmark_stage_result> query_exact_stage{};
        if (!options.skip_exact) {
            query_exact_stage = run_stage(
                "query_exact", "queries/s", options.query_count, options, mesh.vertex_x.size(),
                mesh.tri_i0.size(), options.query_count, stream, build_bvh, [&]() noexcept {
                return run_exact_query(
                    bvh, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), stream
                );
            }
            );
            emit_result(*query_exact_stage);
        }

        auto const query_antipodal_crossing_stage = run_stage(
            "query_antipodal_crossings", "queries/s", options.query_count, options,
            mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream, build_bvh,
            [&]() noexcept {
            return run_antipodal_crossing<StackCapacity>(
                bvh.accessor(), d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(), stream
            );
        }
        );
        emit_result(query_antipodal_crossing_stage);

        auto const query_antipodal_boundary_stage = run_stage(
            "query_antipodal_boundary", "queries/s", options.query_count, options,
            mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream, setup_boundary,
            [&]() noexcept {
            return run_antipodal_boundary_query(
                geometry.accessor(), boundary_chain.accessor(), d_qx.span(), d_qy.span(),
                d_qz.span(), d_out.span(), stream
            );
        }
        );
        emit_result(query_antipodal_boundary_stage);

        auto const query_antipodal_gradient_stage = run_stage(
            "query_antipodal_gradient", "queries/s", options.query_count, options,
            mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream, setup_boundary,
            [&]() noexcept {
            return run_antipodal_gradient_query(
                geometry, boundary_chain, d_qx.span(), d_qy.span(), d_qz.span(), d_grad_x.span(),
                d_grad_y.span(), d_grad_z.span(), stream
            );
        }
        );
        emit_result(query_antipodal_gradient_stage);

        auto const query_antipodal_stage = measure_query_antipodal();
        emit_result(query_antipodal_stage);
        if (query_o1_stage.success && query_antipodal_stage.success &&
            query_antipodal_stage.latency_mean_ms > 0.0) {
            double const speedup =
                query_o1_stage.latency_mean_ms / query_antipodal_stage.latency_mean_ms;
            std::cout << "    taylor_o1_vs_antipodal_speedup=" << std::fixed << std::setprecision(3)
                      << speedup << std::defaultfloat << "\n";
        }
        if (query_o2_stage.success && query_antipodal_stage.success &&
            query_antipodal_stage.latency_mean_ms > 0.0) {
            double const speedup =
                query_o2_stage.latency_mean_ms / query_antipodal_stage.latency_mean_ms;
            std::cout << "    taylor_o2_vs_antipodal_speedup=" << std::fixed << std::setprecision(3)
                      << speedup << std::defaultfloat << "\n";
        }
        if (query_exact_stage.has_value() && query_exact_stage->success &&
            query_antipodal_stage.success && query_antipodal_stage.latency_mean_ms > 0.0) {
            double const speedup =
                query_exact_stage->latency_mean_ms / query_antipodal_stage.latency_mean_ms;
            std::cout << "    current_exact_point_kernel_vs_antipodal_speedup=" << std::fixed
                      << std::setprecision(3) << speedup << std::defaultfloat << "\n";
        }
        if (query_antipodal_crossing_stage.success && query_antipodal_boundary_stage.success &&
            query_antipodal_crossing_stage.latency_mean_ms > 0.0) {
            double const boundary_to_crossing = query_antipodal_boundary_stage.latency_mean_ms /
                                                query_antipodal_crossing_stage.latency_mean_ms;
            std::cout << "    antipodal_boundary_vs_crossing_time=" << std::fixed
                      << std::setprecision(3) << boundary_to_crossing << std::defaultfloat << "\n";
        }

        auto const ray_first_hit_stage = run_stage(
            "query_ray_first_hit", "rays/s", options.query_count, options, mesh.vertex_x.size(),
            mesh.tri_i0.size(), options.query_count, stream, setup_bvh_and_validate_rays,
            [&]() noexcept {
            return run_ray_first_hit<StackCapacity>(
                bvh, d_ray_ox.span(), d_ray_oy.span(), d_ray_oz.span(), d_ray_dx.span(),
                d_ray_dy.span(), d_ray_dz.span(), d_ray_hit.span(), stream
            );
        }
        );
        emit_result(ray_first_hit_stage);
        if (ray_first_hit_stage.success) {
            std::cout << "    ray_mix_hit_ratio=" << std::fixed << std::setprecision(3)
                      << ray_mix_hit_ratio << std::defaultfloat << "\n";
        }

        auto const dynamic_refit_query_o2_stage = run_stage(
            "dynamic_refit_query_o2", "queries/s", options.query_count, options,
            mesh.vertex_x.size(), mesh.tri_i0.size(), options.query_count, stream, build_bvh,
            [&]() noexcept {
            GWN_RETURN_ON_ERROR(
                gwn::gwn_update_geometry(
                    geometry, gwn::gwn_host_span<Real const>(dynamic_x.data(), dynamic_x.size()),
                    gwn::gwn_host_span<Real const>(dynamic_y.data(), dynamic_y.size()),
                    gwn::gwn_host_span<Real const>(dynamic_z.data(), dynamic_z.size()), stream
                )
            );
            GWN_RETURN_ON_ERROR(gwn::gwn_refit_bvh(geometry, bvh, stream));
            GWN_RETURN_ON_ERROR(gwn::gwn_refit_bvh_moment<2>(bvh, moment_tree_o2, stream));
            return run_taylor_query<StackCapacity, 2>(
                bvh, moment_tree_o2, d_qx.span(), d_qy.span(), d_qz.span(), d_out.span(),
                options.accuracy_scale, stream
            );
        }
        );
        emit_result(dynamic_refit_query_o2_stage);

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
}

} // namespace

int main(int argc, char **argv) try {
    benchmark_cli_options options{};
    std::string model_path;
    std::string model_dir;
    std::string csv_path{options.csv_path.string()};
    std::string builder{"hploc"};
    int stream_sync_per_iter = options.stream_sync_per_iter ? 1 : 0;

    CLI::App app{"Benchmark smallgwn build, refit, and query stages"};
    CLI::Option *const model_option =
        app.add_option("--model", model_path, "One PLY model")->check(CLI::ExistingFile);
    CLI::Option *const model_dir_option =
        app.add_option("--model-dir", model_dir, "Directory of PLY models")
            ->check(CLI::ExistingDirectory);
    model_option->excludes(model_dir_option);
    model_dir_option->excludes(model_option);
    app.add_option("--queries", options.query_count, "Query count")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--warmup", options.warmup_iters, "Warmup iterations")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    app.add_option("--iters", options.measure_iters, "Measured iterations")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--accuracy-scale", options.accuracy_scale, "Taylor accuracy scale")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--stack-capacity", options.stack_capacity, "Traversal stack capacity")
        ->check(CLI::IsMember(std::vector<int>{16, 24, 32, 48, 64, 96}))
        ->capture_default_str();
    app.add_option("--stream-sync-per-iter", stream_sync_per_iter, "Synchronize each iteration")
        ->check(CLI::IsMember(std::vector<int>{0, 1}))
        ->capture_default_str();
    app.add_option("--builder", builder, "BVH builder")
        ->check(CLI::IsMember({"lbvh", "hploc"}))
        ->capture_default_str();
    app.add_option("--csv", csv_path, "CSV output path")->capture_default_str();
    app.add_option("--seed", options.seed, "RNG seed")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_flag("--skip-exact", options.skip_exact, "Skip the exact winding stage");
    app.add_flag(
        "--winding-query-only", options.winding_query_only,
        "Measure only order-1 Taylor and complete Antipodal query stages"
    );

    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError const &error) { return app.exit(error); }

    if (model_path.empty() == model_dir.empty()) {
        std::cerr << "Exactly one of --model or --model-dir must be provided.\n";
        return 1;
    }

    if (!model_path.empty())
        options.model_path = std::filesystem::path(model_path);
    else
        options.model_dir = std::filesystem::path(model_dir);
    options.csv_path = std::filesystem::path(csv_path);
    options.builder =
        builder == "lbvh" ? benchmark_bvh_builder::k_lbvh : benchmark_bvh_builder::k_hploc;
    options.stream_sync_per_iter = stream_sync_per_iter != 0;

    std::vector<std::filesystem::path> const models = collect_models(options);
    if (models.empty())
        return 1;

    switch (options.stack_capacity) {
    case 16: return run_benchmark<16>(options, models);
    case 24: return run_benchmark<24>(options, models);
    case 32: return run_benchmark<32>(options, models);
    case 48: return run_benchmark<48>(options, models);
    case 64: return run_benchmark<64>(options, models);
    case 96: return run_benchmark<96>(options, models);
    default: return 1;
    }
} catch (std::exception const &e) {
    std::cerr << "Unhandled std::exception: " << e.what() << "\n";
    return 1;
} catch (...) {
    std::cerr << "Unhandled unknown exception.\n";
    return 1;
}

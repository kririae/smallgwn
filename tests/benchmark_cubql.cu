#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

// clang-format off
#include <cuBQL/bvh.h>
#include <cuBQL/builder/cuda.h>
// clang-format on
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include <cuBQL/traversal/rayQueries.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "benchmark_csv.cuh"

namespace {

using Real = gwn::bench::Real;
using Index = gwn::bench::Index;
using HostMesh = gwn::bench::HostMesh;
using Hit = gwn::gwn_ray_first_hit_result<Real, Index>;

constexpr int k_bvh_width = 4;
constexpr int k_stack_capacity = 64;
constexpr int k_block_size = 256;
constexpr int k_default_warmup = 10;
constexpr int k_default_iters = 30;
constexpr std::size_t k_default_ray_count = 1'000'000;
constexpr std::uint64_t k_default_seed = 0xB3A9E5D4ULL;

struct benchmark_options {
    std::vector<std::filesystem::path> models{};
    std::filesystem::path csv_path{"ray_comparison.csv"};
    std::size_t ray_count{k_default_ray_count};
    int warmup_iters{k_default_warmup};
    int measure_iters{k_default_iters};
    std::uint64_t seed{k_default_seed};
};

struct implementation_result {
    std::string implementation{};
    std::string bvh{};
    std::string builder{};
    gwn::bench::gwn_benchmark_stage_stats build{};
    gwn::bench::gwn_benchmark_stage_stats refit{};
    bool supports_refit{false};
    gwn::bench::gwn_benchmark_stage_stats trace{};
    double throughput_mrays_s{0.0};
    double hit_ratio{0.0};
    double reference_mismatch_rate{0.0};
    double hit_mismatch_rate{0.0};
    double t_mismatch_rate{0.0};
    double primitive_id_disagreement_rate{0.0};
    double matched_hit_mean_relative_t_error{0.0};
    double matched_hit_max_relative_t_error{0.0};
    bool success{false};
    std::string error{};
    std::vector<Hit> output{};
};

[[nodiscard]] gwn::gwn_status synchronize(cudaStream_t const stream) noexcept {
    return gwn::gwn_cuda_to_status(cudaStreamSynchronize(stream));
}

template <class Fn>
[[nodiscard]] gwn::gwn_status measure_trace(
    benchmark_options const &options, cudaStream_t const stream, Fn &&fn,
    gwn::bench::gwn_benchmark_stage_stats &stats
) noexcept {
    std::vector<float> latency_ms{};
    GWN_RETURN_ON_ERROR(
        gwn::bench::gwn_measure_stage_latency_ms(
            options.warmup_iters, options.measure_iters, stream, true, std::forward<Fn>(fn),
            latency_ms
        )
    );
    stats = gwn::bench::summarize_stage_ms(latency_ms);
    return gwn::gwn_status::ok();
}

template <class PrepareFn, class Fn>
[[nodiscard]] gwn::gwn_status measure_host_stage(
    benchmark_options const &options, cudaStream_t const stream, PrepareFn &&prepare, Fn &&fn,
    gwn::bench::gwn_benchmark_stage_stats &stats
) noexcept {
    std::vector<float> latency_ms;
    latency_ms.reserve(static_cast<std::size_t>(options.measure_iters));
    int const total_iters = options.warmup_iters + options.measure_iters;
    for (int iteration = 0; iteration < total_iters; ++iteration) {
        // Prepare and its synchronization stay outside the interval. This excludes object reset
        // from rebuild timing while ensuring every measured iteration starts from a settled state.
        GWN_RETURN_ON_ERROR(prepare());
        GWN_RETURN_ON_ERROR(synchronize(stream));

        auto const start = std::chrono::steady_clock::now();
        GWN_RETURN_ON_ERROR(fn());
        GWN_RETURN_ON_ERROR(synchronize(stream));
        auto const stop = std::chrono::steady_clock::now();
        if (iteration >= options.warmup_iters) {
            latency_ms.push_back(
                static_cast<float>(std::chrono::duration<double, std::milli>(stop - start).count())
            );
        }
    }
    stats = gwn::bench::summarize_stage_ms(latency_ms);
    return gwn::gwn_status::ok();
}

template <class Fn>
[[nodiscard]] gwn::gwn_status measure_host_stage(
    benchmark_options const &options, cudaStream_t const stream, Fn &&fn,
    gwn::bench::gwn_benchmark_stage_stats &stats
) noexcept {
    return measure_host_stage(options, stream, []() noexcept {
        return gwn::gwn_status::ok();
    }, std::forward<Fn>(fn), stats);
}

template <class T>
void upload_vector(
    std::vector<T> const &host, gwn::detail::gwn_device_array<T> &device, cudaStream_t const stream
) {
    device.copy_from_host(cuda::std::span<T const>(host.data(), host.size()), stream);
}

__global__ void fill_cubql_triangle_bounds(
    cuBQL::box3f *const boxes, std::size_t const triangle_count, cuBQL::vec3i const *const indices,
    cuBQL::vec3f const *const vertices
) {
    std::size_t const triangle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_id >= triangle_count)
        return;
    cuBQL::vec3i const index = indices[triangle_id];
    boxes[triangle_id] = cuBQL::box3f()
                             .including(vertices[index.x])
                             .including(vertices[index.y])
                             .including(vertices[index.z]);
}

template <class Bvh>
__global__ void cubql_ray_first_hit_kernel(
    Bvh const bvh, cuBQL::vec3i const *const indices, cuBQL::vec3f const *const vertices,
    std::size_t const ray_count, Real const *const ray_ox, Real const *const ray_oy,
    Real const *const ray_oz, Real const *const ray_dx, Real const *const ray_dy,
    Real const *const ray_dz, Hit *const output
) {
    std::size_t const ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= ray_count)
        return;

    cuBQL::ray3f ray{
        cuBQL::vec3f(ray_ox[ray_id], ray_oy[ray_id], ray_oz[ray_id]),
        cuBQL::vec3f(ray_dx[ray_id], ray_dy[ray_id], ray_dz[ray_id]), Real(0),
        std::numeric_limits<Real>::infinity()
    };
    // The traversal receives the same numeric interval as smallgwn. cuBQL's triangle intersector
    // applies its own strict lower-endpoint and determinant-threshold semantics inside compute().
    Hit best{};
    auto const intersect_primitive = [&](std::uint32_t const candidate_id) {
        cuBQL::vec3i const index = indices[candidate_id];
        cuBQL::Triangle const triangle{vertices[index.x], vertices[index.y], vertices[index.z]};
        cuBQL::RayTriangleIntersection intersection{};
        if (!intersection.compute(ray, triangle))
            return ray.tMax;
        best.t = intersection.t;
        best.primitive_id = static_cast<Index>(candidate_id);
        best.u = intersection.u;
        best.v = intersection.v;
        best.geometric_normal_x = intersection.N.x;
        best.geometric_normal_y = intersection.N.y;
        best.geometric_normal_z = intersection.N.z;
        best.status = gwn::gwn_ray_first_hit_status::k_hit;
        // shrinkingRayQuery uses the returned value as the next traversal tMax, so later node
        // visits cannot replace this record with a farther primitive.
        return intersection.t;
    };

    cuBQL::shrinkingRayQuery::forEachPrim(intersect_primitive, bvh, ray);
    output[ray_id] = best;
}

template <class Bvh>
[[nodiscard]] gwn::gwn_status launch_cubql_ray_first_hit(
    Bvh const bvh, gwn::detail::gwn_device_array<cuBQL::vec3i> const &indices,
    gwn::detail::gwn_device_array<cuBQL::vec3f> const &vertices, std::size_t const ray_count,
    gwn::detail::gwn_device_array<Real> const &ray_ox,
    gwn::detail::gwn_device_array<Real> const &ray_oy,
    gwn::detail::gwn_device_array<Real> const &ray_oz,
    gwn::detail::gwn_device_array<Real> const &ray_dx,
    gwn::detail::gwn_device_array<Real> const &ray_dy,
    gwn::detail::gwn_device_array<Real> const &ray_dz, gwn::detail::gwn_device_array<Hit> &output,
    cudaStream_t const stream
) noexcept {
    std::size_t const block_count = (ray_count + k_block_size - 1) / k_block_size;
    cubql_ray_first_hit_kernel<<<static_cast<unsigned int>(block_count), k_block_size, 0, stream>>>(
        bvh, indices.data(), vertices.data(), ray_count, ray_ox.data(), ray_oy.data(),
        ray_oz.data(), ray_dx.data(), ray_dy.data(), ray_dz.data(), output.data()
    );
    return gwn::gwn_cuda_to_status(cudaGetLastError());
}

[[nodiscard]] gwn::gwn_status launch_smallgwn_ray_first_hit(
    gwn::gwn_bvh4_object<Real, Index> const &bvh, gwn::detail::gwn_device_array<Real> const &ray_ox,
    gwn::detail::gwn_device_array<Real> const &ray_oy,
    gwn::detail::gwn_device_array<Real> const &ray_oz,
    gwn::detail::gwn_device_array<Real> const &ray_dx,
    gwn::detail::gwn_device_array<Real> const &ray_dy,
    gwn::detail::gwn_device_array<Real> const &ray_dz, gwn::detail::gwn_device_array<Hit> &output,
    cudaStream_t const stream
) noexcept {
    constexpr gwn::gwn_query_batch_config config{
        .block_size = gwn::k_gwn_default_query_batch_block_size,
        .stack_capacity = k_stack_capacity,
    };
    return gwn::gwn_compute_ray_first_hit_batch<config, k_bvh_width, Real, Index>(
        bvh, gwn::tests::device_input_span(ray_ox.span()),
        gwn::tests::device_input_span(ray_oy.span()), gwn::tests::device_input_span(ray_oz.span()),
        gwn::tests::device_input_span(ray_dx.span()), gwn::tests::device_input_span(ray_dy.span()),
        gwn::tests::device_input_span(ray_dz.span()), gwn::tests::device_span(output.span()),
        Real(0), std::numeric_limits<Real>::infinity(), stream
    );
}

[[nodiscard]] double hit_ratio(std::vector<Hit> const &output) {
    std::size_t const hits = static_cast<std::size_t>(
        std::count_if(output.begin(), output.end(), [](Hit const &hit) { return hit.hit(); })
    );
    return static_cast<double>(hits) / static_cast<double>(output.size());
}

void compare_with_reference(implementation_result const &reference, implementation_result &result) {
    std::size_t hit_mismatches = 0;
    std::size_t t_mismatches = 0;
    std::size_t primitive_id_disagreements = 0;
    std::size_t matched_hits = 0;
    double relative_error_sum = 0.0;
    double relative_error_max = 0.0;

    for (std::size_t i = 0; i < reference.output.size(); ++i) {
        Hit const &reference_record = reference.output[i];
        Hit const &result_record = result.output[i];
        bool const reference_hit = reference_record.hit();
        bool const result_hit = result_record.hit();
        if (reference_hit != result_hit) {
            ++hit_mismatches;
            continue;
        }
        if (!reference_hit)
            continue;

        double const scale = std::max(
            1.0, std::max(
                     std::abs(static_cast<double>(reference_record.t)),
                     std::abs(static_cast<double>(result_record.t))
                 )
        );
        double const relative_error =
            std::abs(static_cast<double>(reference_record.t - result_record.t)) / scale;
        // Primitive IDs are reported separately because shared-edge and equal-distance hits can
        // select different triangles without changing the first-hit distance.
        if (relative_error > 2e-4)
            ++t_mismatches;
        if (reference_record.primitive_id != result_record.primitive_id)
            ++primitive_id_disagreements;
        relative_error_sum += relative_error;
        relative_error_max = std::max(relative_error_max, relative_error);
        ++matched_hits;
    }

    double const ray_count = static_cast<double>(reference.output.size());
    result.hit_mismatch_rate = static_cast<double>(hit_mismatches) / ray_count;
    result.t_mismatch_rate = static_cast<double>(t_mismatches) / ray_count;
    result.reference_mismatch_rate = static_cast<double>(hit_mismatches + t_mismatches) / ray_count;
    if (matched_hits > 0) {
        result.primitive_id_disagreement_rate =
            static_cast<double>(primitive_id_disagreements) / static_cast<double>(matched_hits);
        result.matched_hit_mean_relative_t_error =
            relative_error_sum / static_cast<double>(matched_hits);
        result.matched_hit_max_relative_t_error = relative_error_max;
    }
}

[[nodiscard]] gwn::gwn_status download_output(
    gwn::detail::gwn_device_array<Hit> &device, std::vector<Hit> &host, cudaStream_t const stream
) {
    device.copy_to_host(cuda::std::span<Hit>(host), stream);
    return synchronize(stream);
}

template <class LaunchFn>
[[nodiscard]] implementation_result run_trace_implementation(
    benchmark_options const &options, std::string implementation, std::string bvh,
    std::string builder, gwn::bench::gwn_benchmark_stage_stats const build,
    gwn::bench::gwn_benchmark_stage_stats const refit, bool const supports_refit,
    gwn::detail::gwn_device_array<Hit> &output, cudaStream_t const stream, LaunchFn &&launch
) {
    implementation_result result{};
    result.implementation = std::move(implementation);
    result.bvh = std::move(bvh);
    result.builder = std::move(builder);
    result.build = build;
    result.refit = refit;
    result.supports_refit = supports_refit;
    result.output.resize(options.ray_count);

    gwn::gwn_status const trace_status = measure_trace(options, stream, launch, result.trace);
    if (!trace_status.is_ok()) {
        result.error = gwn::bench::gwn_status_to_string(trace_status);
        return result;
    }
    gwn::gwn_status const download_status = download_output(output, result.output, stream);
    if (!download_status.is_ok()) {
        result.error = gwn::bench::gwn_status_to_string(download_status);
        return result;
    }
    result.throughput_mrays_s = static_cast<double>(options.ray_count) / result.trace.mean_ms / 1e3;
    result.hit_ratio = hit_ratio(result.output);
    result.success = true;
    return result;
}

void write_csv_header(std::ofstream &csv) {
    csv << "gpu,model,vertices,triangles,rays,implementation,bvh,builder,"
           "build_mean_ms,build_p50_ms,build_p95_ms,refit_supported,"
           "refit_mean_ms,refit_p50_ms,refit_p95_ms,"
           "trace_mean_ms,trace_p50_ms,trace_p95_ms,throughput_mrays_s,hit_ratio,"
           "reference_mismatch_rate,hit_mismatch_rate,t_mismatch_rate,"
           "primitive_id_disagreement_rate,matched_hit_mean_relative_t_error,"
           "matched_hit_max_relative_t_error,success,error\n";
}

void write_csv_result(
    std::ofstream &csv, std::string_view const gpu, std::filesystem::path const &model,
    HostMesh const &mesh, benchmark_options const &options, implementation_result const &result
) {
    csv << gwn::bench::gwn_escape_csv(gpu) << ','
        << gwn::bench::gwn_escape_csv(model.filename().string()) << ',' << mesh.vertex_x.size()
        << ',' << mesh.tri_i0.size() << ',' << options.ray_count << ','
        << gwn::bench::gwn_escape_csv(result.implementation) << ','
        << gwn::bench::gwn_escape_csv(result.bvh) << ','
        << gwn::bench::gwn_escape_csv(result.builder) << ',' << std::fixed << std::setprecision(6)
        << result.build.mean_ms << ',' << result.build.p50_ms << ',' << result.build.p95_ms << ','
        << (result.supports_refit ? 1 : 0) << ',' << result.refit.mean_ms << ','
        << result.refit.p50_ms << ',' << result.refit.p95_ms << ',' << result.trace.mean_ms << ','
        << result.trace.p50_ms << ',' << result.trace.p95_ms << ',' << result.throughput_mrays_s
        << ',' << result.hit_ratio << ',' << result.reference_mismatch_rate << ','
        << result.hit_mismatch_rate << ',' << result.t_mismatch_rate << ','
        << result.primitive_id_disagreement_rate << ',' << result.matched_hit_mean_relative_t_error
        << ',' << result.matched_hit_max_relative_t_error << ',' << (result.success ? 1 : 0) << ','
        << gwn::bench::gwn_escape_csv(result.error) << '\n';
}

void print_result(implementation_result const &result) {
    std::cout << "  " << std::left << std::setw(28)
              << (result.implementation + " " + result.bvh + " " + result.builder);
    if (!result.success) {
        std::cout << "FAILED: " << result.error << '\n';
        return;
    }
    std::cout << std::right << std::fixed << std::setprecision(3) << " build=" << std::setw(8)
              << result.build.mean_ms << " ms refit=";
    if (result.supports_refit)
        std::cout << std::setw(8) << result.refit.mean_ms << " ms";
    else
        std::cout << std::setw(11) << "n/a";
    std::cout << " trace=" << std::setw(8) << result.trace.mean_ms
              << " ms throughput=" << std::setw(9) << result.throughput_mrays_s
              << " Mray/s mismatch=" << std::scientific << std::setprecision(2)
              << result.reference_mismatch_rate << std::defaultfloat << '\n';
}

[[nodiscard]] gwn::gwn_status validate_cubql_index_range(HostMesh const &mesh) noexcept {
    if (mesh.vertex_x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        mesh.tri_i0.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return gwn::gwn_status::invalid_argument("cuBQL adapter index range exceeded.");
    }
    return gwn::gwn_status::ok();
}

int run_model(
    benchmark_options const &options, std::filesystem::path const &model_path,
    std::string_view const gpu_name, std::ofstream &csv
) {
    std::optional<HostMesh> const loaded = gwn::tests::load_ply_mesh(model_path);
    if (!loaded.has_value()) {
        std::cerr << "Could not load model: " << model_path << '\n';
        return 1;
    }
    HostMesh const &mesh = *loaded;
    gwn::gwn_status const index_status = validate_cubql_index_range(mesh);
    if (!index_status.is_ok()) {
        std::cerr << index_status.message() << '\n';
        return 1;
    }

    cudaStream_t stream{};
    cudaError_t const stream_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (stream_status != cudaSuccess) {
        std::cerr << "Could not create CUDA stream: " << cudaGetErrorString(stream_status) << '\n';
        return 1;
    }
    auto const destroy_stream =
        gwn::gwn_make_scope_exit([&]() noexcept { (void)cudaStreamDestroy(stream); });

    auto const rays = gwn::bench::gwn_make_mixed_ray_soa(mesh, options.ray_count, options.seed);
    gwn::detail::gwn_device_array<Real> d_ray_ox(stream), d_ray_oy(stream), d_ray_oz(stream);
    gwn::detail::gwn_device_array<Real> d_ray_dx(stream), d_ray_dy(stream), d_ray_dz(stream);
    gwn::detail::gwn_device_array<Hit> d_output(stream);
    gwn::gwn_status const ray_upload_status = [&]() {
        upload_vector(rays.origin[0], d_ray_ox, stream);
        upload_vector(rays.origin[1], d_ray_oy, stream);
        upload_vector(rays.origin[2], d_ray_oz, stream);
        upload_vector(rays.direction[0], d_ray_dx, stream);
        upload_vector(rays.direction[1], d_ray_dy, stream);
        upload_vector(rays.direction[2], d_ray_dz, stream);
        d_output.resize(options.ray_count, stream);
        return synchronize(stream);
    }();
    if (!ray_upload_status.is_ok()) {
        std::cerr << "Ray upload failed: " << ray_upload_status.message() << '\n';
        return 1;
    }

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status const geometry_status = gwn::gwn_upload_geometry(
        geometry, gwn::gwn_host_span<Real const>(mesh.vertex_x.data(), mesh.vertex_x.size()),
        gwn::gwn_host_span<Real const>(mesh.vertex_y.data(), mesh.vertex_y.size()),
        gwn::gwn_host_span<Real const>(mesh.vertex_z.data(), mesh.vertex_z.size()),
        gwn::gwn_host_span<Index const>(mesh.tri_i0.data(), mesh.tri_i0.size()),
        gwn::gwn_host_span<Index const>(mesh.tri_i1.data(), mesh.tri_i1.size()),
        gwn::gwn_host_span<Index const>(mesh.tri_i2.data(), mesh.tri_i2.size()), stream
    );
    if (!geometry_status.is_ok() || !synchronize(stream).is_ok()) {
        std::cerr << "smallgwn geometry upload failed.\n";
        return 1;
    }

    std::vector<cuBQL::vec3f> cubql_vertices(mesh.vertex_x.size());
    for (std::size_t i = 0; i < cubql_vertices.size(); ++i)
        cubql_vertices[i] = cuBQL::vec3f(mesh.vertex_x[i], mesh.vertex_y[i], mesh.vertex_z[i]);
    std::vector<cuBQL::vec3i> cubql_indices(mesh.tri_i0.size());
    for (std::size_t i = 0; i < cubql_indices.size(); ++i) {
        cubql_indices[i] = cuBQL::vec3i(
            static_cast<int>(mesh.tri_i0[i]), static_cast<int>(mesh.tri_i1[i]),
            static_cast<int>(mesh.tri_i2[i])
        );
    }

    gwn::detail::gwn_device_array<cuBQL::vec3f> d_cubql_vertices(stream);
    gwn::detail::gwn_device_array<cuBQL::vec3i> d_cubql_indices(stream);
    gwn::detail::gwn_device_array<cuBQL::box3f> d_cubql_bounds(stream);
    gwn::gwn_status const cubql_upload_status = [&]() {
        upload_vector(cubql_vertices, d_cubql_vertices, stream);
        upload_vector(cubql_indices, d_cubql_indices, stream);
        d_cubql_bounds.resize(mesh.tri_i0.size(), stream);
        return synchronize(stream);
    }();
    if (!cubql_upload_status.is_ok()) {
        std::cerr << "cuBQL geometry upload failed.\n";
        return 1;
    }

    gwn::gwn_bvh4_object<Real, Index> lbvh;
    gwn::bench::gwn_benchmark_stage_stats lbvh_build{};
    gwn::gwn_status const lbvh_build_status = measure_host_stage(options, stream, [&]() noexcept {
        lbvh.clear();
        return gwn::gwn_status::ok();
    }, [&]() noexcept {
        return gwn::gwn_build_bvh(
            geometry, lbvh, gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_lbvh},
            stream
        );
    }, lbvh_build);
    if (!lbvh_build_status.is_ok()) {
        std::cerr << "smallgwn LBVH build failed: " << lbvh_build_status.message() << '\n';
        return 1;
    }

    gwn::bench::gwn_benchmark_stage_stats lbvh_refit{};
    gwn::gwn_status const lbvh_refit_status = measure_host_stage(options, stream, [&]() noexcept {
        return gwn::gwn_refit_bvh(geometry, lbvh, stream);
    }, lbvh_refit);
    if (!lbvh_refit_status.is_ok()) {
        std::cerr << "smallgwn LBVH refit failed: " << lbvh_refit_status.message() << '\n';
        return 1;
    }

    gwn::gwn_bvh4_object<Real, Index> hploc;
    gwn::bench::gwn_benchmark_stage_stats hploc_build{};
    gwn::gwn_status const hploc_build_status = measure_host_stage(options, stream, [&]() noexcept {
        hploc.clear();
        return gwn::gwn_status::ok();
    }, [&]() noexcept {
        return gwn::gwn_build_bvh(
            geometry, hploc,
            gwn::gwn_bvh_build_options{.method = gwn::gwn_bvh_build_method::k_hploc}, stream
        );
    }, hploc_build);
    if (!hploc_build_status.is_ok()) {
        std::cerr << "smallgwn H-PLOC build failed: " << hploc_build_status.message() << '\n';
        return 1;
    }
    gwn::bench::gwn_benchmark_stage_stats hploc_refit{};
    gwn::gwn_status const hploc_refit_status = measure_host_stage(options, stream, [&]() noexcept {
        return gwn::gwn_refit_bvh(geometry, hploc, stream);
    }, hploc_refit);
    if (!hploc_refit_status.is_ok()) {
        std::cerr << "smallgwn H-PLOC refit failed: " << hploc_refit_status.message() << '\n';
        return 1;
    }

    std::size_t const bounds_blocks = (mesh.tri_i0.size() + k_block_size - 1) / k_block_size;
    auto const fill_bounds = [&]() noexcept -> gwn::gwn_status {
        fill_cubql_triangle_bounds<<<
            static_cast<unsigned int>(bounds_blocks), k_block_size, 0, stream>>>(
            d_cubql_bounds.data(), mesh.tri_i0.size(), d_cubql_indices.data(),
            d_cubql_vertices.data()
        );
        return gwn::gwn_cuda_to_status(cudaGetLastError());
    };

    cuBQL::bvh3f cubql_binary{};
    auto reset_cubql_binary = [&]() noexcept -> gwn::gwn_status {
        try {
            if (cubql_binary.nodes != nullptr)
                cuBQL::cuda::free(cubql_binary, stream);
            cubql_binary = {};
            return gwn::gwn_status::ok();
        } catch (std::exception const &error) {
            return gwn::gwn_status::internal_error(error.what());
        }
    };
    gwn::bench::gwn_benchmark_stage_stats cubql_binary_build{};
    gwn::gwn_status const cubql_binary_build_status =
        measure_host_stage(options, stream, reset_cubql_binary, [&]() noexcept -> gwn::gwn_status {
        // Primitive-bound generation is part of the build contract on both sides; only resident
        // geometry upload and the preceding object reset are excluded from the measured interval.
        GWN_RETURN_ON_ERROR(fill_bounds());
        try {
            cuBQL::gpuBuilder(
                cubql_binary, d_cubql_bounds.data(), static_cast<std::uint32_t>(mesh.tri_i0.size()),
                cuBQL::BuildConfig{}, stream
            );
        } catch (std::exception const &error) {
            return gwn::gwn_status::internal_error(error.what());
        }
        return gwn::gwn_status::ok();
    }, cubql_binary_build);
    if (!cubql_binary_build_status.is_ok()) {
        std::cerr << "cuBQL BinaryBVH build failed: " << cubql_binary_build_status.message()
                  << '\n';
        return 1;
    }
    auto const free_cubql_binary = gwn::gwn_make_scope_exit([&]() noexcept {
        if (cubql_binary.nodes != nullptr)
            cuBQL::cuda::free(cubql_binary, stream);
    });
    gwn::bench::gwn_benchmark_stage_stats cubql_binary_refit{};
    gwn::gwn_status const cubql_binary_refit_status =
        measure_host_stage(options, stream, [&]() noexcept -> gwn::gwn_status {
        // cuBQL refit updates BinaryBVH bounds. Unlike smallgwn refit, its query path continues to
        // gather triangle vertices and therefore does not refresh a leaf-ordered triangle record.
        GWN_RETURN_ON_ERROR(fill_bounds());
        try {
            cuBQL::cuda::refit(cubql_binary, d_cubql_bounds.data(), stream);
        } catch (std::exception const &error) {
            return gwn::gwn_status::internal_error(error.what());
        }
        return gwn::gwn_status::ok();
    }, cubql_binary_refit);
    if (!cubql_binary_refit_status.is_ok()) {
        std::cerr << "cuBQL BinaryBVH refit failed: " << cubql_binary_refit_status.message()
                  << '\n';
        return 1;
    }

    cuBQL::WideBVH<float, 3, 4> cubql_wide4{};
    auto reset_cubql_wide4 = [&]() noexcept -> gwn::gwn_status {
        try {
            if (cubql_wide4.nodes != nullptr)
                cuBQL::cuda::free(cubql_wide4, stream);
            cubql_wide4 = {};
            return gwn::gwn_status::ok();
        } catch (std::exception const &error) {
            return gwn::gwn_status::internal_error(error.what());
        }
    };
    gwn::bench::gwn_benchmark_stage_stats cubql_wide4_build{};
    gwn::gwn_status const cubql_wide4_build_status =
        measure_host_stage(options, stream, reset_cubql_wide4, [&]() noexcept -> gwn::gwn_status {
        GWN_RETURN_ON_ERROR(fill_bounds());
        try {
            cuBQL::gpuBuilder(
                cubql_wide4, d_cubql_bounds.data(), static_cast<std::uint32_t>(mesh.tri_i0.size()),
                cuBQL::BuildConfig{}, stream
            );
        } catch (std::exception const &error) {
            return gwn::gwn_status::internal_error(error.what());
        }
        return gwn::gwn_status::ok();
    }, cubql_wide4_build);
    if (!cubql_wide4_build_status.is_ok()) {
        std::cerr << "cuBQL WideBVH4 build failed: " << cubql_wide4_build_status.message() << '\n';
        return 1;
    }
    auto const free_cubql_wide4 = gwn::gwn_make_scope_exit([&]() noexcept {
        if (cubql_wide4.nodes != nullptr)
            cuBQL::cuda::free(cubql_wide4, stream);
    });

    std::cout << "Model: " << model_path.filename().string() << " (V=" << mesh.vertex_x.size()
              << ", F=" << mesh.tri_i0.size() << ", rays=" << options.ray_count << ")\n";

    std::vector<implementation_result> results{};
    results.push_back(run_trace_implementation(
        options, "smallgwn", "BVH4", "LBVH", lbvh_build, lbvh_refit, true, d_output, stream,
        [&]() noexcept {
        return launch_smallgwn_ray_first_hit(
            lbvh, d_ray_ox, d_ray_oy, d_ray_oz, d_ray_dx, d_ray_dy, d_ray_dz, d_output, stream
        );
    }
    ));
    results.push_back(run_trace_implementation(
        options, "smallgwn", "BVH4", "H-PLOC", hploc_build, hploc_refit, true, d_output, stream,
        [&]() noexcept {
        return launch_smallgwn_ray_first_hit(
            hploc, d_ray_ox, d_ray_oy, d_ray_oz, d_ray_dx, d_ray_dy, d_ray_dz, d_output, stream
        );
    }
    ));
    results.push_back(run_trace_implementation(
        options, "cuBQL", "BinaryBVH", "spatial-median", cubql_binary_build, cubql_binary_refit,
        true, d_output, stream, [&]() noexcept {
        return launch_cubql_ray_first_hit(
            cubql_binary, d_cubql_indices, d_cubql_vertices, options.ray_count, d_ray_ox, d_ray_oy,
            d_ray_oz, d_ray_dx, d_ray_dy, d_ray_dz, d_output, stream
        );
    }
    ));
    results.push_back(run_trace_implementation(
        options, "cuBQL", "WideBVH4", "spatial-median", cubql_wide4_build, {}, false, d_output,
        stream, [&]() noexcept {
        return launch_cubql_ray_first_hit(
            cubql_wide4, d_cubql_indices, d_cubql_vertices, options.ray_count, d_ray_ox, d_ray_oy,
            d_ray_oz, d_ray_dx, d_ray_dy, d_ray_dz, d_output, stream
        );
    }
    ));

    if (!results.front().success) {
        std::cerr << "Reference implementation failed: " << results.front().error << '\n';
        return 1;
    }
    for (implementation_result &result : results) {
        // LBVH is the stable cross-implementation reference for reporting only. Independent CPU
        // query references remain correctness-test responsibilities, outside timed benchmark work.
        if (result.success)
            compare_with_reference(results.front(), result);
        print_result(result);
        write_csv_result(csv, gpu_name, model_path, mesh, options, result);
    }
    std::cout << '\n';
    return std::any_of(
               results.begin(), results.end(),
               [](implementation_result const &result) {
        return !result.success || result.reference_mismatch_rate > 1e-3;
    }
           )
               ? 1
               : 0;
}

} // namespace

int main(int argc, char **argv) try {
    benchmark_options options{};
    std::vector<std::string> model_paths;
    std::string csv_path{options.csv_path.string()};

    CLI::App app{"Compare smallgwn and cuBQL ray-triangle first-hit traversal"};
    app.add_option("--model", model_paths, "PLY model")->required()->check(CLI::ExistingFile);
    app.add_option("--rays", options.ray_count, "Ray count per model")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--warmup", options.warmup_iters, "Warmup launches")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--iters", options.measure_iters, "Measured launches")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--seed", options.seed, "Ray generator seed")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--csv", csv_path, "CSV output path")->capture_default_str();

    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError const &error) { return app.exit(error); }

    options.csv_path = std::filesystem::path(csv_path);
    options.models.reserve(model_paths.size());
    for (std::string const &path : model_paths)
        options.models.emplace_back(path);

    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        std::cerr << "CUDA device unavailable.\n";
        return 1;
    }

    std::ofstream csv(options.csv_path, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Could not open CSV output: " << options.csv_path << '\n';
        return 1;
    }
    write_csv_header(csv);

    std::cout << "Ray-triangle first-hit comparison\n"
              << "GPU: " << properties.name << "\n"
              << "Warmup: " << options.warmup_iters << ", measured: " << options.measure_iters
              << ", seed: " << options.seed << "\n\n";

    int failures = 0;
    for (std::filesystem::path const &model : options.models)
        failures += run_model(options, model, properties.name, csv);
    std::cout << "CSV: " << options.csv_path << '\n';
    return failures == 0 ? 0 : 1;
} catch (std::exception const &error) {
    std::cerr << "Unhandled exception: " << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Unhandled unknown exception.\n";
    return 1;
}

#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark_utils.cuh"

namespace gwn::bench {

[[nodiscard]] inline std::string gwn_escape_csv(std::string_view const value) {
    if (value.find_first_of("\",\n\r") == std::string_view::npos)
        return std::string(value);

    std::string escaped;
    escaped.reserve(value.size() + 4);
    escaped.push_back('"');
    for (char const c : value) {
        if (c == '"')
            escaped.push_back('"');
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

struct gwn_benchmark_csv_context {
    std::string timestamp{};
    std::string gpu_name{};
    int cuda_runtime_version{0};
    std::string builder{"hploc"};
    float accuracy_scale{2.0f};
    int stack_capacity{32};
    bool stream_sync_per_iter{true};
    std::uint64_t seed{0};
    std::size_t boundary_edge_count{0};
};

class gwn_benchmark_csv_writer {
public:
    explicit gwn_benchmark_csv_writer(std::string const &csv_path)
        : stream_(csv_path, std::ios::out | std::ios::trunc) {}

    [[nodiscard]] bool good() const noexcept { return stream_.good(); }

    void write_header() {
        stream_
            << "timestamp,gpu_name,cuda_runtime_version,model_path,vertex_count,"
               "triangle_count,boundary_edge_count,query_count,stage,warmup_iters,measure_iters,"
               "latency_mean_ms,latency_p50_ms,latency_p95_ms,throughput_per_s,unit,"
               "builder,accuracy_scale,stack_capacity,stream_sync_per_iter,seed,success,"
               "error_message\n";
    }

    void append_row(
        gwn_benchmark_csv_context const &context, std::string_view const model_path,
        gwn_benchmark_stage_result const &result
    ) {
        stream_ << gwn_escape_csv(context.timestamp) << "," << gwn_escape_csv(context.gpu_name)
                << "," << context.cuda_runtime_version << "," << gwn_escape_csv(model_path) << ","
                << result.vertex_count << "," << result.triangle_count << ","
                << context.boundary_edge_count << "," << result.query_count << ","
                << gwn_escape_csv(result.stage) << "," << result.warmup_iters << ","
                << result.measure_iters << "," << format_double(result.latency_mean_ms) << ","
                << format_double(result.latency_p50_ms) << ","
                << format_double(result.latency_p95_ms) << ","
                << format_double(result.throughput_per_s) << "," << gwn_escape_csv(result.unit)
                << "," << gwn_escape_csv(context.builder) << "," << context.accuracy_scale << ","
                << context.stack_capacity << "," << (context.stream_sync_per_iter ? 1 : 0) << ","
                << context.seed << "," << (result.success ? 1 : 0) << ","
                << gwn_escape_csv(result.error_message) << "\n";
    }

private:
    [[nodiscard]] static std::string format_double(double const value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << value;
        return out.str();
    }

    std::ofstream stream_;
};

} // namespace gwn::bench

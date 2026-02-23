#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark_utils.hpp"

namespace gwn::bench {

struct gwn_benchmark_csv_context {
    std::string timestamp{};
    std::string gpu_name{};
    int cuda_runtime_version{0};
    float accuracy_scale{2.0f};
    int stack_capacity{32};
    std::uint64_t seed{0};
};

class gwn_benchmark_csv_writer {
public:
    explicit gwn_benchmark_csv_writer(std::string const &csv_path)
        : stream_(csv_path, std::ios::out | std::ios::trunc) {}

    [[nodiscard]] bool good() const noexcept { return stream_.good(); }

    void write_header() {
        stream_ << "timestamp,gpu_name,cuda_runtime_version,model_path,vertex_count,"
                   "triangle_count,query_count,stage,warmup_iters,measure_iters,"
                   "latency_mean_ms,latency_p50_ms,latency_p95_ms,throughput_per_s,unit,"
                   "accuracy_scale,stack_capacity,seed,success,error_message\n";
    }

    void append_row(
        gwn_benchmark_csv_context const &context, std::string_view const model_path,
        gwn_benchmark_stage_result const &result
    ) {
        stream_ << escape_csv(context.timestamp) << "," << escape_csv(context.gpu_name) << ","
                << context.cuda_runtime_version << "," << escape_csv(model_path) << ","
                << result.vertex_count << "," << result.triangle_count << "," << result.query_count
                << "," << escape_csv(result.stage) << "," << result.warmup_iters << ","
                << result.measure_iters << "," << format_double(result.latency_mean_ms) << ","
                << format_double(result.latency_p50_ms) << ","
                << format_double(result.latency_p95_ms) << ","
                << format_double(result.throughput_per_s) << "," << escape_csv(result.unit) << ","
                << context.accuracy_scale << "," << context.stack_capacity << "," << context.seed
                << "," << (result.success ? 1 : 0) << "," << escape_csv(result.error_message)
                << "\n";
    }

private:
    [[nodiscard]] static std::string escape_csv(std::string_view const value) {
        bool needs_quotes = false;
        for (char const c : value) {
            if (c == '"' || c == ',' || c == '\n' || c == '\r') {
                needs_quotes = true;
                break;
            }
        }
        if (!needs_quotes)
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

    [[nodiscard]] static std::string format_double(double const value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << value;
        return out.str();
    }

    std::ofstream stream_;
};

} // namespace gwn::bench

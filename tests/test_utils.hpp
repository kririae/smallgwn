#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gwn/gwn_utils.cuh>

namespace gwn::tests {

// Common type aliases for tests.

using Real = float;
using Index = std::uint32_t;

// SoA host mesh.

struct HostMesh {
    std::vector<Real> vertex_x;
    std::vector<Real> vertex_y;
    std::vector<Real> vertex_z;
    std::vector<Index> tri_i0;
    std::vector<Index> tri_i1;
    std::vector<Index> tri_i2;
};

// CUDA runtime availability checks.

[[nodiscard]] inline bool is_cuda_runtime_unavailable(cudaError_t const result) noexcept {
    return result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
           result == cudaErrorOperatingSystem || result == cudaErrorSystemNotReady;
}

[[nodiscard]] inline bool
is_cuda_runtime_unavailable_message(std::string_view const message) noexcept {
    return message.find("cudaErrorNoDevice") != std::string_view::npos ||
           message.find("cudaErrorInsufficientDriver") != std::string_view::npos ||
           message.find("cudaErrorInitializationError") != std::string_view::npos ||
           message.find("cudaErrorSystemDriverMismatch") != std::string_view::npos ||
           message.find("cudaErrorOperatingSystem") != std::string_view::npos ||
           message.find("cudaErrorSystemNotReady") != std::string_view::npos ||
           message.find("cudaErrorNotSupported") != std::string_view::npos;
}

// gwn_status debug formatting.

[[nodiscard]] inline std::string status_to_debug_string(gwn::gwn_status const &status) {
    std::ostringstream out;
    out << status.message();
    if (status.has_location()) {
        std::source_location const loc = status.location();
        out << " at " << loc.file_name() << ":" << loc.line();
    }
    return out.str();
}

// Environment helpers.

[[nodiscard]] inline int get_env_positive_int(char const *name, int const default_value) {
    char const *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;
    int parsed = 0;
    char const *end = value + std::char_traits<char>::length(value);
    auto const [ptr, ec] = std::from_chars(value, end, parsed);
    if (ec != std::errc() || ptr != end || parsed <= 0)
        return default_value;
    return parsed;
}

[[nodiscard]] inline std::size_t
get_env_positive_size_t(char const *name, std::size_t default_value) {
    char const *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;
    std::size_t parsed = 0;
    char const *end = value + std::char_traits<char>::length(value);
    auto const [ptr, ec] = std::from_chars(value, end, parsed);
    if (ec != std::errc() || ptr != end || parsed == 0)
        return default_value;
    return parsed;
}

// OBJ loading.

[[nodiscard]] inline std::string_view trim_left(std::string_view const line) noexcept {
    std::size_t start = 0;
    while (start < line.size() &&
           (line[start] == ' ' || line[start] == '\t' || line[start] == '\r')) {
        ++start;
    }
    return line.substr(start);
}

[[nodiscard]] inline std::optional<Index>
parse_obj_index(std::string_view const token, std::size_t const vertex_count) {
    if (token.empty())
        return std::nullopt;

    std::size_t const slash = token.find('/');
    std::string_view const index_token =
        (slash == std::string_view::npos) ? token : token.substr(0, slash);
    if (index_token.empty())
        return std::nullopt;

    std::int64_t raw = 0;
    char const *begin = index_token.data();
    char const *end = index_token.data() + index_token.size();
    auto const [ptr, ec] = std::from_chars(begin, end, raw);
    if (ec != std::errc() || ptr != end || raw == 0)
        return std::nullopt;

    auto const vertex_count_i64 = static_cast<std::int64_t>(vertex_count);
    std::int64_t const resolved = (raw > 0) ? (raw - 1) : (vertex_count_i64 + raw);
    if (resolved < 0)
        return std::nullopt;

    auto const resolved_u64 = static_cast<std::uint64_t>(resolved);
    if (resolved_u64 >= static_cast<std::uint64_t>(vertex_count) ||
        resolved_u64 > static_cast<std::uint64_t>(std::numeric_limits<Index>::max())) {
        return std::nullopt;
    }

    return static_cast<Index>(resolved_u64);
}

[[nodiscard]] inline std::optional<HostMesh> load_obj_mesh(std::filesystem::path const &path) {
    std::ifstream input(path);
    if (!input.is_open())
        return std::nullopt;

    HostMesh mesh;
    std::string line;
    while (std::getline(input, line)) {
        std::string_view const trimmed = trim_left(line);
        if (trimmed.size() < 2 || trimmed[0] == '#')
            continue;

        if (trimmed.starts_with("v ")) {
            std::istringstream in(std::string(trimmed.substr(2)));
            Real x = Real(0);
            Real y = Real(0);
            Real z = Real(0);
            if (!(in >> x >> y >> z))
                continue;
            mesh.vertex_x.push_back(x);
            mesh.vertex_y.push_back(y);
            mesh.vertex_z.push_back(z);
            continue;
        }

        if (!trimmed.starts_with("f "))
            continue;

        std::istringstream in(std::string(trimmed.substr(2)));
        std::vector<Index> polygon{};
        std::string token;
        while (in >> token) {
            std::optional<Index> const parsed = parse_obj_index(token, mesh.vertex_x.size());
            if (parsed.has_value())
                polygon.push_back(*parsed);
        }
        if (polygon.size() < 3)
            continue;

        Index const first = polygon[0];
        for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
            mesh.tri_i0.push_back(first);
            mesh.tri_i1.push_back(polygon[corner]);
            mesh.tri_i2.push_back(polygon[corner + 1]);
        }
    }

    if (mesh.vertex_x.empty() || mesh.tri_i0.empty())
        return std::nullopt;

    return mesh;
}

// Model directory and path collection.

[[nodiscard]] inline std::optional<std::filesystem::path> find_model_data_dir() {
    if (char const *env = std::getenv("SMALLGWN_MODEL_DATA_DIR"); env != nullptr && *env != '\0') {
        std::filesystem::path const path(env);
        if (std::filesystem::is_directory(path))
            return path;
    }

    std::filesystem::path const default_path("/tmp/common-3d-test-models/data");
    if (std::filesystem::is_directory(default_path))
        return default_path;

    return std::nullopt;
}

[[nodiscard]] inline std::vector<std::filesystem::path>
collect_obj_model_paths(std::filesystem::path const &model_dir) {
    std::vector<std::filesystem::path> model_paths{};
    for (auto const &entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file())
            continue;
        std::filesystem::path const path = entry.path();
        if (path.extension() == ".obj")
            model_paths.push_back(path);
    }
    std::sort(model_paths.begin(), model_paths.end());
    model_paths.erase(std::unique(model_paths.begin(), model_paths.end()), model_paths.end());
    return model_paths;
}

[[nodiscard]] inline std::vector<std::filesystem::path> collect_model_paths() {
    std::vector<std::filesystem::path> model_paths{};

    if (char const *path_env = std::getenv("SMALLGWN_MODEL_PATH");
        path_env != nullptr && *path_env != '\0') {
        std::filesystem::path const path(path_env);
        if (std::filesystem::is_regular_file(path) && path.extension() == ".obj")
            model_paths.push_back(path);
    }

    if (char const *dir_env = std::getenv("SMALLGWN_MODEL_DATA_DIR");
        dir_env != nullptr && *dir_env != '\0') {
        std::filesystem::path const path(dir_env);
        if (std::filesystem::is_directory(path)) {
            auto const dir_models = collect_obj_model_paths(path);
            model_paths.insert(model_paths.end(), dir_models.begin(), dir_models.end());
        }
    } else {
        std::filesystem::path const default_path("/tmp/common-3d-test-models/data");
        if (std::filesystem::is_directory(default_path)) {
            auto const dir_models = collect_obj_model_paths(default_path);
            model_paths.insert(model_paths.end(), dir_models.begin(), dir_models.end());
        }
    }

    std::sort(model_paths.begin(), model_paths.end());
    model_paths.erase(std::unique(model_paths.begin(), model_paths.end()), model_paths.end());
    return model_paths;
}

// CUDA skip macros for GTest.

#define SMALLGWN_SKIP_IF_NO_CUDA()                                                                 \
    do {                                                                                           \
        cudaError_t const __cuda_check_result = cudaFree(nullptr);                                 \
        if (gwn::tests::is_cuda_runtime_unavailable(__cuda_check_result))                          \
            GTEST_SKIP() << "CUDA runtime unavailable: "                                           \
                         << cudaGetErrorString(__cuda_check_result);                               \
    } while (false)

#define SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status)                                           \
    do {                                                                                           \
        if (!(status).is_ok() && (status).error() == gwn::gwn_error::cuda_runtime_error &&         \
            gwn::tests::is_cuda_runtime_unavailable_message((status).message())) {                 \
            GTEST_SKIP() << "CUDA runtime unavailable: " << (status).message();                    \
        }                                                                                          \
    } while (false)

} // namespace gwn::tests

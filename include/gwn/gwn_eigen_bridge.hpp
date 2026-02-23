#pragma once

#include "gwn_geometry.cuh"

#if !__has_include(<Eigen/Core>)
#error "gwn_eigen_bridge.hpp requires Eigen/Core in the include path."
#endif

#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>

#include <Eigen/Core>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace gwn {

template <class Real = float, class Index = std::uint32_t, class DerivedV, class DerivedF>
gwn_status gwn_upload_from_eigen(
    gwn_geometry_object<Real, Index> &object, Eigen::MatrixBase<DerivedV> const &vertices,
    Eigen::MatrixBase<DerivedF> const &triangles, cudaStream_t const stream = gwn_default_stream()
) noexcept try {
    if (vertices.cols() != 3 || triangles.cols() != 3)
        return gwn_status::invalid_argument("Eigen inputs must be Nx3 vertices and Mx3 triangles.");

    Eigen::Index const vertex_count = vertices.rows();
    Eigen::Index const triangle_count = triangles.rows();
    if (vertex_count < 0 || triangle_count < 0)
        return gwn_status::invalid_argument("Eigen inputs cannot have negative sizes.");

    std::size_t const vertex_count_u = static_cast<std::size_t>(vertex_count);
    std::size_t const triangle_count_u = static_cast<std::size_t>(triangle_count);
    if (vertex_count_u > 0 &&
        (vertex_count_u - 1) > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        return gwn_status::invalid_argument("Vertex count exceeds index range.");
    }

    auto x = std::make_unique_for_overwrite<Real[]>(vertex_count_u);
    auto y = std::make_unique_for_overwrite<Real[]>(vertex_count_u);
    auto z = std::make_unique_for_overwrite<Real[]>(vertex_count_u);
    if (vertex_count > 0) {
        tbb::parallel_for(
            tbb::blocked_range<Eigen::Index>(0, vertex_count),
            [&](tbb::blocked_range<Eigen::Index> const &range) {
            for (Eigen::Index i = range.begin(); i < range.end(); ++i) {
                x[static_cast<std::size_t>(i)] = static_cast<Real>(vertices(i, 0));
                y[static_cast<std::size_t>(i)] = static_cast<Real>(vertices(i, 1));
                z[static_cast<std::size_t>(i)] = static_cast<Real>(vertices(i, 2));
            }
        }
        );
    }

    auto i0 = std::make_unique_for_overwrite<Index[]>(triangle_count_u);
    auto i1 = std::make_unique_for_overwrite<Index[]>(triangle_count_u);
    auto i2 = std::make_unique_for_overwrite<Index[]>(triangle_count_u);
    auto const try_convert_index = [vertex_count_u](auto const value, Index &out) noexcept {
        using value_type = std::decay_t<decltype(value)>;
        if constexpr (std::is_floating_point_v<value_type>) {
            if (!std::isfinite(value) || std::floor(value) != value)
                return false;
        }
        if constexpr (std::is_signed_v<value_type>) {
            if (value < value_type(0))
                return false;
        }

        std::uint64_t const value_u64 = static_cast<std::uint64_t>(value);
        if (value_u64 > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
            value_u64 >= static_cast<std::uint64_t>(vertex_count_u)) {
            return false;
        }
        out = static_cast<Index>(value_u64);
        return true;
    };

    std::atomic<bool> invalid_triangle_indices{false};
    if (triangle_count > 0) {
        tbb::parallel_for(
            tbb::blocked_range<Eigen::Index>(0, triangle_count),
            [&](tbb::blocked_range<Eigen::Index> const &range) {
            for (Eigen::Index i = range.begin(); i < range.end(); ++i) {
                Index tri0 = 0;
                Index tri1 = 0;
                Index tri2 = 0;
                bool const is_valid = try_convert_index(triangles(i, 0), tri0) &&
                                      try_convert_index(triangles(i, 1), tri1) &&
                                      try_convert_index(triangles(i, 2), tri2);
                if (!is_valid) {
                    invalid_triangle_indices.store(true, std::memory_order_relaxed);
                    continue;
                }
                i0[static_cast<std::size_t>(i)] = tri0;
                i1[static_cast<std::size_t>(i)] = tri1;
                i2[static_cast<std::size_t>(i)] = tri2;
            }
        }
        );
        if (invalid_triangle_indices.load(std::memory_order_relaxed)) {
            return gwn_status::invalid_argument(
                "Eigen triangle indices must be integer-valued in [0, vertex_count)."
            );
        }
    }

    return object.upload(
        cuda::std::span<Real const>(x.get(), vertex_count_u),
        cuda::std::span<Real const>(y.get(), vertex_count_u),
        cuda::std::span<Real const>(z.get(), vertex_count_u),
        cuda::std::span<Index const>(i0.get(), triangle_count_u),
        cuda::std::span<Index const>(i1.get(), triangle_count_u),
        cuda::std::span<Index const>(i2.get(), triangle_count_u), stream
    );
} catch (std::exception const &) {
    return gwn_status::internal_error("Unhandled std::exception in gwn_upload_from_eigen.");
} catch (...) {
    return gwn_status::internal_error("Unhandled unknown exception in gwn_upload_from_eigen.");
}

} // namespace gwn

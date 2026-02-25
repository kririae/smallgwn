#pragma once

#include <cmath>

#include "../gwn_utils.cuh"

namespace gwn {
namespace detail {

template <gwn_real_type Real> struct gwn_query_vec3 {
    Real x{};
    Real y{};
    Real z{};

    __host__ __device__ constexpr gwn_query_vec3() noexcept = default;
    __host__
        __device__ constexpr gwn_query_vec3(Real const x_, Real const y_, Real const z_) noexcept
        : x(x_), y(y_), z(z_) {}

    __host__ __device__ constexpr gwn_query_vec3 &operator+=(gwn_query_vec3 const &rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    __host__ __device__ constexpr gwn_query_vec3 &operator-=(gwn_query_vec3 const &rhs) noexcept {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    __host__ __device__ constexpr gwn_query_vec3 &operator*=(Real const s) noexcept {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    __host__ __device__ constexpr gwn_query_vec3 &operator/=(Real const s) noexcept {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }
};

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
operator+(gwn_query_vec3<Real> lhs, gwn_query_vec3<Real> const &rhs) noexcept {
    lhs += rhs;
    return lhs;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
operator-(gwn_query_vec3<Real> lhs, gwn_query_vec3<Real> const &rhs) noexcept {
    lhs -= rhs;
    return lhs;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
operator-(gwn_query_vec3<Real> const &v) noexcept {
    return {-v.x, -v.y, -v.z};
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
operator*(gwn_query_vec3<Real> lhs, Real const s) noexcept {
    lhs *= s;
    return lhs;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
operator*(Real const s, gwn_query_vec3<Real> rhs) noexcept {
    rhs *= s;
    return rhs;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
operator/(gwn_query_vec3<Real> lhs, Real const s) noexcept {
    lhs /= s;
    return lhs;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr Real
gwn_query_dot(gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr gwn_query_vec3<Real>
gwn_query_cross(gwn_query_vec3<Real> const &a, gwn_query_vec3<Real> const &b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ constexpr Real
gwn_query_squared_norm(gwn_query_vec3<Real> const &v) noexcept {
    return gwn_query_dot(v, v);
}

template <gwn_real_type Real>
[[nodiscard]] __host__ __device__ inline Real
gwn_query_norm(gwn_query_vec3<Real> const &v) noexcept {
    using std::sqrt;
    return sqrt(gwn_query_squared_norm(v));
}

} // namespace detail
} // namespace gwn

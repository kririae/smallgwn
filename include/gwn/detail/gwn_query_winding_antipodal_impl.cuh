#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../gwn_boundary.cuh"
#include "../gwn_bvh.cuh"
#include "../gwn_geometry.cuh"
#include "gwn_query_common_impl.cuh"
#include "gwn_query_vec3_impl.cuh"
#include "gwn_query_winding_impl.cuh"

namespace gwn {
namespace detail {

// The Antipodal winding number (Martens et al. 2026) casts one ray along +z and evaluates
//   w = signed crossings of the ray + (1/4pi) * sum over boundary edges of Area(-z, start, end).
// Both terms decide their integer jumps from the same projected-edge predicates. Those
// predicates are exact signs on the query-relative coordinates `vertex - query`, each rounded
// once to Real, under one symbolic perturbation of the query (new term: Z-dominant perturbation)
//   q + d3 * z + d1 * x + d2 * y  with  d3 >> d1 >> d2 > 0 infinitesimal.
// Every predicate therefore describes the same generic point, which lies on no triangle plane
// and on no branch cut, so the two terms agree and every query, including one on the surface,
// receives a value.

template <gwn_real_type Real> struct gwn_query_vec2 {
    Real x{};
    Real y{};
};

/// \brief Outcome of one Antipodal evaluation pass.
///
/// \c k_fallback_required (new term) reports that a rounded predicate could not certify its sign.
/// The rounded pass stops, and the complete query reruns with exact predicate fallbacks.
enum class gwn_antipodal_status : std::uint8_t {
    k_done,
    k_fallback_required,
};

template <gwn_real_type Real> struct gwn_antipodal_value_result {
    Real value{std::numeric_limits<Real>::quiet_NaN()};
    gwn_antipodal_status status{gwn_antipodal_status::k_done};
};

/// \brief Sign of a 2x2 determinant together with its rounded value.
template <gwn_real_type Real> struct gwn_antipodal_determinant {
    int sign{0};
    gwn_antipodal_status status{gwn_antipodal_status::k_done};
    Real value{};
};

[[nodiscard]] __host__ __device__ constexpr int
gwn_antipodal_sign_impl(double const value) noexcept {
    return (value > 0.0) - (value < 0.0);
}

/// \brief High and low parts of an error-free double transformation.
struct gwn_antipodal_expansion_pair {
    double high{};
    double low{};
};

/// \brief Return `a + b` as a rounded sum and its exact rounding error (Knuth TwoSum).
///
/// Explicit-rounding intrinsics keep NVCC from contracting neighboring products into FMAs, which
/// would make the low part inexact.
[[nodiscard]] __device__ inline gwn_antipodal_expansion_pair
gwn_antipodal_expansion_sum_impl(double const a, double const b) noexcept {
    double const sum = __dadd_rn(a, b);
    double const b_virtual = __dsub_rn(sum, a);
    double const a_virtual = __dsub_rn(sum, b_virtual);
    return {sum, __dadd_rn(__dsub_rn(a, a_virtual), __dsub_rn(b, b_virtual))};
}

/// \brief Return `a * b` as a rounded product and its exact rounding error.
[[nodiscard]] __device__ inline gwn_antipodal_expansion_pair
gwn_antipodal_expansion_product_impl(double const a, double const b) noexcept {
    double const product = __dmul_rn(a, b);
    return {product, __fma_rn(a, b, -product)};
}

/// \brief Exact sign of a sum of doubles accumulated as nonoverlapping error-free components.
///
/// Each accumulated value is folded through TwoSum, and nonzero rounding errors are retained in
/// increasing magnitude order (Shewchuk's grow-expansion with zero elimination). The largest
/// retained component then determines the sign of the exact sum.
template <int Capacity> struct gwn_antipodal_expansion_sign_accumulator {
    double components[Capacity]{};
    int size{0};

    __device__ void add(double const value) noexcept {
        if (value == 0.0)
            return;
        double carry = value;
        int retained = 0;
        for (int component_id = 0; component_id < size; ++component_id) {
            gwn_antipodal_expansion_pair const sum =
                gwn_antipodal_expansion_sum_impl(carry, components[component_id]);
            carry = sum.high;
            if (sum.low != 0.0)
                components[retained++] = sum.low;
        }
        if (carry != 0.0)
            components[retained++] = carry;
        size = retained;
    }

    __device__ void add_product(double const a, double const b) noexcept {
        gwn_antipodal_expansion_pair const product = gwn_antipodal_expansion_product_impl(a, b);
        add(product.low);
        add(product.high);
    }

    [[nodiscard]] __device__ int sign() const noexcept {
        if (size == 0)
            return 0;
        return components[size - 1] > 0.0 ? 1 : -1;
    }
};

/// \brief Widen one determinant row to double, rescaled for the expansions.
///
/// A determinant is multilinear in its rows, so scaling each row by its own power of two keeps
/// its sign. Float rows need no scaling because every product they form is exact in double.
/// Double rows are brought near unit magnitude, so the expansions stay exact while the nonzero
/// coordinates of one row lie within a factor of 2^300 of each other.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_query_vec3<double>
gwn_antipodal_expansion_row_impl(Real const x, Real const y, Real const z) noexcept {
    gwn_query_vec3<double> const row(
        static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)
    );
    if constexpr (std::is_same_v<Real, double>)
        return gwn_query_rescale_impl(row).value;
    return row;
}

/// \brief Exact sign of the 2x2 determinant `a.x * b.y - a.y * b.x`.
template <gwn_real_type Real>
[[nodiscard]] __device__ __noinline__ int gwn_antipodal_expansion_determinant_sign_impl(
    gwn_query_vec2<Real> const a, gwn_query_vec2<Real> const b
) noexcept {
    gwn_query_vec3<double> const first = gwn_antipodal_expansion_row_impl(a.x, a.y, Real(0));
    gwn_query_vec3<double> const second = gwn_antipodal_expansion_row_impl(b.x, b.y, Real(0));
    gwn_antipodal_expansion_sign_accumulator<4> accumulator{};
    accumulator.add_product(first.x, second.y);
    accumulator.add_product(-first.y, second.x);
    return accumulator.sign();
}

/// \brief Exact sign of the triple product `[a, b, c]` of query-relative vertices.
///
/// The triple product is expanded along z as
/// `height_a * D(b, c) + height_b * D(c, a) + height_c * D(a, b)`, where `D` is the projected
/// determinant. Its sign tells which side of the triangle plane the query lies on.
template <gwn_real_type Real>
[[nodiscard]] __device__ __noinline__ int gwn_antipodal_expansion_plane_side_impl(
    gwn_query_vec2<Real> const a, Real const height_a, gwn_query_vec2<Real> const b,
    Real const height_b, gwn_query_vec2<Real> const c, Real const height_c
) noexcept {
    gwn_query_vec3<double> const vertex_a = gwn_antipodal_expansion_row_impl(a.x, a.y, height_a);
    gwn_query_vec3<double> const vertex_b = gwn_antipodal_expansion_row_impl(b.x, b.y, height_b);
    gwn_query_vec3<double> const vertex_c = gwn_antipodal_expansion_row_impl(c.x, c.y, height_c);

    gwn_antipodal_expansion_sign_accumulator<24> accumulator{};
    auto const add_term = [&](double const height, gwn_query_vec3<double> const &u,
                              gwn_query_vec3<double> const &v) noexcept {
        gwn_antipodal_expansion_pair const left = gwn_antipodal_expansion_product_impl(u.x, v.y);
        gwn_antipodal_expansion_pair const right = gwn_antipodal_expansion_product_impl(u.y, v.x);
        accumulator.add_product(height, left.low);
        accumulator.add_product(height, left.high);
        accumulator.add_product(-height, right.low);
        accumulator.add_product(-height, right.high);
    };
    add_term(vertex_a.z, vertex_b, vertex_c);
    add_term(vertex_b.z, vertex_c, vertex_a);
    add_term(vertex_c.z, vertex_a, vertex_b);
    return accumulator.sign();
}

/// \brief Return the sign of `a.x * b.y - a.y * b.x` (new term: certified sign).
///
/// The rounded value comes from \ref gwn_query_kahan_determinant_impl. A right product at or
/// above `4 * min / epsilon` has an exact rounding error, and every later
/// rounding stays relative or is dominated by it; below, only a result above that magnitude
/// certifies the sign. Exact zero factors decide the sign directly. Any other uncertified sign,
/// including a NaN from an overflowing product, takes the expansion fallback with
/// \p ExactFallback and reports \c k_fallback_required without it.
template <bool ExactFallback, gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_determinant<Real> gwn_antipodal_determinant_impl(
    gwn_query_vec2<Real> const a, gwn_query_vec2<Real> const b
) noexcept {
    Real const right = a.y * b.x;
    Real const value = gwn_query_kahan_determinant_impl(a.x, a.y, b.x, b.y);
    Real const certified_magnitude =
        Real(4) * std::numeric_limits<Real>::min() / std::numeric_limits<Real>::epsilon();

    int sign = (value > Real(0)) - (value < Real(0));
    bool const certified = fabs(right) >= certified_magnitude || fabs(value) > certified_magnitude;
    if (certified && (sign != 0 || value == Real(0)))
        return {sign, gwn_antipodal_status::k_done, value};

    if (a.y == Real(0) || b.x == Real(0))
        sign = gwn_antipodal_sign_impl(a.x) * gwn_antipodal_sign_impl(b.y);
    else if (a.x == Real(0) || b.y == Real(0))
        sign = -gwn_antipodal_sign_impl(a.y) * gwn_antipodal_sign_impl(b.x);
    else if constexpr (ExactFallback)
        sign = gwn_antipodal_expansion_determinant_sign_impl(a, b);
    else
        return {0, gwn_antipodal_status::k_fallback_required, value};
    return {sign, gwn_antipodal_status::k_done, value};
}

/// \brief Sign of a projected edge determinant that is exactly zero, under the perturbation.
///
/// Moving the projected query by (d1, d2) adds `d1 * (a.y - b.y) + d2 * (b.x - a.x)` to the
/// determinant of the query-relative endpoints, so these coefficients decide it in lexicographic
/// order. They are rounded differences with exact signs, antisymmetric under swapping \p a and
/// \p b, so every triangle and boundary edge sharing two vertices agrees. Zero remains only for an
/// edge that projects to a point.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline int gwn_antipodal_perturbed_sign_impl(
    gwn_query_vec2<Real> const a, gwn_query_vec2<Real> const b
) noexcept {
    int const first = gwn_antipodal_sign_impl(a.y - b.y);
    return first != 0 ? first : gwn_antipodal_sign_impl(b.x - a.x);
}

/// \brief Perturbed projected-edge sign and rounded determinant for triangle crossings.
template <gwn_real_type Real> struct gwn_antipodal_projected_edge_classification {
    int sign{0};
    gwn_antipodal_status status{gwn_antipodal_status::k_done};
    Real projected_determinant{};
};

/// \brief Classify the projected query against one oriented projected edge.
///
/// \param a First endpoint relative to the query point, projected along z.
/// \param b Second endpoint relative to the query point, projected along z.
/// \return The perturbed exact sign and the rounded projected determinant, or
///         \c k_fallback_required.
template <bool ExactFallback, gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_projected_edge_classification<Real>
gwn_antipodal_projected_edge_classify_impl(
    gwn_query_vec2<Real> const a, gwn_query_vec2<Real> const b
) noexcept {
    auto const determinant = gwn_antipodal_determinant_impl<ExactFallback>(a, b);
    int const sign =
        determinant.sign != 0 ? determinant.sign : gwn_antipodal_perturbed_sign_impl(a, b);
    return {sign, determinant.status, determinant.value};
}

/// \brief Classify one oriented triangle's +z crossing contribution.
///
/// The crossing count is the exact count for the perturbed query, so rays through shared edges
/// or vertices are counted once at any distance from the surface. Without \p ExactFallback, an
/// uncertified predicate that could change the result reports \c k_fallback_required.
template <bool ExactFallback, gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_status gwn_ray_triangle_crossing_antipodal_impl(
    gwn_query_vec3<Real> const &origin, gwn_bvh_triangle<Real> const &triangle, int &crossing
) noexcept {
    // Query-relative coordinates round each component once, exactly as the boundary term does for
    // the same geometry vertices. The z coordinates are read only for projected hits, which keeps
    // them out of the register-heavy edge tests.
    gwn_query_vec2<Real> const pa{triangle.v0_x - origin.x, triangle.v0_y - origin.y};
    gwn_query_vec2<Real> const pb{triangle.v1_x - origin.x, triangle.v1_y - origin.y};
    gwn_query_vec2<Real> const pc{triangle.v2_x - origin.x, triangle.v2_y - origin.y};
    auto const is_resolved = [](auto const &edge_class) noexcept {
        return edge_class.status == gwn_antipodal_status::k_done;
    };

    // Matching perturbed edge signs mean the projected query lies inside the oriented triangle.
    // A zero perturbed sign marks an edge that projects to a point, so the projected triangle has
    // no area. Rejecting after two edges skips the third determinant for most candidates.
    auto const edge_ab = gwn_antipodal_projected_edge_classify_impl<ExactFallback>(pa, pb);
    auto const edge_bc = gwn_antipodal_projected_edge_classify_impl<ExactFallback>(pb, pc);
    if (is_resolved(edge_ab) && is_resolved(edge_bc) &&
        (edge_ab.sign == 0 || edge_ab.sign != edge_bc.sign)) {
        return gwn_antipodal_status::k_done;
    }
    auto const edge_ca = gwn_antipodal_projected_edge_classify_impl<ExactFallback>(pc, pa);
    if constexpr (!ExactFallback) {
        // Unresolved signs only arise from underflowing or overflowing products, so the rare
        // restart with exact fallbacks is cheaper than reasoning about partial evidence here.
        if (!is_resolved(edge_ab) || !is_resolved(edge_bc) || !is_resolved(edge_ca))
            return gwn_antipodal_status::k_fallback_required;
    }
    if (edge_ca.sign != edge_ab.sign)
        return gwn_antipodal_status::k_done;

    // The projected triangle orientation is the sum of the three edge determinants, whose
    // perturbation terms cancel, so equal perturbed edge signs fix the sign of the normal's z
    // component exactly. It is both the crossing contribution and the plane side required for an
    // intersection ahead of the query.
    int const orientation = edge_ab.sign;

    // The intersection parameter is t = [pa, pb, pc] / normal_z. Expanding the triple product
    // along z weights the vertex heights by the projected determinants, which are the
    // unnormalized barycentric weights of the projected query. Vertices all ahead of the query, or
    // all behind it, therefore fix the plane side without arithmetic.
    Real const height_a = triangle.v0_z - origin.z;
    Real const height_b = triangle.v1_z - origin.z;
    Real const height_c = triangle.v2_z - origin.z;
    if (height_a < Real(0) && height_b < Real(0) && height_c < Real(0))
        return gwn_antipodal_status::k_done;
    if (!(height_a > Real(0) && height_b > Real(0) && height_c > Real(0))) {
        Real const term_a = height_a * edge_bc.projected_determinant;
        Real const term_b = height_b * edge_ca.projected_determinant;
        Real const term_c = height_c * edge_ab.projected_determinant;
        Real const plane_offset = term_a + term_b + term_c;
        // Each determinant carries relative error 2u, each product adds u, and the two sums add 2u
        // relative to the absolute sum, 5u in total. 3 * epsilon = 6u also covers rounding the
        // bound. The height-scaled minimum covers determinant terms that underflowed. Non-finite
        // terms fail the comparison and take the exact path.
        Real const absolute_sum = fabs(term_a) + fabs(term_b) + fabs(term_c);
        Real const error_bound = Real(3) * std::numeric_limits<Real>::epsilon() * absolute_sum +
                                 (fabs(height_a) + fabs(height_b) + fabs(height_c) + Real(1)) *
                                     std::numeric_limits<Real>::min();
        int plane_side = 0;
        if (fabs(plane_offset) > error_bound) {
            plane_side = plane_offset > Real(0) ? 1 : -1;
        } else if constexpr (ExactFallback) {
            plane_side =
                gwn_antipodal_expansion_plane_side_impl(pa, height_a, pb, height_b, pc, height_c);
        } else {
            return gwn_antipodal_status::k_fallback_required;
        }
        // A query on the triangle plane moves by d3 along +z, which dominates the in-plane
        // perturbation: the triple product becomes -normal_z * d3, whose sign is -orientation.
        // The perturbed query therefore lies above the triangle and the ray misses it.
        if (plane_side != orientation)
            return gwn_antipodal_status::k_done;
    }

    crossing = orientation;
    return gwn_antipodal_status::k_done;
}

/// \brief Signed spherical area of an edge whose projection is collinear with the query.
///
/// The projected determinant is exactly zero, so the unperturbed atan2 numerator vanishes and the
/// area takes its limit at the perturbed query:
/// - An endpoint on the query's vertical line above it tends to +z. The area tends to
///   `2 * atan2(e.y, -e.x)` for the other endpoint `e`, with a zero `e.y` read as `-0` when
///   `e.x > 0`.
/// - An endpoint on the line at or below the query tends to -z, as does an edge along the line.
///   The area is zero.
/// - When the projected query lies strictly inside the projected segment, the edge passes the
///   query's vertical line at a height whose sign is that of `s.z * |e.k| + e.z * |s.k|` for a
///   nonzero coordinate `k`. Above the query, the perturbed sign selects the atan2 branch, which
///   yields -2pi for a positive sign and 2pi for a negative one. At or below the query, the area
///   is zero because the perturbed query lies above the edge.
/// - Otherwise the edge lies on one side of the query and the area is zero.
template <bool ExactFallback, gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_antipodal_collinear_edge_area_impl(
    gwn_query_vec3<Real> const &start, gwn_query_vec3<Real> const &end
) noexcept {
    Real const two_pi = Real(6.283185307179586476925286766559005768);
    auto const done = [](Real const area) noexcept {
        return gwn_antipodal_value_result<Real>{area, gwn_antipodal_status::k_done};
    };
    auto const endpoint_limit = [](gwn_query_vec3<Real> const &other) noexcept {
        Real const numerator = other.y == Real(0) ? copysign(Real(0), -other.x) : other.y;
        return Real(2) * atan2(numerator, -other.x);
    };
    bool const start_on_line = start.x == Real(0) && start.y == Real(0);
    bool const end_on_line = end.x == Real(0) && end.y == Real(0);

    if (start_on_line && end_on_line)
        return done(Real(0));
    if (start_on_line)
        return done(start.z > Real(0) ? endpoint_limit(end) : Real(0));
    if (end_on_line)
        return done(end.z > Real(0) ? -endpoint_limit(start) : Real(0));

    bool const same_direction =
        gwn_antipodal_sign_impl(start.x) == gwn_antipodal_sign_impl(end.x) &&
        gwn_antipodal_sign_impl(start.y) == gwn_antipodal_sign_impl(end.y);
    if (same_direction)
        return done(Real(0));

    // The endpoints point opposite ways, so start.k and end.k are both nonzero for this k.
    bool const use_x = end.x != Real(0);
    Real const start_extent = fabs(use_x ? start.x : start.y);
    Real const end_extent = fabs(use_x ? end.x : end.y);
    auto const height = gwn_antipodal_determinant_impl<ExactFallback>(
        gwn_query_vec2<Real>{start.z, start_extent}, gwn_query_vec2<Real>{-end.z, end_extent}
    );
    if (height.status != gwn_antipodal_status::k_done)
        return {std::numeric_limits<Real>::quiet_NaN(), height.status};
    if (height.sign <= 0)
        return done(Real(0));
    int const perturbed_sign = gwn_antipodal_perturbed_sign_impl(
        gwn_query_vec2<Real>{start.x, start.y}, gwn_query_vec2<Real>{end.x, end.y}
    );
    return done(perturbed_sign > 0 ? -two_pi : two_pi);
}

/// \brief Signed spherical area and whether its products stayed within the accurate range.
template <gwn_real_type Compute> struct gwn_antipodal_area_terms {
    Compute value{};
    bool in_range{};
};

/// \brief Signed spherical area of the triangle (-z, start, end) from Martens et al.
///
/// The principal atan2 branch jumps where the +z ray crosses the edge, exactly where the crossing
/// count changes. The scaled denominator `|s||e| - s.z|e| - e.z|s| + s.e` factors as
/// `gap(s) * gap(e) + s_xy . e_xy` with `gap(v) = |v| - v.z`. For an endpoint almost straight
/// along +z, `|v| - v.z` cancels and a rounded denominator could move the area by 2pi, so such
/// endpoints use `gap(v) = |v_xy|^2 / (|v| + v.z)` instead. atan2 is invariant under a common
/// positive scale, so multiplying numerator and denominator by `(|v| + v.z) / |v|` in (1, 2]
/// removes that division.
///
/// With \p CheckRange, the result also reports whether every product stayed clear of underflow
/// and overflow; without it, the caller has established that for the whole query.
///
/// \param projected_determinant Projected determinant of \p start and \p end with its exact
///        nonzero sign; a magnitude that underflowed is a zero of that sign.
template <bool CheckRange, gwn_real_type Compute>
[[nodiscard]] __device__ inline gwn_antipodal_area_terms<Compute>
gwn_antipodal_signed_spherical_area_impl(
    gwn_query_vec3<Compute> const &start, gwn_query_vec3<Compute> const &end,
    Compute const projected_determinant
) noexcept {
    struct endpoint_terms {
        Compute gap_numerator;
        Compute scale;
    };
    auto const prepare = [](gwn_query_vec3<Compute> const &value, Compute const norm2) noexcept {
        Compute const inverse_norm = rsqrt(norm2);
        if (value.z > Compute(0)) {
            return endpoint_terms{
                (value.x * value.x + value.y * value.y) * inverse_norm,
                Compute(1) + value.z * inverse_norm,
            };
        }
        return endpoint_terms{norm2 * inverse_norm - value.z, Compute(1)};
    };

    Compute const start_norm2 = gwn_query_dot(start, start);
    Compute const end_norm2 = gwn_query_dot(end, end);
    endpoint_terms const start_terms = prepare(start, start_norm2);
    endpoint_terms const end_terms = prepare(end, end_norm2);
    Compute const common_scale = start_terms.scale * end_terms.scale;

    // With x1 = -z, the atan2 numerator is the negated projected determinant. Its sign is exact, so
    // even a zero magnitude selects the branch that matches the crossing count.
    Compute const numerator = -projected_determinant * common_scale;
    Compute const denominator = start_terms.gap_numerator * end_terms.gap_numerator +
                                (start.x * end.x + start.y * end.y) * common_scale;

    Compute const area = Compute(2) * atan2(numerator, denominator);
    if constexpr (!CheckRange)
        return {area, true};
    // Accurate magnitudes need endpoint norms well inside the normal range and an atan2 argument
    // well above the underflow threshold; unit-length endpoints near the z axis can still make
    // both atan2 arguments underflow. NaN fails every comparison.
    Compute const lower = gwn_query_range_floor_impl<Compute>();
    bool const in_range = fmin(start_norm2, end_norm2) >= lower &&
                          fmax(start_norm2, end_norm2) <= Compute(1) / lower &&
                          fabs(numerator) + fabs(denominator) >= lower;
    return {area, in_range};
}

/// \brief Evaluate one boundary edge's spherical area from rescaled endpoints.
///
/// The fallback for edges whose rounded area could lose accuracy to underflow or overflow. The
/// area is homogeneous of degree zero in each endpoint, so each endpoint can be scaled
/// independently. Float endpoints are evaluated in double, where every finite float product is
/// normal; double endpoints are rescaled by powers of two to unit magnitude.
template <gwn_real_type Real>
[[nodiscard]] __device__ __noinline__ Real gwn_antipodal_rescaled_spherical_area_impl(
    gwn_query_vec3<Real> const &start, gwn_query_vec3<Real> const &end, int const sign
) noexcept {
    auto const widen = [](gwn_query_vec3<Real> const &value) noexcept {
        gwn_query_vec3<double> const wide(double(value.x), double(value.y), double(value.z));
        if constexpr (std::is_same_v<Real, double>)
            return gwn_query_rescale_impl(wide).value;
        return wide;
    };
    gwn_query_vec3<double> const wide_start = widen(start);
    gwn_query_vec3<double> const wide_end = widen(end);
    double const magnitude =
        fabs(gwn_query_kahan_determinant_impl(wide_start.x, wide_start.y, wide_end.x, wide_end.y));
    return static_cast<Real>(gwn_antipodal_signed_spherical_area_impl<false>(
                                 wide_start, wide_end, copysign(magnitude, double(sign))
    )
                                 .value);
}

/// \brief Return the signed spherical area of one oriented boundary edge.
///
/// The rounded pass relies on the query-level condition of
/// \ref gwn_antipodal_rounded_range_clear_impl: every nonzero projected product exceeds the
/// certified magnitude without overflowing, so the rounded Kahan determinant already carries the
/// exact sign and a zero is an exact zero. With \p ExactFallback, each edge certifies its sign,
/// checks its own range, and rescales when needed.
template <bool ExactFallback, gwn_real_type Real>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_antipodal_boundary_edge_area_impl(
    gwn_query_vec3<Real> const &start, gwn_query_vec3<Real> const &end
) noexcept {
    if constexpr (!ExactFallback) {
        Real const determinant = gwn_query_kahan_determinant_impl(start.x, start.y, end.x, end.y);
        if (determinant == Real(0))
            return gwn_antipodal_collinear_edge_area_impl<false>(start, end);
        return {
            gwn_antipodal_signed_spherical_area_impl<false>(start, end, determinant).value,
            gwn_antipodal_status::k_done
        };
    } else {
        auto const determinant = gwn_antipodal_determinant_impl<true>(
            gwn_query_vec2<Real>{start.x, start.y}, gwn_query_vec2<Real>{end.x, end.y}
        );
        if (determinant.sign == 0)
            return gwn_antipodal_collinear_edge_area_impl<true>(start, end);
        Real const signed_determinant =
            copysign(fabs(determinant.value), static_cast<Real>(determinant.sign));
        auto const area =
            gwn_antipodal_signed_spherical_area_impl<true>(start, end, signed_determinant);
        if (area.in_range)
            return {area.value, gwn_antipodal_status::k_done};
        return {
            gwn_antipodal_rescaled_spherical_area_impl(start, end, determinant.sign),
            gwn_antipodal_status::k_done
        };
    }
}

/// \brief Largest offset of the query from a face of \p bounds.
///
/// Bounds that contain every vertex make this an upper bound on every query-relative coordinate
/// magnitude. A NaN query coordinate is dropped by `fmax`, so callers test the query separately.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline Real gwn_antipodal_bounds_offset_impl(
    gwn_aabb<Real> const &bounds, gwn_query_vec3<Real> const &query
) noexcept {
    return fmax(
        fmax(
            fmax(fabs(bounds.min_x - query.x), fabs(bounds.max_x - query.x)),
            fmax(fabs(bounds.min_y - query.y), fabs(bounds.max_y - query.y))
        ),
        fmax(fabs(bounds.min_z - query.z), fabs(bounds.max_z - query.z))
    );
}

/// \brief Whether the rounded boundary areas of a query can neither underflow nor overflow.
///
/// Rounded areas lose accuracy only when a product underflows or overflows, and per-edge range
/// checks cost the boundary loop measurably, so the rounded pass excludes both once per query.
/// When the query's x and y magnitudes are at least `64 * sqrt(floor) / epsilon`, each nonzero
/// projected component of `vertex - query` is at least `16 * sqrt(floor)` (distinct floats near
/// the query differ by at least `|q| * epsilon / 4`), so every product of projected components
/// stays above the range floor and the Kahan determinant certifies itself. Subnormal gaps remain
/// possible, but they only matter within rounding of the edge. When \p bounds_offset, the
/// largest offset of the query from a face of bounds that contain every boundary vertex, is at
/// most `sqrt(1 / floor) / 2`, no squared length exceeds `1 / floor`.
template <gwn_real_type Real>
[[nodiscard]] __device__ inline bool gwn_antipodal_rounded_range_clear_impl(
    gwn_query_vec3<Real> const &query, Real const bounds_offset
) noexcept {
    Real const range_floor = gwn_query_range_floor_impl<Real>();
    Real const coordinate_floor =
        Real(64) * sqrt(range_floor) / std::numeric_limits<Real>::epsilon();
    Real const offset_ceiling = Real(0.5) * sqrt(Real(1) / range_floor);
    return fabs(query.x) >= coordinate_floor && fabs(query.y) >= coordinate_floor &&
           bounds_offset <= offset_ceiling;
}

/// \brief Sum the boundary spherical-area contribution for one predicate mode.
///
/// Without \p ExactFallback, the query must satisfy the range condition of
/// \ref gwn_antipodal_rounded_range_clear_impl, under which no rounded area product underflows
/// or overflows. With it, each edge checks its own range.
template <bool ExactFallback, gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_antipodal_boundary_contribution_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query
) noexcept {
    Real const pi = Real(3.141592653589793238462643383279502884);

    Real area_sum = Real(0);
    Real area_compensation = Real(0);
    for (std::size_t edge_id = 0; edge_id < boundary_chain.edge_count(); ++edge_id) {
        // Boundary rows come from gwn_build_boundary_chain.
        auto const start = static_cast<std::size_t>(boundary_chain.start_vertex[edge_id]);
        auto const end = static_cast<std::size_t>(boundary_chain.end_vertex[edge_id]);
        // These query-relative coordinates are bitwise identical to the ones the crossing term
        // derives from BVH triangle records, so both terms classify each edge identically.
        gwn_query_vec3<Real> const edge_start(
            geometry.vertex_x[start] - query.x, geometry.vertex_y[start] - query.y,
            geometry.vertex_z[start] - query.z
        );
        gwn_query_vec3<Real> const edge_end(
            geometry.vertex_x[end] - query.x, geometry.vertex_y[end] - query.y,
            geometry.vertex_z[end] - query.z
        );
        auto const edge_area =
            gwn_antipodal_boundary_edge_area_impl<ExactFallback>(edge_start, edge_end);
        if (edge_area.status != gwn_antipodal_status::k_done)
            return edge_area;
        // Kahan summation keeps boundary-heavy meshes stable in Real precision.
        Real const term = static_cast<Real>(boundary_chain.multiplicity[edge_id]) * edge_area.value;
        Real const compensated_term = term - area_compensation;
        Real const next_sum = area_sum + compensated_term;
        area_compensation = (next_sum - area_sum) - compensated_term;
        area_sum = next_sum;
    }

    return {area_sum / (Real(4) * pi), gwn_antipodal_status::k_done};
}

/// \brief Evaluate the Antipodal winding gradient from the boundary chain.
///
/// The crossing count is locally constant, so the gradient is the derivative of the boundary
/// spherical-area sum. Supp. Alg. 1 of Martens et al. differentiates each auxiliary triangle, whose
/// terms carry opposite poles on the vertical line through each boundary vertex that cancel only
/// in exact arithmetic. Summed over the closed boundary chain, those terms equal the Biot-Savart
/// field of the chain, evaluated here per edge without a ray axis or retries.
template <gwn_real_type Real, gwn_index_type Index>
[[nodiscard]] __device__ inline gwn_query_vec3<Real> gwn_winding_gradient_point_antipodal_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query
) noexcept {
    Real const pi = Real(3.141592653589793238462643383279502884);

    gwn_query_vec3<Real> gradient_sum(Real(0), Real(0), Real(0));
    gwn_query_vec3<Real> gradient_compensation(Real(0), Real(0), Real(0));
    for (std::size_t edge_id = 0; edge_id < boundary_chain.edge_count(); ++edge_id) {
        auto const start = static_cast<std::size_t>(boundary_chain.start_vertex[edge_id]);
        auto const end = static_cast<std::size_t>(boundary_chain.end_vertex[edge_id]);
        gwn_query_vec3<Real> const edge_start(
            geometry.vertex_x[start] - query.x, geometry.vertex_y[start] - query.y,
            geometry.vertex_z[start] - query.z
        );
        gwn_query_vec3<Real> const edge_end(
            geometry.vertex_x[end] - query.x, geometry.vertex_y[end] - query.y,
            geometry.vertex_z[end] - query.z
        );
        gwn_query_vec3<Real> const term =
            static_cast<Real>(boundary_chain.multiplicity[edge_id]) *
            gwn_gradient_solid_angle_edge_impl<true>(edge_start, edge_end);
        // Vector Kahan summation limits cancellation error on long oriented boundary chains.
        gwn_query_vec3<Real> const compensated_term = term - gradient_compensation;
        gwn_query_vec3<Real> const next_sum = gradient_sum + compensated_term;
        gradient_compensation = (next_sum - gradient_sum) - compensated_term;
        gradient_sum = next_sum;
    }

    return gradient_sum / (Real(4) * pi);
}

/// \brief Sum signed +z triangle crossings through the canonical BVH for one predicate mode.
template <
    bool ExactFallback, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback, bool ValidateAccessor = true>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real> gwn_signed_ray_crossing_count_impl(
    gwn_bvh_accessor<Width, Real, Index> const &bvh, gwn_query_vec3<Real> const &origin,
    OverflowCallback const &overflow_callback = {}
) noexcept {
    static_assert(Width >= 2, "BVH width requires at least 2.");
    static_assert(StackCapacity > 0, "Traversal stack capacity requires a positive value.");

    if constexpr (ValidateAccessor)
        if (!bvh.is_valid())
            return {std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_status::k_done};

    // An integer count stays exact past the float mantissa when crossings of both signs cancel.
    std::int64_t crossing_sum = 0;
    auto const intersects_positive_z_ray = [&](gwn_aabb<Real> const &bounds) noexcept {
        // The +z ray's two zero-direction slabs reduce to containment tests, while the z slab only
        // needs to extend beyond the origin. A culled child has every vertex strictly on one side
        // of the query in some coordinate, so the exact predicates could not report a crossing for
        // it either.
        return origin.x >= bounds.min_x && origin.x <= bounds.max_x && origin.y >= bounds.min_y &&
               origin.y <= bounds.max_y && bounds.max_z >= origin.z;
    };
    auto visit_leaf = [&](gwn_bvh_child<Real> const &leaf) noexcept {
        for (std::uint32_t primitive_offset = 0; primitive_offset < leaf.primitive_count();
             ++primitive_offset) {
            std::uint64_t const sorted_index = leaf.offset() + primitive_offset;
            if (sorted_index >= bvh.triangles.size())
                continue;

            auto const &triangle = bvh.triangles[static_cast<std::size_t>(sorted_index)];
            int crossing = 0;
            // An uncertified predicate discards the partial count; the caller reruns the query.
            if (gwn_ray_triangle_crossing_antipodal_impl<ExactFallback>(
                    origin, triangle, crossing
                ) != gwn_antipodal_status::k_done) {
                return gwn_antipodal_status::k_fallback_required;
            }
            crossing_sum += crossing;
        }
        return gwn_antipodal_status::k_done;
    };

    if (bvh.has_leaf_root()) {
        if (visit_leaf(bvh.root) != gwn_antipodal_status::k_done)
            return {
                std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_status::k_fallback_required
            };
        return {static_cast<Real>(crossing_sum), gwn_antipodal_status::k_done};
    }

    // Only internal node offsets survive across iterations, so the topology's Index type is the
    // complete pending state and keeps the default uint32_t stack compact.
    Index stack[StackCapacity];
    int stack_size = 0;
    auto node_index = static_cast<Index>(bvh.root.offset());
    while (true) {
        Index next_node = gwn_invalid_index<Index>();
        if (gwn_index_in_bounds(node_index, bvh.nodes.size())) {
            auto const &node = bvh.nodes[static_cast<std::size_t>(node_index)];
            GWN_DETAIL_PRAGMA_UNROLL
            for (int child_slot = 0; child_slot < Width; ++child_slot) {
                auto const &child = node.child(child_slot);
                if (!child.is_valid())
                    continue;
                if (!intersects_positive_z_ray(child.bounds))
                    continue;

                // Leaf work stays in the current node so only internal node offsets consume the
                // topology-bounded stack shared by every canonical traversal.
                if (child.is_leaf()) {
                    if (visit_leaf(child) != gwn_antipodal_status::k_done) {
                        return {
                            std::numeric_limits<Real>::quiet_NaN(),
                            gwn_antipodal_status::k_fallback_required
                        };
                    }
                    continue;
                }
                if (!child.is_internal())
                    continue;
                if (!gwn_is_invalid_index(next_node)) {
                    if (stack_size >= StackCapacity) {
                        overflow_callback();
                        return {
                            std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_status::k_done
                        };
                    }
                    // Keep the previously selected child pending so the final internal child can
                    // be entered directly without a matching stack store and load.
                    stack[stack_size++] = next_node;
                }
                next_node = static_cast<Index>(child.offset());
            }
        }

        if (!gwn_is_invalid_index(next_node)) {
            node_index = next_node;
            continue;
        }
        if (stack_size == 0)
            break;
        node_index = stack[--stack_size];
    }
    return {static_cast<Real>(crossing_sum), gwn_antipodal_status::k_done};
}

/// \brief Combine the Antipodal crossing and boundary terms for one predicate mode.
///
/// Without \p ExactFallback, an uncertified predicate in either term reports
/// \c k_fallback_required. A stack overflow invokes the callback and yields NaN.
template <
    bool ExactFallback, int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback>
[[nodiscard]] __device__ inline gwn_antipodal_value_result<Real>
gwn_winding_number_antipodal_evaluate_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query,
    OverflowCallback const &overflow_callback
) noexcept {
    // Every vertex lies in the BVH root bounds, so the largest root offset bounds every
    // query-relative coordinate. The fallback returns NaN for a non-finite offset or query, and an
    // open mesh sends queries outside the boundary range condition there before any traversal.
    if constexpr (!ExactFallback) {
        Real const offset = gwn_antipodal_bounds_offset_impl(bvh.root.bounds, query);
        bool const finite =
            !isnan(query.x + query.y + query.z) && offset <= std::numeric_limits<Real>::max();
        if (!finite || (boundary_chain.edge_count() != 0 &&
                        !gwn_antipodal_rounded_range_clear_impl(query, offset))) {
            return {
                std::numeric_limits<Real>::quiet_NaN(), gwn_antipodal_status::k_fallback_required
            };
        }
    }

    auto const crossings = gwn_signed_ray_crossing_count_impl<
        ExactFallback, Width, Real, Index, StackCapacity, OverflowCallback, false>(
        bvh, query, overflow_callback
    );
    // A NaN count reports an overflow that already invoked the callback. Returning before the
    // boundary term keeps such a query out of the fallback, so the callback fires once.
    if (crossings.status != gwn_antipodal_status::k_done || isnan(crossings.value))
        return crossings;

    auto const boundary =
        gwn_antipodal_boundary_contribution_impl<ExactFallback>(geometry, boundary_chain, query);
    if (boundary.status != gwn_antipodal_status::k_done)
        return boundary;
    return {crossings.value + boundary.value, gwn_antipodal_status::k_done};
}

/// \brief Evaluate the complete Antipodal winding number with exact predicate fallbacks.
///
/// Query-relative coordinates must be finite for the predicates to describe the geometry. Every
/// vertex lies in the BVH root bounds, so finite root offsets bound every query-relative
/// coordinate; a query outside that range returns NaN.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback>
[[nodiscard]] __device__ __noinline__ Real gwn_winding_number_antipodal_fallback_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, gwn_query_vec3<Real> const &query,
    OverflowCallback const &overflow_callback
) noexcept {
    Real const extent = gwn_antipodal_bounds_offset_impl(bvh.root.bounds, query);
    if (isnan(query.x + query.y + query.z) || !(extent <= std::numeric_limits<Real>::max()))
        return std::numeric_limits<Real>::quiet_NaN();
    return gwn_winding_number_antipodal_evaluate_impl<
               true, Width, Real, Index, StackCapacity, OverflowCallback>(
               geometry, bvh, boundary_chain, query, overflow_callback
    )
        .value;
}

/// \brief Evaluate the Antipodal winding number of one query.
///
/// The rounded pass handles almost every query. An uncertified predicate reruns the complete
/// query out of line with exact predicate fallbacks.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
[[nodiscard]] __device__ inline Real gwn_winding_number_antipodal_impl(
    gwn_geometry_accessor<Real, Index> const &geometry,
    gwn_bvh_accessor<Width, Real, Index> const &bvh,
    gwn_boundary_chain_accessor<Index> const &boundary_chain, Real const qx, Real const qy,
    Real const qz, OverflowCallback const &overflow_callback = {}
) noexcept {
    gwn_query_vec3<Real> const query(qx, qy, qz);
    auto const result = gwn_winding_number_antipodal_evaluate_impl<
        false, Width, Real, Index, StackCapacity, OverflowCallback>(
        geometry, bvh, boundary_chain, query, overflow_callback
    );
    if (result.status == gwn_antipodal_status::k_done)
        return result.value;
    return gwn_winding_number_antipodal_fallback_impl<Width, Real, Index, StackCapacity>(
        geometry, bvh, boundary_chain, query, overflow_callback
    );
}

/// \brief NaN payload that marks a batch output whose rounded pass requires the fallback.
///
/// The payload differs from `quiet_NaN()` and from the canonical NaN of GPU arithmetic, so the
/// fallback pass recomputes only marked outputs.
template <gwn_real_type Real> struct gwn_antipodal_fallback_marker;

template <> struct gwn_antipodal_fallback_marker<float> {
    static constexpr std::uint32_t k_bits = 0x7fc0a171u;
    [[nodiscard]] __device__ static float value() noexcept { return __uint_as_float(k_bits); }
    [[nodiscard]] __device__ static bool matches(float const output) noexcept {
        return __float_as_uint(output) == k_bits;
    }
};

template <> struct gwn_antipodal_fallback_marker<double> {
    static constexpr std::uint64_t k_bits = 0x7ff800000000a171ull;
    [[nodiscard]] __device__ static double value() noexcept {
        return __longlong_as_double(static_cast<long long>(k_bits));
    }
    [[nodiscard]] __device__ static bool matches(double const output) noexcept {
        return static_cast<std::uint64_t>(__double_as_longlong(output)) == k_bits;
    }
};

/// \brief Run the rounded Antipodal pass for one batch element.
///
/// The kernel contains no exact arithmetic, so its register allocation stays that of the rounded
/// traversal. Queries that need the fallback receive the marker for the fallback pass.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_winding_number_antipodal_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_winding{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        // Host validation covers all shared objects once.
        gwn_query_vec3<Real> const query(query_x[query_id], query_y[query_id], query_z[query_id]);
        auto const result = gwn_winding_number_antipodal_evaluate_impl<
            false, Width, Real, Index, StackCapacity, OverflowCallback>(
            geometry, bvh, boundary_chain, query, overflow_callback
        );
        out_winding[query_id] = result.status == gwn_antipodal_status::k_done
                                    ? result.value
                                    : gwn_antipodal_fallback_marker<Real>::value();
    }
};

/// \brief Rerun marked batch elements with exact predicate fallbacks.
///
/// Outputs that overflowed in the rounded pass hold `quiet_NaN()`, not the marker, so the
/// overflow callback fires at most once per query. A marked query may still reach its first
/// overflow here, because the rounded pass stops at the first uncertified predicate.
template <
    int Width, gwn_real_type Real, gwn_index_type Index, int StackCapacity,
    typename OverflowCallback = gwn_traversal_overflow_trap_callback>
struct gwn_winding_number_antipodal_fallback_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_bvh_accessor<Width, Real, Index> bvh{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_winding{};
    OverflowCallback overflow_callback{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        if (!gwn_antipodal_fallback_marker<Real>::matches(out_winding[query_id]))
            return;
        gwn_query_vec3<Real> const query(query_x[query_id], query_y[query_id], query_z[query_id]);
        out_winding[query_id] =
            gwn_winding_number_antipodal_fallback_impl<Width, Real, Index, StackCapacity>(
                geometry, bvh, boundary_chain, query, overflow_callback
            );
    }
};

/// \brief Invoke Antipodal boundary-gradient evaluation for one batch element.
template <gwn_real_type Real, gwn_index_type Index>
struct gwn_winding_gradient_antipodal_batch_functor {
    gwn_geometry_accessor<Real, Index> geometry{};
    gwn_boundary_chain_accessor<Index> boundary_chain{};
    cuda::std::span<Real const> query_x{};
    cuda::std::span<Real const> query_y{};
    cuda::std::span<Real const> query_z{};
    cuda::std::span<Real> out_gradient_x{};
    cuda::std::span<Real> out_gradient_y{};
    cuda::std::span<Real> out_gradient_z{};

    void __device__ operator()(std::size_t const query_id) const noexcept {
        gwn_query_vec3<Real> const query(query_x[query_id], query_y[query_id], query_z[query_id]);
        gwn_query_vec3<Real> const gradient =
            gwn_winding_gradient_point_antipodal_impl(geometry, boundary_chain, query);
        out_gradient_x[query_id] = gradient.x;
        out_gradient_y[query_id] = gradient.y;
        out_gradient_z[query_id] = gradient.z;
    }
};

} // namespace detail
} // namespace gwn

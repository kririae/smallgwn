#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn_geometry.cuh>

#include "test_fixtures.cuh"
#include "test_utils.cuh"

namespace {

using Real = gwn::tests::Real;
using gwn::tests::CudaFixture;
using GwnGeometryTest = gwn::tests::CudaFixture;
using GwnGeometryStreamTest = gwn::tests::CudaStreamFixture;

template <class Index> class GwnGeometryIndexTest : public CudaFixture {};
using GeometryIndexTypes = ::testing::Types<std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(GwnGeometryIndexTest, GeometryIndexTypes);

template <class T> [[nodiscard]] std::vector<T> copy_to_host(cuda::std::span<T> const input) {
    std::vector<T> output(input.size());
    gwn::detail::gwn_copy_d2h(
        cuda::std::span<T>(output), cuda::std::span<T const>(input.data(), input.size()),
        cudaStreamLegacy
    );
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(cudaStreamLegacy));
    return output;
}

template <gwn::gwn_index_type Index> void expect_upload_and_update_contract() {
    std::array<Real, 4> x{0, 1, 0, 0};
    std::array<Real, 4> y{0, 0, 1, 0};
    std::array<Real, 4> z{0, 0, 0, 1};
    std::array<Index, 2> i0{0, 0};
    std::array<Index, 2> i1{1, 2};
    std::array<Index, 2> i2{2, 3};

    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_status status = gwn::gwn_upload_geometry(
        geometry, gwn::tests::host_span(cuda::std::span<Real const>(x)),
        gwn::tests::host_span(cuda::std::span<Real const>(y)),
        gwn::tests::host_span(cuda::std::span<Real const>(z)),
        gwn::tests::host_span(cuda::std::span<Index const>(i0)),
        gwn::tests::host_span(cuda::std::span<Index const>(i1)),
        gwn::tests::host_span(cuda::std::span<Index const>(i2))
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    ASSERT_TRUE(geometry.accessor().is_valid());
    EXPECT_TRUE(geometry.has_data());
    EXPECT_EQ(geometry.vertex_count(), x.size());
    EXPECT_EQ(geometry.triangle_count(), i0.size());
    EXPECT_EQ(copy_to_host(geometry.accessor().tri_i0), (std::vector<Index>{0, 0}));
    EXPECT_EQ(copy_to_host(geometry.accessor().tri_i1), (std::vector<Index>{1, 2}));
    EXPECT_EQ(copy_to_host(geometry.accessor().tri_i2), (std::vector<Index>{2, 3}));

    x = {4, 5, 4, 4};
    y = {6, 6, 7, 6};
    z = {8, 8, 8, 9};
    status = gwn::gwn_update_geometry(
        geometry, gwn::tests::host_span(cuda::std::span<Real const>(x)),
        gwn::tests::host_span(cuda::std::span<Real const>(y)),
        gwn::tests::host_span(cuda::std::span<Real const>(z))
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_EQ(copy_to_host(geometry.accessor().vertex_x), (std::vector<Real>{4, 5, 4, 4}));
    EXPECT_EQ(copy_to_host(geometry.accessor().vertex_y), (std::vector<Real>{6, 6, 7, 6}));
    EXPECT_EQ(copy_to_host(geometry.accessor().vertex_z), (std::vector<Real>{8, 8, 8, 9}));
    // Position updates must not rewrite topology.
    EXPECT_EQ(copy_to_host(geometry.accessor().tri_i1), (std::vector<Index>{1, 2}));
}

template <gwn::gwn_index_type Index>
__global__ void read_geometry_after_wait(
    gwn::gwn_geometry_accessor<Real, Index> const geometry, Real *const output
) {
    output[0] = geometry.vertex_x[0];
    output[1] = geometry.vertex_y[0];
    output[2] = geometry.vertex_z[0];
}

TEST(gwn_geometry, default_object_is_empty) {
    gwn::gwn_geometry_object<Real> geometry;
    EXPECT_FALSE(geometry.has_data());
    EXPECT_EQ(geometry.vertex_count(), 0u);
    EXPECT_EQ(geometry.triangle_count(), 0u);
    EXPECT_TRUE(geometry.accessor().is_valid());
}

TYPED_TEST(GwnGeometryIndexTest, upload_and_position_update_preserve_geometry_contract) {
    expect_upload_and_update_contract<TypeParam>();
}

TEST_F(GwnGeometryTest, failed_upload_preserves_previous_geometry) {
    using Index = std::uint32_t;
    std::array<Real, 3> const x{0, 1, 0};
    std::array<Real, 3> const y{0, 0, 1};
    std::array<Real, 3> const z{0, 0, 0};
    std::array<Index, 1> const i0{0};
    std::array<Index, 1> const i1{1};
    std::array<Index, 1> const i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, gwn::tests::host_span(cuda::std::span<Real const>(x)),
            gwn::tests::host_span(cuda::std::span<Real const>(y)),
            gwn::tests::host_span(cuda::std::span<Real const>(z)),
            gwn::tests::host_span(cuda::std::span<Index const>(i0)),
            gwn::tests::host_span(cuda::std::span<Index const>(i1)),
            gwn::tests::host_span(cuda::std::span<Index const>(i2))
        )
            .is_ok()
    );

    std::array<Index, 1> const invalid_i2{3};
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, gwn::tests::host_span(cuda::std::span<Real const>(x)),
        gwn::tests::host_span(cuda::std::span<Real const>(y)),
        gwn::tests::host_span(cuda::std::span<Real const>(z)),
        gwn::tests::host_span(cuda::std::span<Index const>(i0)),
        gwn::tests::host_span(cuda::std::span<Index const>(i1)),
        gwn::tests::host_span(cuda::std::span<Index const>(invalid_i2))
    );
    EXPECT_EQ(status.error(), gwn::gwn_error::invalid_argument);
    EXPECT_TRUE(geometry.accessor().is_valid());
    EXPECT_EQ(geometry.vertex_count(), 3u);
    EXPECT_EQ(copy_to_host(geometry.accessor().tri_i2), (std::vector<Index>{2}));
}

TEST_F(GwnGeometryTest, upload_and_update_reject_mismatched_spans) {
    using Index = std::uint32_t;
    std::array<Real, 3> const xyz{0, 1, 0};
    std::array<Real, 2> const short_xyz{0, 1};
    std::array<Index, 1> const index{0};
    cuda::std::span<Real const> const vertices(xyz);
    cuda::std::span<Index const> const triangles(index);

    gwn::gwn_geometry_object<Real, Index> geometry;
    EXPECT_EQ(
        gwn::gwn_upload_geometry(
            geometry, gwn::tests::host_span(vertices),
            gwn::tests::host_span(cuda::std::span<Real const>(short_xyz)),
            gwn::tests::host_span(vertices), gwn::tests::host_span(triangles),
            gwn::tests::host_span(triangles), gwn::tests::host_span(triangles)
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );

    std::array<Index, 1> const i0{0};
    std::array<Index, 1> const i1{1};
    std::array<Index, 1> const i2{2};
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, gwn::tests::host_span(vertices), gwn::tests::host_span(vertices),
            gwn::tests::host_span(vertices),
            gwn::tests::host_span(cuda::std::span<Index const>(i0)),
            gwn::tests::host_span(cuda::std::span<Index const>(i1)),
            gwn::tests::host_span(cuda::std::span<Index const>(i2))
        )
            .is_ok()
    );
    EXPECT_EQ(
        gwn::gwn_update_geometry(
            geometry, gwn::tests::host_span(cuda::std::span<Real const>(short_xyz)),
            gwn::tests::host_span(cuda::std::span<Real const>(short_xyz)),
            gwn::tests::host_span(cuda::std::span<Real const>(short_xyz))
        )
            .error(),
        gwn::gwn_error::invalid_argument
    );
    EXPECT_EQ(geometry.vertex_count(), 3u);
}

TEST_F(GwnGeometryTest, move_transfers_storage_and_clear_releases_it) {
    gwn::tests::SingleTriangleMesh mesh;
    gwn::gwn_geometry_object<Real> source;
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            source, gwn::tests::host_span(cuda::std::span<Real const>(mesh.vx)),
            gwn::tests::host_span(cuda::std::span<Real const>(mesh.vy)),
            gwn::tests::host_span(cuda::std::span<Real const>(mesh.vz)),
            gwn::tests::host_span(cuda::std::span<gwn::tests::Index const>(mesh.i0)),
            gwn::tests::host_span(cuda::std::span<gwn::tests::Index const>(mesh.i1)),
            gwn::tests::host_span(cuda::std::span<gwn::tests::Index const>(mesh.i2))
        )
            .is_ok()
    );

    gwn::gwn_geometry_object<Real> destination(std::move(source));
    EXPECT_FALSE(source.has_data());
    EXPECT_TRUE(destination.has_data());
    destination.clear();
    EXPECT_FALSE(destination.has_data());
    EXPECT_TRUE(destination.accessor().is_valid());
}

TEST_F(GwnGeometryStreamTest, replacement_releases_old_storage_on_its_bound_stream) {
    using Index = std::uint32_t;
    std::array<Real, 3> const x{Real(1), Real(2), Real(3)};
    std::array<Real, 3> const y{Real(4), Real(5), Real(6)};
    std::array<Real, 3> const z{Real(7), Real(8), Real(9)};
    std::array<Index, 1> const i0{0};
    std::array<Index, 1> const i1{1};
    std::array<Index, 1> const i2{2};

    gwn::gwn_geometry_object<Real, Index> geometry;
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, gwn::tests::host_span(cuda::std::span<Real const>(x)),
            gwn::tests::host_span(cuda::std::span<Real const>(y)),
            gwn::tests::host_span(cuda::std::span<Real const>(z)),
            gwn::tests::host_span(cuda::std::span<Index const>(i0)),
            gwn::tests::host_span(cuda::std::span<Index const>(i1)),
            gwn::tests::host_span(cuda::std::span<Index const>(i2)), stream_a_
        )
            .is_ok()
    );

    gwn::detail::gwn_device_array<Real> output(stream_a_);
    output.resize(3, stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

    cudaEvent_t release_event = nullptr;
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&release_event, cudaEventDisableTiming));
    auto const release_event_cleanup =
        gwn::gwn_make_scope_exit([&]() noexcept { (void)cudaEventDestroy(release_event); });
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(stream_a_, release_event));
    auto const old_accessor = geometry.accessor();
    read_geometry_after_wait<<<1, 1, 0, stream_a_>>>(old_accessor, output.data());
    cudaError_t const launch_status = cudaGetLastError();

    std::array<Real, 3> const replacement_x{Real(10), Real(11), Real(12)};
    gwn::gwn_status const replacement_status = gwn::gwn_upload_geometry(
        geometry, gwn::tests::host_span(cuda::std::span<Real const>(replacement_x)),
        gwn::tests::host_span(cuda::std::span<Real const>(y)),
        gwn::tests::host_span(cuda::std::span<Real const>(z)),
        gwn::tests::host_span(cuda::std::span<Index const>(i0)),
        gwn::tests::host_span(cuda::std::span<Index const>(i1)),
        gwn::tests::host_span(cuda::std::span<Index const>(i2)), stream_b_
    );

    std::array<Real, 3> actual{};
    cudaError_t const release_status = cudaEventRecord(release_event, stream_b_);
    output.copy_to_host(cuda::std::span<Real>(actual), stream_a_);
    ASSERT_EQ(cudaSuccess, launch_status);
    ASSERT_TRUE(replacement_status.is_ok());
    ASSERT_EQ(cudaSuccess, release_status);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    EXPECT_EQ(actual, (std::array<Real, 3>{x[0], y[0], z[0]}));
    EXPECT_EQ(geometry.stream(), stream_b_);
}

} // namespace

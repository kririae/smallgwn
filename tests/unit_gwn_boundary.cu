#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <gwn/detail/gwn_device_array.cuh>
#include <gwn/gwn.cuh>

#include "test_fixtures.cuh"
#include "test_meshes.hpp"
#include "test_utils.cuh"

using Real = gwn::tests::Real;
using Index = gwn::tests::Index;
using gwn::tests::CubeMesh;
using GwnBoundaryTest = gwn::tests::CudaFixture;
using GwnBoundaryStreamTest = gwn::tests::CudaStreamFixture;
using gwn::tests::OpenCubeMesh;

namespace {

struct BoundaryRow {
    Index start{};
    Index end{};
    std::uint64_t multiplicity{};
};

std::vector<BoundaryRow> copy_boundary_rows(gwn::gwn_boundary_chain_object<Index> const &boundary) {
    auto const &accessor = boundary.accessor();
    std::vector<Index> start(accessor.edge_count());
    std::vector<Index> end(accessor.edge_count());
    std::vector<std::uint64_t> multiplicity(accessor.edge_count());

    EXPECT_EQ(
        cudaSuccess, cudaMemcpyAsync(
                         start.data(), accessor.start_vertex.data(), start.size() * sizeof(Index),
                         cudaMemcpyDeviceToHost, boundary.stream()
                     )
    );
    EXPECT_EQ(
        cudaSuccess, cudaMemcpyAsync(
                         end.data(), accessor.end_vertex.data(), end.size() * sizeof(Index),
                         cudaMemcpyDeviceToHost, boundary.stream()
                     )
    );
    EXPECT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            multiplicity.data(), accessor.multiplicity.data(),
            multiplicity.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, boundary.stream()
        )
    );
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(boundary.stream()));

    std::vector<BoundaryRow> rows;
    for (std::size_t i = 0; i < start.size(); ++i)
        rows.push_back(BoundaryRow{start[i], end[i], multiplicity[i]});
    std::sort(rows.begin(), rows.end(), [](BoundaryRow const &a, BoundaryRow const &b) {
        if (a.start != b.start)
            return a.start < b.start;
        if (a.end != b.end)
            return a.end < b.end;
        return a.multiplicity < b.multiplicity;
    });
    return rows;
}

template <class Mesh>
void upload_mesh(Mesh const &mesh, gwn::gwn_geometry_object<Real, Index> &geometry) {
    gwn::gwn_status const status = gwn::gwn_upload_geometry(
        geometry, cuda::std::span<Real const>(mesh.vx.data(), mesh.vx.size()),
        cuda::std::span<Real const>(mesh.vy.data(), mesh.vy.size()),
        cuda::std::span<Real const>(mesh.vz.data(), mesh.vz.size()),
        cuda::std::span<Index const>(mesh.i0.data(), mesh.i0.size()),
        cuda::std::span<Index const>(mesh.i1.data(), mesh.i1.size()),
        cuda::std::span<Index const>(mesh.i2.data(), mesh.i2.size())
    );
    SMALLGWN_SKIP_IF_STATUS_CUDA_UNAVAILABLE(status);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
}

__global__ void read_boundary_after_wait(
    gwn::gwn_boundary_chain_accessor<Index> const boundary, Index *const output
) {
    output[0] = boundary.start_vertex[0];
    output[1] = boundary.end_vertex[0];
}

} // namespace

TEST_F(GwnBoundaryTest, boundary_chain_empty_geometry_is_built_empty) {
    gwn::gwn_geometry_object<Real, Index> geometry;
    gwn::gwn_boundary_chain_object<Index> boundary;

    gwn::gwn_status const status = gwn::gwn_build_boundary_chain(geometry, boundary);

    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);
    EXPECT_TRUE(boundary.has_data());
    EXPECT_TRUE(boundary.accessor().is_valid());
    EXPECT_TRUE(boundary.accessor().empty());
    EXPECT_EQ(boundary.accessor().mesh_vertex_count, 0u);
    EXPECT_EQ(boundary.accessor().mesh_triangle_count, 0u);
}

TEST_F(GwnBoundaryTest, boundary_chain_closed_mesh_is_built_empty) {
    CubeMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    upload_mesh(mesh, geometry);

    gwn::gwn_boundary_chain_object<Index> boundary;
    gwn::gwn_status const status = gwn::gwn_build_boundary_chain(geometry, boundary);
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    EXPECT_TRUE(boundary.has_data());
    EXPECT_TRUE(boundary.accessor().is_valid());
    EXPECT_TRUE(boundary.accessor().empty());
    EXPECT_EQ(boundary.accessor().edge_count(), 0u);
    EXPECT_EQ(boundary.accessor().mesh_vertex_count, mesh.Nv);
    EXPECT_EQ(boundary.accessor().mesh_triangle_count, mesh.Nt);
}

TEST_F(GwnBoundaryTest, boundary_chain_open_cube_has_four_unit_edges) {
    OpenCubeMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    upload_mesh(mesh, geometry);

    gwn::gwn_boundary_chain_object<Index> boundary;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(geometry, boundary).is_ok());

    std::vector<BoundaryRow> const rows = copy_boundary_rows(boundary);
    ASSERT_EQ(rows.size(), 4u);
    for (BoundaryRow const &row : rows)
        EXPECT_EQ(row.multiplicity, 1u);

    std::vector<BoundaryRow> const expected{
        BoundaryRow{4, 7, 1},
        BoundaryRow{5, 4, 1},
        BoundaryRow{6, 5, 1},
        BoundaryRow{7, 6, 1},
    };
    EXPECT_EQ(rows.size(), expected.size());
    for (std::size_t i = 0; i < rows.size() && i < expected.size(); ++i) {
        EXPECT_EQ(rows[i].start, expected[i].start) << "row " << i;
        EXPECT_EQ(rows[i].end, expected[i].end) << "row " << i;
        EXPECT_EQ(rows[i].multiplicity, expected[i].multiplicity) << "row " << i;
    }
}

TEST_F(GwnBoundaryTest, boundary_chain_uses_net_orientation) {
    std::array<Index, 4> const i0{0, 1, 0, 0};
    std::array<Index, 4> const i1{1, 0, 1, 1};
    std::array<Index, 4> const i2{2, 3, 2, 2};

    gwn::gwn_boundary_chain_object<Index> boundary;
    gwn::gwn_status const status = gwn::gwn_build_boundary_chain(
        4, cuda::std::span<Index const>(i0.data(), i0.size()),
        cuda::std::span<Index const>(i1.data(), i1.size()),
        cuda::std::span<Index const>(i2.data(), i2.size()), boundary
    );
    ASSERT_TRUE(status.is_ok()) << gwn::tests::status_to_debug_string(status);

    std::vector<BoundaryRow> const rows = copy_boundary_rows(boundary);
    auto const it = std::find_if(rows.begin(), rows.end(), [](BoundaryRow const &row) {
        return row.start == 0 && row.end == 1;
    });
    ASSERT_NE(it, rows.end());
    EXPECT_EQ(it->multiplicity, 2u);

    auto const cancelled = std::find_if(rows.begin(), rows.end(), [](BoundaryRow const &row) {
        return (row.start == 1 && row.end == 0);
    });
    EXPECT_EQ(cancelled, rows.end());
}

TEST_F(GwnBoundaryTest, boundary_chain_rejects_bad_inputs) {
    std::array<Index, 1> const i0{0};
    std::array<Index, 2> const i1{1, 1};
    std::array<Index, 1> const i2{2};

    gwn::gwn_boundary_chain_object<Index> boundary;
    EXPECT_FALSE(
        gwn::gwn_build_boundary_chain(
            3, cuda::std::span<Index const>(i0.data(), i0.size()),
            cuda::std::span<Index const>(i1.data(), i1.size()),
            cuda::std::span<Index const>(i2.data(), i2.size()), boundary
        )
            .is_ok()
    );

    std::array<Index, 1> const bad_i0{0};
    std::array<Index, 1> const bad_i1{1};
    std::array<Index, 1> const bad_i2{4};
    EXPECT_FALSE(
        gwn::gwn_build_boundary_chain(
            3, cuda::std::span<Index const>(bad_i0.data(), bad_i0.size()),
            cuda::std::span<Index const>(bad_i1.data(), bad_i1.size()),
            cuda::std::span<Index const>(bad_i2.data(), bad_i2.size()), boundary
        )
            .is_ok()
    );
}

TEST_F(GwnBoundaryTest, boundary_chain_failed_rebuild_preserves_previous_chain) {
    OpenCubeMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    upload_mesh(mesh, geometry);

    gwn::gwn_boundary_chain_object<Index> boundary;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(geometry, boundary).is_ok());
    std::vector<BoundaryRow> const before = copy_boundary_rows(boundary);
    ASSERT_EQ(before.size(), 4u);

    std::array<Index, 1> const bad_i0{0};
    std::array<Index, 1> const bad_i1{1};
    std::array<Index, 1> const bad_i2{static_cast<Index>(mesh.Nv)};
    EXPECT_FALSE(
        gwn::gwn_build_boundary_chain(
            mesh.Nv, cuda::std::span<Index const>(bad_i0.data(), bad_i0.size()),
            cuda::std::span<Index const>(bad_i1.data(), bad_i1.size()),
            cuda::std::span<Index const>(bad_i2.data(), bad_i2.size()), boundary
        )
            .is_ok()
    );

    EXPECT_TRUE(boundary.has_data());
    EXPECT_EQ(copy_boundary_rows(boundary).size(), before.size());
}

TEST_F(GwnBoundaryTest, boundary_chain_move_transfers_accessors) {
    OpenCubeMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    upload_mesh(mesh, geometry);

    gwn::gwn_boundary_chain_object<Index> source;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(geometry, source).is_ok());

    gwn::gwn_boundary_chain_object<Index> moved(std::move(source));
    EXPECT_TRUE(moved.has_data());
    EXPECT_EQ(moved.accessor().edge_count(), 4u);

    gwn::gwn_boundary_chain_object<Index> assigned;
    assigned = std::move(moved);
    EXPECT_TRUE(assigned.has_data());
    EXPECT_EQ(assigned.accessor().edge_count(), 4u);
}

TEST_F(GwnBoundaryTest, default_boundary_chain_is_not_built) {
    gwn::gwn_boundary_chain_object<Index> boundary;
    EXPECT_FALSE(boundary.has_data());
    EXPECT_FALSE(boundary.accessor().is_valid());
}

TEST_F(GwnBoundaryStreamTest, replacement_releases_old_storage_on_its_bound_stream) {
    OpenCubeMesh mesh;
    gwn::gwn_geometry_object<Real, Index> geometry;
    ASSERT_TRUE(
        gwn::gwn_upload_geometry(
            geometry, cuda::std::span<Real const>(mesh.vx), cuda::std::span<Real const>(mesh.vy),
            cuda::std::span<Real const>(mesh.vz), cuda::std::span<Index const>(mesh.i0),
            cuda::std::span<Index const>(mesh.i1), cuda::std::span<Index const>(mesh.i2), stream_a_
        )
            .is_ok()
    );
    gwn::gwn_boundary_chain_object<Index> boundary;
    ASSERT_TRUE(gwn::gwn_build_boundary_chain(geometry, boundary, stream_a_).is_ok());
    ASSERT_FALSE(boundary.accessor().empty());

    std::array<Index, 2> expected{};
    ASSERT_EQ(
        cudaSuccess, cudaMemcpyAsync(
                         expected.data(), boundary.accessor().start_vertex.data(), sizeof(Index),
                         cudaMemcpyDeviceToHost, stream_a_
                     )
    );
    ASSERT_EQ(
        cudaSuccess, cudaMemcpyAsync(
                         expected.data() + 1, boundary.accessor().end_vertex.data(), sizeof(Index),
                         cudaMemcpyDeviceToHost, stream_a_
                     )
    );

    gwn::detail::gwn_device_array<Index> output(stream_a_);
    output.resize(2, stream_a_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));

    cudaEvent_t release_event = nullptr;
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&release_event, cudaEventDisableTiming));
    auto const release_event_cleanup =
        gwn::gwn_make_scope_exit([&]() noexcept { (void)cudaEventDestroy(release_event); });
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(stream_a_, release_event));
    auto const old_accessor = boundary.accessor();
    read_boundary_after_wait<<<1, 1, 0, stream_a_>>>(old_accessor, output.data());
    cudaError_t const launch_status = cudaGetLastError();
    gwn::gwn_status const replacement_status =
        gwn::gwn_build_boundary_chain(geometry, boundary, stream_b_);

    std::array<Index, 2> actual{};
    cudaError_t const release_status = cudaEventRecord(release_event, stream_b_);
    output.copy_to_host(cuda::std::span<Index>(actual), stream_a_);
    ASSERT_EQ(cudaSuccess, launch_status);
    ASSERT_TRUE(replacement_status.is_ok());
    ASSERT_EQ(cudaSuccess, release_status);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_a_));
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(boundary.stream(), stream_b_);
}

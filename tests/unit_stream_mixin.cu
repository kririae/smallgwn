#include <cstdint>

#include <gtest/gtest.h>

#include <gwn/gwn.cuh>

#include "test_fixtures.hpp"

// ---------------------------------------------------------------------------
// Stream binding tests for owning objects (geometry, bvh topology, bvh data).
// Migrated from parity_scaffold and expanded.
// ---------------------------------------------------------------------------

using gwn::tests::CudaStreamFixture;

// ---------------------------------------------------------------------------
// gwn_stream_mixin basic behavior.
// ---------------------------------------------------------------------------

TEST(smallgwn_unit_stream_mixin, default_stream_is_legacy) {
    gwn::gwn_stream_mixin mixin;
    EXPECT_EQ(mixin.stream(), gwn::gwn_default_stream());
}

TEST(smallgwn_unit_stream_mixin, set_stream_changes_value) {
    gwn::gwn_stream_mixin mixin;
    // Use a sentinel value cast — we don't need a real stream for this test.
    cudaStream_t const fake = reinterpret_cast<cudaStream_t>(uintptr_t(0xDEAD));
    mixin.set_stream(fake);
    EXPECT_EQ(mixin.stream(), fake);
}

TEST(smallgwn_unit_stream_mixin, swap_exchanges_streams) {
    gwn::gwn_stream_mixin a;
    gwn::gwn_stream_mixin b;
    cudaStream_t const fake_a = reinterpret_cast<cudaStream_t>(uintptr_t(0xAAAA));
    cudaStream_t const fake_b = reinterpret_cast<cudaStream_t>(uintptr_t(0xBBBB));
    a.set_stream(fake_a);
    b.set_stream(fake_b);
    swap(a, b);
    EXPECT_EQ(a.stream(), fake_b);
    EXPECT_EQ(b.stream(), fake_a);
}

// ---------------------------------------------------------------------------
// gwn_geometry_object stream binding.
// ---------------------------------------------------------------------------

TEST_F(CudaStreamFixture, geometry_object_default_stream) {
    gwn::gwn_geometry_object<float, std::uint32_t> geometry;
    EXPECT_EQ(geometry.stream(), gwn::gwn_default_stream());
}

TEST_F(CudaStreamFixture, geometry_object_set_stream) {
    gwn::gwn_geometry_object<float, std::uint32_t> geometry;
    geometry.set_stream(stream_a_);
    EXPECT_EQ(geometry.stream(), stream_a_);
}

TEST_F(CudaStreamFixture, geometry_object_clear_rebinds_stream) {
    gwn::gwn_geometry_object<float, std::uint32_t> geometry;
    geometry.set_stream(stream_a_);
    EXPECT_EQ(geometry.stream(), stream_a_);

    geometry.clear(stream_b_);
    EXPECT_EQ(geometry.stream(), stream_b_);
}

// ---------------------------------------------------------------------------
// gwn_bvh_object (topology) stream binding.
// ---------------------------------------------------------------------------

TEST_F(CudaStreamFixture, bvh_object_default_stream) {
    gwn::gwn_bvh_object<float, std::uint32_t> bvh;
    EXPECT_EQ(bvh.stream(), gwn::gwn_default_stream());
}

TEST_F(CudaStreamFixture, bvh_object_set_stream) {
    gwn::gwn_bvh_object<float, std::uint32_t> bvh;
    bvh.set_stream(stream_a_);
    EXPECT_EQ(bvh.stream(), stream_a_);
}

TEST_F(CudaStreamFixture, bvh_object_clear_rebinds_stream) {
    gwn::gwn_bvh_object<float, std::uint32_t> bvh;
    bvh.set_stream(stream_a_);
    bvh.clear(stream_b_);
    EXPECT_EQ(bvh.stream(), stream_b_);
}

// ---------------------------------------------------------------------------
// gwn_bvh_moment_object stream binding.
// ---------------------------------------------------------------------------

TEST_F(CudaStreamFixture, data_object_default_stream) {
    gwn::gwn_bvh_moment_object<float, std::uint32_t> data;
    EXPECT_EQ(data.stream(), gwn::gwn_default_stream());
}

TEST_F(CudaStreamFixture, data_object_set_stream) {
    gwn::gwn_bvh_moment_object<float, std::uint32_t> data;
    data.set_stream(stream_a_);
    EXPECT_EQ(data.stream(), stream_a_);
}

TEST_F(CudaStreamFixture, data_object_clear_rebinds_stream) {
    gwn::gwn_bvh_moment_object<float, std::uint32_t> data;
    data.set_stream(stream_a_);
    data.clear(stream_b_);
    EXPECT_EQ(data.stream(), stream_b_);
}

// ---------------------------------------------------------------------------
// Move semantics for owning objects.
// ---------------------------------------------------------------------------

TEST_F(CudaStreamFixture, geometry_move_construct) {
    gwn::gwn_geometry_object<float, std::uint32_t> src;
    src.set_stream(stream_a_);
    gwn::gwn_geometry_object<float, std::uint32_t> dst(std::move(src));
    EXPECT_EQ(dst.stream(), stream_a_);
}

TEST_F(CudaStreamFixture, bvh_move_construct) {
    gwn::gwn_bvh_object<float, std::uint32_t> src;
    src.set_stream(stream_a_);
    gwn::gwn_bvh_object<float, std::uint32_t> dst(std::move(src));
    EXPECT_EQ(dst.stream(), stream_a_);
}

TEST_F(CudaStreamFixture, data_move_construct) {
    gwn::gwn_bvh_moment_object<float, std::uint32_t> src;
    src.set_stream(stream_a_);
    gwn::gwn_bvh_moment_object<float, std::uint32_t> dst(std::move(src));
    EXPECT_EQ(dst.stream(), stream_a_);
}

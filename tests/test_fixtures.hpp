#pragma once

#include <gtest/gtest.h>

#include <gwn/gwn_utils.cuh>

#include "test_utils.hpp"

namespace gwn::tests {

// ---------------------------------------------------------------------------
// CudaStreamFixture — provides two non-blocking streams and auto-cleanup.
// Skips tests automatically when CUDA is unavailable.
// ---------------------------------------------------------------------------

class CudaStreamFixture : public ::testing::Test {
protected:
    cudaStream_t stream_a_{nullptr};
    cudaStream_t stream_b_{nullptr};

    void SetUp() override {
        cudaError_t result = cudaStreamCreateWithFlags(&stream_a_, cudaStreamNonBlocking);
        if (is_cuda_runtime_unavailable(result))
            GTEST_SKIP() << "CUDA runtime unavailable: " << cudaGetErrorString(result);
        ASSERT_EQ(cudaSuccess, result);

        result = cudaStreamCreateWithFlags(&stream_b_, cudaStreamNonBlocking);
        ASSERT_EQ(cudaSuccess, result);
    }

    void TearDown() override {
        if (stream_a_ != nullptr) {
            (void)cudaStreamSynchronize(stream_a_);
            (void)cudaStreamDestroy(stream_a_);
            stream_a_ = nullptr;
        }
        if (stream_b_ != nullptr) {
            (void)cudaStreamSynchronize(stream_b_);
            (void)cudaStreamDestroy(stream_b_);
            stream_b_ = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// CudaFixture — provides a single default stream + CUDA skip guard.
// ---------------------------------------------------------------------------

class CudaFixture : public ::testing::Test {
protected:
    void SetUp() override { SMALLGWN_SKIP_IF_NO_CUDA(); }
};

} // namespace gwn::tests

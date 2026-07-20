cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(
    smallgwn
    VERSION 0.1.0
    LANGUAGES CXX CUDA)

# ----------------------------------------------------------
# User options
# ----------------------------------------------------------
option(SMALLGWN_BUILD_EIGEN_BRIDGE "Create the optional gwn::smallgwn_eigen_bridge interface target"
       ${PROJECT_IS_TOP_LEVEL})
option(SMALLGWN_BUILD_TESTS "Build the smallgwn correctness tests" ${PROJECT_IS_TOP_LEVEL})
option(SMALLGWN_BUILD_BENCHMARKS "Build the smallgwn benchmarks" ${PROJECT_IS_TOP_LEVEL})

# ----------------------------------------------------------
# Dependencies
# ----------------------------------------------------------
include(FetchContent)

if(NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit REQUIRED)
endif()

if((SMALLGWN_BUILD_EIGEN_BRIDGE
    OR SMALLGWN_BUILD_TESTS
    OR SMALLGWN_BUILD_BENCHMARKS)
   AND NOT TARGET Eigen3::Eigen)
    set(EIGEN_BUILD_DOC OFF)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_TESTING OFF)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Eigen)
endif()

if((SMALLGWN_BUILD_EIGEN_BRIDGE OR SMALLGWN_BUILD_TESTS) AND NOT TARGET TBB::tbb)
    set(TBB_EXAMPLES OFF)
    set(TBB_STRICT OFF)
    set(TBB_TEST OFF)
    FetchContent_Declare(
        oneTBB
        GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
        GIT_TAG v2022.2.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(oneTBB)
endif()

if(SMALLGWN_BUILD_TESTS AND NOT TARGET GTest::gtest)
    set(INSTALL_GTEST OFF)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
endif()

if((SMALLGWN_BUILD_TESTS OR SMALLGWN_BUILD_BENCHMARKS) AND NOT TARGET igl::core)
    set(LIBIGL_BUILD_TESTS OFF)
    set(LIBIGL_BUILD_TUTORIALS OFF)
    set(LIBIGL_INSTALL OFF)
    FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG v2.6.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(libigl)
endif()

if(SMALLGWN_BUILD_BENCHMARKS AND NOT TARGET CLI11::CLI11)
    set(CLI11_BUILD_EXAMPLES OFF)
    set(CLI11_BUILD_TESTS OFF)
    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.5.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(CLI11)
endif()

if(SMALLGWN_BUILD_BENCHMARKS AND NOT TARGET cuBQL_cuda_float3_static)
    FetchContent_Declare(
        cubql
        GIT_REPOSITORY https://github.com/NVIDIA/cuBQL.git
        GIT_TAG d7e61eb37f1188af87332bf317d5b76b1258ffd8)
    FetchContent_MakeAvailable(cubql)
endif()

# ----------------------------------------------------------
# Library targets
# ----------------------------------------------------------
add_library(smallgwn INTERFACE)
add_library(gwn::smallgwn ALIAS smallgwn)
target_include_directories(smallgwn INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                              $<INSTALL_INTERFACE:include>)
target_compile_features(smallgwn INTERFACE cxx_std_20 cuda_std_20)
target_link_libraries(smallgwn INTERFACE CUDA::cudart)
target_compile_options(smallgwn INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
if(MSVC)
    target_compile_options(smallgwn INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

if(SMALLGWN_BUILD_EIGEN_BRIDGE)
    add_library(smallgwn_eigen_bridge INTERFACE)
    add_library(gwn::smallgwn_eigen_bridge ALIAS smallgwn_eigen_bridge)
    target_link_libraries(smallgwn_eigen_bridge INTERFACE gwn::smallgwn Eigen3::Eigen TBB::tbb)
endif()

# ----------------------------------------------------------
# Tests and benchmarks
# ----------------------------------------------------------
if(SMALLGWN_BUILD_TESTS)
    enable_testing()
endif()

if(SMALLGWN_BUILD_TESTS OR SMALLGWN_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()

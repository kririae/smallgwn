cmake_minimum_required(VERSION 3.24)

project(smallgwn LANGUAGES CXX CUDA)

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PCH_WARN_INVALID ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

add_library(smallgwn INTERFACE)
add_library(gwn::smallgwn ALIAS smallgwn)

target_include_directories(smallgwn
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(smallgwn INTERFACE cxx_std_20)
target_link_libraries(smallgwn INTERFACE CUDA::cudart Eigen3::Eigen TBB::tbb)
target_compile_options(smallgwn INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

option(SMALLGWN_BUILD_TESTS "Build smallgwn tests" ON)
if(SMALLGWN_BUILD_TESTS)
    include(CTest)
    enable_testing()

    find_package(GTest QUIET)
    if(GTest_FOUND)
        # ---------------------------------------------------------------
        # Common include directories for all test targets.
        # ---------------------------------------------------------------
        set(SMALLGWN_TEST_INCLUDE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_hdk"
        )

        # ---------------------------------------------------------------
        # Helper: define a unit/integration test target.
        # ---------------------------------------------------------------
        function(smallgwn_add_test TARGET_NAME)
            cmake_parse_arguments(ARG "" "" "SOURCES;EXTRA_SOURCES" ${ARGN})
            add_executable(${TARGET_NAME} ${ARG_SOURCES} ${ARG_EXTRA_SOURCES})
            target_link_libraries(${TARGET_NAME} PRIVATE gwn::smallgwn GTest::gtest GTest::gtest_main TBB::tbb)
            target_include_directories(${TARGET_NAME} PRIVATE ${SMALLGWN_TEST_INCLUDE_DIRS})
            add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
        endfunction()

        # ---------------------------------------------------------------
        # Unit tests.
        # ---------------------------------------------------------------
        smallgwn_add_test(smallgwn_unit_status         SOURCES tests/unit_status.cu)
        smallgwn_add_test(smallgwn_unit_device_array   SOURCES tests/unit_device_array.cu)
        smallgwn_add_test(smallgwn_unit_stream_mixin   SOURCES tests/unit_stream_mixin.cu)
        smallgwn_add_test(smallgwn_unit_geometry       SOURCES tests/unit_geometry.cu)
        smallgwn_add_test(smallgwn_unit_bvh_topology   SOURCES tests/unit_bvh_topology.cu)
        smallgwn_add_test(smallgwn_unit_bvh_taylor     SOURCES tests/unit_bvh_taylor.cu)

        smallgwn_add_test(smallgwn_unit_winding_exact  SOURCES tests/unit_winding_exact.cu
            EXTRA_SOURCES tests/reference_hdk/UT_Array.cpp tests/reference_hdk/UT_SolidAngle.cpp)
        smallgwn_add_test(smallgwn_unit_winding_taylor SOURCES tests/unit_winding_taylor.cu)

        # ---------------------------------------------------------------
        # Integration tests.
        # ---------------------------------------------------------------
        smallgwn_add_test(smallgwn_integration_model_parity SOURCES tests/integration_model_parity.cu
            EXTRA_SOURCES tests/reference_hdk/UT_Array.cpp tests/reference_hdk/UT_SolidAngle.cpp)
        smallgwn_add_test(smallgwn_integration_correctness  SOURCES tests/integration_correctness.cu)
    else()
        message(STATUS "GTest not found; test targets are skipped.")
    endif()
endif()

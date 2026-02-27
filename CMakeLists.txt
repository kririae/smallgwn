cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(smallgwn LANGUAGES CXX CUDA)

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PCH_WARN_INVALID ON)

find_package(CUDAToolkit REQUIRED)

add_library(smallgwn INTERFACE)
add_library(gwn::smallgwn ALIAS smallgwn)

target_include_directories(smallgwn
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(smallgwn INTERFACE cxx_std_20)
target_link_libraries(smallgwn INTERFACE CUDA::cudart)
target_compile_options(smallgwn INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

option(
    SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET
    "Create gwn::smallgwn_eigen_bridge helper target (requires Eigen3 and Intel TBB)"
    ON
)
if(SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET)
    find_package(Eigen3 QUIET)
    find_package(TBB QUIET COMPONENTS tbb)
    if(Eigen3_FOUND AND TBB_FOUND)
        add_library(smallgwn_eigen_bridge INTERFACE)
        add_library(gwn::smallgwn_eigen_bridge ALIAS smallgwn_eigen_bridge)
        target_link_libraries(smallgwn_eigen_bridge INTERFACE gwn::smallgwn Eigen3::Eigen TBB::tbb)
    else()
        message(STATUS "Eigen3 or TBB not found; gwn::smallgwn_eigen_bridge target is skipped.")
    endif()
endif()

option(SMALLGWN_BUILD_BENCHMARKS "Build smallgwn benchmark executable" ON)
if(SMALLGWN_BUILD_BENCHMARKS)
    add_executable(smallgwn_benchmark benchmarks/benchmark_main.cu)
    target_link_libraries(smallgwn_benchmark PRIVATE gwn::smallgwn)
    target_include_directories(smallgwn_benchmark
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
endif()

option(SMALLGWN_BUILD_APPS "Build smallgwn sample applications" ON)
if(SMALLGWN_BUILD_APPS)
    add_executable(smallgwn_harnack_debug_app apps/harnack_debug/main.cu)
    target_link_libraries(smallgwn_harnack_debug_app PRIVATE gwn::smallgwn)
    target_include_directories(smallgwn_harnack_debug_app
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb"
    )
endif()

option(SMALLGWN_BUILD_TESTS "Build smallgwn tests" ON)
if(SMALLGWN_BUILD_TESTS)
    include(CTest)
    enable_testing()

    find_package(GTest QUIET)
    if(GTest_FOUND)
        find_package(Eigen3 REQUIRED)
        find_package(TBB REQUIRED COMPONENTS tbb)

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
            cmake_parse_arguments(ARG "NO_CTEST" "" "SOURCES;EXTRA_SOURCES" ${ARGN})
            add_executable(${TARGET_NAME} ${ARG_SOURCES} ${ARG_EXTRA_SOURCES})
            target_link_libraries(${TARGET_NAME} PRIVATE gwn::smallgwn Eigen3::Eigen GTest::gtest GTest::gtest_main)
            target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
            target_compile_definitions(${TARGET_NAME} PRIVATE GWN_ENABLE_ASSERTS)
            target_include_directories(${TARGET_NAME} PRIVATE ${SMALLGWN_TEST_INCLUDE_DIRS})
            if(NOT ARG_NO_CTEST)
                add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
            endif()
        endfunction()

        # ---------------------------------------------------------------
        # Unit tests.
        # ---------------------------------------------------------------
        smallgwn_add_test(smallgwn_unit_assert         SOURCES tests/unit_assert.cu)
        smallgwn_add_test(smallgwn_unit_status         SOURCES tests/unit_status.cu)
        smallgwn_add_test(smallgwn_unit_device_array   SOURCES tests/unit_device_array.cu)
        smallgwn_add_test(smallgwn_unit_stream_mixin   SOURCES tests/unit_stream_mixin.cu)
        smallgwn_add_test(smallgwn_unit_geometry       SOURCES tests/unit_geometry.cu)
        smallgwn_add_test(smallgwn_unit_bvh_topology   SOURCES tests/unit_bvh_topology.cu)
        smallgwn_add_test(smallgwn_unit_uint64_compile SOURCES tests/unit_uint64_compile.cu)
        smallgwn_add_test(smallgwn_unit_bvh_taylor     SOURCES tests/unit_bvh_taylor.cu)
        smallgwn_add_test(smallgwn_unit_winding_taylor SOURCES tests/unit_winding_taylor.cu)
        smallgwn_add_test(smallgwn_unit_winding_gradient SOURCES tests/unit_winding_gradient.cu)
        smallgwn_add_test(smallgwn_unit_harnack_trace    SOURCES tests/unit_harnack_trace.cu)

        # SDF tests (libigl parity).
        find_package(libigl QUIET HINTS /usr/lib/cmake/igl)
        if(libigl_FOUND)
            smallgwn_add_test(smallgwn_unit_sdf SOURCES tests/unit_sdf.cu EXTRA_SOURCES tests/libigl_reference.cpp)
            target_link_libraries(smallgwn_unit_sdf PRIVATE igl::igl_core)
        else()
            message(STATUS "libigl not found; SDF parity tests are skipped.")
        endif()

        # ---------------------------------------------------------------
        # Integration tests.
        # ---------------------------------------------------------------
        smallgwn_add_test(smallgwn_integration_model_parity SOURCES tests/integration_model_parity.cu
            EXTRA_SOURCES tests/reference_hdk/UT_Array.cpp tests/reference_hdk/UT_SolidAngle.cpp)
        smallgwn_add_test(smallgwn_integration_correctness  SOURCES tests/integration_correctness.cu)
        smallgwn_add_test(smallgwn_integration_gradient     SOURCES tests/integration_gradient.cu)
        smallgwn_add_test(smallgwn_integration_harnack_trace SOURCES tests/integration_harnack_trace.cu)
        smallgwn_add_test(smallgwn_integration_harnack_behavior_match SOURCES tests/integration_harnack_behavior_match.cu)
        smallgwn_add_test(smallgwn_integration_hploc_performance SOURCES tests/integration_hploc_performance.cu)

        smallgwn_add_test(smallgwn_integration_taylor_matrix NO_CTEST
            SOURCES tests/integration_taylor_matrix.cu
            EXTRA_SOURCES tests/reference_hdk/UT_Array.cpp tests/reference_hdk/UT_SolidAngle.cpp)

        add_test(NAME smallgwn_integration_taylor_matrix_light
            COMMAND smallgwn_integration_taylor_matrix --gtest_color=no
                --gtest_filter=smallgwn_integration_taylor_matrix.light_*)
        set_tests_properties(smallgwn_integration_taylor_matrix_light PROPERTIES
            LABELS "integration;light"
            FAIL_REGULAR_EXPRESSION "0 tests from 0 test suites;did not match any test"
            ENVIRONMENT "SMALLGWN_MATRIX_MODEL_LIMIT=2;SMALLGWN_MATRIX_TOTAL_POINTS=200000;SMALLGWN_MATRIX_ORDER_MAX=1")

        add_test(NAME smallgwn_integration_taylor_matrix_heavy
            COMMAND smallgwn_integration_taylor_matrix --gtest_color=no
                --gtest_filter=smallgwn_integration_taylor_matrix.heavy_*)
        set_tests_properties(smallgwn_integration_taylor_matrix_heavy PROPERTIES
            LABELS "integration;heavy"
            FAIL_REGULAR_EXPRESSION "0 tests from 0 test suites;did not match any test"
            ENVIRONMENT "SMALLGWN_MATRIX_MODEL_LIMIT=8;SMALLGWN_MATRIX_TOTAL_POINTS=2000000;SMALLGWN_MATRIX_ORDER_MAX=2")
    else()
        message(STATUS "GTest not found; test targets are skipped.")
    endif()
endif()

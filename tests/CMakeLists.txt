find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(libigl REQUIRED)

# ---------------------------------------------------------------
# Common include directories for all test targets.
# ---------------------------------------------------------------
set(SMALLGWN_TEST_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)
set(SMALLGWN_TEST_SYSTEM_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_hdk"
)

# ---------------------------------------------------------------
# Helper: define a unit/integration test target.
# ---------------------------------------------------------------
function(smallgwn_add_test_base TARGET_NAME)
    cmake_parse_arguments(
        ARG
        "NO_CTEST;RUN_SERIAL"
        ""
        "SOURCES;EXTRA_SOURCES;LABELS;EXTRA_LIBRARIES"
        ${ARGN}
    )
    add_executable(${TARGET_NAME} ${ARG_SOURCES} ${ARG_EXTRA_SOURCES})
    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
            gwn::smallgwn
            GTest::gtest
            GTest::gtest_main
            ${ARG_EXTRA_LIBRARIES}
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE GWN_ENABLE_ASSERTS)
    target_include_directories(${TARGET_NAME} PRIVATE ${SMALLGWN_TEST_INCLUDE_DIRS})
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${SMALLGWN_TEST_SYSTEM_INCLUDE_DIRS})
    smallgwn_apply_strict_warnings(${TARGET_NAME})
    if(NOT ARG_NO_CTEST)
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
        if(ARG_LABELS)
            set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${ARG_LABELS}")
        endif()
        if(ARG_RUN_SERIAL)
            set_tests_properties(${TARGET_NAME} PROPERTIES RUN_SERIAL TRUE)
        endif()
    endif()
endfunction()

function(smallgwn_add_cuda_test TARGET_NAME)
    smallgwn_add_test_base(${TARGET_NAME} ${ARGN})
endfunction()

function(smallgwn_add_eigen_tbb_test TARGET_NAME)
    smallgwn_add_test_base(
        ${TARGET_NAME}
        ${ARGN}
        EXTRA_LIBRARIES Eigen3::Eigen TBB::tbb
    )
endfunction()

function(smallgwn_add_tbb_test TARGET_NAME)
    smallgwn_add_test_base(
        ${TARGET_NAME}
        ${ARGN}
        EXTRA_LIBRARIES TBB::tbb
    )
endfunction()

function(smallgwn_add_libigl_test TARGET_NAME)
    smallgwn_add_test_base(
        ${TARGET_NAME}
        ${ARGN}
        EXTRA_LIBRARIES igl::igl_core
    )
endfunction()

# ---------------------------------------------------------------
# Unit tests.
# ---------------------------------------------------------------
smallgwn_add_cuda_test(smallgwn_unit_assert         LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_assert.cu)
smallgwn_add_cuda_test(smallgwn_unit_status         LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_status.cu)
smallgwn_add_cuda_test(smallgwn_unit_device_array   LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_device_array.cu)
smallgwn_add_cuda_test(smallgwn_unit_kernel_utils   LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_kernel_utils.cu)
smallgwn_add_cuda_test(smallgwn_unit_stream_mixin   LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_stream_mixin.cu)
smallgwn_add_cuda_test(smallgwn_unit_geometry       LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_geometry.cu)
smallgwn_add_cuda_test(smallgwn_unit_boundary_chain LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_boundary_chain.cu)
smallgwn_add_cuda_test(smallgwn_unit_bvh_topology   LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_bvh_topology.cu)
smallgwn_add_cuda_test(smallgwn_unit_uint64_compile LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_uint64_compile.cu)
smallgwn_add_cuda_test(smallgwn_unit_bvh_taylor     LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_bvh_taylor.cu)
smallgwn_add_eigen_tbb_test(smallgwn_unit_eigen_bridge
    LABELS "unit;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_eigen_bridge.cu)
smallgwn_add_eigen_tbb_test(smallgwn_unit_winding_taylor
    LABELS "unit;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_winding_taylor.cu)
smallgwn_add_eigen_tbb_test(smallgwn_unit_winding_gradient
    LABELS "unit;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_winding_gradient.cu)
smallgwn_add_cuda_test(smallgwn_unit_winding_antipodal
    LABELS "unit;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_winding_antipodal.cu)
smallgwn_add_cuda_test(smallgwn_unit_harnack_trace  LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_harnack_trace.cu)
smallgwn_add_cuda_test(smallgwn_unit_edge_distance  LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_edge_distance.cu)
smallgwn_add_cuda_test(smallgwn_unit_ray_query      LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_ray_query.cu)
smallgwn_add_cuda_test(smallgwn_unit_hybrid_trace   LABELS "unit;cuda" SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_hybrid_trace.cu)

# SDF tests (libigl parity).
smallgwn_add_libigl_test(smallgwn_unit_sdf
    LABELS "unit;cuda;libigl"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/unit_sdf.cu
    EXTRA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/libigl_reference.cpp)

# ---------------------------------------------------------------
# Integration tests.
# ---------------------------------------------------------------
smallgwn_add_cuda_test(smallgwn_integration_builtin_models
    LABELS "integration;cuda;fixtures"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_builtin_models.cu)
smallgwn_add_eigen_tbb_test(smallgwn_integration_model_parity
    LABELS "integration;cuda;models;slow"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_model_parity.cu
    EXTRA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/reference_hdk/UT_Array.cpp ${CMAKE_CURRENT_SOURCE_DIR}/reference_hdk/UT_SolidAngle.cpp)
smallgwn_add_cuda_test(smallgwn_integration_antipodal_models
    LABELS "integration;cuda;models;slow"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_antipodal_models.cu)
smallgwn_add_eigen_tbb_test(smallgwn_integration_correctness
    LABELS "integration;cuda;models;slow"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_correctness.cu)
smallgwn_add_eigen_tbb_test(smallgwn_integration_gradient
    LABELS "integration;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_gradient.cu)
smallgwn_add_cuda_test(smallgwn_integration_harnack_trace
    LABELS "integration;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_harnack_trace.cu)
smallgwn_add_cuda_test(smallgwn_integration_harnack_behavior_match
    LABELS "integration;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_harnack_behavior_match.cu)
smallgwn_add_cuda_test(smallgwn_integration_harnack_iteration_stats
    LABELS "integration;cuda"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_harnack_iteration_stats.cu)
smallgwn_add_cuda_test(smallgwn_integration_hploc_performance
    RUN_SERIAL
    LABELS "integration;cuda;perf;slow"
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_hploc_performance.cu)

smallgwn_add_tbb_test(smallgwn_integration_taylor_matrix
    NO_CTEST
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/integration_taylor_matrix.cu
    EXTRA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/reference_hdk/UT_Array.cpp ${CMAKE_CURRENT_SOURCE_DIR}/reference_hdk/UT_SolidAngle.cpp)

add_test(NAME smallgwn_integration_taylor_matrix_light
    COMMAND smallgwn_integration_taylor_matrix --gtest_color=no
        --gtest_filter=smallgwn_integration_taylor_matrix.light_*)
set_tests_properties(smallgwn_integration_taylor_matrix_light PROPERTIES
    LABELS "integration;cuda;models;slow;light"
    FAIL_REGULAR_EXPRESSION "0 tests from 0 test suites;did not match any test"
    ENVIRONMENT "SMALLGWN_MATRIX_MODEL_LIMIT=2;SMALLGWN_MATRIX_TOTAL_POINTS=200000;SMALLGWN_MATRIX_ORDER_MAX=1")

add_test(NAME smallgwn_integration_taylor_matrix_heavy
    COMMAND smallgwn_integration_taylor_matrix --gtest_color=no
        --gtest_filter=smallgwn_integration_taylor_matrix.heavy_*)
set_tests_properties(smallgwn_integration_taylor_matrix_heavy PROPERTIES
    LABELS "integration;cuda;models;slow;heavy"
    FAIL_REGULAR_EXPRESSION "0 tests from 0 test suites;did not match any test"
    ENVIRONMENT "SMALLGWN_MATRIX_MODEL_LIMIT=8;SMALLGWN_MATRIX_TOTAL_POINTS=2000000;SMALLGWN_MATRIX_ORDER_MAX=2")

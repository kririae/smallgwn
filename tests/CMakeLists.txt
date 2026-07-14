find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)

set(SMALLGWN_TEST_STACK_CAPACITY "64" CACHE STRING
    "Traversal stack capacity used by integration and E2E dataset validation (64 or 128)"
)
set_property(CACHE SMALLGWN_TEST_STACK_CAPACITY PROPERTY STRINGS 64 128)
if(NOT SMALLGWN_TEST_STACK_CAPACITY MATCHES "^(64|128)$")
    message(FATAL_ERROR "SMALLGWN_TEST_STACK_CAPACITY must be 64 or 128")
endif()

function(smallgwn_apply_strict_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xcudafe=--diag_suppress=177>
        )
    elseif(MSVC)
        target_compile_options(${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    endif()
endfunction()

function(smallgwn_configure_test_target target)
    target_link_libraries(${target}
        PRIVATE
            gwn::smallgwn
            GTest::gtest
    )
    target_compile_definitions(${target} PRIVATE GWN_ENABLE_ASSERTS)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(${target} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reference_hdk")
    smallgwn_apply_strict_warnings(${target})
endfunction()

add_executable(smallgwn_unit
    unit_gwn_assert.cu
    unit_gwn_boundary.cu
    unit_gwn_bvh.cu
    unit_gwn_bvh_moment.cu
    unit_gwn_device_array.cu
    unit_gwn_eigen_bridge.cu
    unit_gwn_geometry.cu
    unit_gwn_kernel_utils.cu
    unit_gwn_query_antipodal.cu
    unit_gwn_query_distance_ray.cu
    unit_gwn_query_gradient.cu
    unit_gwn_query_stack.cu
    unit_gwn_query_winding.cu
    unit_gwn_status.cu
    unit_gwn_stream.cu
)
smallgwn_configure_test_target(smallgwn_unit)
target_link_libraries(smallgwn_unit PRIVATE GTest::gtest_main Eigen3::Eigen TBB::tbb)
add_test(NAME smallgwn_unit COMMAND smallgwn_unit)
set_tests_properties(smallgwn_unit PROPERTIES LABELS "unit;cuda")

add_executable(smallgwn_integration
    integration_gwn_builder.cu
    integration_gwn_dynamic.cu
    integration_gwn_static.cu
    test_mesh_main.cpp
)
smallgwn_configure_test_target(smallgwn_integration)
target_compile_definitions(
    smallgwn_integration PRIVATE SMALLGWN_TEST_STACK_CAPACITY=${SMALLGWN_TEST_STACK_CAPACITY}
)
target_link_libraries(smallgwn_integration PRIVATE Eigen3::Eigen TBB::tbb)
add_test(
    NAME smallgwn_integration
    COMMAND smallgwn_integration --mesh-dir "${CMAKE_CURRENT_SOURCE_DIR}/data"
)
set_tests_properties(smallgwn_integration PROPERTIES LABELS "integration;cuda;fixtures")

add_library(smallgwn_reference_hdk OBJECT
    reference_hdk/UT_Array.cpp
    reference_hdk/UT_SolidAngle.cpp
)
target_compile_features(smallgwn_reference_hdk PRIVATE cxx_std_20)
target_link_libraries(smallgwn_reference_hdk PRIVATE TBB::tbb)
target_include_directories(smallgwn_reference_hdk SYSTEM PRIVATE reference_hdk)
smallgwn_apply_strict_warnings(smallgwn_reference_hdk)

# E2E is built for explicit dataset validation but is intentionally absent from default CTest.
add_executable(smallgwn_e2e
    e2e_gwn.cu
    test_mesh_main.cpp
    $<TARGET_OBJECTS:smallgwn_reference_hdk>
)
smallgwn_configure_test_target(smallgwn_e2e)
target_compile_definitions(
    smallgwn_e2e PRIVATE SMALLGWN_TEST_STACK_CAPACITY=${SMALLGWN_TEST_STACK_CAPACITY}
)
target_link_libraries(smallgwn_e2e PRIVATE TBB::tbb)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(smallgwn_e2e PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
    )
endif()

# Benchmark belongs to test tooling but never becomes a correctness test.
add_executable(smallgwn_benchmark benchmark_main.cu)
target_link_libraries(smallgwn_benchmark PRIVATE gwn::smallgwn)
target_include_directories(smallgwn_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
smallgwn_apply_strict_warnings(smallgwn_benchmark)

option(SMALLGWN_BUILD_CUBQL_BENCHMARK "Build the opt-in cuBQL comparison benchmark" OFF)
set(SMALLGWN_CUBQL_SOURCE "" CACHE PATH "Path to a local cuBQL source checkout")
if(SMALLGWN_BUILD_CUBQL_BENCHMARK)
    if(NOT EXISTS "${SMALLGWN_CUBQL_SOURCE}/CMakeLists.txt")
        message(FATAL_ERROR
            "SMALLGWN_BUILD_CUBQL_BENCHMARK requires SMALLGWN_CUBQL_SOURCE to name a cuBQL checkout"
        )
    endif()

    add_subdirectory(
        "${SMALLGWN_CUBQL_SOURCE}"
        "${CMAKE_CURRENT_BINARY_DIR}/cubql"
        EXCLUDE_FROM_ALL
    )
    add_executable(smallgwn_benchmark_cubql benchmark_cubql.cu)
    target_link_libraries(
        smallgwn_benchmark_cubql
        PRIVATE
            gwn::smallgwn
            cuBQL_cuda_float3_static
    )
    target_include_directories(
        smallgwn_benchmark_cubql
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_include_directories(
        smallgwn_benchmark_cubql
        SYSTEM PRIVATE "${SMALLGWN_CUBQL_SOURCE}"
    )
    smallgwn_apply_strict_warnings(smallgwn_benchmark_cubql)
endif()

# ----------------------------------------------------------
# Shared support
# ----------------------------------------------------------
function(smallgwn_apply_strict_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            ${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
                              $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xcudafe=--diag_suppress=177>)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>
                                                 $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    endif()
endfunction()

function(smallgwn_configure_test_target target)
    target_link_libraries(${target} PRIVATE gwn::smallgwn GTest::gtest)
    target_compile_definitions(${target} PRIVATE GWN_ENABLE_ASSERTS)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(${target} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reference_hdk")
    smallgwn_apply_strict_warnings(${target})
    if(MSVC)
        target_link_options(${target} PRIVATE $<$<LINK_LANGUAGE:CUDA>:/SUBSYSTEM:CONSOLE>)
    endif()
endfunction()

add_library(smallgwn_test_mesh_io STATIC test_mesh_io.cpp)
target_compile_features(smallgwn_test_mesh_io PRIVATE cxx_std_20)
target_include_directories(smallgwn_test_mesh_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(smallgwn_test_mesh_io SYSTEM
                           PRIVATE $<TARGET_PROPERTY:igl::core,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(smallgwn_test_mesh_io PRIVATE Eigen3::Eigen igl::core)
smallgwn_apply_strict_warnings(smallgwn_test_mesh_io)

if(SMALLGWN_BUILD_TESTS)
    # ----------------------------------------------------------
    # Unit tests
    # ----------------------------------------------------------
    add_executable(
        smallgwn_unit
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
        unit_gwn_stream.cu)
    smallgwn_configure_test_target(smallgwn_unit)
    target_link_libraries(smallgwn_unit PRIVATE GTest::gtest_main Eigen3::Eigen TBB::tbb)
    add_test(NAME smallgwn_unit COMMAND smallgwn_unit)
    set_tests_properties(smallgwn_unit PROPERTIES LABELS "unit;cuda")

    # ----------------------------------------------------------
    # Integration tests
    # ----------------------------------------------------------
    add_executable(smallgwn_integration integration_gwn_builder.cu integration_gwn_dynamic.cu integration_gwn_static.cu
                                        test_mesh_main.cpp)
    smallgwn_configure_test_target(smallgwn_integration)
    target_link_libraries(smallgwn_integration PRIVATE Eigen3::Eigen TBB::tbb smallgwn_test_mesh_io)
    add_test(NAME smallgwn_integration COMMAND smallgwn_integration --mesh-dir "${CMAKE_CURRENT_SOURCE_DIR}/data")
    set_tests_properties(smallgwn_integration PROPERTIES LABELS "integration;cuda;fixtures")

    # ----------------------------------------------------------
    # E2E validation
    # ----------------------------------------------------------
    add_library(smallgwn_reference_hdk OBJECT reference_hdk/UT_Array.cpp reference_hdk/UT_SolidAngle.cpp)
    target_compile_features(smallgwn_reference_hdk PRIVATE cxx_std_20)
    target_link_libraries(smallgwn_reference_hdk PRIVATE TBB::tbb)
    target_include_directories(smallgwn_reference_hdk SYSTEM PRIVATE reference_hdk)
    smallgwn_apply_strict_warnings(smallgwn_reference_hdk)

    add_executable(smallgwn_e2e e2e_gwn.cu test_mesh_main.cpp $<TARGET_OBJECTS:smallgwn_reference_hdk>)
    smallgwn_configure_test_target(smallgwn_e2e)
    target_link_libraries(smallgwn_e2e PRIVATE TBB::tbb smallgwn_test_mesh_io)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(smallgwn_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
    endif()
endif()

if(SMALLGWN_BUILD_BENCHMARKS)
    # ----------------------------------------------------------
    # Benchmarks
    # ----------------------------------------------------------
    add_executable(smallgwn_benchmark benchmark_main.cu)
    target_link_libraries(smallgwn_benchmark PRIVATE gwn::smallgwn CLI11::CLI11 smallgwn_test_mesh_io)
    target_include_directories(smallgwn_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    smallgwn_apply_strict_warnings(smallgwn_benchmark)

    add_executable(smallgwn_benchmark_cubql benchmark_cubql.cu)
    target_include_directories(smallgwn_benchmark_cubql SYSTEM
                               PRIVATE $<TARGET_PROPERTY:cuBQL_cuda_float3_static,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(smallgwn_benchmark_cubql PRIVATE gwn::smallgwn CLI11::CLI11 cuBQL_cuda_float3_static
                                                           smallgwn_test_mesh_io)
    target_include_directories(smallgwn_benchmark_cubql PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    smallgwn_apply_strict_warnings(smallgwn_benchmark_cubql)
endif()

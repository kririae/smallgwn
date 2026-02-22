cmake_minimum_required(VERSION 3.24)

project(smallgwn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

    add_executable(smallgwn_smoke_compile tests/smoke_compile.cu)
    target_link_libraries(smallgwn_smoke_compile PRIVATE gwn::smallgwn)
    add_test(NAME smallgwn_smoke_compile COMMAND smallgwn_smoke_compile)

    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_executable(smallgwn_parity_scaffold
            tests/parity_scaffold.cu
            tests/reference_hdk/UT_Array.cpp
            tests/reference_hdk/UT_SolidAngle.cpp
        )
        target_link_libraries(smallgwn_parity_scaffold PRIVATE gwn::smallgwn GTest::gtest GTest::gtest_main TBB::tbb)
        target_include_directories(smallgwn_parity_scaffold PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_hdk"
        )
        add_test(NAME smallgwn_parity_scaffold COMMAND smallgwn_parity_scaffold)

        add_executable(smallgwn_bvh_model_parity
            tests/bvh_model_parity.cu
        )
        target_link_libraries(smallgwn_bvh_model_parity PRIVATE gwn::smallgwn GTest::gtest GTest::gtest_main TBB::tbb)
        target_include_directories(smallgwn_bvh_model_parity PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )
        add_test(NAME smallgwn_bvh_model_parity COMMAND smallgwn_bvh_model_parity)
    else()
        message(STATUS "GTest not found; parity scaffold target is skipped.")
    endif()
endif()

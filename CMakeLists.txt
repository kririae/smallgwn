cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(smallgwn VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(smallgwn INTERFACE)
add_library(gwn::smallgwn ALIAS smallgwn)
target_include_directories(smallgwn
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(smallgwn INTERFACE cxx_std_20 cuda_std_20)
target_link_libraries(smallgwn INTERFACE CUDA::cudart)
target_compile_options(smallgwn INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

option(
    SMALLGWN_BUILD_EIGEN_BRIDGE
    "Create the optional gwn::smallgwn_eigen_bridge interface target"
    ${PROJECT_IS_TOP_LEVEL}
)
if(SMALLGWN_BUILD_EIGEN_BRIDGE)
    find_package(Eigen3 REQUIRED)
    find_package(TBB REQUIRED COMPONENTS tbb)

    add_library(smallgwn_eigen_bridge INTERFACE)
    add_library(gwn::smallgwn_eigen_bridge ALIAS smallgwn_eigen_bridge)
    target_link_libraries(smallgwn_eigen_bridge INTERFACE gwn::smallgwn Eigen3::Eigen TBB::tbb)
endif()

option(SMALLGWN_BUILD_TESTS "Build the smallgwn tests and benchmark" ${PROJECT_IS_TOP_LEVEL})
if(SMALLGWN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

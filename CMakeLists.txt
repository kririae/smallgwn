cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(smallgwn LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PCH_WARN_INVALID ON)

find_package(CUDAToolkit REQUIRED)

# ----------------------------------------------------------------------------
# Core header-only library target.
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Strict-warnings helper.
# ----------------------------------------------------------------------------
function(smallgwn_apply_strict_warnings TARGET_NAME)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_NAME}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Xcudafe=--diag_suppress=177>
        )
    elseif(MSVC)
        target_compile_options(${TARGET_NAME}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    endif()
endfunction()

# ----------------------------------------------------------------------------
# Options: default ON when this is the top-level project, OFF otherwise
#          (e.g. when consumed via FetchContent).
# ----------------------------------------------------------------------------

option(
    SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET
    "Create gwn::smallgwn_eigen_bridge helper target (requires Eigen3 and Intel TBB)"
    ${PROJECT_IS_TOP_LEVEL}
)
if(SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET)
    find_package(Eigen3 REQUIRED)
    find_package(TBB REQUIRED COMPONENTS tbb)

    add_library(smallgwn_eigen_bridge INTERFACE)
    add_library(gwn::smallgwn_eigen_bridge ALIAS smallgwn_eigen_bridge)
    target_link_libraries(smallgwn_eigen_bridge INTERFACE gwn::smallgwn Eigen3::Eigen TBB::tbb)
endif()

option(
    SMALLGWN_BUILD_BENCHMARKS "Build smallgwn benchmark executable"
    ${PROJECT_IS_TOP_LEVEL}
)
if(SMALLGWN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

option(
    SMALLGWN_BUILD_TESTS "Build smallgwn tests"
    ${PROJECT_IS_TOP_LEVEL}
)
if(SMALLGWN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(smallgwn_harnack_debug LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GetCPM.cmake)

# Mirror winding_studio style: this app can be consumed as a third-party project
# that depends on smallgwn via CPM.
if(NOT TARGET gwn::smallgwn)
    set(
        SMALLGWN_HARNACK_DEBUG_SMALLGWN_REPOSITORY
        "kririae/smallgwn"
        CACHE STRING
        "GitHub repository path used by CPM to fetch smallgwn."
    )
    set(
        SMALLGWN_HARNACK_DEBUG_SMALLGWN_TAG
        "main"
        CACHE STRING
        "Git tag/branch used by CPM to fetch smallgwn."
    )

    CPMAddPackage(
        NAME smallgwn
        GITHUB_REPOSITORY ${SMALLGWN_HARNACK_DEBUG_SMALLGWN_REPOSITORY}
        GIT_TAG ${SMALLGWN_HARNACK_DEBUG_SMALLGWN_TAG}
        OPTIONS
            "SMALLGWN_BUILD_BENCHMARKS OFF"
            "SMALLGWN_BUILD_APPS OFF"
            "SMALLGWN_BUILD_TESTS OFF"
            "SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET OFF"
    )
endif()

CPMAddPackage(
    NAME stb
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

add_executable(smallgwn_harnack_debug_app main.cu)
target_link_libraries(smallgwn_harnack_debug_app PRIVATE gwn::smallgwn)
target_include_directories(
    smallgwn_harnack_debug_app
    PRIVATE
        ${stb_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(winding_studio LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------
# Dependencies
# ----------------------------------------------------------
include(GetCPM.cmake)

# ----------------------------------------------------------
# smallgwn (header-only CUDA winding number library)
# ----------------------------------------------------------
CPMAddPackage(
    NAME smallgwn
    GITHUB_REPOSITORY kririae/smallgwn
    GIT_TAG main
    OPTIONS
        "SMALLGWN_BUILD_BENCHMARKS OFF"
        "SMALLGWN_BUILD_APPS OFF"
        "SMALLGWN_BUILD_TESTS OFF"
        "SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET OFF")

# ----------------------------------------------------------
# Eigen3
# ----------------------------------------------------------
CPMAddPackage(
    NAME Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    OPTIONS "BUILD_TESTING OFF")

# ----------------------------------------------------------
# libigl (mesh I/O)
# ----------------------------------------------------------
CPMAddPackage(
    NAME libigl
    GITHUB_REPOSITORY libigl/libigl
    GIT_TAG v2.5.0
    OPTIONS
        "LIBIGL_BUILD_TESTS OFF"
        "LIBIGL_BUILD_TUTORIALS OFF"
        "LIBIGL_INSTALL OFF")

# ----------------------------------------------------------
# ImGui (no CMakeLists.txt, build manually)
# ----------------------------------------------------------
CPMAddPackage(
    NAME imgui
    GITHUB_REPOSITORY ocornut/imgui
    GIT_TAG v1.91.8
    DOWNLOAD_ONLY YES)

# ----------------------------------------------------------
# stb (header-only image writing)
# ----------------------------------------------------------
CPMAddPackage(
    NAME stb
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG master
    DOWNLOAD_ONLY YES)

# ----------------------------------------------------------
# glfw3, OpenGL, GLEW (system packages)
# ----------------------------------------------------------
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

# ----------------------------------------------------------
# ImGui static library
# ----------------------------------------------------------
add_library(imgui_backend STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_backend
    PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_backend
    PUBLIC glfw OpenGL::GL GLEW::GLEW)
target_compile_definitions(imgui_backend
    PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

# ----------------------------------------------------------
# Winding Studio
# ----------------------------------------------------------
add_executable(winding_studio
    main.cpp
    harnack_tracer.cu)
target_include_directories(winding_studio
    PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(winding_studio
    PRIVATE
        gwn::smallgwn
        imgui_backend
        Eigen3::Eigen
        igl::core)
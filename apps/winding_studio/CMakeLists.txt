cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(winding_studio LANGUAGES CXX CUDA)

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(libigl REQUIRED HINTS /usr/lib/cmake/igl)

# gwn (header-only, parent project)
if(NOT TARGET gwn::smallgwn)
    find_path(GWN_INCLUDE_DIR gwn/gwn.cuh
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    )
    if(NOT GWN_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find gwn headers. Set GWN_INCLUDE_DIR or build from the parent project.")
    endif()
    add_library(smallgwn INTERFACE)
    add_library(gwn::smallgwn ALIAS smallgwn)
    target_include_directories(smallgwn INTERFACE "${GWN_INCLUDE_DIR}")
    target_compile_features(smallgwn INTERFACE cxx_std_20)
    target_link_libraries(smallgwn INTERFACE CUDA::cudart)
    target_compile_options(smallgwn INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
endif()

# ImGui (fetched)
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(winding_studio_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(winding_studio_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(winding_studio_imgui PUBLIC glfw OpenGL::GL GLEW::GLEW)
target_compile_definitions(winding_studio_imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

# stb (header-only, from parent repo third_party/)
add_library(winding_studio_stb INTERFACE)
target_include_directories(winding_studio_stb
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/stb"
)

# App target
add_executable(winding_studio
    main.cpp
    harnack_tracer.cu
)
set_target_properties(winding_studio PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(winding_studio
    PRIVATE
        winding_studio_imgui
        winding_studio_stb
        gwn::smallgwn
        glfw
        OpenGL::GL
        GLEW::GLEW
        Eigen3::Eigen
        igl::igl_core
)

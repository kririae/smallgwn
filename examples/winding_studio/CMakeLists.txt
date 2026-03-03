cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(winding_studio LANGUAGES C CXX CUDA)

option(
    WINDING_STUDIO_ENABLE_IMGUI_TEST_ENGINE
    "Enable Dear ImGui Test Engine based UI contract tests"
    ON)
set(
    WINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR
    ""
    CACHE PATH
    "Optional local path to imgui_test_engine source (skip network fetch)")

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
get_filename_component(
    WINDING_STUDIO_SMALLGWN_PARENT_DEFAULT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE
)
set(
    WINDING_STUDIO_SMALLGWN_SOURCE_DIR
    "${WINDING_STUDIO_SMALLGWN_PARENT_DEFAULT}"
    CACHE PATH
    "Local smallgwn source path. Defaults to parent repo root. Set empty to force CPM remote fetch."
)

if(WINDING_STUDIO_SMALLGWN_SOURCE_DIR AND EXISTS "${WINDING_STUDIO_SMALLGWN_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS
        "Using local smallgwn source from: ${WINDING_STUDIO_SMALLGWN_SOURCE_DIR}")
    CPMAddPackage(
        NAME smallgwn
        SOURCE_DIR "${WINDING_STUDIO_SMALLGWN_SOURCE_DIR}"
        OPTIONS
            "SMALLGWN_BUILD_BENCHMARKS OFF"
            "SMALLGWN_BUILD_APPS OFF"
            "SMALLGWN_BUILD_TESTS OFF"
            "SMALLGWN_BUILD_EIGEN_BRIDGE_TARGET OFF")
else()
    message(FATAL_ERROR
        "Local smallgwn source not found at WINDING_STUDIO_SMALLGWN_SOURCE_DIR='"
        "${WINDING_STUDIO_SMALLGWN_SOURCE_DIR}'. Set this cache variable to a valid smallgwn root.")
endif()

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
    GIT_TAG v1.92.6
    DOWNLOAD_ONLY YES)

# ----------------------------------------------------------
# Dear ImGui Test Engine (optional, for UI contract tests)
# ----------------------------------------------------------
if(WINDING_STUDIO_ENABLE_IMGUI_TEST_ENGINE)
    if(WINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR)
        if(NOT EXISTS "${WINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR}")
            message(FATAL_ERROR
                "WINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR does not exist: "
                "${WINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR}")
        endif()
        CPMAddPackage(
            NAME imgui_test_engine
            SOURCE_DIR "${WINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR}"
            DOWNLOAD_ONLY YES)
    else()
        CPMAddPackage(
            NAME imgui_test_engine
            GITHUB_REPOSITORY ocornut/imgui_test_engine
            GIT_TAG v1.92.6
            DOWNLOAD_ONLY YES)
    endif()
endif()

# ----------------------------------------------------------
# stb (header-only image writing)
# ----------------------------------------------------------
CPMAddPackage(
    NAME stb
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG master
    DOWNLOAD_ONLY YES)

# ----------------------------------------------------------
# glm (header-only math)
# ----------------------------------------------------------
CPMAddPackage(
    NAME glm
    GITHUB_REPOSITORY g-truc/glm
    GIT_TAG 1.0.1)

# ----------------------------------------------------------
# oneTBB
# ----------------------------------------------------------
CPMAddPackage(
    NAME oneTBB
    GITHUB_REPOSITORY oneapi-src/oneTBB
    GIT_TAG v2022.3.0
    OPTIONS
        "TBB_TEST OFF"
        "TBB_STRICT OFF")

# ----------------------------------------------------------
# GLAD2 (OpenGL loader)
# ----------------------------------------------------------
CPMAddPackage(
    NAME glad
    GITHUB_REPOSITORY Dav1dde/glad
    GIT_TAG v2.0.8
    SOURCE_SUBDIR cmake)

# ----------------------------------------------------------
# glfw3, OpenGL (system packages)
# ----------------------------------------------------------
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

glad_add_library(
    glad_gl_core_33
    REPRODUCIBLE
    EXCLUDE_FROM_ALL
    LOADER
    API gl:core=3.3)

# ----------------------------------------------------------
# ImGui core static library
# ----------------------------------------------------------
add_library(imgui_core STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp)
target_include_directories(imgui_core
    PUBLIC ${imgui_SOURCE_DIR})

# ----------------------------------------------------------
# ImGui backend static library
# ----------------------------------------------------------
add_library(imgui_backend STATIC
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_backend
    PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_backend
    PUBLIC imgui_core glfw OpenGL::GL glad_gl_core_33)
target_compile_definitions(imgui_backend
    PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

# ----------------------------------------------------------
# Winding Studio
# ----------------------------------------------------------
add_executable(winding_studio
    main.cpp
    studio_app.cpp
    studio_activation.cpp
    studio_cli.cpp
    studio_mesh_library.cpp
    studio_mesh_validate.cpp
    studio_render.cpp
    studio_trace.cpp
    studio_ui.cpp
    studio_voxel.cpp
    harnack_tracer.cu
    voxelizer.cu)
target_include_directories(winding_studio
    PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(winding_studio
    PRIVATE
        gwn::smallgwn
        imgui_backend
        glm::glm
        TBB::tbb
        Eigen3::Eigen
        igl::core)

include(CTest)
if(BUILD_TESTING)
    add_executable(winding_studio_test_mesh_validate
        tests/test_mesh_validate.cpp
        studio_mesh_validate.cpp)
    target_include_directories(winding_studio_test_mesh_validate
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME winding_studio_test_mesh_validate COMMAND winding_studio_test_mesh_validate)
    set_tests_properties(winding_studio_test_mesh_validate PROPERTIES LABELS "unit")

    add_executable(winding_studio_test_mesh_library
        tests/test_mesh_library.cpp
        studio_mesh_library.cpp)
    target_include_directories(winding_studio_test_mesh_library
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(winding_studio_test_mesh_library
        PRIVATE
            TBB::tbb
            Eigen3::Eigen
            igl::core)
    add_test(NAME winding_studio_test_mesh_library COMMAND winding_studio_test_mesh_library)
    set_tests_properties(winding_studio_test_mesh_library PROPERTIES LABELS "unit")

    if(WINDING_STUDIO_ENABLE_IMGUI_TEST_ENGINE)
        add_library(imgui_core_test STATIC
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp)
        target_include_directories(imgui_core_test
            PUBLIC ${imgui_SOURCE_DIR})
        target_compile_definitions(imgui_core_test
            PUBLIC IMGUI_ENABLE_TEST_ENGINE)

        add_library(imgui_test_engine_lib STATIC
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_capture_tool.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_context.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_coroutine.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_engine.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_exporters.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_perftool.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_ui.cpp
            ${imgui_test_engine_SOURCE_DIR}/imgui_test_engine/imgui_te_utils.cpp)
        target_include_directories(imgui_test_engine_lib
            PUBLIC
                ${imgui_SOURCE_DIR}
                ${imgui_test_engine_SOURCE_DIR})
        target_compile_definitions(imgui_test_engine_lib
            PUBLIC
                IMGUI_ENABLE_TEST_ENGINE
                IMGUI_TEST_ENGINE_ENABLE_COROUTINE_STDTHREAD_IMPL=1)
        target_link_libraries(imgui_test_engine_lib
            PUBLIC imgui_core_test)

        add_executable(winding_studio_test_ui_contract
            tests/ui_contract_runner.cpp
            tests/ui_contract_tests.cpp
            studio_ui.cpp
            studio_mesh_library.cpp)
        target_include_directories(winding_studio_test_ui_contract
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(winding_studio_test_ui_contract
            PRIVATE IMGUI_ENABLE_TEST_ENGINE)
        target_link_libraries(winding_studio_test_ui_contract
            PRIVATE
                imgui_test_engine_lib
                TBB::tbb
                Eigen3::Eigen
                igl::core)
        add_test(NAME winding_studio_test_ui_contract COMMAND winding_studio_test_ui_contract)
        set_tests_properties(winding_studio_test_ui_contract PROPERTIES LABELS "ui_contract")
    endif()
endif()

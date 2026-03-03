#include "studio_app.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <glad/gl.h>

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "harnack_tracer.hpp"
#include "imgui.h"
#include "mesh_loader.hpp"
#include "studio_activation.hpp"
#include "studio_mesh_library.hpp"
#include "studio_mesh_validate.hpp"
#include "studio_render.hpp"
#include "studio_trace.hpp"
#include "studio_ui.hpp"
#include "studio_voxel.hpp"
#include "voxelizer.hpp"

namespace winding_studio::app {

namespace {

void glfw_error_callback(int const error, char const *description) {
    std::cerr << "[GLFW] (" << error << ") " << description << "\n";
}

} // namespace

int run_app(CliOptions const &cli) {
    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == GLFW_FALSE)
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    GLFWwindow *window =
        glfwCreateWindow(cli.width, cli.height, "Winding Studio", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("glfwCreateWindow failed");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0)
        throw std::runtime_error("gladLoadGL failed");
    glGetError();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    float dpi_scale = 1.0f;
    if (GLFWmonitor *monitor = glfwGetPrimaryMonitor()) {
        float xscale = 1.0f;
        float yscale = 1.0f;
        glfwGetMonitorContentScale(monitor, &xscale, &yscale);
        dpi_scale = std::max(xscale, yscale);
    }

    apply_engine_style(dpi_scale);

    float const base_font_size = 15.0f;
    float const scaled_font_size = base_font_size * dpi_scale;
    bool font_loaded = false;
    for (char const *path : {
             "/usr/share/fonts/TTF/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
         }) {
        if (std::filesystem::exists(path)) {
            io.Fonts->AddFontFromFileTTF(path, scaled_font_size);
            font_loaded = true;
            break;
        }
    }
    if (!font_loaded) {
        ImFontConfig cfg;
        cfg.SizePixels = 13.0f * dpi_scale;
        io.Fonts->AddFontDefault(&cfg);
    }

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true))
        throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
    if (!ImGui_ImplOpenGL3_Init("#version 330"))
        throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");

    AppState state{};
    state.view_mode = cli.view_mode;
    if (cli.voxel_dx > 0.0f)
        state.voxel_dx = cli.voxel_dx;
    if (cli.camera_distance > 0.0f)
        state.camera_radius = cli.camera_distance;
    if (cli.harnack_resolution_scale > 0.0f)
        state.harnack_resolution_scale = cli.harnack_resolution_scale;
    if (cli.harnack_target_w > 0.0f)
        state.target_winding = cli.harnack_target_w;

    MeshRenderer renderer{};
    TextureRenderer harnack_texture_renderer{};
    VoxelRenderer voxel_renderer{};
    winding_studio::HarnackTracer harnack_tracer{};
    winding_studio::Voxelizer voxelizer{};
    winding_studio::HarnackTraceImages trace_images{};
    winding_studio::VoxelizeStats voxel_stats{};

    auto const add_and_activate = [&](MeshData mesh, std::string name, bool is_builtin,
                                      std::string const &status_prefix) -> bool {
        std::string validation_error;
        if (!validate_mesh_data(mesh, validation_error)) {
            state.status_line = "Load failed: " + validation_error;
            return false;
        }

        int const mesh_index =
            add_mesh_to_library(state, std::move(mesh), std::move(name), is_builtin);
        std::string activation_error;
        if (!activate_mesh_by_index(
                state, mesh_index, status_prefix, renderer, harnack_tracer, voxelizer,
                activation_error
            )) {
            remove_mesh_from_library(state, mesh_index);
            if (!has_active_mesh(state))
                harnack_texture_renderer.clear();
            state.status_line = "Activate failed: " + activation_error;
            return false;
        }
        return true;
    };

    if (!add_and_activate(build_default_mesh(), "Half Octahedron", true, ""))
        throw std::runtime_error("Failed to initialize default mesh for Harnack tracer.");
    (void)add_mesh_to_library(state, build_closed_octa_mesh(), "Closed Octahedron", true);
    state.status_line = "Ready";

    if (!cli.mesh_file.empty()) {
        winding_studio::LoadedMesh loaded{};
        std::string load_error;
        if (!winding_studio::load_mesh_from_file(cli.mesh_file, loaded, load_error))
            throw std::runtime_error("Failed to load --mesh-file: " + load_error);

        std::string const name = std::filesystem::path(cli.mesh_file).filename().string();
        if (!add_and_activate(to_mesh_data(loaded), name, false, "Loaded mesh: "))
            throw std::runtime_error("Failed to upload --mesh-file into tracer/voxelizer.");
    }

    float last_time = static_cast<float>(glfwGetTime());
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        float const now = static_cast<float>(glfwGetTime());
        float const dt = std::max(now - last_time, 1.0f / 240.0f);
        last_time = now;
        if (state.auto_rotate)
            state.yaw += 0.55f * dt;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        UiLayoutResult const ui_layout = draw_editor_layout(state, dt, dpi_scale);
        if (ui_layout.request_harnack_refresh)
            state.force_harnack_refresh = true;
        if (ui_layout.harnack_params_changed && state.harnack_live_update)
            state.force_harnack_refresh = true;
        if (ui_layout.voxel_params_changed || ui_layout.request_voxel_refresh)
            state.force_voxel_refresh = true;

        if (ui_layout.remove_mesh_index >= 0) {
            remove_mesh_from_library(state, ui_layout.remove_mesh_index);
            if (!has_active_mesh(state))
                harnack_texture_renderer.clear();
        }

        if (ui_layout.activate_mesh_index >= 0 &&
            ui_layout.activate_mesh_index != state.active_mesh_index) {
            std::string activation_error;
            if (!activate_mesh_by_index(
                    state, ui_layout.activate_mesh_index, "Active mesh: ", renderer, harnack_tracer,
                    voxelizer, activation_error
                )) {
                if (!has_active_mesh(state))
                    harnack_texture_renderer.clear();
                state.status_line = "Activate failed: " + activation_error;
            }
        }

        if (!ui_layout.mesh_file_to_add.empty()) {
            winding_studio::LoadedMesh loaded{};
            std::string load_error;
            if (winding_studio::load_mesh_from_file(
                    ui_layout.mesh_file_to_add, loaded, load_error
                )) {
                std::string const name =
                    std::filesystem::path(ui_layout.mesh_file_to_add).filename().string();
                (void)add_and_activate(to_mesh_data(loaded), name, false, "Loaded mesh: ");
            } else {
                state.status_line = "Load failed: " + load_error;
            }
        }

        int fb_w = 0;
        int fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        FramebufferRect const viewport =
            ui_viewport_to_framebuffer(ui_layout, io, *ImGui::GetMainViewport(), fb_w, fb_h);

        update_harnack_trace_if_needed(
            state, viewport, harnack_tracer, harnack_texture_renderer, trace_images
        );
        update_voxelization_if_needed(state, voxelizer, voxel_renderer, voxel_stats);

        glViewport(0, 0, fb_w, fb_h);
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.018f, 0.022f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render_viewport(state, renderer, harnack_texture_renderer, voxel_renderer, viewport);

        ImGui::Render();
        glViewport(0, 0, fb_w, fb_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

} // namespace winding_studio::app

#include "ws_app.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "harnack_tracer.hpp"
#include "imgui.h"
#include "mesh_loader.hpp"
#include "voxelizer.hpp"
#include "ws_core.hpp"
#include "ws_render.hpp"
#include "ws_ui.hpp"

namespace winding_studio::app {

static void glfw_error_callback(int error, char const *description) {
    std::cerr << "[GLFW] (" << error << ") " << description << "\n";
}

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
        glfwCreateWindow(cli.width, cli.height, "Winding Studio (Stage 1)", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("glfwCreateWindow failed");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");
    glGetError();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    float dpi_scale = 1.0f;
    if (GLFWmonitor *monitor = glfwGetPrimaryMonitor()) {
        float xscale = 1.0f, yscale = 1.0f;
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

    auto clear_active_mesh = [&]() {
        state.active_mesh_index = -1;
        state.active_mesh_name = "None";
        state.triangle_count = 0;
        state.force_harnack_refresh = false;
        state.force_voxel_refresh = false;
        state.harnack_hit_count = 0;
        state.harnack_pixel_count = 0;
        state.last_harnack_ms = 0.0f;
        state.voxel_occupied_count = 0;
        state.voxel_grid_total = 0;
        state.voxel_grid_nx = 1;
        state.voxel_grid_ny = 1;
        state.voxel_grid_nz = 1;
        state.last_voxel_ms = 0.0f;
        state.mesh_bounds = winding_studio::voxel::MeshBounds{};
    };

    auto make_unique_mesh_name = [&](std::string name) {
        if (name.empty())
            name = "Imported Mesh";
        auto name_exists = [&](std::string const &candidate) {
            for (MeshLibraryEntry const &entry : state.mesh_library)
                if (entry.name == candidate)
                    return true;
            return false;
        };
        if (!name_exists(name))
            return name;
        std::string const base = name;
        int suffix = 2;
        while (true) {
            std::ostringstream oss;
            oss << base << " (" << suffix << ")";
            std::string const candidate = oss.str();
            if (!name_exists(candidate))
                return candidate;
            ++suffix;
        }
    };

    auto add_mesh_to_library = [&](MeshData mesh, std::string name, bool const is_builtin) {
        MeshLibraryEntry entry{};
        entry.name = make_unique_mesh_name(std::move(name));
        entry.triangle_count = triangle_count(mesh);
        entry.mesh = std::move(mesh);
        entry.is_builtin = is_builtin;
        state.mesh_library.push_back(std::move(entry));
        return static_cast<int>(state.mesh_library.size() - 1u);
    };

    auto activate_mesh_by_index = [&](int const index, std::string const &status_prefix) {
        if (!has_valid_mesh_index(state, index))
            return false;
        int const previous_active_index = state.active_mesh_index;
        MeshLibraryEntry const &entry = state.mesh_library[static_cast<std::size_t>(index)];

        auto restore_previous_uploads = [&]() {
            if (!has_valid_mesh_index(state, previous_active_index))
                return;
            MeshLibraryEntry const &previous_entry =
                state.mesh_library[static_cast<std::size_t>(previous_active_index)];
            winding_studio::HostMeshSoA const previous_host = to_host_mesh_soa(previous_entry.mesh);
            std::string ignored_error;
            (void)harnack_tracer.upload_mesh(previous_host, ignored_error);
            ignored_error.clear();
            (void)voxelizer.upload_mesh(previous_host, ignored_error);
            renderer.upload_mesh(previous_entry.mesh);
        };

        winding_studio::HostMeshSoA const host_mesh = to_host_mesh_soa(entry.mesh);
        std::string tracer_error;
        if (!harnack_tracer.upload_mesh(host_mesh, tracer_error)) {
            restore_previous_uploads();
            state.status_line = "Harnack upload failed: " + tracer_error;
            return false;
        }
        std::string voxel_error;
        if (!voxelizer.upload_mesh(host_mesh, voxel_error)) {
            restore_previous_uploads();
            state.status_line = "Voxel upload failed: " + voxel_error;
            return false;
        }
        renderer.upload_mesh(entry.mesh);
        state.active_mesh_index = index;
        state.selected_mesh_index = index;
        state.active_mesh_name = entry.name;
        state.triangle_count = entry.triangle_count;
        state.force_harnack_refresh = true;
        state.force_voxel_refresh = true;
        state.harnack_hit_count = 0;
        state.harnack_pixel_count = 0;
        state.voxel_occupied_count = 0;
        state.voxel_grid_total = 0;
        state.voxel_grid_nx = 1;
        state.voxel_grid_ny = 1;
        state.voxel_grid_nz = 1;
        state.mesh_bounds = winding_studio::voxel::compute_mesh_bounds(
            entry.mesh.positions.data(), entry.mesh.positions.size() / 3u
        );
        if (!status_prefix.empty())
            state.status_line = status_prefix + entry.name;
        return true;
    };

    auto remove_mesh_from_library = [&](int const index) {
        if (!has_valid_mesh_index(state, index))
            return;
        MeshLibraryEntry removed = std::move(state.mesh_library[static_cast<std::size_t>(index)]);
        bool const removed_active = (index == state.active_mesh_index);
        state.mesh_library.erase(state.mesh_library.begin() + index);

        auto adjust_index_after_erase = [&](int &value) {
            if (value > index)
                --value;
            else if (value == index)
                value = -1;
        };
        adjust_index_after_erase(state.selected_mesh_index);
        adjust_index_after_erase(state.active_mesh_index);

        if (state.mesh_library.empty()) {
            clear_active_mesh();
            state.selected_mesh_index = -1;
            state.status_line = "Removed mesh: " + removed.name + ". Library is empty.";
            return;
        }
        if (removed_active) {
            clear_active_mesh();
            state.selected_mesh_index =
                std::clamp(index, 0, static_cast<int>(state.mesh_library.size()) - 1);
            state.status_line = "Removed mesh: " + removed.name + ". No active mesh.";
            return;
        }
        if (!has_valid_mesh_index(state, state.selected_mesh_index)) {
            state.selected_mesh_index =
                std::clamp(index, 0, static_cast<int>(state.mesh_library.size()) - 1);
        }
        state.status_line = "Removed mesh: " + removed.name;
    };

    int const half_octa_index = add_mesh_to_library(build_default_mesh(), "Half Octahedron", true);
    add_mesh_to_library(build_closed_octa_mesh(), "Closed Octahedron", true);
    state.selected_mesh_index = half_octa_index;
    if (!activate_mesh_by_index(half_octa_index, ""))
        throw std::runtime_error("Failed to initialize default mesh for Harnack tracer.");
    state.status_line = "Ready";

    if (!cli.mesh_file.empty()) {
        winding_studio::LoadedMesh loaded{};
        std::string load_error;
        if (!winding_studio::load_mesh_from_file(cli.mesh_file, loaded, load_error))
            throw std::runtime_error("Failed to load --mesh-file: " + load_error);
        std::string const name = std::filesystem::path(cli.mesh_file).filename().string();
        int const imported_index = add_mesh_to_library(to_mesh_data(loaded), name, false);
        if (!activate_mesh_by_index(imported_index, "Loaded mesh: "))
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

        if (ui_layout.remove_mesh_index >= 0)
            remove_mesh_from_library(ui_layout.remove_mesh_index);

        if (ui_layout.activate_mesh_index >= 0 &&
            ui_layout.activate_mesh_index != state.active_mesh_index) {
            if (!activate_mesh_by_index(ui_layout.activate_mesh_index, "Active mesh: "))
                state.status_line = "Activate failed.";
        }

        if (!ui_layout.mesh_file_to_add.empty()) {
            std::string const &mesh_path = ui_layout.mesh_file_to_add;
            winding_studio::LoadedMesh loaded{};
            std::string load_error;
            if (winding_studio::load_mesh_from_file(mesh_path, loaded, load_error)) {
                std::string const name = std::filesystem::path(mesh_path).filename().string();
                int const imported_index = add_mesh_to_library(to_mesh_data(loaded), name, false);
                if (!activate_mesh_by_index(imported_index, "Loaded mesh: ")) {
                    remove_mesh_from_library(imported_index);
                    state.status_line = "Upload failed: " + name;
                }
            } else {
                state.status_line = "Load failed: " + load_error;
            }
        }

        int fb_w = 0;
        int fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        FramebufferRect const viewport =
            ui_viewport_to_framebuffer(ui_layout, io, *ImGui::GetMainViewport(), fb_w, fb_h);

        bool const needs_harnack =
            state.view_mode == ViewMode::k_split || state.view_mode == ViewMode::k_harnack;
        if (needs_harnack && has_active_mesh(state) && harnack_tracer.has_mesh() &&
            viewport.w > 0 && viewport.h > 0) {
            bool const should_trace = state.force_harnack_refresh || state.harnack_live_update;
            if (should_trace) {
                int const base_w =
                    (state.view_mode == ViewMode::k_split) ? (viewport.w / 2) : viewport.w;
                int const trace_w = std::clamp(
                    static_cast<int>(std::lround(base_w * state.harnack_resolution_scale)), 64, 1600
                );
                int const trace_h = std::clamp(
                    static_cast<int>(std::lround(viewport.h * state.harnack_resolution_scale)), 64,
                    1200
                );

                winding_studio::CameraFrame const camera =
                    make_harnack_camera_frame(state, trace_w, trace_h);
                winding_studio::HarnackTraceConfig const config{
                    state.target_winding, state.epsilon,        state.max_iterations,
                    state.t_max,          state.accuracy_scale,
                };

                auto const trace_begin = std::chrono::steady_clock::now();
                std::string trace_error;
                bool const ok = harnack_tracer.trace(camera, config, trace_images, trace_error);
                auto const trace_end = std::chrono::steady_clock::now();
                state.last_harnack_ms =
                    std::chrono::duration<float, std::milli>(trace_end - trace_begin).count();

                if (ok) {
                    harnack_texture_renderer.upload_rgba(
                        trace_images.width, trace_images.height, trace_images.harnack_rgba
                    );
                    state.harnack_hit_count = trace_images.hit_count;
                    state.harnack_pixel_count = static_cast<std::size_t>(trace_images.width) *
                                                static_cast<std::size_t>(trace_images.height);
                } else {
                    state.status_line = "Harnack trace failed: " + trace_error;
                }
                state.force_harnack_refresh = false;
            }
        }

        bool const needs_voxel = state.view_mode == ViewMode::k_voxel;
        if (needs_voxel && has_active_mesh(state) && voxelizer.has_mesh() &&
            state.force_voxel_refresh) {
            winding_studio::voxel::VoxelGridSpec const grid =
                winding_studio::voxel::make_voxel_grid_from_dx(
                    state.mesh_bounds, state.voxel_dx, state.voxel_max_voxels
                );
            state.voxel_grid_nx = grid.nx;
            state.voxel_grid_ny = grid.ny;
            state.voxel_grid_nz = grid.nz;
            state.voxel_grid_total = static_cast<std::size_t>(grid.total_voxels);
            state.voxel_actual_dx = grid.actual_dx;

            if (!voxel_renderer.ensure_instance_capacity(state.voxel_grid_total)) {
                state.status_line = "Voxelize failed: cannot allocate instance buffer";
            } else {
                winding_studio::VoxelizeConfig const config{
                    state.voxel_target_w,
                    state.accuracy_scale,
                };

                auto const voxel_begin = std::chrono::steady_clock::now();
                std::string voxel_error;
                bool const ok = voxelizer.voxelize(
                    grid, config, voxel_renderer.instance_buffer(),
                    voxel_renderer.instance_capacity(), voxel_stats, voxel_error
                );
                auto const voxel_end = std::chrono::steady_clock::now();
                state.last_voxel_ms =
                    std::chrono::duration<float, std::milli>(voxel_end - voxel_begin).count();

                if (ok) {
                    state.voxel_occupied_count = voxel_stats.occupied_count;
                    state.voxel_grid_total = voxel_stats.total_voxels;
                } else {
                    state.status_line = "Voxelize failed: " + voxel_error;
                    state.voxel_occupied_count = 0;
                }
            }
            state.force_voxel_refresh = false;
        }

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

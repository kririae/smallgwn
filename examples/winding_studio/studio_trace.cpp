#include "studio_trace.hpp"

#include "studio_math.hpp"
#include "studio_mesh_library.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <string>

namespace winding_studio::app {

namespace {

[[nodiscard]] winding_studio::CameraFrame
make_harnack_camera_frame(AppState const &state, int const width, int const height) {
    CameraBasis const basis = build_camera_basis(state);
    winding_studio::CameraFrame camera{};
    camera.origin_x = basis.eye.x;
    camera.origin_y = basis.eye.y;
    camera.origin_z = basis.eye.z;
    camera.forward_x = basis.forward.x;
    camera.forward_y = basis.forward.y;
    camera.forward_z = basis.forward.z;
    camera.right_x = basis.right.x;
    camera.right_y = basis.right.y;
    camera.right_z = basis.right.z;
    camera.up_x = basis.ortho_up.x;
    camera.up_y = basis.ortho_up.y;
    camera.up_z = basis.ortho_up.z;
    camera.tan_half_fov = std::tan(0.5f * state.camera_fovy_radians);
    camera.aspect = static_cast<float>(width) / static_cast<float>(std::max(height, 1));
    camera.width = width;
    camera.height = height;
    return camera;
}

} // namespace

void update_harnack_trace_if_needed(
    AppState &state, FramebufferRect const viewport, winding_studio::HarnackTracer &harnack_tracer,
    TextureRenderer &harnack_texture_renderer, winding_studio::HarnackTraceImages &trace_images
) {
    bool const needs_harnack = state.view_mode == ViewMode::k_split || state.view_mode == ViewMode::k_harnack;
    if (!(needs_harnack && has_active_mesh(state) && harnack_tracer.has_mesh() && viewport.w > 0 && viewport.h > 0))
        return;

    bool const should_trace = state.force_harnack_refresh || state.harnack_live_update;
    if (!should_trace)
        return;

    int const base_w = (state.view_mode == ViewMode::k_split) ? (viewport.w / 2) : viewport.w;
    int const trace_w = std::clamp(
        static_cast<int>(std::lround(base_w * state.harnack_resolution_scale)), 64, 1600
    );
    int const trace_h = std::clamp(
        static_cast<int>(std::lround(viewport.h * state.harnack_resolution_scale)), 64, 1200
    );

    winding_studio::CameraFrame const camera = make_harnack_camera_frame(state, trace_w, trace_h);
    winding_studio::HarnackTraceConfig const config{
        state.target_winding,
        state.epsilon,
        state.max_iterations,
        state.t_max,
        state.accuracy_scale,
    };

    auto const trace_begin = std::chrono::steady_clock::now();
    std::string trace_error;
    bool const ok = harnack_tracer.trace(camera, config, trace_images, trace_error);
    auto const trace_end = std::chrono::steady_clock::now();
    state.last_harnack_ms = std::chrono::duration<float, std::milli>(trace_end - trace_begin).count();

    if (ok) {
        try {
            harnack_texture_renderer.upload_rgba(
                trace_images.width, trace_images.height, trace_images.harnack_rgba
            );
            state.harnack_hit_count = trace_images.hit_count;
            state.harnack_pixel_count = static_cast<std::size_t>(trace_images.width) *
                                        static_cast<std::size_t>(trace_images.height);
        } catch (std::exception const &e) {
            harnack_texture_renderer.clear();
            state.harnack_hit_count = 0;
            state.harnack_pixel_count = 0;
            state.status_line = std::string("Harnack texture upload failed: ") + e.what();
        }
    } else {
        harnack_texture_renderer.clear();
        state.harnack_hit_count = 0;
        state.harnack_pixel_count = 0;
        state.status_line = "Harnack trace failed: " + trace_error;
    }

    state.force_harnack_refresh = false;
}

} // namespace winding_studio::app

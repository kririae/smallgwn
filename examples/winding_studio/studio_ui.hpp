#pragma once

#include "studio_state.hpp"

#include "imgui.h"

namespace winding_studio::app {

/**
 * @brief Apply global UI style parameters.
 */
void apply_engine_style(float dpi_scale = 1.0f);

/**
 * @brief Draw the full editor layout and collect user actions for this frame.
 */
[[nodiscard]] UiLayoutResult draw_editor_layout(AppState &state, float dt, float ui_scale = 1.0f);

/**
 * @brief Convert UI viewport rectangle into framebuffer pixel coordinates.
 */
[[nodiscard]] FramebufferRect ui_viewport_to_framebuffer(
    UiLayoutResult const &layout, ImGuiIO const &io, ImGuiViewport const &main_viewport, int fb_w, int fb_h
);

} // namespace winding_studio::app

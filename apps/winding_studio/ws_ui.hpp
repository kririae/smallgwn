#pragma once

#include "ws_types.hpp"

namespace winding_studio::app {

void apply_engine_style(float dpi_scale = 1.0f);
[[nodiscard]] UiLayoutResult draw_editor_layout(AppState &state, float dt, float ui_scale = 1.0f);
[[nodiscard]] FramebufferRect ui_viewport_to_framebuffer(
    UiLayoutResult const &layout, ImGuiIO const &io, ImGuiViewport const &main_viewport, int fb_w, int fb_h
);

} // namespace winding_studio::app

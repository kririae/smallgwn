#pragma once

#include "harnack_tracer.hpp"
#include "studio_render.hpp"
#include "studio_state.hpp"

namespace winding_studio::app {

/**
 * @brief Refresh Harnack tracing output for current frame when needed.
 *
 * This updates `state` counters/status and clears stale texture content on trace failures.
 */
void update_harnack_trace_if_needed(
    AppState &state, FramebufferRect viewport, winding_studio::HarnackTracer &harnack_tracer,
    TextureRenderer &harnack_texture_renderer, winding_studio::HarnackTraceImages &trace_images
);

} // namespace winding_studio::app

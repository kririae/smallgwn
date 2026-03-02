#pragma once

#include "studio_domain.hpp"

#include <string>

namespace winding_studio::app {

/**
 * @brief Validate mesh layout and index bounds before upload/render.
 *
 * Returns false and fills @p error when any invariant is violated.
 */
[[nodiscard]] bool validate_mesh_data(MeshData const &mesh, std::string &error) noexcept;

} // namespace winding_studio::app

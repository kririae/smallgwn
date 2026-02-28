#pragma once

#include "ws_types.hpp"

#include <GL/glew.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace winding_studio::app {

class MeshRenderer final {
public:
    MeshRenderer();
    MeshRenderer(MeshRenderer const &) = delete;
    MeshRenderer &operator=(MeshRenderer const &) = delete;
    ~MeshRenderer();

    void upload_mesh(MeshData const &mesh);
    void draw(Mat4 const &mvp, Mat4 const &model, bool wireframe) const;

private:
    GLuint vao_{};
    GLuint vbo_{};
    GLuint program_{};
    GLsizei vertex_count_{};
};

class TextureRenderer final {
public:
    TextureRenderer();
    TextureRenderer(TextureRenderer const &) = delete;
    TextureRenderer &operator=(TextureRenderer const &) = delete;
    ~TextureRenderer();

    void upload_rgba(int width, int height, std::vector<std::uint8_t> const &rgba);
    [[nodiscard]] bool has_texture() const noexcept;
    void draw(int x, int y, int w, int h) const;

private:
    GLuint program_{};
    GLuint vao_{};
    GLuint texture_{};
    int tex_w_{0};
    int tex_h_{0};
    bool has_texture_{false};
};

class VoxelRenderer final {
public:
    VoxelRenderer();
    VoxelRenderer(VoxelRenderer const &) = delete;
    VoxelRenderer &operator=(VoxelRenderer const &) = delete;
    ~VoxelRenderer();

    [[nodiscard]] bool ensure_instance_capacity(std::size_t required);
    [[nodiscard]] unsigned int instance_buffer() const noexcept;
    [[nodiscard]] std::size_t instance_capacity() const noexcept;

    void draw(Mat4 const &vp, float voxel_size, std::size_t instance_count, bool wireframe) const;

private:
    struct Face {
        Vec3 normal;
        std::array<Vec3, 4> corners;
    };

    void upload_unit_cube();

    GLuint program_{0};
    GLuint vao_{0};
    GLuint vertex_vbo_{0};
    GLuint instance_vbo_{0};
    GLsizei vertex_count_{0};
    std::size_t instance_capacity_{0};
};

void render_viewport(
    AppState const &state, MeshRenderer const &renderer, TextureRenderer const &harnack_texture,
    VoxelRenderer const &voxel_renderer, FramebufferRect viewport
);

} // namespace winding_studio::app

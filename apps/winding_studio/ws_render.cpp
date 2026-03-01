#include "ws_render.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>
#include <tbb/parallel_for.h>

#include "ws_core.hpp"
#include "ws_math.hpp"

namespace winding_studio::app {

[[nodiscard]] static GLuint compile_shader(GLenum const type, char const *source) {
    GLuint const shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE)
        return shader;

    GLint log_len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<std::size_t>(std::max(log_len, 1)), '\0');
    glGetShaderInfoLog(shader, log_len, nullptr, log.data());
    std::ostringstream oss;
    oss << "Shader compile failed: " << log;
    glDeleteShader(shader);
    throw std::runtime_error(oss.str());
}

[[nodiscard]] static GLuint create_program() {
    static constexpr char const *k_vs = R"GLSL(
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
void main() {
    v_normal = normalize(mat3(u_model) * a_normal);
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
)GLSL";

    static constexpr char const *k_fs = R"GLSL(
#version 330 core
in vec3 v_normal;
out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);

    // Blender-style three-light setup.
    vec3 key_dir  = normalize(vec3( 0.5, 0.7,  0.5));
    vec3 fill_dir = normalize(vec3(-0.6, 0.3,  0.4));

    float key  = max(dot(n, key_dir), 0.0);
    float fill = max(dot(n, fill_dir), 0.0);
    float rim  = pow(clamp(1.0 - abs(dot(n, vec3(0.0, 0.0, 1.0))), 0.0, 1.0), 3.0);

    vec3 base     = vec3(0.80, 0.80, 0.82);
    vec3 key_col  = vec3(1.00, 0.98, 0.92);
    vec3 fill_col = vec3(0.50, 0.60, 0.78);
    vec3 rim_col  = vec3(0.85, 0.88, 0.95);

    vec3 lit = base * 0.18
             + base * key_col  * (key  * 0.62)
             + base * fill_col * (fill * 0.35)
             + rim_col * (rim * 0.22);

    frag_color = vec4(clamp(lit, 0.0, 1.0), 1.0);
}
)GLSL";

    GLuint const vs = compile_shader(GL_VERTEX_SHADER, k_vs);
    GLuint const fs = compile_shader(GL_FRAGMENT_SHADER, k_fs);

    GLuint const program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE)
        return program;

    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<std::size_t>(std::max(log_len, 1)), '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    std::ostringstream oss;
    oss << "Program link failed: " << log;
    glDeleteProgram(program);
    throw std::runtime_error(oss.str());
}

MeshRenderer::MeshRenderer() {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    program_ = create_program();
}

MeshRenderer::~MeshRenderer() {
    if (program_ != 0)
        glDeleteProgram(program_);
    if (vbo_ != 0)
        glDeleteBuffers(1, &vbo_);
    if (vao_ != 0)
        glDeleteVertexArrays(1, &vao_);
}

void MeshRenderer::upload_mesh(MeshData const &mesh) {
    std::size_t const tri_count = mesh.indices.size() / 3u;
    std::vector<float> verts(tri_count * 3u * 6u);
    float const *positions = mesh.positions.data();
    std::uint32_t const *indices = mesh.indices.data();

    tbb::parallel_for(std::size_t{0}, tri_count, [&](std::size_t const t) {
        std::uint32_t const idx[3] = {
            indices[t * 3u + 0u],
            indices[t * 3u + 1u],
            indices[t * 3u + 2u],
        };
        float const *a = &positions[idx[0] * 3u];
        float const *b = &positions[idx[1] * 3u];
        float const *c = &positions[idx[2] * 3u];

        float const ex = b[0] - a[0];
        float const ey = b[1] - a[1];
        float const ez = b[2] - a[2];
        float const fx = c[0] - a[0];
        float const fy = c[1] - a[1];
        float const fz = c[2] - a[2];
        float nx = ey * fz - ez * fy;
        float ny = ez * fx - ex * fz;
        float nz = ex * fy - ey * fx;
        float const len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-12f) {
            nx /= len;
            ny /= len;
            nz /= len;
        }

        std::size_t const base = t * 18u;
        for (int v = 0; v < 3; ++v) {
            float const *p = &positions[idx[v] * 3u];
            std::size_t const out = base + static_cast<std::size_t>(v) * 6u;
            verts[out + 0u] = p[0];
            verts[out + 1u] = p[1];
            verts[out + 2u] = p[2];
            verts[out + 3u] = nx;
            verts[out + 4u] = ny;
            verts[out + 5u] = nz;
        }
    });

    vertex_count_ = static_cast<GLsizei>(tri_count * 3u);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(float)), verts.data(),
        GL_STATIC_DRAW
    );
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
        reinterpret_cast<void const *>(3 * sizeof(float))
    );
    glBindVertexArray(0);
}

void MeshRenderer::draw(Mat4 const &mvp, Mat4 const &model, bool const wireframe) const {
    glUseProgram(program_);
    glBindVertexArray(vao_);

    glUniformMatrix4fv(glGetUniformLocation(program_, "u_mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(
        glGetUniformLocation(program_, "u_model"), 1, GL_FALSE, glm::value_ptr(model)
    );

    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindVertexArray(0);
    glUseProgram(0);
}

[[nodiscard]] static GLuint create_texture_program() {
    static constexpr char const *k_vs = R"GLSL(
#version 330 core
out vec2 v_uv;
void main() {
    vec2 pos;
    if (gl_VertexID == 0)
        pos = vec2(-1.0, -1.0);
    else if (gl_VertexID == 1)
        pos = vec2(3.0, -1.0);
    else
        pos = vec2(-1.0, 3.0);
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = 0.5 * (pos + 1.0);
}
)GLSL";

    static constexpr char const *k_fs = R"GLSL(
#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 frag_color;
void main() {
    frag_color = texture(u_texture, vec2(v_uv.x, 1.0 - v_uv.y));
}
)GLSL";

    GLuint const vs = compile_shader(GL_VERTEX_SHADER, k_vs);
    GLuint const fs = compile_shader(GL_FRAGMENT_SHADER, k_fs);
    GLuint const program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE)
        return program;

    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<std::size_t>(std::max(log_len, 1)), '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    std::ostringstream oss;
    oss << "Texture program link failed: " << log;
    glDeleteProgram(program);
    throw std::runtime_error(oss.str());
}

TextureRenderer::TextureRenderer() {
    program_ = create_texture_program();
    glGenVertexArrays(1, &vao_);
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

TextureRenderer::~TextureRenderer() {
    if (texture_ != 0)
        glDeleteTextures(1, &texture_);
    if (vao_ != 0)
        glDeleteVertexArrays(1, &vao_);
    if (program_ != 0)
        glDeleteProgram(program_);
}

void TextureRenderer::upload_rgba(
    int const width, int const height, std::vector<std::uint8_t> const &rgba
) {
    if (width <= 0 || height <= 0)
        return;
    if (rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u)
        throw std::runtime_error("upload_rgba size mismatch");

    glBindTexture(GL_TEXTURE_2D, texture_);
    if (width != tex_w_ || height != tex_h_) {
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data()
        );
        tex_w_ = width;
        tex_h_ = height;
    } else {
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data()
        );
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    has_texture_ = true;
}

[[nodiscard]] bool TextureRenderer::has_texture() const noexcept { return has_texture_; }

void TextureRenderer::draw(int const x, int const y, int const w, int const h) const {
    if (!has_texture_ || w <= 0 || h <= 0)
        return;
    glViewport(x, y, w, h);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(program_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(glGetUniformLocation(program_, "u_texture"), 0);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

[[nodiscard]] static GLuint create_voxel_program() {
    static constexpr char const *k_vs = R"GLSL(
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_instance;
uniform mat4 u_vp;
uniform float u_voxel_size;
out vec3 v_normal;
void main() {
    vec3 world_pos = a_instance.xyz + a_pos * (u_voxel_size * 0.5);
    v_normal = a_normal;
    gl_Position = u_vp * vec4(world_pos, 1.0);
}
)GLSL";

    static constexpr char const *k_fs = R"GLSL(
#version 330 core
in vec3 v_normal;
out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);

    vec3 key_dir  = normalize(vec3( 0.5, 0.7,  0.5));
    vec3 fill_dir = normalize(vec3(-0.6, 0.3,  0.4));

    float key  = max(dot(n, key_dir), 0.0);
    float fill = max(dot(n, fill_dir), 0.0);
    float rim  = pow(clamp(1.0 - abs(dot(n, vec3(0.0, 0.0, 1.0))), 0.0, 1.0), 3.0);

    vec3 base     = vec3(0.80, 0.82, 0.86);
    vec3 key_col  = vec3(1.00, 0.98, 0.92);
    vec3 fill_col = vec3(0.50, 0.60, 0.78);
    vec3 rim_col  = vec3(0.85, 0.88, 0.95);

    vec3 lit = base * 0.18
             + base * key_col  * (key  * 0.62)
             + base * fill_col * (fill * 0.35)
             + rim_col * (rim * 0.22);

    frag_color = vec4(clamp(lit, 0.0, 1.0), 1.0);
}
)GLSL";

    GLuint const vs = compile_shader(GL_VERTEX_SHADER, k_vs);
    GLuint const fs = compile_shader(GL_FRAGMENT_SHADER, k_fs);
    GLuint const program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE)
        return program;

    GLint log_len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
    std::string log(static_cast<std::size_t>(std::max(log_len, 1)), '\0');
    glGetProgramInfoLog(program, log_len, nullptr, log.data());
    std::ostringstream oss;
    oss << "Voxel program link failed: " << log;
    glDeleteProgram(program);
    throw std::runtime_error(oss.str());
}

VoxelRenderer::VoxelRenderer() {
    program_ = create_voxel_program();
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vertex_vbo_);
    glGenBuffers(1, &instance_vbo_);
    upload_unit_cube();
    if (!ensure_instance_capacity(1))
        throw std::runtime_error("Failed to allocate initial voxel instance buffer.");
}

VoxelRenderer::~VoxelRenderer() {
    if (instance_vbo_ != 0)
        glDeleteBuffers(1, &instance_vbo_);
    if (vertex_vbo_ != 0)
        glDeleteBuffers(1, &vertex_vbo_);
    if (vao_ != 0)
        glDeleteVertexArrays(1, &vao_);
    if (program_ != 0)
        glDeleteProgram(program_);
}

[[nodiscard]] bool VoxelRenderer::ensure_instance_capacity(std::size_t const required) {
    if (required <= instance_capacity_)
        return true;
    std::size_t new_capacity = std::max<std::size_t>(required, 1024u);
    if (new_capacity > std::numeric_limits<GLsizeiptr>::max() / sizeof(float) / 4u)
        return false;

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(new_capacity * sizeof(float) * 4u), nullptr,
        GL_DYNAMIC_DRAW
    );
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glVertexAttribDivisor(2, 1);
    glBindVertexArray(0);

    instance_capacity_ = new_capacity;
    return true;
}

[[nodiscard]] unsigned int VoxelRenderer::instance_buffer() const noexcept { return instance_vbo_; }

[[nodiscard]] std::size_t VoxelRenderer::instance_capacity() const noexcept {
    return instance_capacity_;
}

void VoxelRenderer::draw(
    Mat4 const &vp, float const voxel_size, std::size_t const instance_count, bool wireframe
) const {
    if (instance_count == 0u || voxel_size <= 0.0f)
        return;

    glEnable(GL_DEPTH_TEST);
    glUseProgram(program_);
    glBindVertexArray(vao_);
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_vp"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform1f(glGetUniformLocation(program_, "u_voxel_size"), voxel_size);

    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
    glDrawArraysInstanced(
        GL_TRIANGLES, 0, vertex_count_,
        static_cast<GLsizei>(
            std::min<std::size_t>(instance_count, std::numeric_limits<GLsizei>::max())
        )
    );
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindVertexArray(0);
    glUseProgram(0);
}

void VoxelRenderer::upload_unit_cube() {
    std::array<Face, 6> const faces{{
        {Vec3{1.0f, 0.0f, 0.0f}, {Vec3{1, -1, -1}, Vec3{1, -1, 1}, Vec3{1, 1, 1}, Vec3{1, 1, -1}}},
        {Vec3{-1.0f, 0.0f, 0.0f},
         {Vec3{-1, -1, 1}, Vec3{-1, -1, -1}, Vec3{-1, 1, -1}, Vec3{-1, 1, 1}}},
        {Vec3{0.0f, 1.0f, 0.0f}, {Vec3{-1, 1, -1}, Vec3{1, 1, -1}, Vec3{1, 1, 1}, Vec3{-1, 1, 1}}},
        {Vec3{0.0f, -1.0f, 0.0f},
         {Vec3{-1, -1, 1}, Vec3{1, -1, 1}, Vec3{1, -1, -1}, Vec3{-1, -1, -1}}},
        {Vec3{0.0f, 0.0f, 1.0f}, {Vec3{-1, -1, 1}, Vec3{-1, 1, 1}, Vec3{1, 1, 1}, Vec3{1, -1, 1}}},
        {Vec3{0.0f, 0.0f, -1.0f},
         {Vec3{1, -1, -1}, Vec3{1, 1, -1}, Vec3{-1, 1, -1}, Vec3{-1, -1, -1}}},
    }};

    std::vector<float> verts{};
    verts.reserve(36u * 6u);
    auto append_vertex = [&](Vec3 const &p, Vec3 const &n) {
        verts.push_back(p.x);
        verts.push_back(p.y);
        verts.push_back(p.z);
        verts.push_back(n.x);
        verts.push_back(n.y);
        verts.push_back(n.z);
    };

    for (Face const &face : faces) {
        append_vertex(face.corners[0], face.normal);
        append_vertex(face.corners[1], face.normal);
        append_vertex(face.corners[2], face.normal);
        append_vertex(face.corners[2], face.normal);
        append_vertex(face.corners[3], face.normal);
        append_vertex(face.corners[0], face.normal);
    }

    vertex_count_ = static_cast<GLsizei>(verts.size() / 6u);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(float)), verts.data(),
        GL_STATIC_DRAW
    );
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
        reinterpret_cast<void const *>(3 * sizeof(float))
    );
    glBindVertexArray(0);
}

void render_viewport(
    AppState const &state, MeshRenderer const &renderer, TextureRenderer const &harnack_texture,
    VoxelRenderer const &voxel_renderer, FramebufferRect const viewport
) {
    if (viewport.w <= 0 || viewport.h <= 0)
        return;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);
    glScissor(viewport.x, viewport.y, viewport.w, viewport.h);
    glClearColor(0.082f, 0.088f, 0.105f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    if (!has_active_mesh(state))
        return;

    CameraBasis const camera = build_camera_basis(state);
    Mat4 const view = mat4_look_at(camera.eye, camera.target, camera.up);
    Mat4 const model = mat4_identity();

    auto render_view = [&](int const x, int const y, int const w, int const h) {
        if (w <= 0 || h <= 0)
            return;
        glViewport(x, y, w, h);
        Mat4 const proj = mat4_perspective(
            45.0f * (k_pi / 180.0f), static_cast<float>(w) / static_cast<float>(h), 0.1f, 30.0f
        );
        Mat4 const vp = mat4_mul(proj, view);
        Mat4 const mvp = mat4_mul(vp, model);
        renderer.draw(mvp, model, state.wireframe);
    };

    switch (state.view_mode) {
    case ViewMode::k_split: {
        int const left_w = viewport.w / 2;
        int const right_w = viewport.w - left_w;
        render_view(viewport.x, viewport.y, left_w, viewport.h);
        if (harnack_texture.has_texture())
            harnack_texture.draw(viewport.x + left_w, viewport.y, right_w, viewport.h);
        else
            render_view(viewport.x + left_w, viewport.y, right_w, viewport.h);

        // Split divider drawn later via ImGui draw list.
    } break;
    case ViewMode::k_raster: render_view(viewport.x, viewport.y, viewport.w, viewport.h); break;
    case ViewMode::k_harnack:
        if (harnack_texture.has_texture())
            harnack_texture.draw(viewport.x, viewport.y, viewport.w, viewport.h);
        else
            render_view(viewport.x, viewport.y, viewport.w, viewport.h);
        break;
    case ViewMode::k_voxel: {
        glViewport(viewport.x, viewport.y, viewport.w, viewport.h);
        Mat4 const proj = mat4_perspective(
            45.0f * (k_pi / 180.0f),
            static_cast<float>(viewport.w) / static_cast<float>(viewport.h), 0.1f, 30.0f
        );
        Mat4 const vp = mat4_mul(proj, view);
        voxel_renderer.draw(vp, state.voxel_actual_dx, state.voxel_occupied_count, state.wireframe);
    } break;
    }
}

} // namespace winding_studio::app

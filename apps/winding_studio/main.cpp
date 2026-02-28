#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "harnack_tracer.hpp"
#include "imfilebrowser.h"
#include "imgui.h"
#include "mesh_loader.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stb_image_write.h"

namespace {

constexpr float k_pi = 3.14159265358979323846f;

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

struct Mat4 {
    // Column-major matrix layout for OpenGL.
    std::array<float, 16> v{};
};

enum class MeshPreset : int {
    k_half_octa = 0,
    k_closed_octa = 1,
    k_external = 2,
};

enum class ViewMode : int {
    k_split = 0,
    k_raster = 1,
    k_harnack = 2,
};

struct CliOptions {
    bool show_help{false};
    int width{1600};
    int height{960};
    int frames{4};
    bool capture_png{false};
    std::string capture_path{"artifacts/opengl-headless/winding_studio_720p.png"};
    MeshPreset mesh{MeshPreset::k_half_octa};
    ViewMode view_mode{ViewMode::k_split};
    std::string mesh_file{};
};

struct AppState {
    MeshPreset mesh = MeshPreset::k_half_octa;
    bool external_mesh_available = false;
    bool harnack_live_update = true;
    ViewMode view_mode = ViewMode::k_split;
    bool auto_rotate = true;
    bool wireframe = false;
    float yaw = 0.0f;
    float pitch = -0.35f;
    float camera_radius = 2.7f;
    Vec3 camera_target{0.0f, 0.0f, 0.0f};
    float epsilon = 1e-3f;
    float t_max = 100.0f;
    int max_iterations = 2048;
    float accuracy_scale = 2.0f;
    float target_winding = 0.5f;
    float harnack_resolution_scale = 0.75f;
    std::size_t triangle_count = 0;
    std::size_t harnack_hit_count = 0;
    std::size_t harnack_pixel_count = 0;
    float last_harnack_ms = 0.0f;
    std::string active_mesh_name{"Open Half Octa"};
    std::string external_mesh_name{};
    std::string status_line{"Ready"};
    char mesh_file_input[1024]{};
    bool force_harnack_refresh = true;
    ImGui::FileBrowser file_browser{
        ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter
    };
    bool file_browser_initialized = false;
};

struct MeshData {
    std::vector<float> positions;
    std::vector<std::uint32_t> indices;
};

[[nodiscard]] winding_studio::HostMeshSoA to_host_mesh_soa(MeshData const &mesh) {
    winding_studio::HostMeshSoA out{};
    std::size_t const vertex_count = mesh.positions.size() / 3u;
    std::size_t const triangle_count = mesh.indices.size() / 3u;
    out.vx.resize(vertex_count);
    out.vy.resize(vertex_count);
    out.vz.resize(vertex_count);
    out.i0.resize(triangle_count);
    out.i1.resize(triangle_count);
    out.i2.resize(triangle_count);

    for (std::size_t i = 0; i < vertex_count; ++i) {
        out.vx[i] = mesh.positions[i * 3u + 0u];
        out.vy[i] = mesh.positions[i * 3u + 1u];
        out.vz[i] = mesh.positions[i * 3u + 2u];
    }
    for (std::size_t i = 0; i < triangle_count; ++i) {
        out.i0[i] = mesh.indices[i * 3u + 0u];
        out.i1[i] = mesh.indices[i * 3u + 1u];
        out.i2[i] = mesh.indices[i * 3u + 2u];
    }
    return out;
}

[[nodiscard]] MeshData to_mesh_data(winding_studio::LoadedMesh const &mesh) {
    MeshData out{};
    out.positions = mesh.positions;
    out.indices = mesh.indices;
    return out;
}

[[nodiscard]] std::size_t triangle_count(MeshData const &mesh) { return mesh.indices.size() / 3u; }

[[nodiscard]] Mat4 mat4_identity() {
    Mat4 m{};
    m.v[0] = 1.0f;
    m.v[5] = 1.0f;
    m.v[10] = 1.0f;
    m.v[15] = 1.0f;
    return m;
}

[[nodiscard]] Mat4 mat4_mul(Mat4 const &a, Mat4 const &b) {
    Mat4 r{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += a.v[k * 4 + row] * b.v[col * 4 + k];
            r.v[col * 4 + row] = sum;
        }
    }
    return r;
}

[[nodiscard]] Vec3 vec3_sub(Vec3 const &a, Vec3 const &b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] float vec3_dot(Vec3 const &a, Vec3 const &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 vec3_cross(Vec3 const &a, Vec3 const &b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

[[nodiscard]] Vec3 vec3_normalize(Vec3 const &a) {
    float const len2 = vec3_dot(a, a);
    if (!(len2 > 0.0f))
        return Vec3{0.0f, 0.0f, 1.0f};
    float const inv_len = 1.0f / std::sqrt(len2);
    return Vec3{a.x * inv_len, a.y * inv_len, a.z * inv_len};
}

[[nodiscard]] Mat4 mat4_rotate_x(float const radians) {
    Mat4 m = mat4_identity();
    float const c = std::cos(radians);
    float const s = std::sin(radians);
    m.v[5] = c;
    m.v[6] = s;
    m.v[9] = -s;
    m.v[10] = c;
    return m;
}

[[nodiscard]] Mat4 mat4_rotate_y(float const radians) {
    Mat4 m = mat4_identity();
    float const c = std::cos(radians);
    float const s = std::sin(radians);
    m.v[0] = c;
    m.v[2] = -s;
    m.v[8] = s;
    m.v[10] = c;
    return m;
}

[[nodiscard]] Mat4 mat4_perspective(
    float const fovy_radians, float const aspect, float const z_near, float const z_far
) {
    Mat4 m{};
    float const f = 1.0f / std::tan(0.5f * fovy_radians);
    m.v[0] = f / aspect;
    m.v[5] = f;
    m.v[10] = (z_far + z_near) / (z_near - z_far);
    m.v[11] = -1.0f;
    m.v[14] = (2.0f * z_far * z_near) / (z_near - z_far);
    return m;
}

[[nodiscard]] Mat4 mat4_look_at(Vec3 const &eye, Vec3 const &target, Vec3 const &up_hint) {
    Vec3 const forward = vec3_normalize(vec3_sub(target, eye));
    Vec3 const right = vec3_normalize(vec3_cross(forward, up_hint));
    Vec3 const up = vec3_cross(right, forward);

    Mat4 m = mat4_identity();
    m.v[0] = right.x;
    m.v[1] = up.x;
    m.v[2] = -forward.x;
    m.v[4] = right.y;
    m.v[5] = up.y;
    m.v[6] = -forward.y;
    m.v[8] = right.z;
    m.v[9] = up.z;
    m.v[10] = -forward.z;
    m.v[12] = -vec3_dot(right, eye);
    m.v[13] = -vec3_dot(up, eye);
    m.v[14] = vec3_dot(forward, eye);
    return m;
}

[[nodiscard]] MeshData build_mesh(MeshPreset const preset) {
    if (preset == MeshPreset::k_closed_octa) {
        MeshData mesh{};
        mesh.positions = {
            1.0f, 0.0f,  0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, -1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f, -1.0f,
        };
        mesh.indices = {
            0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
        };
        return mesh;
    }

    MeshData mesh{};
    mesh.positions = {
        1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    mesh.indices = {
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
    };
    return mesh;
}

[[nodiscard]] GLuint compile_shader(GLenum const type, char const *source) {
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

[[nodiscard]] GLuint create_program() {
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

class MeshRenderer final {
public:
    MeshRenderer() {
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        program_ = create_program();
    }

    MeshRenderer(MeshRenderer const &) = delete;
    MeshRenderer &operator=(MeshRenderer const &) = delete;

    ~MeshRenderer() {
        if (program_ != 0)
            glDeleteProgram(program_);
        if (vbo_ != 0)
            glDeleteBuffers(1, &vbo_);
        if (vao_ != 0)
            glDeleteVertexArrays(1, &vao_);
    }

    void upload_mesh(MeshData const &mesh) {
        std::size_t const tri_count = mesh.indices.size() / 3u;
        std::vector<float> verts;
        verts.reserve(tri_count * 3u * 6u);

        for (std::size_t t = 0; t < tri_count; ++t) {
            std::uint32_t const idx[3] = {
                mesh.indices[t * 3 + 0],
                mesh.indices[t * 3 + 1],
                mesh.indices[t * 3 + 2],
            };
            float const *a = &mesh.positions[idx[0] * 3];
            float const *b = &mesh.positions[idx[1] * 3];
            float const *c = &mesh.positions[idx[2] * 3];

            float const ex = b[0] - a[0], ey = b[1] - a[1], ez = b[2] - a[2];
            float const fx = c[0] - a[0], fy = c[1] - a[1], fz = c[2] - a[2];
            float nx = ey * fz - ez * fy;
            float ny = ez * fx - ex * fz;
            float nz = ex * fy - ey * fx;
            float const len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-12f) {
                nx /= len;
                ny /= len;
                nz /= len;
            }

            for (int v = 0; v < 3; ++v) {
                float const *p = &mesh.positions[idx[v] * 3];
                verts.insert(verts.end(), {p[0], p[1], p[2], nx, ny, nz});
            }
        }

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

    void draw(Mat4 const &mvp, Mat4 const &model, bool const wireframe) const {
        glUseProgram(program_);
        glBindVertexArray(vao_);

        glUniformMatrix4fv(glGetUniformLocation(program_, "u_mvp"), 1, GL_FALSE, mvp.v.data());
        glUniformMatrix4fv(glGetUniformLocation(program_, "u_model"), 1, GL_FALSE, model.v.data());

        glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
        glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glBindVertexArray(0);
        glUseProgram(0);
    }

private:
    GLuint vao_{};
    GLuint vbo_{};
    GLuint program_{};
    GLsizei vertex_count_{};
};

[[nodiscard]] GLuint create_texture_program() {
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

class TextureRenderer final {
public:
    TextureRenderer() {
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

    TextureRenderer(TextureRenderer const &) = delete;
    TextureRenderer &operator=(TextureRenderer const &) = delete;

    ~TextureRenderer() {
        if (texture_ != 0)
            glDeleteTextures(1, &texture_);
        if (vao_ != 0)
            glDeleteVertexArrays(1, &vao_);
        if (program_ != 0)
            glDeleteProgram(program_);
    }

    void upload_rgba(int const width, int const height, std::vector<std::uint8_t> const &rgba) {
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

    [[nodiscard]] bool has_texture() const noexcept { return has_texture_; }

    void draw(int const x, int const y, int const w, int const h) const {
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

private:
    GLuint program_{};
    GLuint vao_{};
    GLuint texture_{};
    int tex_w_{0};
    int tex_h_{0};
    bool has_texture_{false};
};

[[nodiscard]] bool parse_int(std::string const &s, int &v) {
    try {
        std::size_t consumed = 0;
        int const parsed = std::stoi(s, &consumed);
        if (consumed != s.size())
            return false;
        v = parsed;
        return true;
    } catch (...) { return false; }
}

[[nodiscard]] bool parse_mesh(std::string const &value, MeshPreset &mesh) {
    if (value == "half") {
        mesh = MeshPreset::k_half_octa;
        return true;
    }
    if (value == "octa") {
        mesh = MeshPreset::k_closed_octa;
        return true;
    }
    return false;
}

[[nodiscard]] bool parse_view(std::string const &value, ViewMode &view_mode) {
    if (value == "split") {
        view_mode = ViewMode::k_split;
        return true;
    }
    if (value == "raster") {
        view_mode = ViewMode::k_raster;
        return true;
    }
    if (value == "harnack") {
        view_mode = ViewMode::k_harnack;
        return true;
    }
    return false;
}

void print_help(char const *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --width <int>           Window/frame width (default: 1600)\n"
        << "  --height <int>          Window/frame height (default: 960)\n"
        << "  --mesh <half|octa>      Initial geometry preset\n"
        << "  --mesh-file <path>      Load external mesh file (libigl formats, OBJ fallback)\n"
        << "  --view <split|raster|harnack>  Initial view mode\n"
        << "  --capture-png <path>    Run N frames then capture OpenGL UI to PNG and exit\n"
        << "  --frames <int>          Frame count before capture (default: 4)\n"
        << "  --help                  Show this message\n";
}

[[nodiscard]] bool parse_cli(int const argc, char **argv, CliOptions &opt) {
    for (int i = 1; i < argc; ++i) {
        std::string const key(argv[i]);
        auto read_value = [&](std::string &out) -> bool {
            if (i + 1 >= argc)
                return false;
            out = argv[++i];
            return true;
        };

        if (key == "--help") {
            opt.show_help = true;
            return true;
        }
        if (key == "--width") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.width))
                return false;
            continue;
        }
        if (key == "--height") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.height))
                return false;
            continue;
        }
        if (key == "--frames") {
            std::string value;
            if (!read_value(value) || !parse_int(value, opt.frames))
                return false;
            continue;
        }
        if (key == "--capture-png") {
            if (!read_value(opt.capture_path))
                return false;
            opt.capture_png = true;
            continue;
        }
        if (key == "--mesh") {
            std::string value;
            if (!read_value(value) || !parse_mesh(value, opt.mesh))
                return false;
            continue;
        }
        if (key == "--mesh-file") {
            if (!read_value(opt.mesh_file))
                return false;
            continue;
        }
        if (key == "--view") {
            std::string value;
            if (!read_value(value) || !parse_view(value, opt.view_mode))
                return false;
            continue;
        }
        return false;
    }
    return opt.width > 0 && opt.height > 0 && opt.frames > 0;
}

void glfw_error_callback(int error, char const *description) {
    std::cerr << "[GLFW] (" << error << ") " << description << "\n";
}

struct UiLayoutResult {
    bool mesh_changed{false};
    bool harnack_params_changed{false};
    bool request_mesh_file_load{false};
    bool request_harnack_refresh{false};
    ImVec2 viewport_pos{0.0f, 0.0f};
    ImVec2 viewport_size{1.0f, 1.0f};
};

struct FramebufferRect {
    int x{0};
    int y{0};
    int w{0};
    int h{0};
};

[[nodiscard]] char const *mesh_name(MeshPreset const mesh) {
    if (mesh == MeshPreset::k_half_octa)
        return "Open Half Octa";
    if (mesh == MeshPreset::k_closed_octa)
        return "Closed Octa";
    return "External Mesh";
}

[[nodiscard]] char const *view_mode_name(ViewMode const view_mode) {
    switch (view_mode) {
    case ViewMode::k_split: return "Split";
    case ViewMode::k_raster: return "Raster";
    case ViewMode::k_harnack: return "Harnack";
    }
    return "Unknown";
}

void apply_engine_style(float const dpi_scale = 1.0f) {
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.PopupRounding = 4.0f;

    style.WindowPadding = ImVec2(10.0f, 8.0f);
    style.FramePadding = ImVec2(8.0f, 5.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.IndentSpacing = 16.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.ScrollbarSize = 12.0f;
    style.ScrollbarRounding = 3.0f;
    style.GrabMinSize = 8.0f;

    ImVec4 *c = style.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.071f, 0.086f, 1.0f);
    c[ImGuiCol_ChildBg] = ImVec4(0.082f, 0.088f, 0.105f, 1.0f);
    c[ImGuiCol_PopupBg] = ImVec4(0.075f, 0.080f, 0.098f, 0.96f);
    c[ImGuiCol_Border] = ImVec4(0.18f, 0.22f, 0.28f, 0.65f);
    c[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TitleBg] = ImVec4(0.065f, 0.072f, 0.092f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.085f, 0.10f, 0.14f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.12f, 0.16f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.14f, 0.17f, 0.24f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.17f, 0.22f, 0.32f, 1.0f);
    c[ImGuiCol_Button] = ImVec4(0.13f, 0.19f, 0.26f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.18f, 0.28f, 0.40f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.22f, 0.38f, 0.56f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.12f, 0.18f, 0.26f, 1.0f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.16f, 0.26f, 0.38f, 1.0f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.34f, 0.50f, 1.0f);
    c[ImGuiCol_Separator] = ImVec4(0.20f, 0.25f, 0.32f, 0.6f);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.30f, 0.45f, 0.65f, 0.8f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.35f, 0.55f, 0.80f, 1.0f);
    c[ImGuiCol_CheckMark] = ImVec4(0.40f, 0.82f, 1.0f, 1.0f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.60f, 0.85f, 1.0f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.40f, 0.75f, 1.0f, 1.0f);
    c[ImGuiCol_Text] = ImVec4(0.86f, 0.90f, 0.94f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.42f, 0.48f, 0.56f, 1.0f);
    c[ImGuiCol_Tab] = c[ImGuiCol_Header];
    c[ImGuiCol_TabHovered] = c[ImGuiCol_HeaderHovered];
    c[ImGuiCol_TabSelected] = ImVec4(0.18f, 0.30f, 0.45f, 1.0f);

    if (dpi_scale > 1.01f)
        style.ScaleAllSizes(dpi_scale);
}

struct CameraBasis {
    Vec3 eye{};
    Vec3 target{};
    Vec3 up{};
    Vec3 forward{};
    Vec3 right{};
    Vec3 ortho_up{};
};

[[nodiscard]] CameraBasis build_camera_basis(AppState const &state) {
    CameraBasis basis{};
    basis.eye = Vec3{
        state.camera_target.x + std::sin(state.yaw) * state.camera_radius,
        state.camera_target.y + std::sin(state.pitch) * state.camera_radius + 0.3f,
        state.camera_target.z + std::cos(state.yaw) * state.camera_radius,
    };
    basis.target = state.camera_target;
    basis.up = Vec3{0.0f, 1.0f, 0.0f};
    basis.forward = vec3_normalize(vec3_sub(basis.target, basis.eye));
    basis.right = vec3_normalize(vec3_cross(basis.forward, basis.up));
    basis.ortho_up = vec3_cross(basis.right, basis.forward);
    return basis;
}

[[nodiscard]] bool begin_collapsing_section(char const *label, bool const default_open = true) {
    ImGui::Spacing();
    ImGuiTreeNodeFlags const flags = default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0;
    bool const open = ImGui::CollapsingHeader(label, flags);
    if (open)
        ImGui::Spacing();
    return open;
}

void item_tooltip(char const *text) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(300.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

bool begin_property_table(char const *id, float label_w = 90.0f) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_NoPadOuterX))
        return false;
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, label_w);
    ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

void property_label(char const *label) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
}

void end_property_table() { ImGui::EndTable(); }

[[nodiscard]] bool
mode_button(char const *label, bool const is_active, ImVec2 const size = ImVec2(0.0f, 0.0f)) {
    if (is_active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.36f, 0.55f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.42f, 0.62f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.28f, 0.48f, 0.70f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.90f, 1.0f, 1.0f));
    }
    bool const pressed = ImGui::Button(label, size);
    if (is_active)
        ImGui::PopStyleColor(4);
    return pressed;
}

// Vertically center the cursor for a single row of widgets inside a
// fixed-height child window.  Call once right after BeginChild.
//   content_h = 0  → use frame height (text + 2*FramePadding, good for bars with buttons)
//   content_h < 0  → use bare text line height (good for text-only bars)
//   content_h > 0  → use that exact value
void vcenter_cursor(float content_h = 0.0f) {
    if (content_h <= 0.0f)
        content_h = (content_h < 0.0f) ? ImGui::GetTextLineHeight() : ImGui::GetFrameHeight();
    float const avail_h = ImGui::GetContentRegionAvail().y;
    if (avail_h > content_h)
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail_h - content_h) * 0.5f);
}

void status_segment(char const *label, char const *value) {
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::TextUnformatted(value);
}

[[nodiscard]] UiLayoutResult
draw_editor_layout(AppState &state, float const dt, float const ui_scale = 1.0f) {
    UiLayoutResult result{};
    (void)dt;

    ImGuiViewport const *main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->Pos);
    ImGui::SetNextWindowSize(main_viewport->Size);

    ImGuiWindowFlags const root_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::Begin("Winding Studio Root", nullptr, root_flags);
    ImGui::PopStyleVar(3);

    ImVec2 const root_avail = ImGui::GetContentRegionAvail();
    float const s = ui_scale;
    float const toolbar_h = std::clamp(root_avail.y * 0.055f, 34.0f * s, 52.0f * s);
    float const statusbar_h = std::clamp(root_avail.y * 0.045f, 28.0f * s, 38.0f * s);
    float right_w = std::clamp(root_avail.x * 0.27f, 280.0f * s, 430.0f * s);
    float const spacing = ImGui::GetStyle().ItemSpacing.x;

    float const min_center_w = 320.0f * s;
    if (right_w > root_avail.x - min_center_w - spacing)
        right_w = std::max(200.0f * s, root_avail.x - min_center_w - spacing);

    ImGui::BeginChild(
        "Toolbar", ImVec2(0.0f, toolbar_h), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );
    vcenter_cursor();
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
    ImGui::TextUnformatted("Winding Studio");
    ImGui::PopStyleColor();

    float const min_mode_button_w = 76.0f * s;
    float const max_mode_button_w = 124.0f * s;
    float const title_reserve = ImGui::CalcTextSize("Winding Studio").x + 28.0f * s;
    float const mode_button_w = std::clamp(
        (ImGui::GetWindowContentRegionMax().x - title_reserve - spacing * 3.0f) / 4.0f,
        min_mode_button_w, max_mode_button_w
    );
    float const buttons_w = mode_button_w * 3.0f + spacing * 2.0f;
    float const btn_start =
        std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - buttons_w);
    ImGui::SameLine(btn_start);
    if (mode_button("Split", state.view_mode == ViewMode::k_split, ImVec2(mode_button_w, 0.0f)))
        state.view_mode = ViewMode::k_split;
    ImGui::SameLine();
    if (mode_button("Raster", state.view_mode == ViewMode::k_raster, ImVec2(mode_button_w, 0.0f)))
        state.view_mode = ViewMode::k_raster;
    ImGui::SameLine();
    if (mode_button("Harnack", state.view_mode == ViewMode::k_harnack, ImVec2(mode_button_w, 0.0f)))
        state.view_mode = ViewMode::k_harnack;

    ImGui::EndChild();

    float const workspace_h =
        std::max(120.0f * s, ImGui::GetContentRegionAvail().y - statusbar_h - spacing);
    ImGui::BeginChild(
        "Workspace", ImVec2(0.0f, workspace_h), false,
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse
    );

    float const panel_h = std::max(120.0f * s, ImGui::GetContentRegionAvail().y);

    float const center_w =
        std::max(120.0f * s, ImGui::GetContentRegionAvail().x - right_w - spacing);
    ImGui::BeginChild(
        "ViewportPanel", ImVec2(center_w, panel_h), false,
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse
    );
    result.viewport_pos = ImGui::GetCursorScreenPos();
    result.viewport_size = ImGui::GetContentRegionAvail();
    result.viewport_size.x = std::max(result.viewport_size.x, 32.0f);
    result.viewport_size.y = std::max(result.viewport_size.y, 32.0f);
    ImGui::InvisibleButton("ViewportCanvas", result.viewport_size);

    // -- Blender-style viewport mouse interaction --
    bool const viewport_hovered = ImGui::IsItemHovered();
    bool const viewport_active = ImGui::IsItemActive();
    ImGuiIO &interact_io = ImGui::GetIO();

    if (viewport_hovered || viewport_active) {
        // Scroll = zoom
        if (interact_io.MouseWheel != 0.0f) {
            state.camera_radius *= (1.0f - interact_io.MouseWheel * 0.1f);
            state.camera_radius = std::clamp(state.camera_radius, 0.5f, 12.0f);
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        }

        // Left-drag: orbit or pan
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
            ImVec2 const delta = interact_io.MouseDelta;
            if (interact_io.KeyShift) {
                // Shift + left-drag = pan
                CameraBasis const cam = build_camera_basis(state);
                float const pan_speed = 0.003f * state.camera_radius;
                state.camera_target.x -=
                    (cam.right.x * delta.x + cam.ortho_up.x * delta.y) * pan_speed;
                state.camera_target.y -=
                    (cam.right.y * delta.x + cam.ortho_up.y * delta.y) * pan_speed;
                state.camera_target.z -=
                    (cam.right.z * delta.x + cam.ortho_up.z * delta.y) * pan_speed;
            } else {
                // Left-drag = orbit
                state.yaw -= delta.x * 0.005f;
                state.pitch -= delta.y * 0.005f;
                state.pitch = std::clamp(state.pitch, -1.4f, 1.4f);
            }
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        }

        // Middle-drag = orbit (Blender convention)
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 2.0f)) {
            ImVec2 const delta = interact_io.MouseDelta;
            if (interact_io.KeyShift) {
                CameraBasis const cam = build_camera_basis(state);
                float const pan_speed = 0.003f * state.camera_radius;
                state.camera_target.x -=
                    (cam.right.x * delta.x + cam.ortho_up.x * delta.y) * pan_speed;
                state.camera_target.y -=
                    (cam.right.y * delta.x + cam.ortho_up.y * delta.y) * pan_speed;
                state.camera_target.z -=
                    (cam.right.z * delta.x + cam.ortho_up.z * delta.y) * pan_speed;
            } else {
                state.yaw -= delta.x * 0.005f;
                state.pitch -= delta.y * 0.005f;
                state.pitch = std::clamp(state.pitch, -1.4f, 1.4f);
            }
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        }
    }

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 const p0 = result.viewport_pos;
    ImVec2 const p1 = ImVec2(
        result.viewport_pos.x + result.viewport_size.x,
        result.viewport_pos.y + result.viewport_size.y
    );
    float const corner_r = ImGui::GetStyle().ChildRounding;
    draw_list->AddRect(p0, p1, IM_COL32(50, 65, 85, 120), corner_r, 0, 0.6f);

    // Split divider line + per-side labels.
    if (state.view_mode == ViewMode::k_split) {
        float const mid_x = p0.x + result.viewport_size.x * 0.5f;
        draw_list->AddLine(
            ImVec2(mid_x, p0.y), ImVec2(mid_x, p1.y), IM_COL32(180, 190, 210, 60), 1.0f
        );
        ImU32 const label_col = IM_COL32(140, 170, 200, 140);
        draw_list->AddText(ImVec2(p0.x + 6.0f * s, p0.y + 4.0f * s), label_col, "Raster");
        ImVec2 const h_size = ImGui::CalcTextSize("Harnack");
        draw_list->AddText(
            ImVec2(p1.x - h_size.x - 6.0f * s, p0.y + 4.0f * s), label_col, "Harnack"
        );
    } else {
        // Mode label in top-left corner of viewport.
        draw_list->AddText(
            ImVec2(p0.x + 6.0f * s, p0.y + 4.0f * s), IM_COL32(140, 170, 200, 140),
            view_mode_name(state.view_mode)
        );
    }

    // Subtle interaction hint at bottom when hovered
    if (viewport_hovered && !viewport_active) {
        draw_list->AddText(
            ImVec2(p0.x + 6.0f * s, p1.y - 18.0f * s), IM_COL32(130, 150, 170, 90),
            "LMB: orbit | Shift: pan | Scroll: zoom"
        );
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("InspectorPanel", ImVec2(0.0f, panel_h), true);

    if (begin_collapsing_section("Geometry", true)) {
        constexpr char const *k_mesh_items[] = {"Open Half Octa", "Closed Octa"};
        int mesh_index = (state.mesh == MeshPreset::k_closed_octa) ? 1 : 0;
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::Combo("##Geometry", &mesh_index, k_mesh_items, IM_ARRAYSIZE(k_mesh_items))) {
            state.mesh = static_cast<MeshPreset>(mesh_index);
            result.mesh_changed = true;
        }
        item_tooltip("Select the built-in geometry preset");

        if (state.external_mesh_available) {
            bool use_external = (state.mesh == MeshPreset::k_external);
            if (ImGui::Checkbox("Use loaded mesh", &use_external)) {
                state.mesh = use_external ? MeshPreset::k_external : MeshPreset::k_half_octa;
                result.mesh_changed = true;
            }
            item_tooltip("Switch to the externally loaded mesh file");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextDisabled("Load external mesh");

        if (!state.file_browser_initialized) {
            state.file_browser.SetTitle("Open Mesh File");
            state.file_browser.SetTypeFilters({".obj", ".ply", ".stl", ".off", ".*"});
            state.file_browser_initialized = true;
        }

        float const fp = ImGui::GetStyle().FramePadding.x;
        float const browse_w = ImGui::CalcTextSize("Browse").x + fp * 2.0f;
        float const load_w = ImGui::CalcTextSize("Load").x + fp * 2.0f;
        float const btns_w = browse_w + spacing + load_w + spacing;
        ImGui::SetNextItemWidth(-btns_w);
        ImGui::InputText("##MeshFile", state.mesh_file_input, sizeof(state.mesh_file_input));
        item_tooltip("Path to an OBJ or other mesh file");
        ImGui::SameLine();
        if (ImGui::Button("Browse"))
            state.file_browser.Open();
        item_tooltip("Open a file browser to select a mesh");
        ImGui::SameLine();
        if (ImGui::Button("Load"))
            result.request_mesh_file_load = true;
        item_tooltip("Load the mesh file from the path above");
    }

    if (begin_collapsing_section("Camera", true)) {
        if (begin_property_table("##CameraProps", 90.0f * s)) {
            property_label("Yaw");
            result.harnack_params_changed |=
                ImGui::SliderFloat("##CameraYaw", &state.yaw, -k_pi, k_pi, "%.3f rad");
            item_tooltip("Horizontal camera rotation angle (radians)");

            property_label("Pitch");
            result.harnack_params_changed |=
                ImGui::SliderFloat("##CameraPitch", &state.pitch, -1.4f, 1.4f, "%.3f rad");
            item_tooltip("Vertical camera tilt angle (radians)");

            property_label("Distance");
            result.harnack_params_changed |=
                ImGui::SliderFloat("##CameraDist", &state.camera_radius, 0.5f, 12.0f, "%.3f");
            item_tooltip("Camera distance from the focus target");

            end_property_table();
        }

        ImGui::Spacing();
        ImGui::Checkbox("Wireframe", &state.wireframe);
        item_tooltip("Toggle wireframe overlay on raster view");
        ImGui::SameLine();
        if (ImGui::Button("Reset Camera")) {
            state.yaw = 0.0f;
            state.pitch = -0.35f;
            state.camera_radius = 2.7f;
            state.camera_target = Vec3{0.0f, 0.0f, 0.0f};
            state.auto_rotate = true;
            result.harnack_params_changed = true;
        }
        item_tooltip("Reset camera to default position and re-enable auto-rotate");

        CameraBasis const camera = build_camera_basis(state);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        if (begin_property_table("##CameraReadout", 60.0f * s)) {
            property_label("Eye");
            ImGui::Text("%.2f  %.2f  %.2f", camera.eye.x, camera.eye.y, camera.eye.z);
            property_label("Target");
            ImGui::Text("%.2f  %.2f  %.2f", camera.target.x, camera.target.y, camera.target.z);
            property_label("Forward");
            ImGui::Text("%.2f  %.2f  %.2f", camera.forward.x, camera.forward.y, camera.forward.z);
            property_label("Rotate");
            ImGui::TextUnformatted(state.auto_rotate ? "Auto" : "Manual");
            end_property_table();
        }
    }

    if (begin_collapsing_section("Harnack Trace", true)) {
        if (begin_property_table("##HarnackProps", 90.0f * s)) {
            property_label("Target W");
            result.harnack_params_changed |=
                ImGui::SliderFloat("##TargetW", &state.target_winding, 0.1f, 0.9f, "%.2f");
            item_tooltip(
                "Target winding number iso-value for surface extraction (0.5 = standard surface)"
            );

            property_label("Epsilon");
            result.harnack_params_changed |= ImGui::SliderFloat(
                "##Epsilon", &state.epsilon, 1e-5f, 1e-2f, "%.5f", ImGuiSliderFlags_Logarithmic
            );
            item_tooltip("Convergence threshold. Smaller = more precise but slower");

            property_label("Max Iters");
            result.harnack_params_changed |=
                ImGui::SliderInt("##MaxIters", &state.max_iterations, 16, 4096);
            item_tooltip("Maximum Harnack iterations per ray");

            property_label("Accuracy");
            result.harnack_params_changed |=
                ImGui::SliderFloat("##Accuracy", &state.accuracy_scale, 0.8f, 6.0f, "%.2f");
            item_tooltip("BVH traversal accuracy multiplier. Higher = more precise but slower");

            property_label("Trace t_max");
            result.harnack_params_changed |=
                ImGui::SliderFloat("##TraceT", &state.t_max, 5.0f, 500.0f, "%.1f");
            item_tooltip("Maximum ray distance. Increase for large scenes");

            property_label("Resolution");
            result.harnack_params_changed |= ImGui::SliderFloat(
                "##Resolution", &state.harnack_resolution_scale, 0.2f, 1.0f, "%.2f"
            );
            item_tooltip("Trace resolution relative to viewport. Lower = faster preview");

            end_property_table();
        }

        ImGui::Spacing();
        result.harnack_params_changed |= ImGui::Checkbox("Live Update", &state.harnack_live_update);
        item_tooltip("Continuously re-trace when parameters change. Disable for manual control");
        ImGui::SameLine();
        if (ImGui::Button("Refresh"))
            result.request_harnack_refresh = true;
        item_tooltip("Force a single Harnack trace refresh");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.42f, 0.48f, 0.56f, 1.0f));
    ImGui::TextWrapped("%s", state.status_line.c_str());
    ImGui::PopStyleColor();

    ImGui::EndChild();

    ImGui::EndChild();

    ImGui::BeginChild(
        "StatusBar", ImVec2(0.0f, statusbar_h), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );
    vcenter_cursor(-1.0f);

    char buf[64];

    status_segment("Mode:", view_mode_name(state.view_mode));
    ImGui::SameLine(0.0f, 16.0f);

    std::snprintf(
        buf, sizeof(buf), "%s (%zu tris)", state.active_mesh_name.c_str(), state.triangle_count
    );
    status_segment("Mesh:", buf);
    ImGui::SameLine(0.0f, 16.0f);

    std::snprintf(buf, sizeof(buf), "%.0f x %.0f", result.viewport_size.x, result.viewport_size.y);
    status_segment("Viewport:", buf);

    bool const show_harnack_stats =
        state.view_mode == ViewMode::k_harnack || state.view_mode == ViewMode::k_split;
    if (show_harnack_stats) {
        ImGui::SameLine(0.0f, 16.0f);
        std::snprintf(
            buf, sizeof(buf), "%zu / %zu", state.harnack_hit_count, state.harnack_pixel_count
        );
        status_segment("Hits:", buf);
        ImGui::SameLine(0.0f, 16.0f);
        std::snprintf(buf, sizeof(buf), "%.2f ms", state.last_harnack_ms);
        status_segment("Trace:", buf);
    }

    ImGui::EndChild();

    ImGui::End();

    state.file_browser.Display();
    if (state.file_browser.HasSelected()) {
        std::string const path = state.file_browser.GetSelected().string();
        std::snprintf(state.mesh_file_input, sizeof(state.mesh_file_input), "%s", path.c_str());
        state.file_browser.ClearSelected();
        result.request_mesh_file_load = true;
    }

    return result;
}

[[nodiscard]] FramebufferRect ui_viewport_to_framebuffer(
    UiLayoutResult const &layout, ImGuiIO const &io, ImGuiViewport const &main_viewport,
    int const fb_w, int const fb_h
) {
    float const scale_x = io.DisplayFramebufferScale.x > 0.0f ? io.DisplayFramebufferScale.x : 1.0f;
    float const scale_y = io.DisplayFramebufferScale.y > 0.0f ? io.DisplayFramebufferScale.y : 1.0f;

    int const x =
        static_cast<int>(std::lround((layout.viewport_pos.x - main_viewport.Pos.x) * scale_x));
    int const y_top =
        static_cast<int>(std::lround((layout.viewport_pos.y - main_viewport.Pos.y) * scale_y));
    int const w = std::max(0, static_cast<int>(std::lround(layout.viewport_size.x * scale_x)));
    int const h = std::max(0, static_cast<int>(std::lround(layout.viewport_size.y * scale_y)));
    int const y = fb_h - (y_top + h);

    int const x0 = std::clamp(x, 0, fb_w);
    int const y0 = std::clamp(y, 0, fb_h);
    int const x1 = std::clamp(x + w, 0, fb_w);
    int const y1 = std::clamp(y + h, 0, fb_h);
    return FramebufferRect{x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
}

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
    camera.tan_half_fov = std::tan(0.5f * 45.0f * (k_pi / 180.0f));
    camera.aspect = static_cast<float>(width) / static_cast<float>(std::max(height, 1));
    camera.width = width;
    camera.height = height;
    return camera;
}

void render_viewport(
    AppState const &state, MeshRenderer const &renderer, TextureRenderer const &harnack_texture,
    FramebufferRect const viewport
) {
    if (viewport.w <= 0 || viewport.h <= 0)
        return;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);
    glScissor(viewport.x, viewport.y, viewport.w, viewport.h);
    glClearColor(0.082f, 0.088f, 0.105f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

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
    }
}

[[nodiscard]] std::vector<std::uint8_t> read_backbuffer_rgba(int const width, int const height) {
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u
    );
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    return pixels;
}

[[nodiscard]] std::vector<std::uint8_t>
flip_vertical_rgba(std::vector<std::uint8_t> const &src, int const width, int const height) {
    std::vector<std::uint8_t> dst(src.size());
    std::size_t const row_size = static_cast<std::size_t>(width) * 4u;
    for (int y = 0; y < height; ++y) {
        std::size_t const src_offset = static_cast<std::size_t>(height - 1 - y) * row_size;
        std::size_t const dst_offset = static_cast<std::size_t>(y) * row_size;
        std::copy_n(src.data() + src_offset, row_size, dst.data() + dst_offset);
    }
    return dst;
}

void ensure_parent_directory(std::string const &path) {
    std::filesystem::path p(path);
    std::filesystem::path const parent = p.parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
}

[[nodiscard]] bool write_png_rgba(
    std::string const &path, int const width, int const height,
    std::vector<std::uint8_t> const &rgba
) {
    return stbi_write_png(path.c_str(), width, height, 4, rgba.data(), width * 4) != 0;
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
    if (cli.capture_png)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow *window =
        glfwCreateWindow(cli.width, cli.height, "Winding Studio (Stage 1)", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("glfwCreateWindow failed");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(cli.capture_png ? 0 : 1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");
    glGetError();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Query DPI scale from the primary monitor.
    float dpi_scale = 1.0f;
    if (GLFWmonitor *monitor = glfwGetPrimaryMonitor()) {
        float xscale = 1.0f, yscale = 1.0f;
        glfwGetMonitorContentScale(monitor, &xscale, &yscale);
        dpi_scale = std::max(xscale, yscale);
    }

    apply_engine_style(dpi_scale);

    // Load a scaled font if available, fall back to default ProggyClean.
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
    state.mesh = cli.mesh;
    state.view_mode = cli.view_mode;
    std::string const default_mesh_path = "";
    std::string const initial_mesh_file = cli.mesh_file.empty() ? default_mesh_path : cli.mesh_file;
    std::snprintf(
        state.mesh_file_input, sizeof(state.mesh_file_input), "%s", initial_mesh_file.c_str()
    );

    MeshRenderer renderer{};
    TextureRenderer harnack_texture_renderer{};
    winding_studio::HarnackTracer harnack_tracer{};
    winding_studio::HarnackTraceImages trace_images{};

    MeshData external_mesh{};

    auto apply_active_mesh = [&](MeshData const &mesh, MeshPreset const preset,
                                 std::string const &name) {
        renderer.upload_mesh(mesh);
        std::string tracer_error;
        if (!harnack_tracer.upload_mesh(to_host_mesh_soa(mesh), tracer_error)) {
            state.status_line = "Harnack upload failed: " + tracer_error;
            return false;
        }
        state.mesh = preset;
        state.active_mesh_name = name;
        state.triangle_count = triangle_count(mesh);
        state.force_harnack_refresh = true;
        state.harnack_hit_count = 0;
        state.harnack_pixel_count = 0;
        state.status_line = "Loaded mesh: " + name;
        return true;
    };

    if (!apply_active_mesh(build_mesh(state.mesh), state.mesh, std::string(mesh_name(state.mesh))))
        throw std::runtime_error("Failed to initialize built-in mesh for Harnack tracer.");
    if (!cli.mesh_file.empty()) {
        winding_studio::LoadedMesh loaded{};
        std::string load_error;
        if (!winding_studio::load_mesh_from_file(cli.mesh_file, loaded, load_error))
            throw std::runtime_error("Failed to load --mesh-file: " + load_error);
        external_mesh = to_mesh_data(loaded);
        state.external_mesh_available = true;
        state.external_mesh_name = std::filesystem::path(cli.mesh_file).filename().string();
        if (!apply_active_mesh(external_mesh, MeshPreset::k_external, state.external_mesh_name))
            throw std::runtime_error("Failed to upload external mesh for Harnack tracer.");
    }

    float last_time = static_cast<float>(glfwGetTime());
    int frame_counter = 0;

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
        if (ui_layout.harnack_params_changed || ui_layout.request_harnack_refresh)
            state.force_harnack_refresh = true;

        if (ui_layout.request_mesh_file_load) {
            std::string const mesh_path(state.mesh_file_input);
            if (mesh_path.empty()) {
                state.status_line = "Mesh load failed: empty path.";
            } else {
                winding_studio::LoadedMesh loaded{};
                std::string load_error;
                if (winding_studio::load_mesh_from_file(mesh_path, loaded, load_error)) {
                    external_mesh = to_mesh_data(loaded);
                    state.external_mesh_available = true;
                    state.external_mesh_name = std::filesystem::path(mesh_path).filename().string();
                    if (!apply_active_mesh(
                            external_mesh, MeshPreset::k_external, state.external_mesh_name
                        )) {
                        state.status_line = "Mesh load succeeded but tracer upload failed: " +
                                            state.external_mesh_name;
                    }
                } else {
                    state.status_line = "Mesh load failed: " + load_error;
                }
            }
        }

        if (ui_layout.mesh_changed) {
            if (state.mesh == MeshPreset::k_half_octa) {
                (void)apply_active_mesh(
                    build_mesh(MeshPreset::k_half_octa), MeshPreset::k_half_octa, "Open Half Octa"
                );
            } else if (state.mesh == MeshPreset::k_closed_octa) {
                (void)apply_active_mesh(
                    build_mesh(MeshPreset::k_closed_octa), MeshPreset::k_closed_octa, "Closed Octa"
                );
            } else if (state.mesh == MeshPreset::k_external && state.external_mesh_available) {
                (void)apply_active_mesh(
                    external_mesh, MeshPreset::k_external, state.external_mesh_name
                );
            }
        }

        int fb_w = 0;
        int fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        FramebufferRect const viewport =
            ui_viewport_to_framebuffer(ui_layout, io, *ImGui::GetMainViewport(), fb_w, fb_h);

        bool const needs_harnack =
            state.view_mode == ViewMode::k_split || state.view_mode == ViewMode::k_harnack;
        if (needs_harnack && harnack_tracer.has_mesh() && viewport.w > 0 && viewport.h > 0) {
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

        glViewport(0, 0, fb_w, fb_h);
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.018f, 0.022f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render_viewport(state, renderer, harnack_texture_renderer, viewport);

        ImGui::Render();
        glViewport(0, 0, fb_w, fb_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);

        ++frame_counter;
        if (cli.capture_png && frame_counter >= cli.frames) {
            std::vector<std::uint8_t> const rgba = read_backbuffer_rgba(fb_w, fb_h);
            std::vector<std::uint8_t> const flipped = flip_vertical_rgba(rgba, fb_w, fb_h);
            ensure_parent_directory(cli.capture_path);
            if (!write_png_rgba(cli.capture_path, fb_w, fb_h, flipped))
                throw std::runtime_error("failed to write PNG: " + cli.capture_path);
            std::cout << "Captured OpenGL UI frame: " << cli.capture_path << "\n";
            std::cout << "Resolution: " << fb_w << "x" << fb_h << "\n";
            break;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        CliOptions cli{};
        if (!parse_cli(argc, argv, cli)) {
            std::cerr << "Invalid options.\n";
            print_help(argv[0]);
            return 1;
        }
        if (cli.show_help) {
            print_help(argv[0]);
            return 0;
        }
        return run_app(cli);
    } catch (std::exception const &e) {
        std::cerr << "error: " << e.what() << "\n";
        glfwTerminate();
        return 1;
    }
}

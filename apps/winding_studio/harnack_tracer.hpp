#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace winding_studio {

struct HostMeshSoA {
    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> vz;
    std::vector<std::uint32_t> i0;
    std::vector<std::uint32_t> i1;
    std::vector<std::uint32_t> i2;
};

struct HarnackTraceConfig {
    float target_winding{0.5f};
    float epsilon{1e-3f};
    int max_iterations{2048};
    float t_max{100.0f};
    float accuracy_scale{2.0f};
};

struct CameraFrame {
    float origin_x{0.0f};
    float origin_y{0.0f};
    float origin_z{2.0f};
    float forward_x{0.0f};
    float forward_y{0.0f};
    float forward_z{-1.0f};
    float right_x{1.0f};
    float right_y{0.0f};
    float right_z{0.0f};
    float up_x{0.0f};
    float up_y{1.0f};
    float up_z{0.0f};
    float tan_half_fov{0.41421356f};
    float aspect{1.0f};
    int width{1};
    int height{1};
};

struct HarnackTraceImages {
    int width{0};
    int height{0};
    std::size_t hit_count{0};
    std::vector<std::uint8_t> harnack_rgba;
};

class HarnackTracer final {
public:
    HarnackTracer();
    ~HarnackTracer();

    HarnackTracer(HarnackTracer &&other) noexcept;
    HarnackTracer &operator=(HarnackTracer &&other) noexcept;

    HarnackTracer(HarnackTracer const &) = delete;
    HarnackTracer &operator=(HarnackTracer const &) = delete;

    [[nodiscard]] bool has_mesh() const noexcept;

    [[nodiscard]] bool upload_mesh(HostMeshSoA const &mesh, std::string &error);
    [[nodiscard]] bool trace(
        CameraFrame const &camera, HarnackTraceConfig const &config, HarnackTraceImages &out,
        std::string &error
    );

private:
    class Impl;
    Impl *impl_{nullptr};
};

} // namespace winding_studio

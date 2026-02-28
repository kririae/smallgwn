#include "ws_cli.hpp"

#include <iostream>
#include <string>

namespace winding_studio::app {

[[nodiscard]] static bool parse_int(std::string const &s, int &v) {
    try {
        std::size_t consumed = 0;
        int const parsed = std::stoi(s, &consumed);
        if (consumed != s.size())
            return false;
        v = parsed;
        return true;
    } catch (...) { return false; }
}

[[nodiscard]] static bool parse_float(std::string const &s, float &v) {
    try {
        std::size_t consumed = 0;
        float const parsed = std::stof(s, &consumed);
        if (consumed != s.size())
            return false;
        v = parsed;
        return true;
    } catch (...) { return false; }
}

[[nodiscard]] static bool parse_view(std::string const &value, ViewMode &view_mode) {
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
    if (value == "voxel") {
        view_mode = ViewMode::k_voxel;
        return true;
    }
    return false;
}

void print_help(char const *argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "Options:\n"
              << "  --width <int>           Window/frame width (default: 1600)\n"
              << "  --height <int>          Window/frame height (default: 960)\n"
              << "  --mesh-file <path>      Load mesh file (libigl formats, OBJ fallback)\n"
              << "  --view <split|raster|harnack|voxel>  Initial view mode\n"
              << "  --voxel-dx <float>      Initial voxel dx (voxel mode)\n"
              << "  --camera-distance <f>   Initial camera distance\n"
              << "  --harnack-resolution <f>  Harnack trace resolution scale [0.2,1.0]\n"
              << "  --harnack-target-w <f>  Initial Harnack target winding\n"
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
        if (key == "--voxel-dx") {
            std::string value;
            if (!read_value(value) || !parse_float(value, opt.voxel_dx) || !(opt.voxel_dx > 0.0f))
                return false;
            continue;
        }
        if (key == "--camera-distance") {
            std::string value;
            if (!read_value(value) || !parse_float(value, opt.camera_distance) ||
                !(opt.camera_distance > 0.0f))
                return false;
            continue;
        }
        if (key == "--harnack-resolution") {
            std::string value;
            if (!read_value(value) || !parse_float(value, opt.harnack_resolution_scale) ||
                !(opt.harnack_resolution_scale >= 0.2f && opt.harnack_resolution_scale <= 1.0f))
                return false;
            continue;
        }
        if (key == "--harnack-target-w") {
            std::string value;
            if (!read_value(value) || !parse_float(value, opt.harnack_target_w) ||
                !(opt.harnack_target_w > 0.0f))
                return false;
            continue;
        }
        return false;
    }
    return opt.width > 0 && opt.height > 0;
}

} // namespace winding_studio::app

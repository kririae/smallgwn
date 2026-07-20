#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include <gtest/gtest.h>

#include "test_utils.cuh"

int main(int argc, char **argv) try {
    std::filesystem::path mesh_directory;
    int output_arg = 1;
    for (int input_arg = 1; input_arg < argc; ++input_arg) {
        std::string_view const argument(argv[input_arg]);
        if (argument == "--mesh-dir") {
            if (++input_arg >= argc)
                throw std::invalid_argument("--mesh-dir requires a directory path");
            mesh_directory = argv[input_arg];
            continue;
        }
        constexpr std::string_view prefix = "--mesh-dir=";
        if (argument.starts_with(prefix)) {
            mesh_directory = argument.substr(prefix.size());
            continue;
        }
        argv[output_arg++] = argv[input_arg];
    }
    argc = output_arg;

    if (mesh_directory.empty())
        throw std::invalid_argument("--mesh-dir is required");
    if (!std::filesystem::is_directory(mesh_directory))
        throw std::invalid_argument("--mesh-dir must name an existing directory");
    gwn::tests::set_mesh_directory(std::move(mesh_directory));
    if (gwn::tests::collect_mesh_paths().empty())
        throw std::invalid_argument("--mesh-dir must contain at least one .ply mesh");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} catch (std::exception const &error) {
    std::cerr << "smallgwn test setup failed: " << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "smallgwn test setup failed with an unknown exception\n";
    return 1;
}

#include <exception>
#include <iostream>

#include "ws_app.hpp"
#include "ws_cli.hpp"

int main(int argc, char **argv) {
    try {
        winding_studio::app::CliOptions cli{};
        if (!winding_studio::app::parse_cli(argc, argv, cli)) {
            std::cerr << "Invalid options.\n";
            winding_studio::app::print_help(argv[0]);
            return 1;
        }
        if (cli.show_help) {
            winding_studio::app::print_help(argv[0]);
            return 0;
        }
        return winding_studio::app::run_app(cli);
    } catch (std::exception const &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

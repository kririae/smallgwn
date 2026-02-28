#pragma once

#include "ws_types.hpp"

namespace winding_studio::app {

void print_help(char const *argv0);
[[nodiscard]] bool parse_cli(int argc, char **argv, CliOptions &opt);

} // namespace winding_studio::app

#pragma once

// Clang CUDA runtime wrapper in clangd may include this legacy CUDA header.
// CUDA 13 toolchains can omit it, so we provide a no-op shim for analysis only.

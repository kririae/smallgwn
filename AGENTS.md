# AGENTS.md

## Scope
These instructions apply to the `smallgwn/` project tree.

## Language and Build Baseline
- Use C++20 and CUDA 12+.
- Target `sm_86+`.
- Keep the project header-only at the library level.

## File Naming Rules
- If a header contains CUDA callable annotations (`__device__`, `__global__`, or mixed host/device kernel interfaces), use the `.cuh` extension.
- Keep pure host-side helpers in `.hpp`.

## Formatting Rules
- Use `clang-format` with Chromium style via `smallgwn/.clang-format`.
- Run format after each code change touching C++/CUDA files.

## Architecture Rules
- Keep public include surface focused on runtime components; CPU reference helpers belong under `tests/`.
- Use SoA layout for geometry (`x/y/z`, `i0/i1/i2`).
- Prefer TBB for trivially parallel CPU-side batch work.

## Validation
- Configure/build: `cmake -S smallgwn -B smallgwn/build && cmake --build smallgwn/build -j`
- Run tests: `ctest --test-dir smallgwn/build --output-on-failure`

## Git Commit Style
- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, etc.).

# Winding Studio E2E Framework

This folder defines the **end-to-end quality pipeline** for `winding_studio`.

The framework follows a layered approach so correctness does not depend on screenshot diffs.

## Testing Layers

1. `unit`
- Scope: pure logic/data behavior
- Examples: mesh validation, mesh library indexing/removal
- Runs fast and deterministic

2. `ui_contract`
- Scope: ImGui interaction contracts with **state assertions**
- Strategy: run scripted interactions through **Dear ImGui Test Engine** and assert emitted intents / mutated UI state
- Examples: toolbar mode switching, mesh list activation intent, remove-button intent
- This is the main replacement for brittle image-only checks

3. `visual` (optional)
- Scope: coarse visual smoke snapshots
- Purpose: quick manual sanity check for rendering regressions
- Not used as primary correctness gate

## Why This Structure

Image comparisons are useful, but they are noisy for behavior checks:
- tiny text/stat changes can pass/fail for the wrong reason
- viewport and font differences create false positives

`ui_contract` tests assert semantics directly (state + intent), while visual smoke remains a lightweight human-facing safety net.

## Dear ImGui Test Engine Integration

- Source: [ocornut/imgui_test_engine](https://github.com/ocornut/imgui_test_engine)
- The test target is `winding_studio_test_ui_contract`
- Registered tests live in `examples/winding_studio/tests/ui_contract_tests.cpp`
- Runner lives in `examples/winding_studio/tests/ui_contract_runner.cpp`

Notes:
- `imgui_test_engine` has its own license terms for the engine folder (free for open-source and small teams; see upstream LICENSE).
- This repository is open-source, so this integration path is valid for your requested workflow.

## Run Locally

```bash
# Run unit + ui_contract layers
examples/winding_studio/e2e/run.sh

# Include optional visual captures
examples/winding_studio/e2e/run.sh --visual
```

Useful options:

```bash
examples/winding_studio/e2e/run.sh --build-dir /tmp/ws-build
examples/winding_studio/e2e/run.sh --skip-build
```

If your network cannot clone dependencies directly, configure with a local checkout:

```bash
cmake -S examples/winding_studio -B examples/winding_studio/build \
  -DBUILD_TESTING=ON \
  -DWINDING_STUDIO_ENABLE_IMGUI_TEST_ENGINE=ON \
  -DWINDING_STUDIO_IMGUI_TEST_ENGINE_SOURCE_DIR=/abs/path/to/imgui_test_engine
```

## CI Recommendation

For CI gate, run:

```bash
examples/winding_studio/e2e/run.sh
```

For nightly/manual jobs, run:

```bash
examples/winding_studio/e2e/run.sh --visual
```

## Artifacts

Visual outputs are written to:

- `examples/winding_studio/e2e/artifacts/`

This directory is git-ignored except for `.gitignore`.

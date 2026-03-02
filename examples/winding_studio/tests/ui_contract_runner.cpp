#include "ui_contract_tests.hpp"

#include <cstdio>
#include <cstdlib>

#include "imgui.h"
#include "imgui_test_engine/imgui_te_engine.h"
#include "imgui_test_engine/imgui_te_exporters.h"
#include "imgui_test_engine/imgui_te_utils.h"

#include "studio_ui.hpp"

int main() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(1280.0f, 720.0f);
    io.Fonts->AddFontDefault();
    unsigned char *font_pixels = nullptr;
    int font_w = 0;
    int font_h = 0;
    io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_w, &font_h);
    io.Fonts->SetTexID(static_cast<ImTextureID>(1));
#if IMGUI_VERSION_NUM >= 19004
    io.ConfigDebugIsDebuggerPresent = ImOsIsDebuggerPresent();
#endif

    winding_studio::app::apply_engine_style(1.0f);

    ImGuiTestEngine *engine = ImGuiTestEngine_CreateContext();
    ImGuiTestEngineIO &test_io = ImGuiTestEngine_GetIO(engine);
    test_io.ConfigRunSpeed = ImGuiTestRunSpeed_Fast;
    test_io.ConfigVerboseLevel = ImGuiTestVerboseLevel_Warning;
    test_io.ConfigVerboseLevelOnError = ImGuiTestVerboseLevel_Debug;
    test_io.ConfigNoThrottle = true;
    test_io.ConfigLogToTTY = true;
    test_io.ConfigWatchdogWarning = 5.0f;
    test_io.ConfigWatchdogKillTest = 10.0f;
    test_io.ConfigWatchdogKillApp = 20.0f;

    if (char const *xml_path = std::getenv("WINDING_STUDIO_UI_CONTRACT_XML"); xml_path != nullptr && xml_path[0] != '\0') {
        test_io.ExportResultsFilename = xml_path;
        test_io.ExportResultsFormat = ImGuiTestEngineExportFormat_JUnitXml;
    }

    RegisterWindingStudioUiContractTests(engine);
    ImGuiTestEngine_Start(engine, ImGui::GetCurrentContext());
    ImGuiTestEngine_InstallDefaultCrashHandler();

    ImGuiTestEngine_QueueTests(
        engine, ImGuiTestGroup_Unknown, "all", ImGuiTestRunFlags_RunFromCommandLine
    );

    bool timed_out = true;
    int constexpr k_max_frames = 12000;
    for (int frame = 0; frame < k_max_frames; ++frame) {
        io.DeltaTime = 1.0f / 60.0f;
        io.DisplaySize = ImVec2(1280.0f, 720.0f);

        ImGui::NewFrame();
        ImGui::Render();

        if (!test_io.IsRunningTests && ImGuiTestEngine_IsTestQueueEmpty(engine)) {
            timed_out = false;
            break;
        }
    }

    ImGuiTestEngine_Stop(engine);

    ImGuiTestEngineResultSummary summary{};
    ImGuiTestEngine_GetResultSummary(engine, &summary);
    ImGuiTestEngine_PrintResultSummary(engine);

    bool const has_failures = summary.CountSuccess < summary.CountTested;
    bool const has_no_tests = summary.CountTested == 0;

    ImGui::DestroyContext();
    ImGuiTestEngine_DestroyContext(engine);

    if (timed_out) {
        std::fprintf(stderr, "[ui-contract] Timed out after %d frames\n", k_max_frames);
        return 2;
    }
    if (has_no_tests) {
        std::fprintf(stderr, "[ui-contract] No tests were executed\n");
        return 3;
    }
    if (has_failures)
        return 1;

    return 0;
}

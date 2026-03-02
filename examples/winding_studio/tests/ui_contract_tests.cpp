#include "ui_contract_tests.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_test_engine/imgui_te_context.h"

#include "studio_mesh_library.hpp"
#include "studio_ui.hpp"

#include <cmath>
#include <cstring>
#include <string>

namespace {

using namespace winding_studio::app;

struct UiContractVars {
    AppState state{};
    UiLayoutResult layout{};
    bool initialized{false};
    bool include_imported_mesh{false};
    bool apply_remove_intent{false};
    int last_activate_mesh_index{-1};
    int last_remove_mesh_index{-1};
    bool saw_harnack_refresh{false};
    bool saw_voxel_params_changed{false};
};

void initialize_state(UiContractVars &vars, bool const include_imported_mesh) {
    vars.state = AppState{};
    int const i0 = add_mesh_to_library(vars.state, build_default_mesh(), "Half Octahedron", true);
    int const i1 = add_mesh_to_library(vars.state, build_closed_octa_mesh(), "Closed Octahedron", true);
    (void)i0;
    (void)i1;

    if (include_imported_mesh) {
        MeshData imported = build_closed_octa_mesh();
        int const i2 = add_mesh_to_library(vars.state, std::move(imported), "Imported Mesh", false);
        vars.state.active_mesh_index = i2;
        vars.state.selected_mesh_index = i2;
        vars.state.active_mesh_name = vars.state.mesh_library[static_cast<std::size_t>(i2)].name;
        vars.state.triangle_count = vars.state.mesh_library[static_cast<std::size_t>(i2)].triangle_count;
    } else {
        vars.state.active_mesh_index = 0;
        vars.state.selected_mesh_index = 0;
        vars.state.active_mesh_name = vars.state.mesh_library[0].name;
        vars.state.triangle_count = vars.state.mesh_library[0].triangle_count;
    }

    vars.state.view_mode = ViewMode::k_split;
    vars.last_activate_mesh_index = -1;
    vars.last_remove_mesh_index = -1;
    vars.saw_harnack_refresh = false;
    vars.saw_voxel_params_changed = false;
    vars.initialized = true;
    vars.include_imported_mesh = include_imported_mesh;
}

void draw_ui(UiContractVars &vars) {
    if (!vars.initialized)
        initialize_state(vars, false);
    vars.layout = draw_editor_layout(vars.state, 1.0f / 60.0f, 1.0f);
    if (vars.layout.activate_mesh_index >= 0)
        vars.last_activate_mesh_index = vars.layout.activate_mesh_index;
    if (vars.layout.remove_mesh_index >= 0) {
        vars.last_remove_mesh_index = vars.layout.remove_mesh_index;
        if (vars.apply_remove_intent)
            remove_mesh_from_library(vars.state, vars.layout.remove_mesh_index);
    }
    if (vars.layout.request_harnack_refresh)
        vars.saw_harnack_refresh = true;
    if (vars.layout.voxel_params_changed)
        vars.saw_voxel_params_changed = true;
}

ImGuiWindow *find_window_with_fragment(char const *fragment) {
    ImGuiContext *context = ImGui::GetCurrentContext();
    for (int i = 0; i < context->Windows.Size; ++i) {
        ImGuiWindow *window = context->Windows[i];
        if (std::strstr(window->Name, fragment) != nullptr)
            return window;
    }
    return nullptr;
}

ImGuiWindow *find_toolbar_window() { return find_window_with_fragment("/Toolbar_"); }

ImGuiWindow *find_inspector_window() { return find_window_with_fragment("/InspectorPanel_"); }

void set_ref_toolbar(ImGuiTestContext *ctx) {
    ImGuiWindow *toolbar = find_toolbar_window();
    IM_CHECK(toolbar != nullptr);
    ctx->SetRef(toolbar);
}

void set_ref_inspector(ImGuiTestContext *ctx) {
    ImGuiWindow *inspector = find_inspector_window();
    IM_CHECK(inspector != nullptr);
    ctx->SetRef(inspector);
}

[[nodiscard]] bool item_exists(ImGuiTestContext *ctx, char const *ref) {
    return ctx->ItemInfo(ref, ImGuiTestOpFlags_NoError).ID != 0;
}

} // namespace

void RegisterWindingStudioUiContractTests(ImGuiTestEngine *engine) {
    ImGuiTest *test = nullptr;

    test = IM_REGISTER_TEST(engine, "winding_studio", "toolbar_modes");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_toolbar(ctx);

        ctx->ItemClick("Raster##ws_mode_raster");
        IM_CHECK_EQ(static_cast<int>(vars.state.view_mode), static_cast<int>(ViewMode::k_raster));

        ctx->ItemClick("Harnack##ws_mode_harnack");
        IM_CHECK_EQ(static_cast<int>(vars.state.view_mode), static_cast<int>(ViewMode::k_harnack));

        ctx->ItemClick("Voxel##ws_mode_voxel");
        IM_CHECK_EQ(static_cast<int>(vars.state.view_mode), static_cast<int>(ViewMode::k_voxel));

        ctx->ItemClick("Split##ws_mode_split");
        IM_CHECK_EQ(static_cast<int>(vars.state.view_mode), static_cast<int>(ViewMode::k_split));
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "mesh_activation_intent");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);

        std::string const expected_row_label =
            mesh_list_label(vars.state.mesh_library[1], 1 == vars.state.active_mesh_index) +
            "##ws_mesh_row_1";
        std::string const expected_row_ref = "**/" + expected_row_label;

        ImGuiTestItemInfo const row_item =
            ctx->ItemInfo(expected_row_ref.c_str(), ImGuiTestOpFlags_NoError);
        if (row_item.ID == 0) {
            ImGuiTestItemList items;
            ctx->GatherItems(&items, ctx->GetRef());
            ctx->LogItemList(&items);
        }
        IM_CHECK(row_item.ID != 0);
        ctx->ItemClick(row_item.ID);
        IM_CHECK_EQ(vars.state.selected_mesh_index, 1);
        IM_CHECK_EQ(vars.last_activate_mesh_index, 1);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "remove_button_intent");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, true);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);
        int const expected_remove_index = vars.state.selected_mesh_index;

        ctx->ItemClick("Remove##ws_remove_selected");
        IM_CHECK_EQ(vars.last_remove_mesh_index, expected_remove_index);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "harnack_refresh_intent");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        vars.state.view_mode = ViewMode::k_harnack;
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);

        ctx->ItemClick("Refresh##ws_harnack_refresh");
        IM_CHECK(vars.saw_harnack_refresh);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "geometry_remove_builtin_disabled");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);

        ImGuiTestItemInfo const remove_button =
            ctx->ItemInfo("Remove##ws_remove_selected", ImGuiTestOpFlags_NoError);
        IM_CHECK(remove_button.ID != 0);
        IM_CHECK((remove_button.ItemFlags & ImGuiItemFlags_Disabled) != 0);
        std::size_t const library_size = vars.state.mesh_library.size();

        ctx->ItemClick("Remove##ws_remove_selected", ImGuiMouseButton_Left, ImGuiTestOpFlags_NoError);
        IM_CHECK_EQ(vars.last_remove_mesh_index, -1);
        IM_CHECK_EQ(vars.state.selected_mesh_index, 0);
        IM_CHECK_EQ(static_cast<int>(vars.state.mesh_library.size()), static_cast<int>(library_size));
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "geometry_remove_imported_reselection");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, true);
        vars.apply_remove_intent = true;
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);

        IM_CHECK_EQ(vars.state.selected_mesh_index, 2);
        IM_CHECK_EQ(vars.state.active_mesh_index, 2);
        IM_CHECK_EQ(static_cast<int>(vars.state.mesh_library.size()), 3);

        ctx->ItemClick("Remove##ws_remove_selected");
        ctx->Yield();

        IM_CHECK_EQ(vars.last_remove_mesh_index, 2);
        IM_CHECK_EQ(static_cast<int>(vars.state.mesh_library.size()), 2);
        IM_CHECK_EQ(vars.state.selected_mesh_index, 1);
        IM_CHECK_EQ(vars.state.active_mesh_index, -1);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "geometry_add_mesh_opens_browser");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        set_ref_inspector(ctx);
        ctx->ItemClick("Add Mesh...##ws_add_mesh");
        ctx->Yield();

        ImGuiWindow *file_browser_window = find_window_with_fragment("Open Mesh File");
        IM_CHECK(file_browser_window != nullptr);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "mode_section_visibility_contract");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        auto assert_sections = [&](bool const expect_harnack, bool const expect_voxel) {
            set_ref_inspector(ctx);
            IM_CHECK_EQ(item_exists(ctx, "**/Harnack Trace"), expect_harnack);
            IM_CHECK_EQ(item_exists(ctx, "**/Voxelization"), expect_voxel);
        };

        assert_sections(true, false); // split

        set_ref_toolbar(ctx);
        ctx->ItemClick("Raster##ws_mode_raster");
        assert_sections(false, false);

        set_ref_toolbar(ctx);
        ctx->ItemClick("Harnack##ws_mode_harnack");
        assert_sections(true, false);

        set_ref_toolbar(ctx);
        ctx->ItemClick("Voxel##ws_mode_voxel");
        assert_sections(false, true);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "harnack_controls_intent_contract");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        vars.state.view_mode = ViewMode::k_harnack;
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);
        IM_CHECK(item_exists(ctx, "Live Update##ws_harnack_live_update"));
        IM_CHECK(item_exists(ctx, "Refresh##ws_harnack_refresh"));

        bool const previous_live_update = vars.state.harnack_live_update;
        ctx->ItemClick("Live Update##ws_harnack_live_update");
        IM_CHECK_EQ(vars.state.harnack_live_update, !previous_live_update);

        ctx->ItemClick("Refresh##ws_harnack_refresh");
        IM_CHECK(vars.saw_harnack_refresh);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "voxel_param_edit_sets_dirty_flag");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        vars.state.view_mode = ViewMode::k_voxel;
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);
        if (!item_exists(ctx, "**/##VoxelDx")) {
            ImGuiTestItemList items;
            ctx->GatherItems(&items, ctx->GetRef());
            ctx->LogItemList(&items);
        }
        IM_CHECK(item_exists(ctx, "**/##VoxelDx"));

        ctx->ItemInput("**/##VoxelDx");
        ctx->KeyChars("0.0300");
        ctx->ItemClick("Recompute");
        IM_CHECK(vars.saw_voxel_params_changed);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "camera_reset_contract");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        vars.state.yaw = 1.1f;
        vars.state.pitch = -0.6f;
        vars.state.camera_radius = 6.2f;
        vars.state.camera_target = Vec3{2.0f, -3.0f, 1.5f};
        ctx->Yield();

        set_ref_inspector(ctx);
        ctx->ItemClick("Reset Camera");
        IM_CHECK(std::abs(vars.state.yaw - 0.7853981634f) < 1e-6f);
        IM_CHECK(std::abs(vars.state.pitch - 0.6154797087f) < 1e-6f);
        IM_CHECK(std::abs(vars.state.camera_radius - 2.7f) < 1e-6f);
        IM_CHECK(std::abs(vars.state.camera_target.x - 0.0f) < 1e-6f);
        IM_CHECK(std::abs(vars.state.camera_target.y - 0.0f) < 1e-6f);
        IM_CHECK(std::abs(vars.state.camera_target.z - 0.0f) < 1e-6f);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "camera_frame_mesh_enabled_state");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, false);
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        clear_active_mesh_state(vars.state);
        ctx->Yield();

        set_ref_inspector(ctx);
        ImGuiTestItemInfo const frame_mesh_disabled =
            ctx->ItemInfo("Frame Mesh", ImGuiTestOpFlags_NoError);
        IM_CHECK(frame_mesh_disabled.ID != 0);
        IM_CHECK((frame_mesh_disabled.ItemFlags & ImGuiItemFlags_Disabled) != 0);

        vars.state.active_mesh_index = 0;
        vars.state.active_mesh_name = vars.state.mesh_library[0].name;
        vars.state.triangle_count = vars.state.mesh_library[0].triangle_count;
        ctx->Yield();

        set_ref_inspector(ctx);
        ImGuiTestItemInfo const frame_mesh_enabled =
            ctx->ItemInfo("Frame Mesh", ImGuiTestOpFlags_NoError);
        IM_CHECK(frame_mesh_enabled.ID != 0);
        IM_CHECK((frame_mesh_enabled.ItemFlags & ImGuiItemFlags_Disabled) == 0);
    };

    test = IM_REGISTER_TEST(engine, "winding_studio", "status_line_contract");
    test->SetVarsDataType<UiContractVars>();
    test->GuiFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        if (!vars.initialized)
            initialize_state(vars, true);
        vars.apply_remove_intent = true;
        draw_ui(vars);
    };
    test->TestFunc = [](ImGuiTestContext *ctx) {
        UiContractVars &vars = ctx->GetVars<UiContractVars>();
        set_ref_inspector(ctx);
        ctx->ItemClick("Remove##ws_remove_selected");
        ctx->Yield();

        IM_CHECK(vars.state.status_line.rfind("Removed mesh:", 0) == 0);
    };
}

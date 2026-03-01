#include "ws_ui.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "imgui.h"
#include "ws_core.hpp"
#include "ws_math.hpp"

namespace winding_studio::app {

[[nodiscard]] static bool
begin_collapsing_section(char const *label, bool const default_open = true) {
    ImGui::Spacing();
    ImGuiTreeNodeFlags const flags = default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0;
    bool const open = ImGui::CollapsingHeader(label, flags);
    if (open)
        ImGui::Spacing();
    return open;
}

static void item_tooltip(char const *text) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(300.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

static bool begin_property_table(char const *id, float label_w = 90.0f) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_NoPadOuterX))
        return false;
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, label_w);
    ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

static void property_label(char const *label) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
}

static void end_property_table() { ImGui::EndTable(); }

[[nodiscard]] static bool mode_button(
    char const *label, bool const is_active, ImVec2 const size = ImVec2(0.0f, 0.0f),
    ImVec4 const *active_palette = nullptr
) {
    if (is_active) {
        ImVec4 const default_palette[4] = {
            ImVec4(0.20f, 0.36f, 0.55f, 1.0f),
            ImVec4(0.24f, 0.42f, 0.62f, 1.0f),
            ImVec4(0.28f, 0.48f, 0.70f, 1.0f),
            ImVec4(0.55f, 0.90f, 1.0f, 1.0f),
        };
        ImVec4 const *palette = (active_palette == nullptr) ? default_palette : active_palette;
        ImGui::PushStyleColor(ImGuiCol_Button, palette[0]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, palette[1]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, palette[2]);
        ImGui::PushStyleColor(ImGuiCol_Text, palette[3]);
    }
    bool const pressed = ImGui::Button(label, size);
    if (is_active)
        ImGui::PopStyleColor(4);
    return pressed;
}

static void vcenter_cursor(float content_h = 0.0f) {
    if (content_h <= 0.0f)
        content_h = (content_h < 0.0f) ? ImGui::GetTextLineHeight() : ImGui::GetFrameHeight();
    float const avail_h = ImGui::GetContentRegionAvail().y;
    if (avail_h > content_h)
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail_h - content_h) * 0.5f);
}

static void status_segment(char const *label, char const *value) {
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::TextUnformatted(value);
}

void apply_engine_style(float const dpi_scale) {
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.PopupRounding = 4.0f;

    style.WindowPadding = ImVec2(10.0f, 8.0f);
    style.FramePadding = ImVec2(8.0f, 5.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.IndentSpacing = 16.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.ScrollbarSize = 12.0f;
    style.ScrollbarRounding = 3.0f;
    style.GrabMinSize = 8.0f;

    ImVec4 *c = style.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.071f, 0.086f, 1.0f);
    c[ImGuiCol_ChildBg] = ImVec4(0.082f, 0.088f, 0.105f, 1.0f);
    c[ImGuiCol_PopupBg] = ImVec4(0.075f, 0.080f, 0.098f, 0.96f);
    c[ImGuiCol_Border] = ImVec4(0.18f, 0.22f, 0.28f, 0.65f);
    c[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TitleBg] = ImVec4(0.065f, 0.072f, 0.092f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.085f, 0.10f, 0.14f, 1.0f);
    c[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.12f, 0.16f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.14f, 0.17f, 0.24f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.17f, 0.22f, 0.32f, 1.0f);
    c[ImGuiCol_Button] = ImVec4(0.13f, 0.19f, 0.26f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.18f, 0.28f, 0.40f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.22f, 0.38f, 0.56f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.12f, 0.18f, 0.26f, 1.0f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.16f, 0.26f, 0.38f, 1.0f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.34f, 0.50f, 1.0f);
    c[ImGuiCol_Separator] = ImVec4(0.20f, 0.25f, 0.32f, 0.6f);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.30f, 0.45f, 0.65f, 0.8f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.35f, 0.55f, 0.80f, 1.0f);
    c[ImGuiCol_CheckMark] = ImVec4(0.40f, 0.82f, 1.0f, 1.0f);
    c[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.60f, 0.85f, 1.0f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.40f, 0.75f, 1.0f, 1.0f);
    c[ImGuiCol_Text] = ImVec4(0.86f, 0.90f, 0.94f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.42f, 0.48f, 0.56f, 1.0f);
    c[ImGuiCol_Tab] = c[ImGuiCol_Header];
    c[ImGuiCol_TabHovered] = c[ImGuiCol_HeaderHovered];
    c[ImGuiCol_TabSelected] = ImVec4(0.18f, 0.30f, 0.45f, 1.0f);

    if (dpi_scale > 1.01f)
        style.ScaleAllSizes(dpi_scale);
}

[[nodiscard]] UiLayoutResult
draw_editor_layout(AppState &state, float const dt, float const ui_scale) {
    UiLayoutResult result{};
    (void)dt;

    ImGuiViewport const *main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->Pos);
    ImGui::SetNextWindowSize(main_viewport->Size);

    ImGuiWindowFlags const root_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::Begin("Winding Studio Root", nullptr, root_flags);
    ImGui::PopStyleVar(3);

    ImVec2 const root_avail = ImGui::GetContentRegionAvail();
    float const s = ui_scale;
    float const toolbar_h = std::clamp(root_avail.y * 0.055f, 34.0f * s, 52.0f * s);
    float const statusbar_h = std::clamp(root_avail.y * 0.045f, 28.0f * s, 38.0f * s);
    float right_w = std::clamp(root_avail.x * 0.27f, 280.0f * s, 430.0f * s);
    float const spacing = ImGui::GetStyle().ItemSpacing.x;

    float const min_center_w = 320.0f * s;
    if (right_w > root_avail.x - min_center_w - spacing)
        right_w = std::max(200.0f * s, root_avail.x - min_center_w - spacing);

    ImGui::BeginChild(
        "Toolbar", ImVec2(0.0f, toolbar_h), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );
    vcenter_cursor();
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
    ImGui::TextUnformatted("Winding Studio");
    ImGui::PopStyleColor();

    float const min_mode_button_w = 76.0f * s;
    float const max_mode_button_w = 124.0f * s;
    float const title_reserve = ImGui::CalcTextSize("Winding Studio").x + 28.0f * s;
    float const mode_button_w = std::clamp(
        (ImGui::GetWindowContentRegionMax().x - title_reserve - spacing * 3.0f) / 4.0f,
        min_mode_button_w, max_mode_button_w
    );
    float const buttons_w = mode_button_w * 4.0f + spacing * 3.0f;
    float const btn_start =
        std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - buttons_w);
    ImGui::SameLine(btn_start);
    if (mode_button("Split", state.view_mode == ViewMode::k_split, ImVec2(mode_button_w, 0.0f)))
        state.view_mode = ViewMode::k_split;
    ImGui::SameLine();
    if (mode_button("Raster", state.view_mode == ViewMode::k_raster, ImVec2(mode_button_w, 0.0f)))
        state.view_mode = ViewMode::k_raster;
    ImGui::SameLine();
    if (mode_button("Harnack", state.view_mode == ViewMode::k_harnack, ImVec2(mode_button_w, 0.0f)))
        state.view_mode = ViewMode::k_harnack;
    ImGui::SameLine();
    ImVec4 const voxel_palette[4] = {
        ImVec4(0.15f, 0.45f, 0.30f, 1.0f),
        ImVec4(0.18f, 0.52f, 0.35f, 1.0f),
        ImVec4(0.22f, 0.58f, 0.40f, 1.0f),
        ImVec4(0.70f, 1.00f, 0.80f, 1.0f),
    };
    if (mode_button(
            "Voxel", state.view_mode == ViewMode::k_voxel, ImVec2(mode_button_w, 0.0f),
            voxel_palette
        ))
        state.view_mode = ViewMode::k_voxel;

    ImGui::EndChild();

    float const workspace_h =
        std::max(120.0f * s, ImGui::GetContentRegionAvail().y - statusbar_h - spacing);
    ImGui::BeginChild(
        "Workspace", ImVec2(0.0f, workspace_h), false,
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse
    );

    float const panel_h = std::max(120.0f * s, ImGui::GetContentRegionAvail().y);

    float const center_w =
        std::max(120.0f * s, ImGui::GetContentRegionAvail().x - right_w - spacing);
    ImGui::BeginChild(
        "ViewportPanel", ImVec2(center_w, panel_h), false,
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse
    );
    result.viewport_pos = ImGui::GetCursorScreenPos();
    result.viewport_size = ImGui::GetContentRegionAvail();
    result.viewport_size.x = std::max(result.viewport_size.x, 32.0f);
    result.viewport_size.y = std::max(result.viewport_size.y, 32.0f);
    ImGui::InvisibleButton("ViewportCanvas", result.viewport_size);

    bool const viewport_hovered = ImGui::IsItemHovered();
    bool const viewport_active = ImGui::IsItemActive();
    ImGuiIO &interact_io = ImGui::GetIO();

    if (viewport_hovered || viewport_active) {
        if (interact_io.MouseWheel != 0.0f) {
            state.camera_radius *= (1.0f - interact_io.MouseWheel * 0.1f);
            state.camera_radius = std::clamp(state.camera_radius, 0.5f, 12.0f);
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        }

        bool const mmb_drag = ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 2.0f);
        bool const alt_lmb_drag =
            interact_io.KeyAlt && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f);
        if (mmb_drag || alt_lmb_drag) {
            ImVec2 const delta = interact_io.MouseDelta;
            if (interact_io.KeyCtrl) {
                state.camera_radius *= (1.0f + delta.y * 0.01f);
                state.camera_radius = std::clamp(state.camera_radius, 0.5f, 12.0f);
            } else if (interact_io.KeyShift) {
                CameraBasis const cam = build_camera_basis(state);
                float const pan_speed = 0.003f * state.camera_radius;
                state.camera_target.x -=
                    (cam.right.x * delta.x + cam.ortho_up.x * delta.y) * pan_speed;
                state.camera_target.y -=
                    (cam.right.y * delta.x + cam.ortho_up.y * delta.y) * pan_speed;
                state.camera_target.z -=
                    (cam.right.z * delta.x + cam.ortho_up.z * delta.y) * pan_speed;
            } else {
                state.yaw -= delta.x * 0.005f;
                state.pitch -= delta.y * 0.005f;
                state.pitch = std::clamp(state.pitch, -1.4f, 1.4f);
            }
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        }
    }

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 const p0 = result.viewport_pos;
    ImVec2 const p1 = ImVec2(
        result.viewport_pos.x + result.viewport_size.x,
        result.viewport_pos.y + result.viewport_size.y
    );
    float const corner_r = ImGui::GetStyle().ChildRounding;
    draw_list->AddRect(p0, p1, IM_COL32(50, 65, 85, 120), corner_r, 0, 0.6f);

    if (state.view_mode == ViewMode::k_split) {
        float const mid_x = p0.x + result.viewport_size.x * 0.5f;
        draw_list->AddLine(
            ImVec2(mid_x, p0.y), ImVec2(mid_x, p1.y), IM_COL32(180, 190, 210, 60), 1.0f
        );
        ImU32 const label_col = IM_COL32(140, 170, 200, 140);
        draw_list->AddText(ImVec2(p0.x + 6.0f * s, p0.y + 4.0f * s), label_col, "Raster");
        ImVec2 const h_size = ImGui::CalcTextSize("Harnack");
        draw_list->AddText(
            ImVec2(p1.x - h_size.x - 6.0f * s, p0.y + 4.0f * s), label_col, "Harnack"
        );
    } else {
        draw_list->AddText(
            ImVec2(p0.x + 6.0f * s, p0.y + 4.0f * s), IM_COL32(140, 170, 200, 140),
            view_mode_name(state.view_mode)
        );
    }

    if (viewport_hovered && !viewport_active) {
        draw_list->AddText(
            ImVec2(p0.x + 6.0f * s, p1.y - 18.0f * s), IM_COL32(130, 150, 170, 90),
            "MMB: orbit | Shift+MMB: pan | Ctrl+MMB/Scroll: zoom | Alt+LMB: orbit"
        );
    }
    if (!has_active_mesh(state)) {
        char const *msg = "No active mesh";
        ImVec2 const text_size = ImGui::CalcTextSize(msg);
        ImVec2 const text_pos{
            p0.x + 0.5f * (result.viewport_size.x - text_size.x),
            p0.y + 0.5f * (result.viewport_size.y - text_size.y),
        };
        draw_list->AddText(text_pos, IM_COL32(150, 165, 190, 160), msg);
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("InspectorPanel", ImVec2(0.0f, panel_h), true);

    if (begin_collapsing_section("Geometry", true)) {
        ImGui::TextUnformatted("Mesh Library");
        ImGui::Spacing();

        float const list_h =
            std::clamp(7.0f * ImGui::GetTextLineHeightWithSpacing(), 96.0f * s, 180.0f * s);
        if (ImGui::BeginListBox("##GeometryLibrary", ImVec2(-FLT_MIN, list_h))) {
            for (int i = 0; i < static_cast<int>(state.mesh_library.size()); ++i) {
                MeshLibraryEntry const &entry = state.mesh_library[static_cast<std::size_t>(i)];
                bool const selected = (state.selected_mesh_index == i);
                std::string const label = mesh_list_label(entry, i == state.active_mesh_index);
                ImGui::PushID(i);
                if (ImGui::Selectable(label.c_str(), selected)) {
                    state.selected_mesh_index = i;
                    result.activate_mesh_index = i;
                }
                if (selected)
                    ImGui::SetItemDefaultFocus();
                ImGui::PopID();
            }
            ImGui::EndListBox();
        }
        if (state.mesh_library.empty())
            ImGui::TextDisabled("No meshes loaded.");
        else
            ImGui::TextDisabled("Click a row to show it in the viewport.");

        if (!state.file_browser_initialized) {
            state.file_browser.SetTitle("Open Mesh File");
            state.file_browser.SetTypeFilters({".obj", ".ply", ".stl", ".off", ".*"});
            if (char const *home = std::getenv("HOME"); home != nullptr && home[0] != '\0')
                (void)state.file_browser.SetPwd(home);
            state.file_browser_initialized = true;
        }

        ImGui::Spacing();
        if (ImGui::Button("Add Mesh..."))
            state.file_browser.Open();
        item_tooltip("Add a mesh file to this session (.obj, .ply, .stl, .off).");
        ImGui::SameLine();
        bool const can_remove_selected =
            has_valid_mesh_index(state, state.selected_mesh_index) &&
            !state.mesh_library[static_cast<std::size_t>(state.selected_mesh_index)].is_builtin;
        if (!can_remove_selected)
            ImGui::BeginDisabled();
        if (ImGui::Button("Remove"))
            result.remove_mesh_index = state.selected_mesh_index;
        if (!can_remove_selected)
            ImGui::EndDisabled();
        item_tooltip("Remove selected imported mesh from this session.");
    }

    if (begin_collapsing_section("Camera", true)) {
        auto reset_camera = [&]() {
            state.yaw = 0.0f;
            state.pitch = 0.35f;
            state.camera_radius = 2.7f;
            state.camera_target = Vec3{0.0f, 0.0f, 0.0f};
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        };
        auto set_camera_preset = [&](float const yaw, float const pitch) {
            state.yaw = yaw;
            state.pitch = std::clamp(pitch, -1.45f, 1.45f);
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        };
        auto frame_camera_to_mesh = [&]() {
            if (!has_active_mesh(state))
                return;
            float const cx = 0.5f * (state.mesh_bounds.min_x + state.mesh_bounds.max_x);
            float const cy = 0.5f * (state.mesh_bounds.min_y + state.mesh_bounds.max_y);
            float const cz = 0.5f * (state.mesh_bounds.min_z + state.mesh_bounds.max_z);
            float const dx = state.mesh_bounds.max_x - state.mesh_bounds.min_x;
            float const dy = state.mesh_bounds.max_y - state.mesh_bounds.min_y;
            float const dz = state.mesh_bounds.max_z - state.mesh_bounds.min_z;
            float const diag = std::sqrt(dx * dx + dy * dy + dz * dz);
            float const fit_distance = std::max(1.35f * diag, 0.8f);
            state.camera_target = Vec3{cx, cy, cz};
            state.camera_radius = std::clamp(fit_distance, 0.5f, 12.0f);
            state.auto_rotate = false;
            result.harnack_params_changed = true;
        };

        if (ImGui::Button("Reset Camera", ImVec2(-FLT_MIN, 0.0f)))
            reset_camera();
        item_tooltip("Reset camera orbit, distance and target");

        float const preset_gap = ImGui::GetStyle().ItemSpacing.x;
        float const preset_w =
            std::max(62.0f * s, (ImGui::GetContentRegionAvail().x - preset_gap * 2.0f) / 3.0f);
        if (ImGui::Button("Front", ImVec2(preset_w, 0.0f)))
            set_camera_preset(0.0f, 0.0f);
        ImGui::SameLine();
        if (ImGui::Button("Back", ImVec2(preset_w, 0.0f)))
            set_camera_preset(k_pi, 0.0f);
        ImGui::SameLine();
        if (ImGui::Button("Top", ImVec2(preset_w, 0.0f)))
            set_camera_preset(state.yaw, 1.45f);

        if (ImGui::Button("Left", ImVec2(preset_w, 0.0f)))
            set_camera_preset(-0.5f * k_pi, 0.0f);
        ImGui::SameLine();
        if (ImGui::Button("Right", ImVec2(preset_w, 0.0f)))
            set_camera_preset(0.5f * k_pi, 0.0f);
        ImGui::SameLine();
        if (ImGui::Button("Bottom", ImVec2(preset_w, 0.0f)))
            set_camera_preset(state.yaw, -1.45f);
        item_tooltip("Blender-style quick views");

        bool const can_frame_mesh = has_active_mesh(state);
        if (!can_frame_mesh)
            ImGui::BeginDisabled();
        if (ImGui::Button("Frame Mesh", ImVec2(-FLT_MIN, 0.0f)))
            frame_camera_to_mesh();
        if (!can_frame_mesh)
            ImGui::EndDisabled();
        item_tooltip("Center target on active mesh bounds and adjust distance to fit");

        if (begin_property_table("##CameraProps", 90.0f * s)) {
            property_label("Orbit Yaw");
            if (ImGui::SliderFloat("##CameraYaw", &state.yaw, -k_pi, k_pi, "%.3f rad")) {
                state.auto_rotate = false;
                result.harnack_params_changed = true;
            }
            item_tooltip("Orbit angle around world-up axis");

            property_label("Orbit Pitch");
            if (ImGui::SliderFloat("##CameraPitch", &state.pitch, -1.45f, 1.45f, "%.3f rad")) {
                state.auto_rotate = false;
                result.harnack_params_changed = true;
            }
            item_tooltip("Orbit elevation angle");

            property_label("Distance");
            if (ImGui::SliderFloat("##CameraDist", &state.camera_radius, 0.5f, 12.0f, "%.3f")) {
                state.auto_rotate = false;
                result.harnack_params_changed = true;
            }
            item_tooltip("Camera distance to target");

            property_label("Target");
            if (ImGui::DragFloat3("##CameraTarget", &state.camera_target.x, 0.01f, -100.0f, 100.0f)) {
                state.auto_rotate = false;
                result.harnack_params_changed = true;
            }
            item_tooltip("Focus target point in world coordinates");

            end_property_table();
        }

        ImGui::Spacing();
        bool const rotate_changed = ImGui::Checkbox("Auto Rotate", &state.auto_rotate);
        result.harnack_params_changed |= rotate_changed;
        item_tooltip("Continuously orbit camera around target");

        CameraBasis const camera = build_camera_basis(state);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        if (begin_property_table("##CameraReadout", 60.0f * s)) {
            property_label("Eye");
            ImGui::Text("%.2f  %.2f  %.2f", camera.eye.x, camera.eye.y, camera.eye.z);
            property_label("Target");
            ImGui::Text("%.2f  %.2f  %.2f", camera.target.x, camera.target.y, camera.target.z);
            property_label("Forward");
            ImGui::Text("%.2f  %.2f  %.2f", camera.forward.x, camera.forward.y, camera.forward.z);
            property_label("Rotate");
            ImGui::TextUnformatted(state.auto_rotate ? "Auto" : "Manual");
            end_property_table();
        }
    }

    if (begin_collapsing_section("Viewport", true)) {
        ImGui::Checkbox("Wireframe", &state.wireframe);
        item_tooltip("Show wireframe for raster and voxel rendering");
        ImGui::TextDisabled("Harnack view is image-based and ignores wireframe.");
    }

    if ((state.view_mode == ViewMode::k_split || state.view_mode == ViewMode::k_harnack) &&
        begin_collapsing_section("Harnack Trace", true)) {
        if (!has_active_mesh(state)) {
            ImGui::TextDisabled("No active mesh. Select one in Geometry.");
        } else {
            if (begin_property_table("##HarnackProps", 90.0f * s)) {
                property_label("Target W");
                result.harnack_params_changed |=
                    ImGui::SliderFloat("##TargetW", &state.target_winding, 0.1f, 0.9f, "%.2f");
                item_tooltip(
                    "Target winding number iso-value for surface extraction (0.5 = standard "
                    "surface)"
                );

                property_label("Epsilon");
                result.harnack_params_changed |= ImGui::SliderFloat(
                    "##Epsilon", &state.epsilon, 1e-5f, 1e-2f, "%.5f", ImGuiSliderFlags_Logarithmic
                );
                item_tooltip("Convergence threshold. Smaller = more precise but slower");

                property_label("Max Iters");
                result.harnack_params_changed |=
                    ImGui::SliderInt("##MaxIters", &state.max_iterations, 16, 4096);
                item_tooltip("Maximum Harnack iterations per ray");

                property_label("Accuracy");
                result.harnack_params_changed |=
                    ImGui::SliderFloat("##Accuracy", &state.accuracy_scale, 0.8f, 6.0f, "%.2f");
                item_tooltip("BVH traversal accuracy multiplier. Higher = more precise but slower");

                property_label("Trace t_max");
                result.harnack_params_changed |=
                    ImGui::SliderFloat("##TraceT", &state.t_max, 5.0f, 500.0f, "%.1f");
                item_tooltip("Maximum ray distance. Increase for large scenes");

                property_label("Resolution");
                result.harnack_params_changed |= ImGui::SliderFloat(
                    "##Resolution", &state.harnack_resolution_scale, 0.2f, 1.0f, "%.2f"
                );
                item_tooltip("Trace resolution relative to viewport. Lower = faster preview");

                end_property_table();
            }

            ImGui::Spacing();
            result.harnack_params_changed |=
                ImGui::Checkbox("Live Update", &state.harnack_live_update);
            item_tooltip(
                "Continuously re-trace when parameters change. Disable for manual control"
            );
            ImGui::SameLine();
            if (ImGui::Button("Refresh"))
                result.request_harnack_refresh = true;
            item_tooltip("Force a single Harnack trace refresh");
        }
    }

    if (state.view_mode == ViewMode::k_voxel && begin_collapsing_section("Voxelization", true)) {
        if (!has_active_mesh(state)) {
            ImGui::TextDisabled("No active mesh. Select one in Geometry.");
        } else {
            if (begin_property_table("##VoxelProps", 90.0f * s)) {
                property_label("Target W");
                ImGui::SliderFloat("##VoxelTargetW", &state.voxel_target_w, 0.1f, 0.9f, "%.2f");
                if (ImGui::IsItemDeactivatedAfterEdit())
                    result.voxel_params_changed = true;
                item_tooltip("Winding threshold for occupancy (inside if W >= Target W)");

                property_label("Voxel dx");
                ImGui::SliderFloat(
                    "##VoxelDx", &state.voxel_dx, 0.002f, 0.25f, "%.4f",
                    ImGuiSliderFlags_Logarithmic
                );
                if (ImGui::IsItemDeactivatedAfterEdit())
                    result.voxel_params_changed = true;
                item_tooltip("Requested voxel size. May be clamped to satisfy voxel-count budget");

                property_label("Max Voxels");
                ImGui::Text("%zu", state.voxel_max_voxels);

                property_label("Actual dx");
                ImGui::Text("%.5f", state.voxel_actual_dx);

                property_label("Grid");
                ImGui::Text(
                    "%zu x %zu x %zu", state.voxel_grid_nx, state.voxel_grid_ny, state.voxel_grid_nz
                );

                end_property_table();
            }

            ImGui::Spacing();
            if (ImGui::Button("Recompute"))
                result.request_voxel_refresh = true;
            item_tooltip("Re-run voxelization with current settings");
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.42f, 0.48f, 0.56f, 1.0f));
    ImGui::TextWrapped("%s", state.status_line.c_str());
    ImGui::PopStyleColor();

    ImGui::EndChild();

    ImGui::EndChild();

    ImGui::BeginChild(
        "StatusBar", ImVec2(0.0f, statusbar_h), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );
    vcenter_cursor(-1.0f);

    char buf[64];

    status_segment("Mode:", view_mode_name(state.view_mode));
    ImGui::SameLine(0.0f, 16.0f);

    std::snprintf(
        buf, sizeof(buf), "%s (%zu tris)", state.active_mesh_name.c_str(), state.triangle_count
    );
    status_segment("Mesh:", buf);
    ImGui::SameLine(0.0f, 16.0f);

    std::snprintf(buf, sizeof(buf), "%.0f x %.0f", result.viewport_size.x, result.viewport_size.y);
    status_segment("Viewport:", buf);

    bool const show_harnack_stats =
        state.view_mode == ViewMode::k_harnack || state.view_mode == ViewMode::k_split;
    if (show_harnack_stats) {
        ImGui::SameLine(0.0f, 16.0f);
        std::snprintf(
            buf, sizeof(buf), "%zu / %zu", state.harnack_hit_count, state.harnack_pixel_count
        );
        status_segment("Hits:", buf);
        ImGui::SameLine(0.0f, 16.0f);
        std::snprintf(buf, sizeof(buf), "%.2f ms", state.last_harnack_ms);
        status_segment("Trace:", buf);
    }
    if (state.view_mode == ViewMode::k_voxel) {
        ImGui::SameLine(0.0f, 16.0f);
        std::snprintf(
            buf, sizeof(buf), "%zu / %zu", state.voxel_occupied_count, state.voxel_grid_total
        );
        status_segment("Voxels:", buf);
        ImGui::SameLine(0.0f, 16.0f);
        std::snprintf(buf, sizeof(buf), "%.2f ms", state.last_voxel_ms);
        status_segment("Voxelize:", buf);
    }

    ImGui::EndChild();

    ImGui::End();

    state.file_browser.Display();
    if (state.file_browser.HasSelected()) {
        result.mesh_file_to_add = state.file_browser.GetSelected().string();
        state.file_browser.ClearSelected();
    }

    return result;
}

[[nodiscard]] FramebufferRect ui_viewport_to_framebuffer(
    UiLayoutResult const &layout, ImGuiIO const &io, ImGuiViewport const &main_viewport,
    int const fb_w, int const fb_h
) {
    float const scale_x = io.DisplayFramebufferScale.x > 0.0f ? io.DisplayFramebufferScale.x : 1.0f;
    float const scale_y = io.DisplayFramebufferScale.y > 0.0f ? io.DisplayFramebufferScale.y : 1.0f;

    int const x =
        static_cast<int>(std::lround((layout.viewport_pos.x - main_viewport.Pos.x) * scale_x));
    int const y_top =
        static_cast<int>(std::lround((layout.viewport_pos.y - main_viewport.Pos.y) * scale_y));
    int const w = std::max(0, static_cast<int>(std::lround(layout.viewport_size.x * scale_x)));
    int const h = std::max(0, static_cast<int>(std::lround(layout.viewport_size.y * scale_y)));
    int const y = fb_h - (y_top + h);

    int const x0 = std::clamp(x, 0, fb_w);
    int const y0 = std::clamp(y, 0, fb_h);
    int const x1 = std::clamp(x + w, 0, fb_w);
    int const y1 = std::clamp(y + h, 0, fb_h);
    return FramebufferRect{x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
}

} // namespace winding_studio::app

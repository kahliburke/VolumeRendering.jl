# CImGui interop for a FieldView: `ShowVolume` draws it as an interactive image and turns
# mouse/keyboard into camera moves (drag orbits, Shift-drag or right-drag pans, wheel zooms); right-click (no drag) toggles a floating `ShowControls` window.

using CImGui
using CImGui: ImVec2
using CImGui.CSyntax   # @c

const ROT_SPEED  = 0.007   # rad per pixel of drag
const ZOOM_SPEED = 0.12    # per mouse-wheel notch

# Per-view interaction state, in a WeakKeyDict so it stays off the FieldView type and GCs with it.
mutable struct _Drag
    rotating::Bool
    panning::Bool
    controls_open::Bool     # is the floating controls window shown for this view?
    rmb_on_widget::Bool     # did the current right-button press start on this image?
    rmb_dragged::Bool       # …and did it turn into a drag (→ pan, not a controls toggle)?
end
const _DRAG = Base.WeakKeyDict{FieldView,_Drag}()
_drag(view::FieldView) = get!(() -> _Drag(false, false, false, false, false), _DRAG, view)

# Draw `view` as an interactive image of `size` (ImGui item-size convention). Renders at device
# resolution, shows the G-buffer V-flipped, and scopes camera input per-view (multi-view safe).
function ShowVolume(view::FieldView; size::ImVec2 = ImVec2(-CImGui.FLT_MIN, -CImGui.FLT_MIN),
                    controls::Bool = true)
    sz = CImGui.CalcItemSize(size, 256.0, 256.0)                 # 256² default if a component is 0
    dw = max(1, round(Int, sz.x)); dh = max(1, round(Int, sz.y)) # on-screen logical points
    fbs = unsafe_load(CImGui.GetIO().DisplayFramebufferScale)    # Retina: (2,2)
    gw = max(1, round(Int, dw * fbs.x)); gh = max(1, round(Int, dh * fbs.y))   # device px
    tex = render!(view, gw, gh)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)                   # restore default FB so imgui draws to screen
    CImGui.PushID(pointer_from_objref(view))                     # per-view id scoping
    ref = CImGui.ImTextureRef(CImGui.ImTextureID(UInt64(tex)))
    CImGui.Image(ref, ImVec2(dw, dh), ImVec2(0, 1), ImVec2(1, 0))   # flip V: GL origin is bottom-left
    _handle_camera_input!(view, CImGui.IsItemHovered(), dw, dh)
    CImGui.PopID()
    controls && _drag(view).controls_open && _controls_window!(view)
    nothing
end

function _handle_camera_input!(view::FieldView, hovered::Bool, w, h)
    io = CImGui.GetIO()
    d = unsafe_load(io.MouseDelta)
    drag = _drag(view); cam = view.camera; params = view.params
    # Left-drag orbits, or pans with Shift held as the drag starts; right- and middle-drag pan.
    if hovered && CImGui.IsMouseClicked(0)
        unsafe_load(io.KeyShift) ? (drag.panning = true) : (drag.rotating = true)
    end
    hovered && (CImGui.IsMouseClicked(1) || CImGui.IsMouseClicked(2)) && (drag.panning = true)
    CImGui.IsMouseDown(0) || (drag.rotating = false)
    (CImGui.IsMouseDown(0) || CImGui.IsMouseDown(1) || CImGui.IsMouseDown(2)) || (drag.panning = false)

    # Right-click WITHOUT drag toggles the controls window; right-DRAG still pans (above).
    hovered && CImGui.IsMouseClicked(1) && (drag.rmb_on_widget = true; drag.rmb_dragged = false)
    CImGui.IsMouseDragging(1) && (drag.rmb_dragged = true)
    if CImGui.IsMouseReleased(1)
        drag.rmb_on_widget && !drag.rmb_dragged && (drag.controls_open = !drag.controls_open)
        drag.rmb_on_widget = false
    end

    if drag.rotating
        dyaw = -d.x * ROT_SPEED; dpitch = -d.y * ROT_SPEED
        params.flymode ? flylook!(cam, dyaw, dpitch) : orbit!(cam, dyaw, -dpitch)
    elseif drag.panning
        pan!(cam, d.x / w, d.y / h)
    end

    if hovered
        wheel = unsafe_load(io.MouseWheel)
        wheel != 0 && zoom!(cam, exp(-ZOOM_SPEED * wheel))
        if params.flymode
            k(key) = CImGui.IsKeyDown(key) ? 1.0 : 0.0
            fwd = k(CImGui.ImGuiKey_W) - k(CImGui.ImGuiKey_S)
            rgt = k(CImGui.ImGuiKey_D) - k(CImGui.ImGuiKey_A)
            upd = k(CImGui.ImGuiKey_Space) - k(CImGui.ImGuiKey_C)
            speed = params.fly_speed * (unsafe_load(io.KeyShift) ? 5.0 : 1.0)   # Shift = sprint
            (fwd != 0 || rgt != 0 || upd != 0) && fly!(cam, fwd, rgt, upd, speed)
        end
    end
end

# Floating per-view controls window (distinct ##id per view), with a × close wired to controls_open.
# Interleaving this Begin/End inside the host's window is allowed by Dear ImGui.
function _controls_window!(view::FieldView)
    drag = _drag(view)
    p_open = Ref(true)
    title = "Controls##$(UInt(pointer_from_objref(view)))"
    if @c CImGui.Begin(title, &p_open[])
        ShowControls(view)
    end
    CImGui.End()
    p_open[] || (drag.controls_open = false)
end

# `ShowControls(view)` emits the RenderParams / TransferFunction / Camera widgets into the current
# ImGui window (imgui.jl's right-click window wraps it). Colormap buttons come from ImPlot, over the
# package's ColorSchemes maps registered into it.

using CImGui, ImPlot
using CImGui: ImVec2, ImVec4
using CImGui.CSyntax          # @c — pass Ref pointers to the widget wrappers
import ColorSchemes
import ImPlotExtra

const _MODES = (("DVR", "direct volume rendering (emission-absorption)"),
                ("MIP", "maximum intensity projection"),
                ("avg", "average intensity"))
const _RES_VALUES = (64, 128, 256, 512, 1024, 2048, 4096)   # discrete "max render px" steps

# A horizontal row of radio buttons; returns the newly-selected 0-based index, or nothing.
function _radio_row(id, opts, cur)
    sel = nothing
    for (i, (label, tip)) in enumerate(opts)
        i > 1 && CImGui.SameLine()
        CImGui.RadioButton("$label##$id", cur == i - 1) && (sel = i - 1)
        tip !== nothing && CImGui.SetItemTooltip(tip)
    end
    sel
end

# ImPlot needs its own context, bound to the host's ImGui context. Created lazily on the render
# thread (where AddColormap/SampleColormap require the context to be live).
function _ensure_implot!()
    if ImPlot.GetCurrentContext() == C_NULL
        ImPlot.CreateContext()
        ImPlot.SetImGuiContext(CImGui.GetCurrentContext())
    end
end

# Register each ColorSchemes map into ImPlot once (so ColormapButton shows the real gradient) and
# cache symbol → ImPlot colormap index. Names are the lowercase ColorSchemes symbols, distinct from
# ImPlot's capitalized built-ins; the GetColormapIndex guard reuses an existing registration (e.g.
# after a Revise reload where this Dict was reset but ImPlot kept the colormap).
const _CMAPS = Dict{Symbol,Int32}()
const _CMAP_N = 32   # sample points; ImPlot interpolates between them (continuous colormap)
function _cmap_index(sym::Symbol)
    get!(_CMAPS, sym) do
        name = String(sym)
        idx = ImPlot.GetColormapIndex(name)
        idx >= 0 && return Int32(idx)
        scheme = ColorSchemes.colorschemes[sym]
        cols = [ImVec4(c.r, c.g, c.b, 1) for c in ColorSchemes.get(scheme, range(0, 1, _CMAP_N))]
        Int32(ImPlot.AddColormap(name, cols, _CMAP_N, false))   # false = continuous, not qualitative
    end
end

# A row of ImPlot ColormapButtons; a click sets the TF colormap, and the selected one is outlined.
function _colormap_buttons!(tf::TransferFunction)
    bw = 80.0
    sp = unsafe_load(CImGui.GetStyle().ItemSpacing).x
    right = CImGui.GetCursorScreenPos().x + CImGui.GetContentRegionAvail().x
    for (i, sym) in enumerate(TF_COLORMAPS)
        # Side by side while the next one fits, on the next row when it does not.
        i > 1 && CImGui.GetItemRectMax().x + sp + bw <= right && CImGui.SameLine()
        if ImPlot.ColormapButton(String(sym), ImVec2(bw, 0), _cmap_index(sym))
            tf.colormap = sym
            tf.dirty = true
        end
        sym === tf.colormap && CImGui.AddRect(
            CImGui.GetWindowDrawList(), CImGui.GetItemRectMin(), CImGui.GetItemRectMax(),
            CImGui.GetColorU32(CImGui.ImGuiCol_DragDropTarget), 0.0f0, 3.0f0)
    end
end

# Interactive value→colour scale for the TF window: scroll zooms, drag pans, mutating tf.lo/tf.hi
# (live shader uniforms, so no re-bake). Mirrors the shader normalization — linear → identity over
# (lo,hi); log → log10 over (10^lo, 10^hi) — so the bar and the render can't disagree.
function _tf_colorbar!(tf::TransferFunction, width::Real)
    scale = tf.logscale ? log10 : identity
    cr = tf.logscale ? (exp10(tf.lo), exp10(tf.hi)) : (tf.lo, tf.hi)
    ref = Ref(cr)
    ImPlotExtra.colorbar!("##tfwin", tf.colormap, ref, scale, width; orientation = :horizontal)
    if ref[] != cr                       # write back only on interaction (no log/exp round-trip drift)
        nlo, nhi = ref[]
        tf.lo, tf.hi = tf.logscale ? (log10(nlo), log10(nhi)) : (nlo, nhi)
    end
    nothing
end

# Emit the controls into the current ImGui window (no Begin/End of its own). Built per field kind:
# color (RGBA) fields omit the mode selector + Transfer-function section and render DVR-only.
function ShowControls(view::FieldView)
    _ensure_implot!()
    params = view.params; tf = view.tf; cam = view.camera
    hw = (CImGui.GetContentRegionAvail().x - 6) / 2   # half content width, for paired items
    tf_field = uses_transfer_function(view.field)

    CImGui.SeparatorText("Rendering")
    if tf_field
        let sel = _radio_row("mode", _MODES, params.mode)
            sel !== nothing && (params.mode = sel)
        end
    else
        params.mode = 0
    end
    if params.mode == 0
        tf_field && CImGui.SameLine()
        CImGui.SetNextItemWidth(hw)
        let v = Ref(Cfloat(params.opacity_scale))
            (@c CImGui.SliderFloat("##opacity", &v[], 0.02f0, 5.0f0, "opacity: %.2f",
                                   CImGui.ImGuiSliderFlags_Logarithmic)) && (params.opacity_scale = v[])
        end
    end
    CImGui.SetNextItemWidth(hw)
    let v = Ref(Cfloat(params.step_scale))
        (@c CImGui.SliderFloat("##stepk", &v[], 1.0f-3, 1.0f3, "step scale: %.4g",
                               CImGui.ImGuiSliderFlags_Logarithmic)) && (params.step_scale = v[])
    end
    CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
    let cur = something(findfirst(==(params.max_render_px), _RES_VALUES), length(_RES_VALUES)),
        v = Ref(Cint(cur - 1))
        (@c CImGui.SliderInt("##maxpx", &v[], 0, length(_RES_VALUES) - 1,
                             "max render px: $(_RES_VALUES[v[]+1])")) &&
            (params.max_render_px = _RES_VALUES[v[]+1])
    end
    let v = Ref(params.interp == 1)
        (@c CImGui.Checkbox("trilinear (vs nearest)", &v[])) && (params.interp = v[] ? 1 : 0)
    end

    if tf_field
        vmin, vmax = value_range(view.field)
        CImGui.SeparatorText("Transfer function")
        _colormap_buttons!(tf)
        let v = Ref(tf.logscale)
            (@c CImGui.Checkbox("log scale", &v[])) &&
                (tf.logscale = v[]; default_window!(tf, (vmin, vmax)))
        end
        _tf_colorbar!(tf, CImGui.GetContentRegionAvail().x)
    end

    CImGui.SeparatorText("View")
    let sel = _radio_row("proj", (("perspective", nothing), ("ortho", nothing)),
                         cam.projection === :perspective ? 0 : 1)
        sel !== nothing && (cam.projection = sel == 0 ? :perspective : :ortho)
    end
    if CImGui.Button("reset view")
        c, r = bounds(framing_region(view))
        reset!(cam, Camera(; center = c, radius = r))
    end
    CImGui.SameLine()
    let v = Ref(params.flymode)
        (@c CImGui.Checkbox("WASD mode", &v[])) && (params.flymode = v[])
    end
    CImGui.SetItemTooltip("W/S forward·back   A/D strafe   Space/C up·down   Shift sprint   drag to look")
    if params.flymode
        CImGui.SameLine(); CImGui.SetNextItemWidth(hw)
        _, r = bounds(framing_region(view))
        let v = Ref(Cfloat(params.fly_speed))
            (@c CImGui.SliderFloat("##flyspeed", &v[], Cfloat(r / 2000), Cfloat(r / 20),
                                   "fly speed: %.4f", CImGui.ImGuiSliderFlags_Logarithmic)) &&
                (params.fly_speed = v[])
        end
    end
    nothing
end

module VolumeRendering

include("gl.jl")
include("camera.jl")
include("transfer_function.jl")
include("region.jl")
include("field_source.jl")
include("vector_field.jl")
include("geometry_renderer.jl")
include("volume_renderer.jl")   # defines the Overlay protocol + GeomContext, used by overlays.jl
include("overlays.jl")
include("imgui.jl")             # CImGui interop: ShowVolume widget + camera input
include("controls.jl")          # ShowControls panel (CImGui + ImPlot)

# ── public API ──
export FieldView, render!, RenderParams, TransferFunction, Camera
export default_window!, normalize_points!   # transfer-function window / opacity-curve helpers
export FieldSource, GLSLField, axis_index_glsl
export Region, BoxRegion, SphereRegion
# interface functions users overload for custom sources / regions:
export field_glsl, region, value_range, fingerprint, upload_field, bind_field!, free_field!
export uses_transfer_function
export region_glsl, bind_region!, bounds
# camera controls (host wires input to these):
export orbit!, flylook!, pan!, zoom!, fly!, reset!, view_proj
# overlay protocol + concrete overlays:
export Overlay, enabled, refresh!, draw!, aabb
export SliceOverlay, BoxOutlineOverlay, SphereOutlineOverlay, SphereOverlay, LineOverlay, AxesOverlay
export StreamlinesOverlay, GlyphsOverlay, default_overlays
# vector fields:
export VectorField, compute_streamlines

# The UI (`VolumeRendering.ShowVolume` widget + `VolumeRendering.ShowControls` panel) lives in
# imgui.jl / controls.jl, included above. Both are intentionally NOT exported — call them qualified.

end # module VolumeRendering

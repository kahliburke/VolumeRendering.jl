# The render core: a full-res geometry G-buffer (overlays → color + depth texture), a low-res volume
# march (clipped at the geometry depth, premultiplied), and a composite (volume over geometry). Ported
# and generalized from the GRMHD viewer: the field is any `FieldSource` (its `sampleField`/`stepSize`
# GLSL is injected into the volume/slice programs), the march region is any `Region` (its
# `intersectRegion` GLSL is injected), and every pass sets ALL the GL state it depends on so multiple
# FieldViews (and any host) can share one GL context without corrupting each other.

using StaticArrays
using ConstructionBase: getfields

# ── Overlay protocol (concrete overlays + GeomContext usage live in overlays.jl) ──
# Every overlay keeps its lazily-built GL resources in a single field named `gl` (a CachedProgram or
# CachedLines) and its image-affecting state in the remaining fields — which is what lets `enabled`
# and `overlay_fingerprint` below be shared instead of rewritten per overlay.
abstract type Overlay end
enabled(o::Overlay) = o.enabled
refresh!(::Overlay, ctx) = nothing      # default: stateless overlays need no per-frame update
# Every input that affects the image, derived from the fields, so a new one can't be forgotten here.
overlay_fingerprint(o::Overlay) = (typeof(o), Base.structdiff(getfields(o), NamedTuple{(:gl,)}))
# draw!(o::Overlay, ctx) has no default — every concrete overlay defines it.

# Shared per-frame inputs handed to each overlay's refresh!/draw! (no field-specific/GRMHD types).
struct GeomContext
    viewproj::Vector{Float32}   # collected view·projection (matches camera.jl view_proj)
    vieww::Int; viewh::Int      # geometry-pass pixel size (device res)
    cam::Camera
    aspect::Float64
    field                       # the active FieldSource
    gpu                         # its uploaded GPU handle (so a SliceOverlay can bind_field! + sample it)
    region::Region
    tf::TransferFunction
    params                      # ::RenderParams
end

# All shader-driving state that isn't the camera, transfer function, field, or region.
@kwdef mutable struct RenderParams
    mode::Int = 0               # 0 emission-absorption, 1 MIP, 2 average
    interp::Int = 1             # 0 nearest, 1 trilinear (the field snippet may honor this)
    step_scale::Float64 = 0.5   # adaptive step dt = step_scale · stepSize(pos) — quality knob
    opacity_scale::Float64 = 1.0  # DVR extinction per unit length
    max_render_px::Int = 2048   # cap the VOLUME long side; geometry/composite are always full res
    refine_samples::Int = 16    # jittered frames averaged while idle for progressive refinement; 1 disables it
    flymode::Bool = false
    fly_speed::Float64 = 0.25
end

# ── shared per-program uniform groups ──
function set_camera_uniforms!(prog, cam::Camera, aspect)
    right, up, fwd = view_basis(cam)
    uni_3f(prog, "camEye", cam.eye...);  uni_3f(prog, "camRight", right...)
    uni_3f(prog, "camUp", up...);        uni_3f(prog, "camFwd", fwd...)
    uni_i(prog, "perspective", cam.projection === :perspective ? 1 : 0)
    uni_f(prog, "fovscale", fovscale(cam)); uni_f(prog, "aspect", aspect)
end
# tf on the engine-reserved unit 14 (fields use 0..13, geometry depth uses 15).
function bind_tf_uniforms!(prog, tf::TransferFunction)
    bind_sampler(prog, "tfTex", 14, GL.GL_TEXTURE_2D, tf.tex)
    uni_i(prog, "logScale", tf.logscale ? 1 : 0); uni_f(prog, "lo", tf.lo); uni_f(prog, "hi", tf.hi)
end

# GL resources shared by a FieldView across frames. `volprog` is (re)built per (field_glsl, region_glsl)
# in `_ensure_built!`; `compprog`/`vao`/framebuffers are stable.
mutable struct VolumeRenderer
    volprog::UInt32          # volume march (field+region snippets injected)
    compprog::UInt32         # composite volume over geometry
    vao::UInt32              # shared fullscreen-triangle VAO
    geom_fb::Framebuffer     # full device-res: color + sampleable depth texture (the G-buffer)
    vol_fb::Framebuffer      # low-res: premultiplied color
    accum_fb::Framebuffer    # full-res RGBA16F: running average of composited frames (progressive refinement)
    nsamp::Int               # frames already averaged into accum_fb for the current render_key
end
function VolumeRenderer()
    compprog = link_program("volume.vert", "composite.frag")
    vao = Ref{GL.GLuint}(0); GL.glGenVertexArrays(1, vao)
    VolumeRenderer(0, compprog, vao[], Framebuffer(16, 16; depth = :texture), Framebuffer(16, 16),
                   Framebuffer(16, 16; color = :rgba16f), 0)
end

_default_region(field) = region(field)   # avoids the `region` accessor being shadowed by the kwarg

# A single interactive field view: the field + region + camera + params + transfer function + overlays,
# plus its own GL resources (built lazily on the first `render!`, in whatever context is then current).
mutable struct FieldView
    field
    region::Region
    camera::Camera
    params::RenderParams
    tf::TransferFunction
    overlays::Vector{Overlay}
    vr::Union{Nothing,VolumeRenderer}     # GL resources, lazy
    gpu::Any                              # upload_field handle, lazy
    built_for::Union{Nothing,Tuple{String,String}}   # (field_glsl, region_glsl) the volprog was built for
    uploaded_fp::Any                      # fingerprint of the field whose texture is currently uploaded
    last_key::Base.RefValue{Union{Nothing,UInt64}}   # render-on-demand fingerprint
end
function FieldView(field; overlays = Overlay[], colormap::Symbol = :viridis,
                   region = _default_region(field))
    c, r = bounds(region)
    cam = Camera(; center = c, radius = r)
    tf = TransferFunction(; colormap = colormap)
    default_window!(tf, value_range(field))
    params = RenderParams(; fly_speed = r / 200)
    FieldView(field, region, cam, params, tf, Vector{Overlay}(overlays),
              nothing, nothing, nothing, nothing, Ref{Union{Nothing,UInt64}}(nothing))
end

# Build/rebuild GL resources to match the current field+region. The volume program bakes the field's
# and region's GLSL, so it is rebuilt (and the field texture re-uploaded) whenever either changes —
# e.g. `view.field = other`. Must be called with a current GL context.
function _ensure_built!(view::FieldView)
    view.vr === nothing && (view.vr = VolumeRenderer())
    fg = field_glsl_full(view.field); rg = region_glsl(view.region)
    if view.built_for != (fg, rg)
        view.vr.volprog != 0 && GL.glDeleteProgram(view.vr.volprog)
        vsrc = read(joinpath(SHADER_DIR, "volume.vert"), String)
        fsrc = read(joinpath(SHADER_DIR, "volume.frag"), String)
        view.vr.volprog = link_program_src(vsrc, fsrc;
            includes = Dict{String,String}("field" => fg, "region" => rg))
        view.built_for = (fg, rg)
    end
    # Re-upload the texture whenever the FIELD OBJECT changes — even to a same-shape, different-data
    # array (which has identical GLSL), so `view.field = other` never leaves a stale texture bound.
    fp = fingerprint(view.field)
    if view.gpu === nothing || view.uploaded_fp != fp
        view.gpu !== nothing && free_field!(view.gpu)
        view.gpu = upload_field(view.field)
        view.uploaded_fp = fp
    end
    view
end

# Render-on-demand fingerprint: a hash of EVERY input that affects the rendered image. If unchanged
# from the last frame, `render_frame!` reuses the cached G-buffer texture instead of re-marching.
render_key(view::FieldView, gw, gh, vw, vh) = hash((
    view.camera.eye, view.camera.lookat, view.camera.up, view.camera.projection,
    view.camera.fov, view.camera.ortho_half,
    view.params.mode, view.params.interp, view.params.step_scale, view.params.opacity_scale,
    view.params.refine_samples,
    fingerprint(view.field), fingerprint(view.region),
    view.tf.colormap, view.tf.logscale, view.tf.lo, view.tf.hi, view.tf.opacity_pts,
    Tuple(overlay_fingerprint(o) for o in view.overlays),
    gw, gh, vw, vh))

function geometry_pass!(view::FieldView, ctx::GeomContext)
    for o in view.overlays
        enabled(o) || continue
        refresh!(o, ctx)
        draw!(o, ctx)
    end
end

# Render one frame at geometry size gw×gh and volume size vw×vh; returns the color texture to display —
# the running-average accum_fb while refining, else the G-buffer. Each pass sets all the GL state it
# depends on (defensive for multi-view / arbitrary host state); the previously-bound FB is restored.
#
# Progressive refinement: while the render_key is unchanged, successive frames re-march with a shifted
# jitter (sampleIdx) and fold into accum_fb's running average, converging the jitter noise over
# refine_samples frames. Any input change bumps the key → nsamp resets → the average restarts.
function render_frame!(view::FieldView, gw, gh, vw, vh)
    key = render_key(view, gw, gh, vw, vh)
    vr = view.vr
    @assert view.params.refine_samples >= 1 "refine_samples must be ≥ 1"
    nmax = view.params.refine_samples
    refining = nmax > 1
    # Nothing changed and the target sample count reached → the finished image is already cached.
    key == view.last_key[] && vr.nsamp >= nmax && return refining ? vr.accum_fb.tex : vr.geom_fb.tex
    if key != view.last_key[]
        view.last_key[] = key
        vr.nsamp = 0
    end

    ensure_baked!(view.tf)
    prev_fbo = Ref{GL.GLint}(0); GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING, prev_fbo)
    resize!(vr.geom_fb, gw, gh); resize!(vr.vol_fb, vw, vh)
    refining && resize!(vr.accum_fb, gw, gh)
    aspect = gw / gh
    center, radius = bounds(view.region)
    vpm = view_proj(view.camera, center, radius, aspect)
    vp  = vec(Matrix{Float32}(vpm))
    ivp = vec(Matrix{Float32}(inv(vpm)))
    ctx = GeomContext(vp, gw, gh, view.camera, aspect, view.field, view.gpu, view.region, view.tf, view.params)

    # ── Pass 1: geometry G-buffer (full device-res) ──
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, vr.geom_fb.fbo); GL.glViewport(0, 0, gw, gh)
    GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE)
    GL.glEnable(GL.GL_DEPTH_TEST); GL.glDepthFunc(GL.GL_LESS); GL.glDepthMask(GL.GL_TRUE)
    GL.glDisable(GL.GL_BLEND)
    GL.glClearColor(0, 0, 0, 1); GL.glClearDepth(1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    geometry_pass!(view, ctx)

    # ── Pass 2: volume march (low-res), clipped at the geometry depth ──
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, vr.vol_fb.fbo); GL.glViewport(0, 0, vw, vh)
    GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE)
    GL.glDisable(GL.GL_DEPTH_TEST); GL.glDisable(GL.GL_BLEND)
    GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(vr.volprog); GL.glBindVertexArray(vr.vao)
    set_camera_uniforms!(vr.volprog, view.camera, aspect)
    bind_field!(view.gpu, vr.volprog); uni_i(vr.volprog, "interp", view.params.interp)
    bind_region!(view.region, vr.volprog)
    bind_tf_uniforms!(vr.volprog, view.tf)
    uni_i(vr.volprog, "mode", view.params.mode)
    uni_f(vr.volprog, "opacityScale", view.params.opacity_scale)
    uni_i(vr.volprog, "steps", 2048)
    uni_f(vr.volprog, "stepScale", view.params.step_scale)
    uni_m4(vr.volprog, "invViewProj", ivp)
    uni_2i(vr.volprog, "geomSize", gw, gh); uni_2i(vr.volprog, "volSize", vw, vh)
    uni_i(vr.volprog, "sampleIdx", vr.nsamp)
    bind_sampler(vr.volprog, "geomDepthTex", 15, GL.GL_TEXTURE_2D, vr.geom_fb.depth)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    # ── Pass 3: composite volume over geometry (into the G-buffer) ──
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, vr.geom_fb.fbo); GL.glViewport(0, 0, gw, gh)
    GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE); GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
    GL.glUseProgram(vr.compprog); GL.glBindVertexArray(vr.vao)
    bind_sampler(vr.compprog, "volTex", 0, GL.GL_TEXTURE_2D, vr.vol_fb.tex)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
    GL.glDisable(GL.GL_BLEND)
    vr.nsamp += 1

    # ── Pass 4: fold the freshly composited frame into accum_fb's running average (only while refining) ──
    if refining
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, vr.accum_fb.fbo); GL.glViewport(0, 0, gw, gh)
        GL.glDisable(GL.GL_SCISSOR_TEST); GL.glDisable(GL.GL_CULL_FACE); GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(vr.compprog); GL.glBindVertexArray(vr.vao)
        bind_sampler(vr.compprog, "volTex", 0, GL.GL_TEXTURE_2D, vr.geom_fb.tex)
        if vr.nsamp == 1
            GL.glDisable(GL.GL_BLEND)                          # first frame: overwrite (also clears stale float)
        else
            GL.glEnable(GL.GL_BLEND)                           # accum ← mix(accum, frame, 1/n): equal-weight mean
            GL.glBlendColor(0f0, 0f0, 0f0, 1f0 / vr.nsamp)
            GL.glBlendFunc(GL.GL_CONSTANT_ALPHA, GL.GL_ONE_MINUS_CONSTANT_ALPHA)
        end
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glDisable(GL.GL_BLEND)
    end

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fbo[])
    refining ? vr.accum_fb.tex : vr.geom_fb.tex
end

# Public L1 entry point: render the view at `w`×`h` device pixels; returns the color texture to display
# (the RGBA16F progressive-refinement average while refining, else the RGBA8 G-buffer). The VOLUME long
# side is capped at `params.max_render_px` (geometry stays full-res). A current GL context is required;
# GL resources are allocated on the first call.
function render!(view::FieldView, w::Integer, h::Integer)
    _ensure_built!(view)
    gw = max(1, Int(w)); gh = max(1, Int(h))
    scale = min(1.0, view.params.max_render_px / max(gw, gh))
    vw = max(1, round(Int, gw * scale)); vh = max(1, round(Int, gh * scale))
    render_frame!(view, gw, gh, vw, vh)
end

# Concrete overlays for the geometry G-buffer. Each is an INDEPENDENT unit (its own GL program + draw)
# that writes opaque color + depth in the shared `viewProj` convention, so `GL_LESS` composes them and
# the volume clips against their union. The protocol (`abstract type Overlay`, `enabled`, `refresh!`,
# `draw!`, `overlay_fingerprint`) lives in volume_renderer.jl; here we only add concrete types + methods.
#
# Every overlay follows the same shape: image-affecting state in plain fields, GL resources in a single
# `gl` field (CachedProgram / CachedLines) that builds itself on first `refresh!` — so an overlay can be
# constructed with no current GL context, and `enabled`/`overlay_fingerprint` come from the protocol.
# Fields carry their own types, so `@kwdef` constructors convert whatever the caller passes.

using StaticArrays

_axis_int(a::Symbol) = a === :x ? 0 : a === :y ? 1 : a === :z ? 2 :
    error("SliceOverlay axis must be :x, :y, or :z; got $(repr(a))")

# ── SliceOverlay: world-space quad on {axis = pos}, sampling the field, TF-colored ──
# slice.frag injects the field snippet, so the program is PER-FIELD and relinked when the field's
# GLSL changes.
@kwdef mutable struct SliceOverlay <: Overlay
    axis::Int                       # 0 = x, 1 = y, 2 = z
    pos::Float64 = 0.0
    enabled::Bool = true
    gl::CachedProgram = CachedProgram()
end
SliceOverlay(axis::Symbol; kwargs...) = SliceOverlay(; axis = _axis_int(axis), kwargs...)

function refresh!(o::SliceOverlay, ctx::GeomContext)
    fg = field_glsl_full(ctx.field)
    refresh!(o.gl, fg) do
        link_program_src(read(joinpath(SHADER_DIR, "slice.vert"), String),
                         read(joinpath(SHADER_DIR, "slice.frag"), String);
                         includes = Dict("field" => fg))
    end
    o
end

function draw!(o::SliceOverlay, ctx::GeomContext)
    lo, hi = aabb(ctx.region)
    prog = use!(o.gl)
    uni_m4(prog, "viewProj", ctx.viewproj)
    uni_i(prog, "sliceAxis", o.axis)
    uni_f(prog, "slicePos", o.pos)
    uni_3f(prog, "boxLo", lo...); uni_3f(prog, "boxHi", hi...)
    uni_i(prog, "interp", ctx.params.interp)
    bind_field!(ctx.gpu, prog)
    bind_tf_uniforms!(prog, ctx.tf)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
end

# ── SurfaceSliceOverlay: a region's near boundary, sampling the field, TF-colored ──
# SliceOverlay on a curved surface. surface_slice.frag injects the region snippet next to the field
# one, so the program is PER-(FIELD, REGION) and relinked when either's GLSL changes.
@kwdef mutable struct SurfaceSliceOverlay <: Overlay
    region::Region
    enabled::Bool = true
    gl::CachedProgram = CachedProgram()
end
SurfaceSliceOverlay(region::Region; kwargs...) = SurfaceSliceOverlay(; region, kwargs...)

function refresh!(o::SurfaceSliceOverlay, ctx::GeomContext)
    fg = field_glsl_full(ctx.field); rg = region_glsl(o.region)
    refresh!(o.gl, (fg, rg)) do
        link_program_src(read(joinpath(SHADER_DIR, "volume.vert"), String),
                         read(joinpath(SHADER_DIR, "surface_slice.frag"), String);
                         includes = Dict("field" => fg, "region" => rg))
    end
    o
end

function draw!(o::SurfaceSliceOverlay, ctx::GeomContext)
    prog = use!(o.gl)
    set_camera_uniforms!(prog, ctx.cam, ctx.aspect)
    uni_m4(prog, "viewProj", ctx.viewproj)
    uni_i(prog, "interp", ctx.params.interp)
    bind_region!(o.region, prog)
    bind_field!(ctx.gpu, prog)
    bind_tf_uniforms!(prog, ctx.tf)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
end

# ── SphereOverlay: analytic ray–sphere impostor over the fullscreen triangle ──
@kwdef mutable struct SphereOverlay <: Overlay
    center::SVector{3,Float64} = SVector(0.0, 0.0, 0.0)
    radius::Float64
    color::RGB{Float64} = RGB(0.06, 0.06, 0.09)
    enabled::Bool = true
    gl::CachedProgram = CachedProgram()
end

refresh!(o::SphereOverlay, ::GeomContext) = (refresh!(o.gl, "volume.vert", "sphere.frag"); o)

function draw!(o::SphereOverlay, ctx::GeomContext)
    prog = use!(o.gl)
    set_camera_uniforms!(prog, ctx.cam, ctx.aspect)
    uni_m4(prog, "viewProj", ctx.viewproj)
    uni_3f(prog, "sphereCenter", o.center...)
    uni_f(prog, "sphereRadius", o.radius)
    uni_3f(prog, "sphereColor", o.color)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
end

# ── LineOverlay: screen-space constant-pixel-width line between two world endpoints ──
@kwdef mutable struct LineOverlay <: Overlay
    a::SVector{3,Float64}
    b::SVector{3,Float64}
    width_px::Float64 = 2.0
    color::RGB{Float64} = RGB(0.8, 0.8, 0.85)
    enabled::Bool = true
    gl::CachedProgram = CachedProgram()
end
LineOverlay(a, b; kwargs...) = LineOverlay(; a, b, kwargs...)

refresh!(o::LineOverlay, ::GeomContext) = (refresh!(o.gl, "line.vert", "line.frag"); o)

function draw!(o::LineOverlay, ctx::GeomContext)
    prog = use!(o.gl)
    uni_m4(prog, "viewProj", ctx.viewproj)
    uni_3f(prog, "axisA", o.a...); uni_3f(prog, "axisB", o.b...)
    uni_2f(prog, "viewportPx", ctx.vieww, ctx.viewh)
    uni_f(prog, "lineWidthPx", o.width_px)
    uni_3f(prog, "lineColor", o.color)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
end

# ── AxesOverlay: three colored screen-space lines through the region center spanning the region ──
@kwdef mutable struct AxesOverlay <: Overlay
    width_px::Float64 = 2.0
    colors::NTuple{3,RGB{Float64}} = (RGB(1.0, 0.3, 0.3), RGB(0.3, 1.0, 0.3), RGB(0.4, 0.6, 1.0))
    enabled::Bool = true
    gl::CachedProgram = CachedProgram()
end

refresh!(o::AxesOverlay, ::GeomContext) = (refresh!(o.gl, "line.vert", "line.frag"); o)

const _AXIS_UNITS = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))

function draw!(o::AxesOverlay, ctx::GeomContext)
    center, radius = bounds(ctx.region)
    prog = use!(o.gl)
    uni_m4(prog, "viewProj", ctx.viewproj)
    uni_2f(prog, "viewportPx", ctx.vieww, ctx.viewh)
    uni_f(prog, "lineWidthPx", o.width_px)
    for (ê, color) in zip(_AXIS_UNITS, o.colors)
        a = center - radius * ê
        b = center + radius * ê
        uni_3f(prog, "axisA", a...); uni_3f(prog, "axisB", b...)
        uni_3f(prog, "lineColor", color)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    end
end

# ── BoxOutlineOverlay: the 12 edges of a region's AABB as GL_LINES (via a GeometryRenderer) ──
@kwdef mutable struct BoxOutlineOverlay <: Overlay
    region::Union{Nothing,Region} = nothing   # explicit region, else ctx.region at draw time
    color::RGB{Float64} = RGB(0.5, 0.5, 0.55)
    enabled::Bool = true
    gl::CachedLines = CachedLines()
end

# Interleaved (x,y,z, r,g,b) vertices for the 12 edges of [lo,hi] (drawn as GL_LINES).
function _boxoutline_verts(lo, hi, color)
    c = RGB{Float32}(color)
    xs = (lo[1], hi[1]); ys = (lo[2], hi[2]); zs = (lo[3], hi[3])
    corner(ix, iy, iz) = (Float32(xs[ix]), Float32(ys[iy]), Float32(zs[iz]))
    edges = (
        ((1, 1, 1), (2, 1, 1)), ((1, 2, 1), (2, 2, 1)),     # 4 edges ∥ x
        ((1, 1, 2), (2, 1, 2)), ((1, 2, 2), (2, 2, 2)),
        ((1, 1, 1), (1, 2, 1)), ((2, 1, 1), (2, 2, 1)),     # 4 edges ∥ y
        ((1, 1, 2), (1, 2, 2)), ((2, 1, 2), (2, 2, 2)),
        ((1, 1, 1), (1, 1, 2)), ((2, 1, 1), (2, 1, 2)),     # 4 edges ∥ z
        ((1, 2, 1), (1, 2, 2)), ((2, 2, 1), (2, 2, 2)),
    )
    out = Float32[]
    for (ca, cb) in edges
        append!(out, (corner(ca...)..., c.r, c.g, c.b, corner(cb...)..., c.r, c.g, c.b))
    end
    out
end

function refresh!(o::BoxOutlineOverlay, ctx::GeomContext)
    lo, hi = aabb(something(o.region, ctx.region))
    refresh!(o.gl, (lo, hi, o.color)) do
        _boxoutline_verts(lo, hi, o.color)
    end
    o
end

draw!(o::BoxOutlineOverlay, ctx::GeomContext) = (draw!(o.gl, ctx.viewproj); o)

# ── SphereOutlineOverlay: lat/lon graticule of a region's bounding sphere, as GL_LINES ──
# The mirror of BoxOutlineOverlay: that one outlines `aabb(region)`, this one `bounds(region)`. An
# arbitrary reference sphere is a `SphereRegion`, whose bounds are itself.
@kwdef mutable struct SphereOutlineOverlay <: Overlay
    region::Union{Nothing,Region} = nothing         # explicit region, else ctx.region at draw time
    parallels::Union{Integer,AbstractVector} = 3    # latitudes, or how many of them
    meridians::Union{Integer,AbstractVector} = 6    # azimuths, or how many of them
    segments::Int = 64                              # segments per full circle
    color::RGB{Float64} = RGB(0.5, 0.5, 0.55)
    enabled::Bool = true
    gl::CachedLines = CachedLines()
end

# Latitudes of the parallels: given explicitly, or a count ⇒ evenly spaced with the poles left out
# (so an odd count includes the equator).
_parallel_lats(n::Integer) = range(-π/2, π/2, n + 2)[2:end-1]
_parallel_lats(lats) = lats

# Azimuths of the meridians: given explicitly, or a count ⇒ evenly spaced over the full turn. A
# meridian is a half great circle pole to pole, so a count of 6 draws as 3 great circles.
_meridian_lons(n::Integer) = range(; start = 0.0, step = 2π / n, length = n)
_meridian_lons(lons) = lons

_sphere_point(center, radius, lat, lon) =
    center + radius * SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))

# Consecutive point pairs of a polyline, as GL_LINES vertices.
function _push_polyline!(out, points, c::RGB{Float32})
    for (a, b) in zip(points, @view points[2:end])
        append!(out, (a..., c.r, c.g, c.b, b..., c.r, c.g, c.b))
    end
    out
end

# Interleaved (x,y,z, r,g,b) vertices for the graticule of the sphere (center, radius): rings at
# fixed latitude plus half great circles pole to pole. Angles are radians — Unitful angles work too,
# since they only ever reach `sin`/`cos`. `segments` counts a FULL circle, so a meridian gets half.
function _sphereoutline_verts(center, radius, parallels, meridians, segments, color)
    c = RGB{Float32}(color)
    out = Float32[]
    for lat in _parallel_lats(parallels)
        _push_polyline!(out, [_sphere_point(center, radius, lat, lon)
                              for lon in range(0, 2π, segments + 1)], c)
    end
    for lon in _meridian_lons(meridians)
        _push_polyline!(out, [_sphere_point(center, radius, lat, lon)
                              for lat in range(-π/2, π/2, segments ÷ 2 + 1)], c)
    end
    out
end

function refresh!(o::SphereOutlineOverlay, ctx::GeomContext)
    center, radius = bounds(something(o.region, ctx.region))
    refresh!(o.gl, (center, radius, o.parallels, o.meridians, o.segments, o.color)) do
        _sphereoutline_verts(center, radius, o.parallels, o.meridians, o.segments, o.color)
    end
    o
end

draw!(o::SphereOutlineOverlay, ctx::GeomContext) = (draw!(o.gl, ctx.viewproj); o)

# ── StreamlinesOverlay: evenly-spaced field lines of a VectorField, colored by |V| ──
@kwdef mutable struct StreamlinesOverlay <: Overlay
    vf::VectorField
    colormap::Symbol = :viridis
    min_density::Int = 1
    max_density::Int = 2
    mag_window::Union{Nothing,Tuple} = nothing
    enabled::Bool = true
    gl::CachedLines = CachedLines()
end
StreamlinesOverlay(vf::VectorField; kwargs...) = StreamlinesOverlay(; vf, kwargs...)

function refresh!(o::StreamlinesOverlay, ::GeomContext)
    refresh!(o.gl, (o.vf, o.colormap, o.min_density, o.max_density, o.mag_window)) do
        data = compute_streamlines(o.vf; o.min_density, o.max_density)
        mw = @something o.mag_window magnitude_window(data)
        streamline_vertices(data, o.colormap, mw...)
    end
    o
end

draw!(o::StreamlinesOverlay, ctx::GeomContext) = (draw!(o.gl, ctx.viewproj); o)

# ── GlyphsOverlay: arrow glyphs subsampled from the streamlines of a VectorField ──
@kwdef mutable struct GlyphsOverlay <: Overlay
    vf::VectorField
    colormap::Symbol = :viridis
    min_density::Int = 1
    max_density::Int = 2
    every::Int = 12
    mag_window::Union{Nothing,Tuple} = nothing
    enabled::Bool = true
    gl::CachedLines = CachedLines()
end
GlyphsOverlay(vf::VectorField; kwargs...) = GlyphsOverlay(; vf, kwargs...)

function refresh!(o::GlyphsOverlay, ::GeomContext)
    refresh!(o.gl, (o.vf, o.colormap, o.min_density, o.max_density, o.every, o.mag_window)) do
        data = compute_streamlines(o.vf; o.min_density, o.max_density)
        mw = @something o.mag_window magnitude_window(data)
        glyph_vertices(data, o.colormap, mw...; o.every)
    end
    o
end

draw!(o::GlyphsOverlay, ctx::GeomContext) = (draw!(o.gl, ctx.viewproj); o)

# Hosts add overlays explicitly; the default stack is empty.
default_overlays(field) = Overlay[]

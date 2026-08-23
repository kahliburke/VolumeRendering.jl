# VolumeRendering.jl

**An embeddable [Dear ImGui](https://github.com/JuliaImGui/CImGui.jl) widget for real-time GPU volume rendering of 3D fields, with composable overlays.**

VolumeRendering turns a 3D scalar field into an interactive, ray-marched image on the GPU, and hands it to you as a single ImGui widget you can drop into your own application — the way you'd drop a plot into a window. Scalar fields are volume-rendered with a configurable transfer function; vector fields are shown as streamlines and glyphs. Slices, bounding boxes, spheres and axes layer on top as overlays.

The recommended way to use `VolumeRendering.jl` is as a Dear ImGui widget, but it can be integrated into other environments that have OpenGL, can render a texture, and pass mouse events.

Simple example, see [demo.jl](examples/demo.jl):

https://github.com/user-attachments/assets/0ff46ebc-9a73-4ef7-b758-9f330e348ea5

In [PlasmaScope.jl](https://github.com/aplavin/PlasmaScope.jl) that uses `VolumeRendering.jl` to display GRHMD snapshots:

https://github.com/user-attachments/assets/fb981f3e-8402-4861-85fb-c061f06532b5

# Quick start

```julia
using VolumeRendering, AxisKeys

# your data: any 3D `KeyedArray` whose axis keys give the world coordinates
A = KeyedArray(...; x = xs, y = ys, z = zs)

view = FieldView(A)

# ...then, inside your Dear ImGui frame (with CImGui loaded):
ShowField(view)
```

`ShowField` fills the current content region with the rendered view and turns mouse/keyboard into camera motion. It becomes available as soon as you `using CImGui` — no window or render-loop boilerplate of our own; you drive the ImGui frame like you already do for any other widget.

The camera has two modes: an **orbit** camera (drag / pan / wheel-zoom) and a **first-person fly** camera (`view.params.flymode = true`, WASD to move).

# Features

- **Scalar fields** — arbitrary grids or analytic fields rendered on GPU
- **Vector fields** — streamlines and arrows
- **Transfer function** — any [ColorSchemes](https://github.com/JuliaGraphics/ColorSchemes.jl) colormap plus an opacity curve
- Composable **overlays** — slices/outlines/geometries/lines/...

# Scalar fields

## Rectilinear grid

Any `KeyedArray{<:Real,3}` is a volume field out of the box — its axis keys define the world coordinates, and its value range sets the default transfer-function window:

```julia
# your data, on any grid:
xs = ys = zs = -1:0.02:1
A = KeyedArray([exp(-(x^2 + y^2 + z^2) / 0.1) for x in xs, y in ys, z in zs];
               x = xs, y = ys, z = zs)

view = FieldView(A)
```

For a non-uniform axis, define one method — `VolumeRendering.axis_index_glsl(k::MyAxis, coord)` returning the GLSL expression that maps a world coordinate to a fractional cell index — and your axis type works inside a `KeyedArray` like any range.

## Analytic

`GLSLField` samples a field defined directly in GLSL — no array, no texture upload:

```julia
f = GLSLField("""
        float sampleField(vec3 p){ return exp(-dot(p, p) * 0.7); }
        float stepSize(vec3 p){ return 0.05; }
        """;
    region = SphereRegion(radius = 3.0),
    value_range = (0.0, 1.0))
view = FieldView(f)
```

## Custom source

Any type can be a volume field by implementing the field interface — the GLSL that samples it plus a small GPU lifecycle:

```julia
field_glsl(f)      # GLSL defining `float sampleField(vec3 p)` and `float stepSize(vec3 p)`
region(f)          # the Region the field lives in
value_range(f)     # (min, max) for the default TF window
fingerprint(f)     # hashable identity, for render caching
upload_field(f)    # once: allocate GL resources, return a handle
bind_field!(h, p)  # per frame: bind the handle's textures/uniforms to program `p`
free_field!(h)     # teardown
```

# Vector fields

A `VectorField` wraps a Julia function `p -> v` over a region. It is drawn as evenly-spaced streamlines and/or arrow glyphs, colored by magnitude — add `StreamlinesOverlay`/`GlyphsOverlay` to a view:

```julia
using StaticArrays

vf = VectorField(p -> SVector(-p[2], p[1], 0.2);            # your field, on any region
                 region = BoxRegion((-1, -1, -1), (1, 1, 1)))

view = FieldView(A; overlays = [
    StreamlinesOverlay(vf),        # field lines
    GlyphsOverlay(vf; every = 8),  # arrows, every 8th sample
])
```

# Extras

## Overlays

Overlays draw into the same scene as the volume and clip against it. Pass them when constructing a view (or push to `view.overlays` later):

```julia
using StaticArrays

view = FieldView(A; overlays = [
    SliceOverlay(:z; pos = 0.0),      # a planar slice through the field
    BoxOutlineOverlay(),              # edges of the region's bounding box
    SphereOutlineOverlay(),           # lat/lon graticule of the region's bounding sphere
    AxesOverlay(),                    # x/y/z axes through the center
    SphereOverlay(radius = 0.4),      # sphere geometry
])
```

Each overlay carries its own `enabled` flag, so a host UI can toggle them per frame.

`BoxOutlineOverlay`/`SphereOutlineOverlay` outline the view's region by default; pass `region` to outline
another one — e.g. `SphereOutlineOverlay(region = SphereRegion(radius = 8.0))` for a reference sphere.
Their `parallels`/`meridians` take either a count or explicit angles, in radians or as Unitful angles:

```julia
SphereOutlineOverlay(parallels = 5, meridians = 12)          # evenly spaced
SphereOutlineOverlay(parallels = [-30u"°", 0u"°", 30u"°"])   # exactly these latitudes
```

## Render parameters

Basic controls UI is invoked by right-click. Parameters can also be changed programmatically on `view`.

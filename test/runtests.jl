# NOTE: this suite uses a single shared GLFW/OpenGL context for all render checks (rather than the
# usual per-@testitem isolation) — every test needs a live GL context, and one shared hidden context
# is both faster and avoids multiple concurrent contexts. All GL work happens inside `withctx`.

using Test
using VolumeRendering, AxisKeys, StaticArrays
using ColorTypes, FixedPointNumbers, Unitful
import GLFW, ModernGL as GL

# ── headless GL context + pixel readback helpers ──
function withctx(f; w = 64, h = 64)
    GLFW.Init()
    GLFW.WindowHint(GLFW.VISIBLE, false)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 4); GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 1)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    win = GLFW.CreateWindow(w, h, "VolumeRendering tests"); GLFW.MakeContextCurrent(win)
    try f() finally GLFW.DestroyWindow(win); GLFW.Terminate() end
end
function renderbuf(view, w)
    tex = render!(view, w, w)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, view.vr.geom_fb.fbo)
    buf = Array{UInt8}(undef, 4, w, w)
    GL.glReadPixels(0, 0, w, w, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buf); GL.glFinish()
    (tex, buf)
end
litfrac(b) = count(>(4), @view b[1:3, :, :]) / (3 * size(b, 2) * size(b, 3))

const W = 256
gaussian(xs, ys, zs; c = (0.0, 0.0, 0.0), s = 0.2) =
    Float64[exp(-((x-c[1])^2 + (y-c[2])^2 + (z-c[3])^2) / s) for x in xs, y in ys, z in zs]

# a custom (non-AbstractRange) uniform axis, to exercise the axis_index_glsl extension seam
struct UniformAxis <: AbstractVector{Float64}; a::Float64; d::Float64; n::Int; end
Base.size(k::UniformAxis) = (k.n,)
Base.getindex(k::UniformAxis, i::Int) = k.a + k.d * (i - 1)
VolumeRendering.axis_index_glsl(k::UniformAxis, c) = "((" * c * ") - " * repr(k.a) * ") * " * repr(inv(k.d))

@testset "VolumeRendering" begin
    withctx() do
        xs = ys = zs = -1.0:0.05:1.0

        @testset "KeyedArray volume renders" begin
            v = FieldView(KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs))
            _, b = renderbuf(v, W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.02
        end

        @testset "coordinate mapping (x>0 field → image right)" begin
            step = Float64[x > 0 ? 1.0 : 0.0 for x in xs, y in ys, z in zs]
            v = FieldView(KeyedArray(step; x = xs, y = ys, z = zs))
            v.params.mode = 1                                   # MIP
            v.camera.projection = :ortho
            v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2
            _, b = renderbuf(v, W)
            lum = sum(Int.(b[1:3, :, :]); dims = 1)[1, :, :]
            left = sum(@view lum[1:W÷2, :]); right = sum(@view lum[W÷2+1:W, :])
            @test right > 3 * max(left, 1)                      # +x maps to camera-right
        end

        @testset "RGBA color field: intrinsic color + placement" begin
            # red where x>0, green where x<0; the field carries its own color (no colormap).
            grid = [RGBA{N0f8}(x > 0 ? 1.0 : 0.0, x > 0 ? 0.0 : 1.0, 0.0, 0.5) for x in xs, y in ys, z in zs]
            A = KeyedArray(grid; x = xs, y = ys, z = zs)
            @test uses_transfer_function(A) == false
            v = FieldView(A)
            v.camera.projection = :ortho
            v.camera.eye = SVector(0.0, 0.0, 3.0); v.camera.lookat = SVector(0.0, 0.0, 0.0)
            v.camera.up = SVector(0.0, 1.0, 0.0); v.camera.ortho_half = 1.2
            _, b = renderbuf(v, W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.02
            R, G = Int.(b[1, :, :]), Int.(b[2, :, :])
            rR, rG = sum(@view R[W÷2+1:W, :]), sum(@view G[W÷2+1:W, :])   # +x half (camera-right)
            lR, lG = sum(@view R[1:W÷2, :]),   sum(@view G[1:W÷2, :])     # -x half
            @test rR > 3 * max(rG, 1)     # right half is red
            @test lG > 3 * max(lR, 1)     # left half is green
        end

        @testset "GLSLField (analytic) + SphereRegion" begin
            f = GLSLField("""
                float sampleField(vec3 p){ return exp(-dot(p,p)*0.7); }
                float stepSize(vec3 p){ return 0.05; }""";
                region = SphereRegion(radius = 3.0), value_range = (0.0, 1.0))
            _, b = renderbuf(FieldView(f), W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.02
        end

        @testset "custom axis type == equivalent range axis (pixel-identical)" begin
            blob = gaussian(xs, ys, zs)
            _, bref = renderbuf(FieldView(KeyedArray(blob; x = xs, y = ys, z = zs)), W)
            ax = UniformAxis(-1.0, 0.05, length(xs))
            _, bcus = renderbuf(FieldView(KeyedArray(blob; x = ax, y = ys, z = zs)), W)
            @test bref == bcus
        end

        @testset "all overlays render without GL error" begin
            A = KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)
            vf = VectorField(p -> SVector(-p[2], p[1], 0.2); region = BoxRegion((-1.,-1.,-1.), (1.,1.,1.)))
            ovs = Overlay[SliceOverlay(:z), BoxOutlineOverlay(), SphereOverlay(radius = 0.4),
                          SphereOutlineOverlay(region = SphereRegion(radius = 0.6)),
                          AxesOverlay(), StreamlinesOverlay(vf), GlyphsOverlay(vf; every = 8)]
            _, b = renderbuf(FieldView(A; overlays = ovs), W)
            @test GL.glGetError() == 0
            @test litfrac(b) > 0.05
        end

        @testset "sphere outline graticule" begin
            @test VolumeRendering._parallel_lats(3) ≈ [-π/4, 0, π/4]
            @test isempty(VolumeRendering._parallel_lats(0))
            @test VolumeRendering._meridian_lons(6) ≈ [0, π/3, 2π/3, π, 4π/3, 5π/3]
            @test isempty(VolumeRendering._meridian_lons(0))
            @test VolumeRendering._parallel_lats([0.1, 0.2]) == [0.1, 0.2]     # explicit angles pass through

            # equator only, radius 2 around (1,0,0): 8 segments ⇒ 8 vertex pairs on the circle z = 0
            v = reshape(VolumeRendering._sphereoutline_verts(
                SVector(1.0, 0.0, 0.0), 2.0, [0.0], 0, 8, RGB(1.0, 0.0, 0.0)), 6, :)
            @test size(v) == (6, 16)
            @test v[3, :] == zeros(16)
            @test hypot.(v[1, :] .- 1, v[2, :]) ≈ fill(2.0, 16)
            @test v[4:6, :] == repeat([1.0, 0.0, 0.0], 1, 16)

            # a meridian is half a circle, so it gets half the segments (9 points ⇒ 8 pairs)
            m = VolumeRendering._sphereoutline_verts(SVector(0.0, 0.0, 0.0), 1.0, 0, [0.0], 16, RGB(1.0, 0.0, 0.0))
            @test length(m) == 6 * 2 * 8

            # angles only ever reach sin/cos, so Unitful angles work with no conversion
            @test VolumeRendering._sphereoutline_verts(SVector(0.0, 0.0, 0.0), 1.0, [30u"°"], 0, 8, RGB(1.0, 0.0, 0.0)) ≈
                  VolumeRendering._sphereoutline_verts(SVector(0.0, 0.0, 0.0), 1.0, [π/6],    0, 8, RGB(1.0, 0.0, 0.0))
        end

        @testset "overlay fingerprint tracks state, not GL resources" begin
            o = BoxOutlineOverlay(color = RGB(1.0, 0.0, 0.0))
            fp = VolumeRendering.overlay_fingerprint(o)
            o.gl.key = (1, 2, 3)                                  # GL cache: not part of the image
            @test VolumeRendering.overlay_fingerprint(o) == fp
            o.color = RGB(0.0, 1.0, 0.0)
            @test VolumeRendering.overlay_fingerprint(o) != fp
        end

        @testset "field switch re-uploads (same-shape, different data)" begin
            # A1/A2 have identical shape+axes → identical field_glsl → the volume program is NOT
            # rebuilt on switch; the texture MUST still be re-uploaded so the image reflects A2.
            A1 = KeyedArray(gaussian(xs, ys, zs; c = (-0.5, 0.0, 0.0), s = 0.1); x = xs, y = ys, z = zs)
            A2 = KeyedArray(gaussian(xs, ys, zs; c = ( 0.5, 0.0, 0.0), s = 0.1); x = xs, y = ys, z = zs)
            @test field_glsl(A1) == field_glsl(A2)     # same GLSL: only the data differs
            v = FieldView(A1); (_, b1) = renderbuf(v, W)
            v.field = A2;      (_, b2) = renderbuf(v, W)
            @test b1 != b2                             # image reflects the new data (no stale texture)
        end

        @testset "multi-view isolation (two views, one context)" begin
            A  = KeyedArray(gaussian(xs, ys, zs); x = xs, y = ys, z = zs)
            A2 = KeyedArray(gaussian(xs, ys, zs; c = (0.5, 0.0, 0.0), s = 0.1); x = xs, y = ys, z = zs)
            va = FieldView(A); vb = FieldView(A2)
            _, ba = renderbuf(va, W)
            renderbuf(vb, W)                                    # render B in between
            va.last_key[] = nothing                             # force A to re-render (same inputs)
            _, ba2 = renderbuf(va, W)
            @test ba == ba2                                     # B did not corrupt A
            @test va.vr.geom_fb.tex != vb.vr.geom_fb.tex        # separate GL resources
        end
    end
end

import Aqua
import CompatHelperLocal as CHL
@testset "_" begin
    Aqua.test_all(VolumeRendering; ambiguities=false)
    Aqua.test_ambiguities(VolumeRendering)
    CHL.@check()
end

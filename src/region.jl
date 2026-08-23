# Region interface + concrete BoxRegion / SphereRegion.
#
# A Region is a ray-interval provider for the volume marcher and the surface overlays: it supplies a
# GLSL `vec2 intersectRegion(vec3 ro, vec3 rd)` (returning the (tnear, tfar) ray parameters of the
# entry and exit of the region), the uniforms that snippet needs, and a bounding sphere used by the
# camera for near/far framing. Both parameters are UNCLAMPED — negative means behind the eye; a
# consumer that must start at the eye clamps for itself.

abstract type Region end

# ---- interface (methods below overload these; see SPEC "Region") -------------------------------
# region_glsl(::Region)::String        — defines `vec2 intersectRegion(vec3 ro, vec3 rd)` + own uniforms
# bind_region!(::Region, prog)         — set that snippet's uniforms
# bounds(::Region)::Tuple{SVector{3,Float64},Float64}  — (center, radius) circumscribing sphere → framing
# fingerprint(::Region)                — hashable, for render_key

# ================================================================================================
# BoxRegion — axis-aligned box; ray/box slab test.
# ================================================================================================
struct BoxRegion <: Region
    lo::SVector{3,Float64}
    hi::SVector{3,Float64}
end
# The default constructor converts each arg to SVector{3,Float64} via `new`, so tuple and SVector
# args both work — no extra constructor needed.

# Slab test, ported from volume.frag (lines 30-33) with boxLo/boxHi replacing ±extent.
region_glsl(::BoxRegion) = """
uniform vec3 boxLo, boxHi;

vec2 intersectRegion(vec3 ro, vec3 rd) {
    vec3 t0 = (boxLo - ro) / rd, t1 = (boxHi - ro) / rd;
    vec3 tmn = min(t0, t1), tmx = max(t0, t1);
    float tnear = max(max(tmn.x, tmn.y), tmn.z);
    float tfar  = min(min(tmx.x, tmx.y), tmx.z);
    return vec2(tnear, tfar);
}
"""

bind_region!(r::BoxRegion, prog) = (uni_3f(prog, "boxLo", r.lo...); uni_3f(prog, "boxHi", r.hi...))

# Circumscribed sphere: midpoint + half the space diagonal (so [-e,e]³ → radius e·√3). The camera's
# framing assumes the circumscribing radius — NOT the inscribed one.
bounds(r::BoxRegion) = ((r.lo + r.hi) / 2, norm(r.hi - r.lo) / 2)

# Axis-aligned bounding box (lo, hi) — used by SliceOverlay/BoxOutlineOverlay to span the region.
aabb(r::BoxRegion) = (r.lo, r.hi)

fingerprint(r::BoxRegion) = (:box, r.lo, r.hi)

# ================================================================================================
# SphereRegion — ray/sphere test.
# ================================================================================================
struct SphereRegion <: Region
    center::SVector{3,Float64}
    radius::Float64
end
SphereRegion(; center = SVector(0.0, 0.0, 0.0), radius) =
    SphereRegion(SVector{3,Float64}(center), Float64(radius))

# Ray vs sphere, ported from horizon.frag's root solve, centered at sphereCenter. Returns both roots
# as (tnear, tfar); a MISS yields (1e30, -1e30) so the marcher's `tfar > tnear` test skips it.
region_glsl(::SphereRegion) = """
uniform vec3  sphereCenter;
uniform float sphereRadius;

vec2 intersectRegion(vec3 ro, vec3 rd) {
    vec3 oc = ro - sphereCenter;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float disc = b * b - c;
    if (disc < 0.0) return vec2(1e30, -1e30);   // miss → tfar < tnear, marcher skips
    float s = sqrt(disc);
    float tnear = -b - s;
    float tfar  = -b + s;
    return vec2(tnear, tfar);
}
"""

bind_region!(r::SphereRegion, prog) =
    (uni_3f(prog, "sphereCenter", r.center...); uni_f(prog, "sphereRadius", r.radius))

bounds(r::SphereRegion) = (r.center, r.radius)

aabb(r::SphereRegion) = (r.center .- r.radius, r.center .+ r.radius)

fingerprint(r::SphereRegion) = (:sphere, r.center, r.radius)

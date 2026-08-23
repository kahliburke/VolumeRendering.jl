# Thin OpenGL helpers (shader compilation, texture upload, FBO, uniforms) for the viewer.
# CImGui-free so the whole render path is testable in a headless GLFW context.

import ModernGL as GL
using ColorTypes, FixedPointNumbers

const SHADER_DIR = joinpath(@__DIR__, "shaders")

glcheck(tag) = (e = GL.glGetError(); e != 0 && error("GL error 0x$(string(e, base=16)) @ $tag"); nothing)

function compile_shader(kind, src::AbstractString)
    s = GL.glCreateShader(kind)
    bytes = Vector{UInt8}(codeunits(src)); push!(bytes, 0x00)
    GC.@preserve bytes GL.glShaderSource(s, 1, Ref(pointer(bytes)), Ref{GL.GLint}(length(bytes) - 1))
    GL.glCompileShader(s)
    ok = Ref{GL.GLint}(0); GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS, ok)
    if ok[] == 0
        len = Ref{GL.GLint}(0); GL.glGetShaderiv(s, GL.GL_INFO_LOG_LENGTH, len)
        buf = Vector{UInt8}(undef, len[]); GL.glGetShaderInfoLog(s, len[], C_NULL, buf)
        error("shader compile failed:\n" * String(buf))
    end
    s
end

# Splice `//#include "name.glsl"` lines from the in-memory `includes[name]` snippet, else
# `shaders/name.glsl` on disk. One level only (snippets don't nest). `#version` stays in the
# including file; snippets carry only declarations + functions.
resolve_includes(src::AbstractString; includes::Dict{String,String} = Dict{String,String}()) = replace(src,
    r"(?m)^[ \t]*//#include[ \t]+\"([^\"]+)\"[ \t]*$" =>
        m -> (name = String(match(r"\"([^\"]+)\"", m).captures[1]);
              haskey(includes, name) ? includes[name] : read(joinpath(SHADER_DIR, name), String)))
_shader_src(file) = resolve_includes(read(joinpath(SHADER_DIR, file), String))

# Compile + link a program from vertex/fragment source strings, resolving `//#include`s.
function link_program_src(vert_src::AbstractString, frag_src::AbstractString; includes::Dict{String,String} = Dict{String,String}())
    vs = compile_shader(GL.GL_VERTEX_SHADER,   resolve_includes(vert_src; includes))
    fs = compile_shader(GL.GL_FRAGMENT_SHADER, resolve_includes(frag_src; includes))
    p = GL.glCreateProgram()
    GL.glAttachShader(p, vs); GL.glAttachShader(p, fs); GL.glLinkProgram(p)
    ok = Ref{GL.GLint}(0); GL.glGetProgramiv(p, GL.GL_LINK_STATUS, ok)
    if ok[] == 0
        len = Ref{GL.GLint}(0); GL.glGetProgramiv(p, GL.GL_INFO_LOG_LENGTH, len)
        buf = Vector{UInt8}(undef, len[]); GL.glGetProgramInfoLog(p, len[], C_NULL, buf)
        error("program link failed:\n" * String(buf))
    end
    GL.glDeleteShader(vs); GL.glDeleteShader(fs)
    p
end

# Build a program from vertex/fragment shader source files under shaders/.
link_program(vert_file::AbstractString, frag_file::AbstractString) =
    link_program_src(read(joinpath(SHADER_DIR, vert_file), String), read(joinpath(SHADER_DIR, frag_file), String))

# A fresh (attribute-less) VAO — core profile still requires one bound to draw generated geometry.
_new_vao() = (r = Ref{GL.GLuint}(0); GL.glGenVertexArrays(1, r); r[])

# The GL state of a shader-driven overlay: its program + VAO, built on first use so the overlay can
# be constructed with no current context. `key` is whatever the source depends on (the field GLSL for
# per-field programs, `nothing` for static ones) — the program relinks only when it changes.
mutable struct CachedProgram
    prog::GL.GLuint
    vao::GL.GLuint
    key::Any
end
CachedProgram() = CachedProgram(0, 0, nothing)

function refresh!(build, gl::CachedProgram, key)
    gl.vao == 0 && (gl.vao = _new_vao())
    if gl.prog == 0 || key != gl.key
        gl.prog != 0 && GL.glDeleteProgram(gl.prog)
        gl.prog = build()
        gl.key = key
    end
    gl
end

# A static program: linked once from these shader files, never relinked.
refresh!(gl::CachedProgram, vert::AbstractString, frag::AbstractString) =
    refresh!(() -> link_program(vert, frag), gl, (vert, frag))

# Bind for drawing; returns the program so the caller can set its uniforms.
use!(gl::CachedProgram) = (GL.glUseProgram(gl.prog); GL.glBindVertexArray(gl.vao); gl.prog)

# `GL_LINEAR` default → one hardware-filtered `texture()` fetch. Wrap is per-axis: `wrap[1]→S,
# wrap[2]→T, wrap[3]→R`, default CLAMP_TO_EDGE (correct for a generic Cartesian grid). (w,h,d)→S,T,R.
function tex3d_r32f(data::AbstractArray{<:Real,3};
                    wrap = (GL.GL_CLAMP_TO_EDGE, GL.GL_CLAMP_TO_EDGE, GL.GL_CLAMP_TO_EDGE),
                    filter = GL.GL_LINEAR)
    arr = data isa Array{Float32,3} ? data : Array{Float32,3}(data)   # contiguous for glTexImage3D
    w, h, d = size(arr)
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_3D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, wrap[1])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, wrap[2])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, wrap[3])
    GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R32F, w, h, d, 0, GL.GL_RED, GL.GL_FLOAT, arr)
    id[]
end

# RGBA 3D texture from a Colorant grid; precision follows eltype (N0f8→RGBA8, float→RGBA32F). Same
# conventions as `tex3d_r32f`. Colorant memory is (r,g,b,a) per element, matching the GL_RGBA upload.
function tex3d_rgba(data::AbstractArray{<:Colorant,3};
                    wrap = (GL.GL_CLAMP_TO_EDGE, GL.GL_CLAMP_TO_EDGE, GL.GL_CLAMP_TO_EDGE),
                    filter = GL.GL_LINEAR)
    dense = data isa Array ? data : Array(data)              # materialize KeyedArray → plain Array
    if eltype(eltype(dense)) <: AbstractFloat
        arr = eltype(dense) === RGBA{Float32} ? dense : RGBA{Float32}.(dense)
        internal, gltype = GL.GL_RGBA32F, GL.GL_FLOAT
    else                                                     # fixed-point (N0f8, …) → 8-bit
        arr = eltype(dense) === RGBA{N0f8} ? dense : RGBA{N0f8}.(dense)
        internal, gltype = GL.GL_RGBA8, GL.GL_UNSIGNED_BYTE
    end
    w, h, d = size(arr)
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_3D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, wrap[1])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, wrap[2])
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, wrap[3])
    GC.@preserve arr GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, internal, w, h, d, 0, GL.GL_RGBA, gltype, pointer(arr))
    id[]
end

function tex2d_r32f(data0::AbstractMatrix{<:Real}; filter = GL.GL_NEAREST)
    data = data0 isa Matrix{Float32} ? data0 : Matrix{Float32}(data0)
    w, h = size(data)
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_2D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0, GL.GL_RED, GL.GL_FLOAT, data)
    id[]
end

# A 256×1 RGBA32F transfer-function texture (LINEAR so lookups interpolate between entries).
function tex_tf()
    id = Ref{GL.GLuint}(0); GL.glGenTextures(1, id); GL.glBindTexture(GL.GL_TEXTURE_2D, id[])
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, 256, 1, 0, GL.GL_RGBA, GL.GL_FLOAT, C_NULL)
    id[]
end
# data: 4×256 Float32 (r,g,b,a per column).
update_tf!(id, data::Matrix{Float32}) = (GL.glBindTexture(GL.GL_TEXTURE_2D, id);
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL.GL_RGBA, GL.GL_FLOAT, data))

# Offscreen render target: an RGBA8 color texture + an optional depth attachment, resized on demand.
# `depth=:texture` makes depth a *sampleable* GL_DEPTH_COMPONENT32F texture (COMPARE_MODE=NONE) so a
# later pass can read the scene depth — the geometry G-buffer. `depth=:none` is color-only (the volume
# target). (Verified on this GL 4.1-over-Metal: depth-texture FBO is complete and samples as raw depth.)
mutable struct Framebuffer
    fbo::GL.GLuint
    tex::GL.GLuint          # color texture, RGBA8 or RGBA16F per `colormode`
    depth::GL.GLuint        # depth texture id, or 0 when depthmode == :none
    depthmode::Symbol       # :none | :texture
    colormode::Symbol       # :rgba8 | :rgba16f
    w::Int
    h::Int
end
function Framebuffer(w, h; depth::Symbol = :none, color::Symbol = :rgba8)
    fb = Framebuffer(0, 0, 0, depth, color, 0, 0)
    r = Ref{GL.GLuint}(0); GL.glGenFramebuffers(1, r); fb.fbo = r[]
    resize!(fb, w, h)
    fb
end
function Base.resize!(fb::Framebuffer, w, h)
    (w == fb.w && h == fb.h) && return fb
    fb.tex   != 0 && GL.glDeleteTextures(1, Ref(fb.tex))
    fb.depth != 0 && GL.glDeleteTextures(1, Ref(fb.depth))
    internal, gltype = fb.colormode === :rgba16f ? (GL.GL_RGBA16F, GL.GL_FLOAT) : (GL.GL_RGBA8, GL.GL_UNSIGNED_BYTE)
    t = Ref{GL.GLuint}(0); GL.glGenTextures(1, t); GL.glBindTexture(GL.GL_TEXTURE_2D, t[])
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, GL.GL_RGBA, gltype, C_NULL)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb.fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t[], 0)
    dtex = GL.GLuint(0)
    if fb.depthmode === :texture
        d = Ref{GL.GLuint}(0); GL.glGenTextures(1, d); GL.glBindTexture(GL.GL_TEXTURE_2D, d[])
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT32F, w, h, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, C_NULL)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_NONE)   # sample raw depth, not shadow
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, d[], 0)
        dtex = d[]
    end
    @assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE "incomplete FBO"
    fb.tex = t[]; fb.depth = dtex; fb.w = w; fb.h = h
    fb
end

# Uniform setters (look up by name each call; few uniforms per shader).
uni_i(p, name, v)     = GL.glUniform1i(GL.glGetUniformLocation(p, name), Int32(v))
uni_f(p, name, v)     = GL.glUniform1f(GL.glGetUniformLocation(p, name), Float32(v))
uni_2f(p, name, x, y) = GL.glUniform2f(GL.glGetUniformLocation(p, name), Float32(x), Float32(y))
uni_2i(p, name, x, y) = GL.glUniform2i(GL.glGetUniformLocation(p, name), Int32(x), Int32(y))
uni_3f(p, name, x, y, z) = GL.glUniform3f(GL.glGetUniformLocation(p, name), Float32(x), Float32(y), Float32(z))
uni_3f(p, name, c::Colorant) = (rgb = RGB{Float32}(c); uni_3f(p, name, rgb.r, rgb.g, rgb.b))
# 4×4 matrix uniform. `m` is column-major (Julia/StaticArrays native), matching GL's expectation.
# A `Vector{Float32}` is uploaded as-is (already collected once per frame — no re-copy per overlay).
uni_m4(p, name, m) = GL.glUniformMatrix4fv(GL.glGetUniformLocation(p, name), 1, GL.GL_FALSE,
                                           collect(Float32, m))
uni_m4(p, name, m::Vector{Float32}) = GL.glUniformMatrix4fv(GL.glGetUniformLocation(p, name), 1, GL.GL_FALSE, m)

# Bind `texid` (of `target`) to texture unit `unit` and point sampler uniform `name` at it.
function bind_sampler(p, name, unit, target, texid)
    GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
    GL.glBindTexture(target, texid)
    uni_i(p, name, unit)
end

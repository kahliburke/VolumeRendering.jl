# GL line geometry for the line/streamline/glyph overlays: one interleaved (x,y,z, r,g,b) VBO drawn
# as GL_LINES with depth testing, so lines composite correctly against the ray-marched volume.
# (`GL` = ModernGL is imported once at module scope in gl.jl.)

mutable struct GeometryRenderer
    program::UInt32
    vao::UInt32
    vbo::UInt32
    nverts::Int
end

function GeometryRenderer()
    prog = link_program("geom.vert", "geom.frag")
    vao = Ref{GL.GLuint}(0); GL.glGenVertexArrays(1, vao)
    vbo = Ref{GL.GLuint}(0); GL.glGenBuffers(1, vbo)
    GL.glBindVertexArray(vao[])
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo[])
    stride = 6 * sizeof(Float32)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, Ptr{Cvoid}(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, Ptr{Cvoid}(3 * sizeof(Float32)))
    GL.glEnableVertexAttribArray(1)
    GL.glBindVertexArray(0)
    GeometryRenderer(prog, vao[], vbo[], 0)
end

# Replace the buffer contents with interleaved (x,y,z, r,g,b) vertices (drawn as GL_LINES).
function upload!(gr::GeometryRenderer, verts::Vector{Float32})
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, gr.vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, sizeof(verts), verts, GL.GL_DYNAMIC_DRAW)
    gr.nverts = length(verts) ÷ 6
    gr
end

# Draw the uploaded lines into the geometry G-buffer. The geometry pass enables GL_LESS depth
# testing, so the lines share one depth space with the other overlays (mutual occlusion) and the
# volume later composites over them. Lines are 1 device-px: macOS core-profile GL clamps glLineWidth
# to 1.0, so thicker lines would need screen-space geometry expansion (as the spin axis does).
function draw!(gr::GeometryRenderer, vp)
    gr.nverts == 0 && return gr
    GL.glUseProgram(gr.program)
    uni_m4(gr.program, "viewProj", vp)
    GL.glBindVertexArray(gr.vao)
    GL.glDrawArrays(GL.GL_LINES, 0, gr.nverts)
    gr
end

# The GL state of a line-geometry overlay: a GeometryRenderer built on first use, its vertices
# regenerated only when `key` — everything they are computed from — changes.
mutable struct CachedLines
    gr::Union{Nothing,GeometryRenderer}
    key::Any
end
CachedLines() = CachedLines(nothing, nothing)

function refresh!(makeverts, gl::CachedLines, key)
    gl.gr === nothing && (gl.gr = GeometryRenderer())
    if key != gl.key
        upload!(gl.gr, makeverts())
        gl.key = key
    end
    gl
end

draw!(gl::CachedLines, vp) = (gl.gr === nothing || draw!(gl.gr, vp); gl)

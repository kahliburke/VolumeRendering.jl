#version 410 core
// slice.frag on a curved surface: an analytic impostor coloring a region's near boundary through the
// same TF as the volume. The injected region snippet supplies the hit, so a SphereRegion gives a
// spherical shell and a BoxRegion the box's front faces.
in vec2 vuv;
out vec4 frag;

//#include "inc_camera.glsl"
//#include "inc_depth.glsl"
//#include "inc_tf.glsl"
//#include "field"
//#include "region"

void main() {
    vec3 ro, rd;
    cameraRay(vuv, ro, rd);
    vec2 tr = intersectRegion(ro, rd);
    if (tr.y <= tr.x) discard;             // ray misses the region
    float t = tr.x > 0.0 ? tr.x : tr.y;    // eye inside → far surface (its inner wall)
    if (t < 0.0) discard;                  // region entirely behind the eye
    vec3 p = ro + rd * t;
    frag = vec4(sample4(p).rgb, 1.0);
    writeDepth(p);
}

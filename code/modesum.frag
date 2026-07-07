// ============================================================
//  modesum.frag -- analytic Beltrami mode-sum, per-pixel (paper section 6)
//
//  Evaluates the divergence-free helical velocity field
//     u(x) = sum_j a_j ( e1_j cos(phi_j) - s_j e2_j sin(phi_j) ),
//     phi_j = k_j . x + theta_j,
//  entirely in the fragment shader: no grid, no FFT. Each term is transverse to
//  its own k_j, so div u == 0 pointwise, and is a Beltrami mode with
//  curl u_j = s_j |k_j| u_j -- the vorticity is the second (commented) sum below,
//  requiring no derivatives.
//
//  Mode data are uploaded once as a small RGBA float texture uMODES (width = NMODES,
//  height = 3): row 0 = (kx,ky,kz, theta), row 1 = (e1x,e1y,e1z, a),
//  row 2 = (e2x,e2y,e2z, s). Build them CPU-side exactly as code/e3_mode_sum.py /
//  e5_performance.py:make_modes (directions uniform on the sphere, a_j=|k_j|^-slope,
//  s_j=+1 with probability (1+p)/2). O(NMODES) per pixel; targets WebGL2 / GLSL ES 3.0
//  (works as desktop GLSL 3.30 with the version line swapped).
// ============================================================
#version 300 es
precision highp float;

uniform sampler2D uMODES;   // NMODES x 3 mode table (see header)
uniform int   uNMODES;      // number of modes M
uniform float uTIME;        // seconds; advects the sample plane along z
uniform vec2  uRES;         // viewport resolution in px
uniform float uZOOM;        // world units across the viewport (default 2*PI)
out vec4 fragColor;

// sample world position for this pixel: a z-slice that drifts with time
vec3 worldPos() {
    vec2 uv = (gl_FragCoord.xy / uRES) * uZOOM;   // [0, uZOOM)
    return vec3(uv, 0.25 * uTIME);                // slice sweeps in z
}

// fetch mode j (texel centres); texture is NMODES wide, 3 tall
void getMode(int j, out vec3 k, out vec3 e1, out vec3 e2,
             out float theta, out float a, out float s) {
    float fx = (float(j) + 0.5) / float(uNMODES);
    vec4 r0 = texture(uMODES, vec2(fx, 0.5 / 3.0));   // (k, theta)
    vec4 r1 = texture(uMODES, vec2(fx, 1.5 / 3.0));   // (e1, a)
    vec4 r2 = texture(uMODES, vec2(fx, 2.5 / 3.0));   // (e2, s)
    k = r0.xyz; theta = r0.w;
    e1 = r1.xyz; a = r1.w;
    e2 = r2.xyz; s = r2.w;
}

// returns velocity u; vorticity w is the same loop with the (commented) s*|k| weight
vec3 fieldAt(vec3 x) {
    vec3 u = vec3(0.0);
    // vec3 w = vec3(0.0);   // uncomment for vorticity (no derivatives needed)
    for (int j = 0; j < uNMODES; ++j) {
        vec3 k, e1, e2; float theta, a, s;
        getMode(j, k, e1, e2, theta, a, s);
        float phi = dot(k, x) + theta;
        vec3 term = a * (e1 * cos(phi) - s * e2 * sin(phi));
        u += term;
        // w += s * length(k) * term;   // curl of a Beltrami mode = s|k| * mode
    }
    return u;
}

void main() {
    vec3 x = worldPos();
    vec3 u = fieldAt(x);
    // visualize: in-plane flow direction as hue-ish RG, out-of-plane u.z as blue
    vec2 f = 0.5 + 0.5 * normalize(u.xy + 1e-6) * clamp(length(u.xy), 0.0, 1.0);
    fragColor = vec4(f, 0.5 + 0.5 * clamp(u.z, -1.0, 1.0), 1.0);
}

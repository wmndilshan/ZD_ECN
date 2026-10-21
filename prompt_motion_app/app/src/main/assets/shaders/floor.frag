#version 310 es
precision highp float;

in vec3 v_WorldPos;

uniform vec2  u_Focus;        // camera target on the floor (xz) — fade centre
uniform vec2  u_ShadowCenter; // character pelvis projected on the floor (xz)
uniform float u_ShadowLift;   // pelvis height above standing height (m), softens shadow in the air

out vec4 fragColor;

void main() {
    vec2 p = v_WorldPos.xz;

    // 0.5 m checker with thin grid lines
    vec2 cell  = floor(p * 2.0);
    float checker = mod(cell.x + cell.y, 2.0);
    vec2 g = abs(fract(p * 2.0) - 0.5);
    float line = 1.0 - smoothstep(0.47, 0.5, max(g.x, g.y));
    vec3 base = mix(vec3(0.085, 0.095, 0.115), vec3(0.11, 0.12, 0.145), checker);
    base = mix(vec3(0.20, 0.23, 0.28), base, line);

    // Soft contact shadow under the body
    float r = length(p - u_ShadowCenter);
    float spread = 0.32 + 0.35 * clamp(u_ShadowLift, 0.0, 1.0);
    float shadow = (1.0 - smoothstep(0.0, spread, r)) * (0.55 - 0.35 * clamp(u_ShadowLift, 0.0, 1.0));
    base *= 1.0 - shadow;

    // Radial fade into the background clear colour
    float d = length(p - u_Focus);
    float fade = 1.0 - smoothstep(2.5, 6.5, d);
    vec3 bg = vec3(0.04, 0.05, 0.07);  // matches glClearColor (already display space)
    fragColor = vec4(mix(bg, pow(base, vec3(1.0 / 1.25)), fade), 1.0);
}

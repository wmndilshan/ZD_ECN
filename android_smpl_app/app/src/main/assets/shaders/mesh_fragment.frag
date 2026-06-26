#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorld;
out vec4 fragColor;
uniform vec3 uColor;
void main() {
    vec3 n = normalize(vNormal);
    vec3 l = normalize(vec3(0.3, 0.7, 0.6));
    float ndl = max(dot(n, l), 0.0);
    vec3 color = uColor * (0.25 + 0.75 * ndl);
    fragColor = vec4(color, 1.0);
}

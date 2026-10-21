#version 310 es
precision highp float;

in vec3 a_Position;   // floor quad in display world (Y-up, y = 0)

uniform mat4 u_ViewProj;

out vec3 v_WorldPos;

void main() {
    v_WorldPos  = a_Position;
    gl_Position = u_ViewProj * vec4(a_Position, 1.0);
}

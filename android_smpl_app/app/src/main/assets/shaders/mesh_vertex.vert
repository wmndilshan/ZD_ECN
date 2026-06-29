#version 300 es
precision highp float;

layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uMvp;
uniform mat4 uModel;
out vec3 vWorld;
out vec3 vNormal;

void main() {
    vec3 position = aPosition;
    position.y -= 0.9;
    vec4 world = uModel * vec4(position, 1.0);
    vWorld = world.xyz;
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uMvp * vec4(position, 1.0);
}

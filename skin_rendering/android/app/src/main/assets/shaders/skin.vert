#version 310 es
precision highp float;

// Per-vertex data from VBOs
in vec3 a_Position;   // v_final from combine.comp (metres)
in vec3 a_Normal;     // per-vertex smooth normal from normals.comp
in vec2 a_TexCoord;   // cylindrical UV

uniform mat4 u_MVP;
uniform mat4 u_Model;   // model → world (for lighting in world space)
uniform mat3 u_NormalMat; // inverse-transpose of model matrix

out vec3 v_Normal;
out vec2 v_TexCoord;
out vec3 v_WorldPos;

void main() {
    vec4 worldPos = u_Model * vec4(a_Position, 1.0);
    v_WorldPos  = worldPos.xyz;
    v_Normal    = normalize(u_NormalMat * a_Normal);
    v_TexCoord  = a_TexCoord;
    gl_Position = u_MVP * vec4(a_Position, 1.0);
}

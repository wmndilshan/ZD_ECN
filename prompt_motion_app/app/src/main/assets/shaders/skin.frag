#version 310 es
precision highp float;

in vec3 v_Normal;
in vec2 v_TexCoord;
in vec3 v_WorldPos;

// Texturing
uniform sampler2D u_SkinTexture;
uniform bool      u_UseTexture;

// Lighting parameters (all in world space)
uniform vec3  u_LightDir;     // normalised, points toward light source
uniform vec3  u_LightColor;   // typically (1.0, 0.97, 0.93) warm white
uniform vec3  u_ViewDir;      // normalised, points toward camera

// Skin material parameters (adjustable from UI)
uniform float u_WrapFactor;   // 0.3 default — SSS wrap breadth
uniform float u_SpecStrength; // 0.05 default — specularity intensity

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_Normal);
    // Flip normal for back-faces so interior triangles look correct
    if (!gl_FrontFacing) N = -N;

    vec3 L = normalize(u_LightDir);
    vec3 V = normalize(u_ViewDir);

    // ── albedo ────────────────────────────────────────────────────────────────
    vec3 albedo;
    if (u_UseTexture) {
        albedo = texture(u_SkinTexture, v_TexCoord).rgb;
    } else {
        albedo = vec3(0.82, 0.65, 0.47);  // neutral warm skin debug colour
    }

    // ── subsurface scattering approximation (wrap-lighting, GPU Gems 1 Ch.16) ─
    // Simulates light bleeding through thin skin (ears, fingers, lips).
    // NdotL=−1 → wrap/(1+wrap) instead of 0, giving non-zero backlit response.
    float NdotL = dot(N, L);
    float wrappedDiffuse = (NdotL + u_WrapFactor) / (1.0 + u_WrapFactor);
    wrappedDiffuse = max(wrappedDiffuse, 0.0);

    // attenuate albedo toward red-orange in the wrap region (blood-colour SSS)
    float subsurf = max(-NdotL * u_WrapFactor, 0.0);
    vec3 sssColor = vec3(0.9, 0.3, 0.1) * subsurf * 0.25;

    vec3 diffuse = albedo * (wrappedDiffuse * u_LightColor + sssColor);

    // ── specular (Blinn-Phong, moist skin surface) ─────────────────────────
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = u_LightColor * (u_SpecStrength * spec);

    // ── rim lighting (warm silhouette, separates body from background) ──────
    float rim = 1.0 - max(dot(N, V), 0.0);
    rim = pow(rim, 3.0);
    vec3 rimColor = vec3(0.18, 0.12, 0.09) * rim;

    // ── hemisphere ambient (sky/ground IBL approximation) ───────────────────
    float hemi = dot(N, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    vec3 ambient = mix(vec3(0.04, 0.03, 0.06),   // dark ground (purple-blue)
                       vec3(0.14, 0.16, 0.20),   // sky (cool blue-white)
                       hemi);
    ambient *= albedo;

    // ── composite ────────────────────────────────────────────────────────────
    vec3 color = diffuse + specular + rimColor + ambient;

    // Gamma correction (linear → sRGB, approximation: pow(x, 1/2.2))
    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}

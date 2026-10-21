// skin_renderer.cpp — JNI entry points for textured skin rendering extension.
// Reuses the validated SmplGpuPipeline from android_smpl_app unchanged.
// Adds: UV VBO, texture loading from PNG asset, PBR skin shader.

#include "skin_gpu_pipeline.h"

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <GLES3/gl31.h>
#include <GLES3/gl3ext.h>
#include <EGL/egl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifndef GL_TIME_ELAPSED_EXT
#define GL_TIME_ELAPSED_EXT 0x88BF
#endif
#ifndef GL_GPU_DISJOINT_EXT
#define GL_GPU_DISJOINT_EXT 0x8FBB
#endif
#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif
#ifndef PFNGLGETQUERYOBJECTUI64VEXTPROC
typedef void (GL_APIENTRYP PFNGLGETQUERYOBJECTUI64VEXTPROC)(GLuint, GLenum, GLuint64*);
#endif

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SkinRenderer", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SkinRenderer", __VA_ARGS__)

namespace {

// ── math helpers (column-major Mat4) ─────────────────────────────────────────
struct Mat4 { float m[16]; };
Mat4 identity() { Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c=0;c<4;c++) for (int row=0;row<4;row++) for (int k=0;k<4;k++)
        r.m[c*4+row] += a.m[k*4+row] * b.m[c*4+k];
    return r;
}
Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
    Mat4 r{}; float f = 1.0f / std::tan(fovy * 0.5f);
    r.m[0]=f/aspect; r.m[5]=f;
    r.m[10]=(zfar+znear)/(znear-zfar); r.m[11]=-1.0f;
    r.m[14]=(2*zfar*znear)/(znear-zfar); return r;
}
Mat4 translate(float x,float y,float z){ Mat4 r=identity(); r.m[12]=x; r.m[13]=y; r.m[14]=z; return r; }
Mat4 rotateY(float a){ Mat4 r=identity(); float c=std::cos(a),s=std::sin(a); r.m[0]=c; r.m[2]=-s; r.m[8]=s; r.m[10]=c; return r; }
Mat4 rotateX(float a){ Mat4 r=identity(); float c=std::cos(a),s=std::sin(a); r.m[5]=c; r.m[6]=s; r.m[9]=-s; r.m[10]=c; return r; }
Mat4 scale3(float s){ Mat4 r=identity(); r.m[0]=r.m[5]=r.m[10]=s; return r; }

// Normal matrix: upper-left 3×3 of model (no non-uniform scale so = model 3×3)
void normalMatrix(const Mat4& model, float nm[9]) {
    nm[0]=model.m[0]; nm[1]=model.m[1]; nm[2]=model.m[2];
    nm[3]=model.m[4]; nm[4]=model.m[5]; nm[5]=model.m[6];
    nm[6]=model.m[8]; nm[7]=model.m[9]; nm[8]=model.m[10];
}

// ── globals ───────────────────────────────────────────────────────────────────
constexpr int kVertexCount = 6890;
constexpr int kFrameCount  = 600;

AAssetManager* g_assets  = nullptr;
SkinGpuPipeline g_pipeline;

GLuint g_skin_program = 0;
GLuint g_ibo = 0, g_vao = 0, g_uv_vbo = 0;
GLuint g_textures[2] = {};   // texture_skin_0.png, texture_skin_1.png
int    g_active_texture = 0;
bool   g_use_texture = true;

// Uniform locations
GLint g_u_mvp = -1, g_u_model = -1, g_u_normal_mat = -1;
GLint g_u_light_dir = -1, g_u_light_color = -1, g_u_view_dir = -1;
GLint g_u_wrap = -1, g_u_spec = -1, g_u_use_tex = -1;
GLint g_u_skin_tex = -1;

int   g_width = 1, g_height = 1;
int   g_backend = 0, g_tier = 0;
bool  g_use_fixed_frame = false;
int   g_fixed_frame = 0;
float g_time = 0.0f;
float g_user_yaw = 0.0f, g_user_pitch = -0.10f, g_user_zoom = 1.0f;
float g_light_angle = 0.5f;  // radians around Y axis
float g_wrap_factor = 0.3f;
float g_spec_strength = 0.05f;
int   g_pose_frame = -1;
bool  g_geometry_ready = false;
size_t g_index_count = 0;
std::string g_status = "not initialized";
std::vector<float>    g_pose_aa;
std::vector<float>    g_pose_rot6d;
std::vector<uint16_t> g_indices;
auto g_start_time = std::chrono::steady_clock::now();

// ── asset helpers ─────────────────────────────────────────────────────────────
std::vector<uint8_t> readAsset(const char* name) {
    if (!g_assets) return {};
    AAsset* a = AAssetManager_open(g_assets, name, AASSET_MODE_BUFFER);
    if (!a) { LOGE("missing asset: %s", name); return {}; }
    const off_t len = AAsset_getLength(a);
    std::vector<uint8_t> data(static_cast<size_t>(len));
    AAsset_read(a, data.data(), data.size());
    AAsset_close(a);
    return data;
}

std::string readText(const char* name) {
    auto b = readAsset(name);
    return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

template <typename T>
bool readBinary(const char* name, std::vector<T>& out) {
    auto b = readAsset(name);
    if (b.empty() || b.size() % sizeof(T) != 0) return false;
    out.resize(b.size() / sizeof(T));
    std::memcpy(out.data(), b.data(), b.size());
    return true;
}

// ── shader compilation ────────────────────────────────────────────────────────
GLuint compile(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof(log), &n, log);
        LOGE("shader compile failed: %.4000s", log);
    }
    return s;
}

GLuint linkProgram(const std::string& vs_src, const std::string& fs_src) {
    GLuint v = compile(GL_VERTEX_SHADER,   vs_src);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    // Bind attribute locations before linking
    glBindAttribLocation(p, 0, "a_Position");
    glBindAttribLocation(p, 1, "a_Normal");
    glBindAttribLocation(p, 2, "a_TexCoord");
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log), &n, log);
        LOGE("program link: %.4000s", log);
        glDeleteProgram(p); p = 0;
    }
    return p;
}

// ── minimal PNG decoder (header + palette-free RGBA only, uses stdlib zlib) ──
// Android system PNG is available via libpng but we avoid adding a dep.
// Instead load as raw RGBA bytes via the smplrenderer stb-compatible path:
// The pipeline receives raw pixel bytes from the asset manager.
// We decode the PNG in Java and pass raw RGBA bytes via JNI.
GLuint uploadTexture(const uint8_t* rgba, int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (!tex) return 0;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    LOGI("uploaded texture %u (%dx%d)", tex, w, h);
    return tex;
}

// ── init sub-routines ─────────────────────────────────────────────────────────
bool createGeometryBuffers() {
    if (!readBinary("animation/animation_poses_aa_f32.bin", g_pose_aa) ||
        !readBinary("animation/animation_poses_rot6d_f32.bin", g_pose_rot6d) ||
        !readBinary("smpl_shader_data/faces_u16.bin", g_indices)) {
        g_status = "animation/faces asset load failed";
        return false;
    }
    if (g_pose_aa.size() != kFrameCount * 72 ||
        g_pose_rot6d.size() != kFrameCount * 144 || g_indices.empty()) {
        g_status = "animation/faces shape mismatch";
        return false;
    }

    // IBO
    glGenBuffers(1, &g_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 g_indices.size() * sizeof(uint16_t),
                 g_indices.data(), GL_STATIC_DRAW);
    g_index_count = g_indices.size();

    // UV VBO (cylindrical UV, 6890 × 2 float32)
    std::vector<float> uv;
    if (!readBinary("smpl_uv_cylindrical_f32.bin", uv) ||
        uv.size() != kVertexCount * 2) {
        g_status = "UV asset load failed (expected 6890×2 floats)";
        return false;
    }
    glGenBuffers(1, &g_uv_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_uv_vbo);
    glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(float), uv.data(), GL_STATIC_DRAW);

    // VAO
    glGenVertexArrays(1, &g_vao);
    return glGetError() == GL_NO_ERROR;
}

bool configureRenderVertexBuffer() {
    const GLuint pos_buf  = g_pipeline.renderBuffer();
    const GLuint norm_buf = g_pipeline.normalBuffer();
    if (!g_vao || !pos_buf || !norm_buf || !g_uv_vbo) return false;

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ibo);

    // attr 0: position (from SSBO used as VBO)
    glBindBuffer(GL_ARRAY_BUFFER, pos_buf);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

    // attr 1: normal (from SSBO used as VBO)
    glBindBuffer(GL_ARRAY_BUFFER, norm_buf);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

    // attr 2: UV (static VBO)
    glBindBuffer(GL_ARRAY_BUFFER, g_uv_vbo);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);

    glBindVertexArray(0);
    return glGetError() == GL_NO_ERROR;
}

bool updateGpuGeometry(int frame) {
    if (!g_pipeline.ready() || frame == g_pose_frame) return g_pipeline.ready();
    g_pose_frame = frame;
    const float* aa    = &g_pose_aa[frame * 72];
    const float* rot6d = &g_pose_rot6d[frame * 144];
    const bool apply_delta = (g_backend != 2);  // 2 = LBS only

    if (!g_pipeline.dispatchLbs(aa)) {
        g_status = "GPU LBS failed: " + g_pipeline.error(); return false;
    }
    if (apply_delta && g_backend == 0) {
        if (!g_pipeline.dispatchZdecn(rot6d)) {
            g_status = "GPU ZD-ECN failed: " + g_pipeline.error(); return false;
        }
    }
    if (!g_pipeline.prepareRenderBuffer(apply_delta)) {
        g_status = "GPU combine failed: " + g_pipeline.error(); return false;
    }
    if (!g_geometry_ready) {
        configureRenderVertexBuffer();
        g_geometry_ready = true;
        LOGI("GPU geometry ready — textured skin renderer active");
    }
    return true;
}

// ── draw ──────────────────────────────────────────────────────────────────────
void drawFrame() {
    glClearColor(0.04f, 0.05f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!g_skin_program || !g_vao || !g_geometry_ready) return;

    const float aspect = static_cast<float>(g_width) / static_cast<float>(g_height);
    const bool portrait = g_height >= g_width;
    const float fov = (portrait ? 48.0f : 55.0f) * 3.14159265f / 180.0f;
    const float cam_dist = (portrait ? 4.80f : 3.35f) * g_user_zoom;
    const float auto_yaw = g_use_fixed_frame ? 0.0f : g_time * 0.08f;

    Mat4 proj  = perspective(fov, aspect, 0.05f, 20.0f);
    Mat4 view  = translate(0, portrait ? 0.0f : -0.03f, -cam_dist);
    Mat4 model = mul(rotateY(g_user_yaw + auto_yaw), rotateX(g_user_pitch));
    model = mul(translate(0.0f, portrait ? 0.95f : 0.08f, 0.0f), model);
    model = mul(scale3(portrait ? 0.98f : 1.28f), model);
    Mat4 mvp = mul(proj, mul(view, model));

    // Light direction (world space, rotating with light_angle)
    float lx = std::sin(g_light_angle), lz = std::cos(g_light_angle);
    float light_dir[3] = { lx * 0.6f, 0.8f, lz * 0.6f };
    float len = std::sqrt(light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1] + light_dir[2]*light_dir[2]);
    for (float& v : light_dir) v /= len;

    // View direction (camera → world, camera at (0,0,cam_dist) ≈ +Z)
    float view_dir[3] = { 0.0f, 0.0f, 1.0f };

    float nm[9];
    normalMatrix(model, nm);

    glUseProgram(g_skin_program);
    glUniformMatrix4fv(g_u_mvp,        1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(g_u_model,      1, GL_FALSE, model.m);
    glUniformMatrix3fv(g_u_normal_mat, 1, GL_FALSE, nm);
    glUniform3fv(g_u_light_dir,   1, light_dir);
    glUniform3f (g_u_light_color,     1.0f, 0.97f, 0.93f);
    glUniform3fv(g_u_view_dir,    1, view_dir);
    glUniform1f (g_u_wrap,  g_wrap_factor);
    glUniform1f (g_u_spec,  g_spec_strength);
    glUniform1i (g_u_use_tex, g_use_texture ? 1 : 0);

    if (g_use_texture && g_textures[g_active_texture]) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_textures[g_active_texture]);
        glUniform1i(g_u_skin_tex, 0);
    }

    glBindVertexArray(g_vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(g_index_count),
                   GL_UNSIGNED_SHORT, nullptr);
    glBindVertexArray(0);
    if (g_use_texture) glBindTexture(GL_TEXTURE_2D, 0);
}

}  // namespace

// ── JNI exports ───────────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeInit(
        JNIEnv* env, jclass, jobject asset_manager) {
    g_assets = AAssetManager_fromJava(env, asset_manager);
    g_start_time = std::chrono::steady_clock::now();

    g_skin_program = linkProgram(
        readText("shaders/skin.vert"),
        readText("shaders/skin.frag"));
    if (!g_skin_program) { g_status = "shader link failed"; return; }

    g_u_mvp        = glGetUniformLocation(g_skin_program, "u_MVP");
    g_u_model      = glGetUniformLocation(g_skin_program, "u_Model");
    g_u_normal_mat = glGetUniformLocation(g_skin_program, "u_NormalMat");
    g_u_light_dir  = glGetUniformLocation(g_skin_program, "u_LightDir");
    g_u_light_color= glGetUniformLocation(g_skin_program, "u_LightColor");
    g_u_view_dir   = glGetUniformLocation(g_skin_program, "u_ViewDir");
    g_u_wrap       = glGetUniformLocation(g_skin_program, "u_WrapFactor");
    g_u_spec       = glGetUniformLocation(g_skin_program, "u_SpecStrength");
    g_u_use_tex    = glGetUniformLocation(g_skin_program, "u_UseTexture");
    g_u_skin_tex   = glGetUniformLocation(g_skin_program, "u_SkinTexture");

    const bool geom_ok = createGeometryBuffers();
    const bool gpu_ok  = g_pipeline.init(g_assets);
    const bool topo_ok = geom_ok && gpu_ok &&
        g_pipeline.configureRenderTopology(g_indices.data(), g_indices.size());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    g_status = topo_ok ? "SkinRenderer initialized" :
               (gpu_ok ? g_pipeline.error() : g_status);
    LOGI("nativeInit: %s", g_status.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeUploadTexture(
        JNIEnv* env, jclass, jint index, jbyteArray rgba_bytes, jint width, jint height) {
    if (index < 0 || index > 1) return;
    jbyte* data = env->GetByteArrayElements(rgba_bytes, nullptr);
    const int n = env->GetArrayLength(rgba_bytes);
    if (n == width * height * 4) {
        if (g_textures[index]) glDeleteTextures(1, &g_textures[index]);
        g_textures[index] = uploadTexture(
            reinterpret_cast<const uint8_t*>(data), width, height);
    } else {
        LOGE("uploadTexture: size mismatch %d vs %dx%dx4=%d", n, width, height, width*height*4);
    }
    env->ReleaseByteArrayElements(rgba_bytes, data, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeResize(
        JNIEnv*, jclass, jint w, jint h) {
    g_width  = w > 0 ? w : 1;
    g_height = h > 0 ? h : 1;
    glViewport(0, 0, g_width, g_height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeDraw(JNIEnv*, jclass) {
    g_time += 1.0f / 60.0f;
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_start_time).count();
    int tier_base = g_tier == 0 ? 0 : (g_tier == 1 ? 200 : 400);
    int frame = g_use_fixed_frame
        ? g_fixed_frame
        : tier_base + (static_cast<int>(elapsed * 10.0) % 200);
    updateGpuGeometry(frame);
    drawFrame();
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetBackend(
        JNIEnv*, jclass, jint backend) {
    // 0=ZD-ECN, 1=ONNX CPU, 2=LBS only
    g_backend = backend;
    g_pose_frame = -1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetTier(
        JNIEnv*, jclass, jint tier) {
    // 0=natural(0-199), 1=moderate(200-399), 2=extreme(400-599)
    g_tier = tier;
    g_pose_frame = -1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetTexture(
        JNIEnv*, jclass, jint index) {
    g_active_texture = std::max(0, std::min(1, index));
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetUseTexture(
        JNIEnv*, jclass, jboolean use_tex) {
    g_use_texture = (use_tex == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetLighting(
        JNIEnv*, jclass, jfloat wrap, jfloat spec, jfloat light_angle) {
    g_wrap_factor  = std::max(0.0f, std::min(0.8f, wrap));
    g_spec_strength= std::max(0.0f, std::min(0.5f, spec));
    g_light_angle  = light_angle;
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetCamera(
        JNIEnv*, jclass, jfloat yaw, jfloat pitch, jfloat zoom) {
    g_user_yaw   = yaw;
    g_user_pitch = std::max(-0.55f, std::min(0.35f, pitch));
    g_user_zoom  = std::max(0.70f,  std::min(1.55f, zoom));
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeSetFixedFrame(
        JNIEnv*, jclass, jint frame, jboolean enabled) {
    g_fixed_frame = std::max(0, std::min(599, frame));
    g_use_fixed_frame = (enabled == JNI_TRUE);
    g_pose_frame = -1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeUploadOnnxDelta(
        JNIEnv* env, jclass, jfloatArray delta_arr) {
    if (!g_pipeline.ready()) return;
    const int n = env->GetArrayLength(delta_arr);
    if (n != 20670) { LOGE("ONNX delta size %d ≠ 20670", n); return; }
    jfloat* data = env->GetFloatArrayElements(delta_arr, nullptr);
    g_pipeline.uploadExternalDelta(data);
    env->ReleaseFloatArrayElements(delta_arr, data, JNI_ABORT);
    // Force the combine pass to re-run with the new delta on the next draw
    // (updateGpuGeometry early-returns when the pose frame is unchanged).
    g_pose_frame = -1;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeGetCurrentFrame(
        JNIEnv*, jclass) {
    return g_pose_frame;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeGetCurrentRot6d(
        JNIEnv* env, jclass) {
    const int frame = std::max(0, std::min(kFrameCount - 1, g_pose_frame));
    if (g_pose_rot6d.size() < static_cast<size_t>((frame + 1) * 144)) return nullptr;
    jfloatArray out = env->NewFloatArray(144);
    if (!out) return nullptr;
    env->SetFloatArrayRegion(out, 0, 144, &g_pose_rot6d[frame * 144]);
    return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeRunFullPipelineBenchmark(
        JNIEnv* env, jclass, jint warmup, jint iterations) {
    // Full-pipeline GPU benchmark: timer wraps shape blend + LBS + ZD-ECN +
    // combine + normals + indexed draw. Swap is outside the timed region
    // (it is issued by GLSurfaceView after this call returns).
    // Protocol mirrors android_smpl_app's nativeRunCombinedBenchmarkSession.
    if (!g_pipeline.ready() || !g_skin_program || !g_geometry_ready ||
        warmup < 0 || iterations <= 0) {
        return env->NewStringUTF("{\"error\":\"benchmark_not_ready\"}");
    }
    const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    if (!extensions || !std::strstr(extensions, "GL_EXT_disjoint_timer_query")) {
        return env->NewStringUTF("{\"error\":\"GL_EXT_disjoint_timer_query_unavailable\"}");
    }
    auto get_query_object_ui64v = reinterpret_cast<PFNGLGETQUERYOBJECTUI64VEXTPROC>(
            eglGetProcAddress("glGetQueryObjectui64vEXT"));
    if (!get_query_object_ui64v) {
        return env->NewStringUTF("{\"error\":\"glGetQueryObjectui64vEXT_unavailable\"}");
    }

    GLuint queries[2] = {};
    glGenQueries(2, queries);
    if (!queries[0] || !queries[1]) {
        return env->NewStringUTF("{\"error\":\"query_allocation_failed\"}");
    }

    const int old_backend = g_backend;
    const bool old_fixed = g_use_fixed_frame;
    const int old_frame = g_fixed_frame;
    g_backend = 0;  // ZD-ECN GLSL path
    g_use_fixed_frame = false;
    g_pipeline.setInternalZdecnTimerEnabled(false);
    auto restore_state = [&]() {
        glDeleteQueries(2, queries);
        g_pipeline.setInternalZdecnTimerEnabled(true);
        g_backend = old_backend;
        g_use_fixed_frame = old_fixed;
        g_fixed_frame = old_frame;
        g_pose_frame = -1;
    };

    auto run_iteration = [&](int frame) -> bool {
        g_pose_frame = -1;  // force full dispatch chain every iteration
        if (!updateGpuGeometry(frame)) return false;
        drawFrame();
        return glGetError() == GL_NO_ERROR;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!run_iteration((i * 37) % kFrameCount)) {
            restore_state();
            return env->NewStringUTF("{\"error\":\"warmup_failed\"}");
        }
        glFinish();
    }

    std::vector<double> gpu_ms(static_cast<size_t>(iterations));
    std::vector<double> cpu_ms(static_cast<size_t>(iterations));
    int disjoint_count = 0;
    for (int i = 0; i < iterations; ++i) {
        const int frame = (i * 113) % kFrameCount;  // covers all pose tiers
        const auto cpu_start = std::chrono::steady_clock::now();
        glBeginQuery(GL_TIME_ELAPSED_EXT, queries[i & 1]);
        const bool ok = run_iteration(frame);
        glEndQuery(GL_TIME_ELAPSED_EXT);
        glFinish();
        const auto cpu_end = std::chrono::steady_clock::now();
        if (!ok) {
            restore_state();
            return env->NewStringUTF("{\"error\":\"measured_iteration_failed\"}");
        }
        GLuint64 elapsed_ns = 0;
        get_query_object_ui64v(queries[i & 1], GL_QUERY_RESULT, &elapsed_ns);
        GLint disjoint = GL_FALSE;
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
        if (disjoint == GL_TRUE) ++disjoint_count;
        gpu_ms[static_cast<size_t>(i)] = static_cast<double>(elapsed_ns) / 1.0e6;
        cpu_ms[static_cast<size_t>(i)] =
                std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    restore_state();

    auto stats_json = [](std::vector<double> v) -> std::string {
        const size_t n = v.size();
        double mean = 0.0;
        for (double x : v) mean += x;
        mean /= static_cast<double>(n);
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        const double stddev = n > 1 ? std::sqrt(var / static_cast<double>(n - 1)) : 0.0;
        std::sort(v.begin(), v.end());
        auto pct = [&](double p) {
            const double idx = p / 100.0 * static_cast<double>(n - 1);
            const size_t lo = static_cast<size_t>(idx);
            const size_t hi = std::min(lo + 1, n - 1);
            const double frac = idx - static_cast<double>(lo);
            return v[lo] * (1.0 - frac) + v[hi] * frac;
        };
        std::ostringstream s;
        s.setf(std::ios::fixed);
        s.precision(6);
        s << "{\"mean\":" << mean << ",\"std\":" << stddev
          << ",\"p50\":" << pct(50) << ",\"p95\":" << pct(95)
          << ",\"p99\":" << pct(99) << ",\"min\":" << v.front()
          << ",\"max\":" << v.back() << "}";
        return s.str();
    };

    double gpu_mean = 0.0;
    for (double x : gpu_ms) gpu_mean += x;
    gpu_mean /= static_cast<double>(gpu_ms.size());

    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(6);
    out << "{\"protocol\":{\"warmup\":" << warmup
        << ",\"iterations\":" << iterations
        << ",\"timer\":\"GL_EXT_disjoint_timer_query\""
        << ",\"boundary\":\"shape_blend+lbs+zdecn+combine+normals+indexed_draw\""
        << ",\"swap_excluded\":true"
        << ",\"backend\":\"zdecn_glsl_fp16\"}"
        << ",\"gpu_ms\":" << stats_json(gpu_ms)
        << ",\"cpu_wall_ms\":" << stats_json(cpu_ms)
        << ",\"fps_from_gpu_mean\":" << (1000.0 / gpu_mean)
        << ",\"disjoint_count\":" << disjoint_count
        << "}";
    const std::string json = out.str();
    LOGI("full pipeline benchmark: %s", json.c_str());
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeGetStatus(
        JNIEnv* env, jclass) {
    return env->NewStringUTF(g_status.c_str());
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_graphicsresearch_skinrenderer_SkinRendererNative_nativeGetZdecnMs(
        JNIEnv*, jclass) {
    return static_cast<jfloat>(g_pipeline.lastZdecnGpuMs());
}

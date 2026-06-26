#include "smpl_gpu_pipeline.h"

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <GLES3/gl31.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SmplRenderer", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SmplRenderer", __VA_ARGS__)

namespace {
struct Vec3 { float x, y, z; };
struct Vertex { float px, py, pz, nx, ny, nz; };
struct Mat4 { float m[16]; };

constexpr int kVertexCount = 6890;
constexpr int kFrameCount = 600;

AAssetManager* g_assets = nullptr;
SmplGpuPipeline g_pipeline;
GLuint g_program = 0, g_vbo = 0, g_ibo = 0, g_vao = 0;
GLint g_u_mvp = -1, g_u_model = -1, g_u_color = -1;
int g_width = 1, g_height = 1;
int g_backend = 0, g_tier = 0;
int g_fixed_frame = 0;
bool g_use_fixed_frame = false;
float g_time = 0.0f;
float g_user_yaw = 0.0f;
float g_user_pitch = -0.10f;
float g_user_zoom = 1.0f;
int g_pose_frame = -1;
size_t g_index_count = 0;
std::string g_status = "not initialized";
std::vector<float> g_pose_aa;
std::vector<float> g_pose_rot6d;
std::vector<uint16_t> g_indices;
std::vector<float> g_positions_xyz;
std::vector<Vertex> g_render_vertices;
SmplGpuFrameStats g_frame_stats;
double g_dispatch_lbs_ms = 0.0;
double g_dispatch_zdecn_ms = 0.0;
auto g_start_time = std::chrono::steady_clock::now();

std::vector<uint8_t> readAsset(const char* name) {
    std::vector<uint8_t> data;
    if (!g_assets) return data;
    AAsset* asset = AAssetManager_open(g_assets, name, AASSET_MODE_BUFFER);
    if (!asset) { LOGE("missing asset %s", name); return data; }
    const off_t len = AAsset_getLength(asset);
    data.resize(static_cast<size_t>(len));
    const int64_t n = AAsset_read(asset, data.data(), data.size());
    AAsset_close(asset);
    if (n <= 0) return {};
    data.resize(static_cast<size_t>(n));
    return data;
}

std::string readText(const char* name) {
    auto bytes = readAsset(name);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

template <typename T>
bool readBinary(const char* name, std::vector<T>& out) {
    auto bytes = readAsset(name);
    if (bytes.empty() || (bytes.size() % sizeof(T)) != 0) return false;
    out.resize(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

GLuint compile(GLenum type, const std::string& src) {
    GLuint shader = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(shader, 1, &c, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetShaderInfoLog(shader, sizeof(log), &n, log);
        LOGE("shader compile failed: %s", log);
    }
    return shader;
}

GLuint linkProgram(const std::string& vs, const std::string& fs) {
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log), &n, log);
        LOGE("program link failed: %s", log);
    }
    return p;
}

Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
void add(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
Vec3 norm(Vec3 a) {
    float l = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    return l > 1e-8f ? Vec3{a.x/l, a.y/l, a.z/l} : Vec3{0,0,1};
}

Mat4 identity() { Mat4 r{}; r.m[0]=r.m[5]=r.m[10]=r.m[15]=1; return r; }
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c=0;c<4;c++) for (int row=0;row<4;row++) for (int k=0;k<4;k++) r.m[c*4+row] += a.m[k*4+row]*b.m[c*4+k];
    return r;
}
Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
    Mat4 r{}; float f = 1.0f / std::tan(fovy * 0.5f);
    r.m[0]=f/aspect; r.m[5]=f; r.m[10]=(zfar+znear)/(znear-zfar); r.m[11]=-1; r.m[14]=(2*zfar*znear)/(znear-zfar); return r;
}
Mat4 translate(float x,float y,float z){ Mat4 r=identity(); r.m[12]=x; r.m[13]=y; r.m[14]=z; return r; }
Mat4 rotateY(float a){ Mat4 r=identity(); float c=std::cos(a), s=std::sin(a); r.m[0]=c; r.m[2]=-s; r.m[8]=s; r.m[10]=c; return r; }
Mat4 rotateX(float a){ Mat4 r=identity(); float c=std::cos(a), s=std::sin(a); r.m[5]=c; r.m[6]=s; r.m[9]=-s; r.m[10]=c; return r; }
Mat4 scale(float s){ Mat4 r=identity(); r.m[0]=r.m[5]=r.m[10]=s; return r; }

bool createRenderBuffers() {
    if (!readBinary("animation/animation_poses_aa_f32.bin", g_pose_aa) ||
        !readBinary("animation/animation_poses_rot6d_f32.bin", g_pose_rot6d) ||
        !readBinary("smpl_shader_data/faces_u16.bin", g_indices)) {
        g_status = "animation/faces asset load failed";
        return false;
    }
    if (g_pose_aa.size() != kFrameCount * 72 || g_pose_rot6d.size() != kFrameCount * 144 || g_indices.empty()) {
        g_status = "animation/faces shape mismatch";
        return false;
    }
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, kVertexCount * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &g_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_indices.size() * sizeof(uint16_t), g_indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
    g_index_count = g_indices.size();
    return true;
}

void uploadVerticesWithNormals() {
    std::vector<Vec3> normals(kVertexCount, {0,0,0});
    for (size_t i = 0; i + 2 < g_indices.size(); i += 3) {
        uint16_t ia = g_indices[i], ib = g_indices[i + 1], ic = g_indices[i + 2];
        Vec3 a{g_positions_xyz[ia*3], g_positions_xyz[ia*3+1], g_positions_xyz[ia*3+2]};
        Vec3 b{g_positions_xyz[ib*3], g_positions_xyz[ib*3+1], g_positions_xyz[ib*3+2]};
        Vec3 c{g_positions_xyz[ic*3], g_positions_xyz[ic*3+1], g_positions_xyz[ic*3+2]};
        Vec3 n = cross(sub(b,a), sub(c,a));
        add(normals[ia], n); add(normals[ib], n); add(normals[ic], n);
    }
    g_render_vertices.resize(kVertexCount);
    for (int i = 0; i < kVertexCount; ++i) {
        Vec3 n = norm(normals[i]);
        g_render_vertices[i] = {g_positions_xyz[i*3], g_positions_xyz[i*3+1] - 0.9f, g_positions_xyz[i*3+2], n.x, n.y, n.z};
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, g_render_vertices.size() * sizeof(Vertex), g_render_vertices.data(), GL_DYNAMIC_DRAW);
}

bool updateGpuGeometry(int frame) {
    if (!g_pipeline.ready() || frame == g_pose_frame) return g_pipeline.ready();
    g_pose_frame = frame;
    const float* aa = &g_pose_aa[frame * 72];
    const float* rot6d = &g_pose_rot6d[frame * 144];
    const bool apply_delta = g_backend != 4;

    auto t0 = std::chrono::steady_clock::now();
    if (!g_pipeline.dispatchLbs(aa)) {
        g_status = "GPU LBS failed: " + g_pipeline.error();
        LOGE("%s", g_status.c_str());
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (apply_delta && (g_backend == 0 || g_backend == 1)) {
        if (!g_pipeline.dispatchZdecn(rot6d)) {
            g_status = "GPU ZD-ECN failed: " + g_pipeline.error();
            LOGE("%s", g_status.c_str());
            return false;
        }
    }
    auto t2 = std::chrono::steady_clock::now();
    if (!g_pipeline.readCombinedVertices(apply_delta, &g_positions_xyz, &g_frame_stats)) {
        g_status = "GPU readback failed: " + g_pipeline.error();
        LOGE("%s", g_status.c_str());
        return false;
    }
    uploadVerticesWithNormals();
    g_dispatch_lbs_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    g_dispatch_zdecn_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    g_status = "GPU SSBO LBS + ZD-ECN renderer active";
    return true;
}

const char* backendName() {
    switch (g_backend) {
        case 0: return "ZD-ECN native";
        case 1: return "ShaderNN-style";
        case 2: return "ONNX CPU";
        case 3: return "ONNX NNAPI";
        default: return "LBS only";
    }
}
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeInit(JNIEnv* env, jclass, jobject assetManager) {
    g_assets = AAssetManager_fromJava(env, assetManager);
    g_program = linkProgram(readText("shaders/mesh_vertex.vert"), readText("shaders/mesh_fragment.frag"));
    g_u_mvp = glGetUniformLocation(g_program, "uMvp");
    g_u_model = glGetUniformLocation(g_program, "uModel");
    g_u_color = glGetUniformLocation(g_program, "uColor");
    const bool render_ok = createRenderBuffers();
    const bool gpu_ok = g_pipeline.init(g_assets);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    g_status = render_ok && gpu_ok ? "GPU SSBO pipeline initialized" : (render_ok ? g_pipeline.error() : g_status);
    LOGI("native init %s", g_status.c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeResize(JNIEnv*, jclass, jint w, jint h) {
    g_width = w > 0 ? w : 1;
    g_height = h > 0 ? h : 1;
    glViewport(0, 0, g_width, g_height);
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeSetBackend(JNIEnv*, jclass, jint backend, jint tier) {
    g_backend = backend;
    g_tier = tier;
    g_pose_frame = -1;
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeSetFixedFrame(JNIEnv*, jclass, jint frame, jboolean enabled) {
    g_fixed_frame = std::max(0, std::min(599, static_cast<int>(frame)));
    g_use_fixed_frame = enabled == JNI_TRUE;
    g_pose_frame = -1;
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeSetCamera(JNIEnv*, jclass, jfloat yaw, jfloat pitch, jfloat zoom) {
    g_user_yaw = yaw;
    g_user_pitch = std::max(-0.55f, std::min(0.35f, static_cast<float>(pitch)));
    g_user_zoom = std::max(0.70f, std::min(1.55f, static_cast<float>(zoom)));
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeDraw(JNIEnv*, jclass) {
    g_time += 1.0f / 60.0f;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - g_start_time).count();
    int tier_base = g_tier == 0 ? 0 : (g_tier == 1 ? 200 : 400);
    int frame = g_use_fixed_frame ? g_fixed_frame : tier_base + (static_cast<int>(elapsed * 10.0) % 200);
    updateGpuGeometry(frame);

    float bg = g_tier == 2 ? 0.045f : (g_tier == 1 ? 0.035f : 0.025f);
    glClearColor(bg, bg + 0.015f, bg + 0.03f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!g_program || !g_vao || g_positions_xyz.empty()) return;

    float aspect = static_cast<float>(g_width) / static_cast<float>(g_height);
    const bool portrait = g_height >= g_width;
    const float fov_deg = portrait ? 48.0f : 55.0f;
    const float camera_distance = (portrait ? 4.80f : 3.35f) * g_user_zoom;
    const float model_scale = portrait ? 0.98f : 1.28f;
    const float auto_yaw = g_use_fixed_frame ? 0.0f : g_time * 0.10f;
    Mat4 proj = perspective(fov_deg * 3.14159265f / 180.0f, aspect, 0.05f, 20.0f);
    Mat4 view = translate(0, portrait ? 0.0f : -0.03f, -camera_distance);
    Mat4 model = mul(rotateY(g_user_yaw + auto_yaw), rotateX(g_user_pitch));
    model = mul(translate(0.0f, portrait ? 0.95f : 0.08f, 0.0f), model);
    model = mul(scale(model_scale), model);
    Mat4 mvp = mul(proj, mul(view, model));

    glUseProgram(g_program);
    glUniformMatrix4fv(g_u_mvp, 1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(g_u_model, 1, GL_FALSE, model.m);
    float colors[5][3]={{0.72f,0.76f,0.82f},{0.72f,0.84f,0.72f},{0.84f,0.72f,0.72f},{0.80f,0.72f,0.88f},{0.95f,0.78f,0.42f}};
    glUniform3fv(g_u_color, 1, colors[g_backend < 0 ? 4 : (g_backend > 4 ? 4 : g_backend)]);
    glBindVertexArray(g_vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(g_index_count), GL_UNSIGNED_SHORT, nullptr);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeStatus(JNIEnv* env, jclass) {
    char buf[768];
    const char* backend_note = g_backend == 4 ? "delta skipped" : "ZD-ECN GLSL dispatch used";
    if (g_backend == 1) backend_note = "currently same GLSL SSBO path; ShaderNN layout pending";
    if (g_backend == 2) backend_note = "ONNX CPU delta uploaded to GL SSBO";
    if (g_backend == 3) backend_note = "ONNX NNAPI delta uploaded to GL SSBO when available";
    std::snprintf(buf, sizeof(buf),
                  "FPS: native | Backend: %s | Tier: %d | frame %d | LBS dispatch+read %.2f ms | ZD dispatch %.2f ms | read %.2f ms | maxΔ %.1f mm | %s | %s",
                  backendName(), g_tier, g_pose_frame, g_dispatch_lbs_ms, g_dispatch_zdecn_ms,
                  g_frame_stats.readback_cpu_ms, g_frame_stats.max_abs_delta_mm, backend_note, g_status.c_str());
    if (g_backend == 0 || g_backend == 1) {
        std::snprintf(buf, sizeof(buf),
                      "FPS: native | Backend: %s | Tier: %d | frame %d | LBS dispatch+read %.2f ms | ZD dispatch %.2f ms | ZD GPU %.3f ms | read %.2f ms | maxΔ %.1f mm | %s | %s",
                      backendName(), g_tier, g_pose_frame, g_dispatch_lbs_ms, g_dispatch_zdecn_ms,
                      g_pipeline.lastZdecnGpuMs(), g_frame_stats.readback_cpu_ms,
                      g_frame_stats.max_abs_delta_mm, backend_note, g_status.c_str());
    }
    return env->NewStringUTF(buf);
}

extern "C" JNIEXPORT void JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeRelease(JNIEnv*, jclass) {
    g_pipeline.release();
    if (g_vbo) glDeleteBuffers(1, &g_vbo);
    if (g_ibo) glDeleteBuffers(1, &g_ibo);
    if (g_vao) glDeleteVertexArrays(1, &g_vao);
    if (g_program) glDeleteProgram(g_program);
    g_vbo = g_ibo = g_vao = g_program = 0;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_graphicsresearch_smplrenderer_RendererNative_nativeUploadExternalDelta(JNIEnv* env, jclass, jfloatArray delta) {
    if (delta == nullptr || env->GetArrayLength(delta) != 20670) {
        return JNI_FALSE;
    }
    jboolean is_copy = JNI_FALSE;
    jfloat* ptr = env->GetFloatArrayElements(delta, &is_copy);
    if (ptr == nullptr) {
        return JNI_FALSE;
    }
    const bool ok = g_pipeline.uploadExternalDelta(ptr);
    env->ReleaseFloatArrayElements(delta, ptr, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

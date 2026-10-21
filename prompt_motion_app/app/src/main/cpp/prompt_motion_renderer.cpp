// prompt_motion_renderer.cpp — JNI entry points for the prompt-to-motion app.
// Plays a prompt-generated SMPL pose clip through the validated SmplGpuPipeline
// (shared with android_smpl_app / skin_rendering, unchanged): per pose frame
// GPU LBS + ZD-ECN correction + combine + normals, then the wrap-lit skin shader.
//
// Frames: the pipeline outputs vertices in the Z-up world ZD-ECN was trained in
// (root orientation is part of the pose). Display world is Y-up; the model matrix
// is D · T(trans[f]) with D = Rx(-90°).

#include "smpl_gpu_pipeline.h"

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

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PromptMotion", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PromptMotion", __VA_ARGS__)

#define JNI_FN(name) Java_com_graphicsresearch_promptmotion_MotionRendererNative_##name

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
Mat4 lookAt(const float eye[3], const float at[3]) {
    float f[3] = {at[0]-eye[0], at[1]-eye[1], at[2]-eye[2]};
    float fl = std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    for (float& v : f) v /= fl;
    float s[3] = {f[1]*0.0f - f[2]*1.0f, f[2]*0.0f - f[0]*0.0f, f[0]*1.0f - f[1]*0.0f};  // f × up(0,1,0)
    float sl = std::sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    for (float& v : s) v /= sl;
    float u[3] = {s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]};
    Mat4 r = identity();
    r.m[0]=s[0]; r.m[4]=s[1]; r.m[8]=s[2];
    r.m[1]=u[0]; r.m[5]=u[1]; r.m[9]=u[2];
    r.m[2]=-f[0]; r.m[6]=-f[1]; r.m[10]=-f[2];
    r.m[12]=-(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
    r.m[13]=-(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
    r.m[14]= (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
    return r;
}
// D = Rx(-90°): Z-up pipeline world (x, y, z) -> Y-up display world (x, z, -y)
Mat4 zUpToYUp() {
    Mat4 r = identity();
    r.m[5]=0; r.m[6]=-1; r.m[9]=1; r.m[10]=0;
    return r;
}

// ── globals ───────────────────────────────────────────────────────────────────
constexpr int kVertexCount = 6890;

AAssetManager*  g_assets = nullptr;
SmplGpuPipeline g_pipeline;

GLuint g_skin_program = 0, g_floor_program = 0;
GLuint g_ibo = 0, g_vao = 0, g_uv_vbo = 0, g_floor_vao = 0, g_floor_vbo = 0;
GLuint g_textures[2] = {};
int    g_active_texture = 0;
bool   g_use_texture = true;

GLint g_u_mvp = -1, g_u_model = -1, g_u_normal_mat = -1;
GLint g_u_light_dir = -1, g_u_light_color = -1, g_u_view_dir = -1;
GLint g_u_wrap = -1, g_u_spec = -1, g_u_use_tex = -1, g_u_skin_tex = -1;
GLint g_uf_viewproj = -1, g_uf_focus = -1, g_uf_shadow = -1, g_uf_lift = -1;

int   g_width = 1, g_height = 1;
int   g_backend = 0;               // 0=ZD-ECN GLSL, 1=ONNX CPU, 2=LBS only
float g_user_yaw = 0.35f, g_user_pitch = 0.0f, g_user_zoom = 1.0f;
float g_light_angle = 0.5f, g_wrap_factor = 0.3f, g_spec_strength = 0.05f;
bool  g_geometry_ready = false;
size_t g_index_count = 0;
std::string g_status = "not initialized";
std::vector<uint16_t> g_indices;

// Motion clip (set from Kotlin on the GL thread via queueEvent)
struct Clip {
    int fps = 30;
    int frames = 0;
    std::vector<float> aa, rot6d, trans;
    float rest_pelvis[3] = {0, 0, 0};
} g_clip;
int    g_pose_frame = -1;           // frame currently in the GPU buffers
double g_play_time = 0.0;
bool   g_playing = true;
float  g_speed = 1.0f;
int    g_fixed_frame = -1;          // >=0 pins playback (screenshots)
auto   g_last_tick = std::chrono::steady_clock::now();
float  g_cam_target[3] = {0.0f, 0.9f, 0.0f};
bool   g_cam_snap = true;

// ── asset helpers ─────────────────────────────────────────────────────────────
std::vector<uint8_t> readAsset(const char* name) {
    if (!g_assets) return {};
    AAsset* a = AAssetManager_open(g_assets, name, AASSET_MODE_BUFFER);
    if (!a) { LOGE("missing asset: %s", name); return {}; }
    std::vector<uint8_t> data(static_cast<size_t>(AAsset_getLength(a)));
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
    GLuint v = compile(GL_VERTEX_SHADER, vs_src);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
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
    return tex;
}

// ── init sub-routines ─────────────────────────────────────────────────────────
bool createGeometryBuffers() {
    if (!readBinary("smpl_shader_data/faces_u16.bin", g_indices) || g_indices.empty()) {
        g_status = "faces asset load failed";
        return false;
    }
    glGenBuffers(1, &g_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_indices.size() * sizeof(uint16_t),
                 g_indices.data(), GL_STATIC_DRAW);
    g_index_count = g_indices.size();

    std::vector<float> uv;
    if (!readBinary("smpl_uv_cylindrical_f32.bin", uv) || uv.size() != kVertexCount * 2) {
        g_status = "UV asset load failed (expected 6890×2 floats)";
        return false;
    }
    glGenBuffers(1, &g_uv_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_uv_vbo);
    glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(float), uv.data(), GL_STATIC_DRAW);
    glGenVertexArrays(1, &g_vao);

    // Floor: 40 m quad at display y = 0, two triangles CCW seen from above
    const float s = 20.0f;
    const float floor_verts[] = { -s,0,-s,  -s,0,s,  s,0,s,   -s,0,-s,  s,0,s,  s,0,-s };
    glGenVertexArrays(1, &g_floor_vao);
    glGenBuffers(1, &g_floor_vbo);
    glBindVertexArray(g_floor_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_floor_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_verts), floor_verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);
    return glGetError() == GL_NO_ERROR;
}

bool configureRenderVertexBuffer() {
    const GLuint pos_buf  = g_pipeline.renderBuffer();
    const GLuint norm_buf = g_pipeline.normalBuffer();
    if (!g_vao || !pos_buf || !norm_buf || !g_uv_vbo) return false;
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ibo);
    glBindBuffer(GL_ARRAY_BUFFER, pos_buf);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, norm_buf);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, g_uv_vbo);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindVertexArray(0);
    return glGetError() == GL_NO_ERROR;
}

bool updateGpuGeometry(int frame) {
    if (!g_pipeline.ready() || g_clip.frames <= 0) return false;
    if (frame == g_pose_frame) return true;
    g_pose_frame = frame;
    const float* aa    = &g_clip.aa[static_cast<size_t>(frame) * 72];
    const float* rot6d = &g_clip.rot6d[static_cast<size_t>(frame) * 144];
    const bool apply_delta = (g_backend != 2);

    if (!g_pipeline.dispatchLbs(aa)) {
        g_status = "GPU LBS failed: " + g_pipeline.error(); return false;
    }
    if (g_backend == 0 && !g_pipeline.dispatchZdecn(rot6d)) {
        g_status = "GPU ZD-ECN failed: " + g_pipeline.error(); return false;
    }
    if (!g_pipeline.prepareRenderBuffer(apply_delta)) {
        g_status = "GPU combine failed: " + g_pipeline.error(); return false;
    }
    if (!g_geometry_ready) {
        configureRenderVertexBuffer();
        g_geometry_ready = true;
        LOGI("GPU geometry ready — prompt motion renderer active");
    }
    return true;
}

int currentFrame() {
    if (g_clip.frames <= 0) return -1;
    if (g_fixed_frame >= 0) return std::min(g_fixed_frame, g_clip.frames - 1);
    const int f = static_cast<int>(g_play_time * g_clip.fps);
    return std::max(0, std::min(f, g_clip.frames - 1));
}

// ── draw ──────────────────────────────────────────────────────────────────────
void drawFrame(int frame, bool follow_camera) {
    glClearColor(0.04f, 0.05f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!g_skin_program || !g_vao || !g_geometry_ready || frame < 0) return;

    const float* t = &g_clip.trans[static_cast<size_t>(frame) * 3];
    // Pelvis in Z-up world, then display (x, z, -y)
    const float px = t[0] + g_clip.rest_pelvis[0];
    const float py = t[1] + g_clip.rest_pelvis[1];
    const float pz = t[2] + g_clip.rest_pelvis[2];
    const float pelvis_disp[3] = {px, pz, -py};

    if (follow_camera) {
        const float k = g_cam_snap ? 1.0f : 0.08f;
        g_cam_target[0] += (pelvis_disp[0] - g_cam_target[0]) * k;
        g_cam_target[2] += (pelvis_disp[2] - g_cam_target[2]) * k;
        g_cam_target[1] = 0.85f;
        g_cam_snap = false;
    }

    const float aspect = static_cast<float>(g_width) / static_cast<float>(g_height);
    const bool portrait = g_height >= g_width;
    const float fov = (portrait ? 50.0f : 42.0f) * 3.14159265f / 180.0f;
    const float dist = (portrait ? 3.7f : 3.0f) * g_user_zoom;
    const float elev = 0.12f + g_user_pitch;
    const float eye[3] = {
        g_cam_target[0] + dist * std::sin(g_user_yaw) * std::cos(elev),
        g_cam_target[1] + dist * std::sin(elev),
        g_cam_target[2] + dist * std::cos(g_user_yaw) * std::cos(elev),
    };
    Mat4 proj = perspective(fov, aspect, 0.05f, 40.0f);
    Mat4 view = lookAt(eye, g_cam_target);
    Mat4 viewproj = mul(proj, view);
    Mat4 model = mul(zUpToYUp(), translate(t[0], t[1], t[2]));
    Mat4 mvp = mul(viewproj, model);

    // Floor first (depth-tested; body is always above it)
    if (g_floor_program) {
        glDisable(GL_CULL_FACE);
        glUseProgram(g_floor_program);
        glUniformMatrix4fv(g_uf_viewproj, 1, GL_FALSE, viewproj.m);
        glUniform2f(g_uf_focus, g_cam_target[0], g_cam_target[2]);
        glUniform2f(g_uf_shadow, pelvis_disp[0], pelvis_disp[2]);
        glUniform1f(g_uf_lift, std::max(0.0f, pelvis_disp[1] - 0.95f) * 2.0f);
        glBindVertexArray(g_floor_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glEnable(GL_CULL_FACE);
    }

    float lx = std::sin(g_light_angle), lz = std::cos(g_light_angle);
    float light_dir[3] = {lx * 0.6f, 0.8f, lz * 0.6f};
    float len = std::sqrt(light_dir[0]*light_dir[0] + light_dir[1]*light_dir[1] + light_dir[2]*light_dir[2]);
    for (float& v : light_dir) v /= len;
    float view_dir[3] = {eye[0]-g_cam_target[0], eye[1]-g_cam_target[1], eye[2]-g_cam_target[2]};
    float vl = std::sqrt(view_dir[0]*view_dir[0] + view_dir[1]*view_dir[1] + view_dir[2]*view_dir[2]);
    for (float& v : view_dir) v /= vl;

    // Model is rotation + translation only: normal matrix = upper-left 3×3
    const float nm[9] = {model.m[0], model.m[1], model.m[2], model.m[4], model.m[5], model.m[6],
                         model.m[8], model.m[9], model.m[10]};

    glUseProgram(g_skin_program);
    glUniformMatrix4fv(g_u_mvp, 1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(g_u_model, 1, GL_FALSE, model.m);
    glUniformMatrix3fv(g_u_normal_mat, 1, GL_FALSE, nm);
    glUniform3fv(g_u_light_dir, 1, light_dir);
    glUniform3f(g_u_light_color, 1.0f, 0.97f, 0.93f);
    glUniform3fv(g_u_view_dir, 1, view_dir);
    glUniform1f(g_u_wrap, g_wrap_factor);
    glUniform1f(g_u_spec, g_spec_strength);
    glUniform1i(g_u_use_tex, g_use_texture ? 1 : 0);
    if (g_use_texture && g_textures[g_active_texture]) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_textures[g_active_texture]);
        glUniform1i(g_u_skin_tex, 0);
    }
    glBindVertexArray(g_vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(g_index_count), GL_UNSIGNED_SHORT, nullptr);
    glBindVertexArray(0);
    if (g_use_texture) glBindTexture(GL_TEXTURE_2D, 0);
}

std::string statsJson(std::vector<double> v) {
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
    s << "{\"mean\":" << mean << ",\"std\":" << stddev << ",\"p50\":" << pct(50)
      << ",\"p95\":" << pct(95) << ",\"p99\":" << pct(99) << ",\"min\":" << v.front()
      << ",\"max\":" << v.back() << "}";
    return s.str();
}

}  // namespace

// ── JNI exports ───────────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeInit)(JNIEnv* env, jclass, jobject asset_manager) {
    g_assets = AAssetManager_fromJava(env, asset_manager);

    g_skin_program = linkProgram(readText("shaders/skin.vert"), readText("shaders/skin.frag"));
    g_floor_program = linkProgram(readText("shaders/floor.vert"), readText("shaders/floor.frag"));
    if (!g_skin_program) { g_status = "shader link failed"; return; }

    g_u_mvp         = glGetUniformLocation(g_skin_program, "u_MVP");
    g_u_model       = glGetUniformLocation(g_skin_program, "u_Model");
    g_u_normal_mat  = glGetUniformLocation(g_skin_program, "u_NormalMat");
    g_u_light_dir   = glGetUniformLocation(g_skin_program, "u_LightDir");
    g_u_light_color = glGetUniformLocation(g_skin_program, "u_LightColor");
    g_u_view_dir    = glGetUniformLocation(g_skin_program, "u_ViewDir");
    g_u_wrap        = glGetUniformLocation(g_skin_program, "u_WrapFactor");
    g_u_spec        = glGetUniformLocation(g_skin_program, "u_SpecStrength");
    g_u_use_tex     = glGetUniformLocation(g_skin_program, "u_UseTexture");
    g_u_skin_tex    = glGetUniformLocation(g_skin_program, "u_SkinTexture");
    if (g_floor_program) {
        g_uf_viewproj = glGetUniformLocation(g_floor_program, "u_ViewProj");
        g_uf_focus    = glGetUniformLocation(g_floor_program, "u_Focus");
        g_uf_shadow   = glGetUniformLocation(g_floor_program, "u_ShadowCenter");
        g_uf_lift     = glGetUniformLocation(g_floor_program, "u_ShadowLift");
    }

    // GLSurfaceView may recreate the context: drop stale GPU state
    g_geometry_ready = false;
    g_pose_frame = -1;
    g_textures[0] = g_textures[1] = 0;

    const bool geom_ok = createGeometryBuffers();
    const bool gpu_ok  = g_pipeline.init(g_assets);
    const bool topo_ok = geom_ok && gpu_ok &&
        g_pipeline.configureRenderTopology(g_indices.data(), g_indices.size());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    g_status = topo_ok ? "renderer initialized" : (gpu_ok ? g_pipeline.error() : g_status);
    LOGI("nativeInit: %s", g_status.c_str());
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeResize)(JNIEnv*, jclass, jint w, jint h) {
    g_width  = w > 0 ? w : 1;
    g_height = h > 0 ? h : 1;
    glViewport(0, 0, g_width, g_height);
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeDraw)(JNIEnv*, jclass) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::min(0.1, std::chrono::duration<double>(now - g_last_tick).count());
    g_last_tick = now;
    if (g_playing && g_clip.frames > 0 && g_fixed_frame < 0) {
        const double duration = static_cast<double>(g_clip.frames) / g_clip.fps;
        g_play_time += dt * g_speed;
        if (g_play_time >= duration) {
            g_play_time = std::fmod(g_play_time, duration);
            g_cam_snap = true;  // clip loops back to its start position
        }
    }
    const int frame = currentFrame();
    if (frame >= 0) updateGpuGeometry(frame);
    drawFrame(frame, true);
}

extern "C" JNIEXPORT jboolean JNICALL
JNI_FN(nativeLoadMotion)(JNIEnv* env, jclass, jfloatArray aa, jfloatArray rot6d,
                         jfloatArray trans, jfloatArray rest_pelvis, jint fps) {
    const jsize n_aa = env->GetArrayLength(aa);
    const jsize frames = n_aa / 72;
    if (frames <= 0 || n_aa != frames * 72 || env->GetArrayLength(rot6d) != frames * 144 ||
        env->GetArrayLength(trans) != frames * 3 || env->GetArrayLength(rest_pelvis) != 3 || fps <= 0) {
        LOGE("nativeLoadMotion: bad shapes");
        return JNI_FALSE;
    }
    g_clip.fps = fps;
    g_clip.frames = frames;
    g_clip.aa.resize(static_cast<size_t>(frames) * 72);
    g_clip.rot6d.resize(static_cast<size_t>(frames) * 144);
    g_clip.trans.resize(static_cast<size_t>(frames) * 3);
    env->GetFloatArrayRegion(aa, 0, frames * 72, g_clip.aa.data());
    env->GetFloatArrayRegion(rot6d, 0, frames * 144, g_clip.rot6d.data());
    env->GetFloatArrayRegion(trans, 0, frames * 3, g_clip.trans.data());
    env->GetFloatArrayRegion(rest_pelvis, 0, 3, g_clip.rest_pelvis);
    g_play_time = 0.0;
    g_pose_frame = -1;
    g_cam_snap = true;
    LOGI("motion loaded: %d frames @ %d fps (%.2f s)", frames, fps, frames / static_cast<float>(fps));
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetPlaying)(JNIEnv*, jclass, jboolean playing) { g_playing = playing == JNI_TRUE; }

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetSpeed)(JNIEnv*, jclass, jfloat speed) { g_speed = std::max(0.1f, std::min(2.0f, speed)); }

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSeek)(JNIEnv*, jclass, jfloat seconds) {
    if (g_clip.frames <= 0) return;
    const double duration = static_cast<double>(g_clip.frames) / g_clip.fps;
    g_play_time = std::max(0.0, std::min(static_cast<double>(seconds), duration - 1e-4));
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetFixedFrame)(JNIEnv*, jclass, jint frame) {
    g_fixed_frame = frame;
    g_cam_snap = true;
}

extern "C" JNIEXPORT jfloat JNICALL
JNI_FN(nativeGetPlaybackTime)(JNIEnv*, jclass) { return static_cast<jfloat>(g_play_time); }

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetBackend)(JNIEnv*, jclass, jint backend) { g_backend = backend; g_pose_frame = -1; }

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetCamera)(JNIEnv*, jclass, jfloat yaw, jfloat pitch, jfloat zoom) {
    g_user_yaw   = yaw;
    g_user_pitch = std::max(-0.10f, std::min(0.9f, pitch));
    g_user_zoom  = std::max(0.45f, std::min(2.2f, zoom));
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeUploadTexture)(JNIEnv* env, jclass, jint index, jbyteArray rgba_bytes, jint width, jint height) {
    if (index < 0 || index > 1) return;
    jbyte* data = env->GetByteArrayElements(rgba_bytes, nullptr);
    const int n = env->GetArrayLength(rgba_bytes);
    if (n == width * height * 4) {
        if (g_textures[index]) glDeleteTextures(1, &g_textures[index]);
        g_textures[index] = uploadTexture(reinterpret_cast<const uint8_t*>(data), width, height);
    } else {
        LOGE("uploadTexture: size mismatch %d vs %dx%dx4", n, width, height);
    }
    env->ReleaseByteArrayElements(rgba_bytes, data, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetTexture)(JNIEnv*, jclass, jint index) { g_active_texture = std::max(0, std::min(1, index)); }

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetUseTexture)(JNIEnv*, jclass, jboolean use_tex) { g_use_texture = (use_tex == JNI_TRUE); }

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeSetLighting)(JNIEnv*, jclass, jfloat wrap, jfloat spec, jfloat light_angle) {
    g_wrap_factor   = std::max(0.0f, std::min(0.8f, wrap));
    g_spec_strength = std::max(0.0f, std::min(0.5f, spec));
    g_light_angle   = light_angle;
}

extern "C" JNIEXPORT void JNICALL
JNI_FN(nativeUploadOnnxDelta)(JNIEnv* env, jclass, jfloatArray delta_arr) {
    if (!g_pipeline.ready()) return;
    if (env->GetArrayLength(delta_arr) != 20670) { LOGE("ONNX delta size ≠ 20670"); return; }
    jfloat* data = env->GetFloatArrayElements(delta_arr, nullptr);
    g_pipeline.uploadExternalDelta(data);
    env->ReleaseFloatArrayElements(delta_arr, data, JNI_ABORT);
    g_pose_frame = -1;  // re-run combine with the new delta
}

extern "C" JNIEXPORT jint JNICALL
JNI_FN(nativeGetCurrentFrame)(JNIEnv*, jclass) { return g_pose_frame; }

extern "C" JNIEXPORT jfloatArray JNICALL
JNI_FN(nativeGetCurrentRot6d)(JNIEnv* env, jclass) {
    if (g_pose_frame < 0 || g_pose_frame >= g_clip.frames) return nullptr;
    jfloatArray out = env->NewFloatArray(144);
    if (!out) return nullptr;
    env->SetFloatArrayRegion(out, 0, 144, &g_clip.rot6d[static_cast<size_t>(g_pose_frame) * 144]);
    return out;
}

extern "C" JNIEXPORT jstring JNICALL
JNI_FN(nativeRunFullPipelineBenchmark)(JNIEnv* env, jclass, jint warmup, jint iterations) {
    // Timer wraps LBS + ZD-ECN + combine + normals + floor + indexed draw on the
    // loaded prompt clip's poses. Swap is outside (issued by GLSurfaceView).
    if (!g_pipeline.ready() || !g_skin_program || g_clip.frames <= 0 || warmup < 0 || iterations <= 0) {
        return env->NewStringUTF("{\"error\":\"benchmark_not_ready\"}");
    }
    const char* ext = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    if (!ext || !std::strstr(ext, "GL_EXT_disjoint_timer_query")) {
        return env->NewStringUTF("{\"error\":\"GL_EXT_disjoint_timer_query_unavailable\"}");
    }
    auto get_query_ui64 = reinterpret_cast<PFNGLGETQUERYOBJECTUI64VEXTPROC>(
            eglGetProcAddress("glGetQueryObjectui64vEXT"));
    if (!get_query_ui64) return env->NewStringUTF("{\"error\":\"glGetQueryObjectui64vEXT_unavailable\"}");

    GLuint queries[2] = {};
    glGenQueries(2, queries);
    const int old_backend = g_backend;
    g_backend = 0;
    g_pipeline.setInternalZdecnTimerEnabled(false);
    auto restore = [&]() {
        glDeleteQueries(2, queries);
        g_pipeline.setInternalZdecnTimerEnabled(true);
        g_backend = old_backend;
        g_pose_frame = -1;
    };
    auto run_iteration = [&](int frame) -> bool {
        g_pose_frame = -1;  // force the full dispatch chain
        if (!updateGpuGeometry(frame)) return false;
        drawFrame(frame, false);
        return glGetError() == GL_NO_ERROR;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!run_iteration((i * 37) % g_clip.frames)) { restore(); return env->NewStringUTF("{\"error\":\"warmup_failed\"}"); }
        glFinish();
    }
    std::vector<double> gpu_ms(static_cast<size_t>(iterations)), cpu_ms(static_cast<size_t>(iterations));
    int disjoint = 0;
    for (int i = 0; i < iterations; ++i) {
        const int frame = (i * 113) % g_clip.frames;
        const auto c0 = std::chrono::steady_clock::now();
        glBeginQuery(GL_TIME_ELAPSED_EXT, queries[i & 1]);
        const bool ok = run_iteration(frame);
        glEndQuery(GL_TIME_ELAPSED_EXT);
        glFinish();
        const auto c1 = std::chrono::steady_clock::now();
        if (!ok) { restore(); return env->NewStringUTF("{\"error\":\"measured_iteration_failed\"}"); }
        GLuint64 ns = 0;
        get_query_ui64(queries[i & 1], GL_QUERY_RESULT, &ns);
        GLint dj = GL_FALSE;
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &dj);
        if (dj == GL_TRUE) ++disjoint;
        gpu_ms[static_cast<size_t>(i)] = static_cast<double>(ns) / 1.0e6;
        cpu_ms[static_cast<size_t>(i)] = std::chrono::duration<double, std::milli>(c1 - c0).count();
    }
    restore();

    double gpu_mean = 0.0;
    for (double x : gpu_ms) gpu_mean += x;
    gpu_mean /= static_cast<double>(gpu_ms.size());
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(6);
    out << "{\"protocol\":{\"warmup\":" << warmup << ",\"iterations\":" << iterations
        << ",\"timer\":\"GL_EXT_disjoint_timer_query\""
        << ",\"boundary\":\"lbs+zdecn+combine+normals+floor+indexed_draw\""
        << ",\"swap_excluded\":true,\"backend\":\"zdecn_glsl_fp16\""
        << ",\"pose_source\":\"prompt_clip\",\"clip_frames\":" << g_clip.frames << "}"
        << ",\"gpu_ms\":" << statsJson(gpu_ms) << ",\"cpu_wall_ms\":" << statsJson(cpu_ms)
        << ",\"fps_from_gpu_mean\":" << (1000.0 / gpu_mean)
        << ",\"disjoint_count\":" << disjoint << "}";
    const std::string json = out.str();
    LOGI("full pipeline benchmark: %s", json.c_str());
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
JNI_FN(nativeGetStatus)(JNIEnv* env, jclass) { return env->NewStringUTF(g_status.c_str()); }

extern "C" JNIEXPORT jfloat JNICALL
JNI_FN(nativeGetZdecnMs)(JNIEnv*, jclass) { return static_cast<jfloat>(g_pipeline.lastZdecnGpuMs()); }

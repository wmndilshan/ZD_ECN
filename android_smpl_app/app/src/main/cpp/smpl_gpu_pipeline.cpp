#include "smpl_gpu_pipeline.h"

#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl3ext.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SmplGpuPipeline", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SmplGpuPipeline", __VA_ARGS__)

#ifndef GL_TIME_ELAPSED_EXT
#define GL_TIME_ELAPSED_EXT 0x88BF
#endif
#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif
#ifndef PFNGLGETQUERYOBJECTUI64VEXTPROC
typedef void (GL_APIENTRYP PFNGLGETQUERYOBJECTUI64VEXTPROC)(GLuint, GLenum, GLuint64*);
#endif

namespace {
constexpr int kVertexCount = 6890;
constexpr int kJointCount = 24;
constexpr int kNumBetas = 10;
constexpr int kPoseBasisCount = 207;
constexpr int kOutputSize = 20670;
constexpr int kHiddenSize = 64;
constexpr int kInputSize = 144;
constexpr int kLocalSizeX = 256;

std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    return a.back() == '/' ? a + b : a + "/" + b;
}

std::vector<uint8_t> readBinaryAsset(AAssetManager* assets, const char* dir, const char* name) {
    if (!assets) return {};
    const std::string path = joinPath(dir, name);
    AAsset* asset = AAssetManager_open(assets, path.c_str(), AASSET_MODE_BUFFER);
    if (!asset) return {};
    const off_t size = AAsset_getLength(asset);
    std::vector<uint8_t> bytes(static_cast<size_t>(std::max<off_t>(size, 0)));
    const int64_t read_count = AAsset_read(asset, bytes.data(), bytes.size());
    AAsset_close(asset);
    if (read_count <= 0) return {};
    bytes.resize(static_cast<size_t>(read_count));
    return bytes;
}

std::string readTextAsset(AAssetManager* assets, const char* dir, const char* name) {
    auto bytes = readBinaryAsset(assets, dir, name);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::string shaderErrorLog(GLuint object, bool program) {
    GLint length = 0;
    if (program) glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
    else glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
    if (length <= 1) return {};
    std::string log(static_cast<size_t>(length), '\0');
    GLsizei written = 0;
    if (program) glGetProgramInfoLog(object, length, &written, log.data());
    else glGetShaderInfoLog(object, length, &written, log.data());
    log.resize(static_cast<size_t>(written));
    return log;
}

std::string prependDefines(const std::string& source,
                           const std::vector<std::pair<std::string, std::string>>& defines) {
    std::ostringstream out;
    out << "#version 310 es\n";
    out << "precision highp float;\n";
    out << "precision highp int;\n";
    for (const auto& def : defines) out << "#define " << def.first << ' ' << def.second << '\n';
    out << source;
    return out.str();
}

GLuint compileComputeProgram(const std::string& source,
                             const std::vector<std::pair<std::string, std::string>>& defines,
                             std::string* error_out) {
    const std::string final_source = prependDefines(source, defines);
    const char* source_ptr = final_source.c_str();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source_ptr, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        if (error_out) *error_out = "Shader compile failed: " + shaderErrorLog(shader, false);
        glDeleteShader(shader);
        return 0;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        if (error_out) *error_out = "Program link failed: " + shaderErrorLog(program, true);
        glDeleteProgram(program);
        glDeleteShader(shader);
        return 0;
    }
    glDetachShader(program, shader);
    glDeleteShader(shader);
    return program;
}
}

bool SmplGpuPipeline::init(AAssetManager* assets) {
    release();
    assets_ = assets;
    const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    LOGI("GL_EXTENSIONS: %s", extensions ? extensions : "NULL");
    const bool has_fp16_arith = extensions &&
                                std::strstr(extensions, "GL_EXT_shader_explicit_arithmetic_types_float16");
    const bool has_fp16_storage = extensions &&
                                  std::strstr(extensions, "GL_EXT_shader_16bit_storage");
    const bool has_half = extensions &&
                          std::strstr(extensions, "GL_AMD_gpu_shader_half_float");
    LOGI("fp16_arith=%d fp16_storage=%d half=%d",
         has_fp16_arith ? 1 : 0,
         has_fp16_storage ? 1 : 0,
         has_half ? 1 : 0);
    // fp16 packed weights are used unconditionally to match android_compute_benchmark.
    // The shaders use core unpackHalf2x16() on uint SSBO words, not float16 arithmetic
    // types, and this path is validated on Mali-G57 Android 14.
    use_fp16_weights_ = true;
    LOGI("fp16 packed-weight path enabled unconditionally via unpackHalf2x16");
    timer_supported_ = extensions && std::strstr(extensions, "GL_EXT_disjoint_timer_query");
    if (timer_supported_) {
        glGenQueries(1, &zdecn_timer_query_);
        if (zdecn_timer_query_ == 0) {
            timer_supported_ = false;
        }
    }
    if (!loadPrograms() || !createBuffers() || !uploadSmplBuffers() || !uploadMlpBuffers()) {
        release();
        return false;
    }
    lbs_readback_.resize(kOutputSize);
    delta_readback_.resize(kOutputSize);
    initialized_ = true;
    return true;
}

void SmplGpuPipeline::release() {
    for (GLuint& program : smpl_programs_) {
        if (program) glDeleteProgram(program);
        program = 0;
    }
    for (GLuint& program : mlp_programs_) {
        if (program) glDeleteProgram(program);
        program = 0;
    }
    glDeleteBuffers(kBufCount, smpl_buffers_);
    glDeleteBuffers(kMlpCount, mlp_buffers_);
    if (zdecn_timer_query_) {
        glDeleteQueries(1, &zdecn_timer_query_);
        zdecn_timer_query_ = 0;
    }
    std::fill(std::begin(smpl_buffers_), std::end(smpl_buffers_), 0);
    std::fill(std::begin(mlp_buffers_), std::end(mlp_buffers_), 0);
    initialized_ = false;
}

bool SmplGpuPipeline::loadPrograms() {
    const char* smpl_names[4] = {
            "smpl_shape_blend.comp",
            "smpl_joint_regression.comp",
            "smpl_pose_kinematics.comp",
            "smpl_skinning.comp",
    };
    for (int i = 0; i < 4; ++i) {
        smpl_programs_[i] = compileComputeAsset(smpl_names[i], {});
        if (!smpl_programs_[i]) return false;
    }
    std::vector<std::pair<std::string, std::string>> l0{{"INPUT_SIZE", "144"}, {"OUTPUT_SIZE", "64"}};
    std::vector<std::pair<std::string, std::string>> l1{{"INPUT_SIZE", "64"}, {"OUTPUT_SIZE", "64"}};
    std::vector<std::pair<std::string, std::string>> l2{{"INPUT_SIZE", "64"}, {"OUTPUT_SIZE", "20670"}, {"OUTPUT_LINEAR", "1"}};
    if (use_fp16_weights_) {
        l0.emplace_back("USE_FP16_WEIGHTS", "1"); l0.emplace_back("USE_FP16_BIASES", "1");
        l1.emplace_back("USE_FP16_WEIGHTS", "1"); l1.emplace_back("USE_FP16_BIASES", "1");
        l2.emplace_back("USE_FP16_WEIGHTS", "1"); l2.emplace_back("USE_FP16_BIASES", "1");
    }
    mlp_programs_[0] = compileComputeAsset("zdecn_l0.comp", l0);
    mlp_programs_[1] = compileComputeAsset("zdecn_l1.comp", l1);
    mlp_programs_[2] = compileComputeAsset("zdecn_l2.comp", l2);
    return mlp_programs_[0] && mlp_programs_[1] && mlp_programs_[2];
}

GLuint SmplGpuPipeline::compileComputeAsset(
        const char* asset_name,
        const std::vector<std::pair<std::string, std::string>>& defines) {
    const std::string source = readTextAsset(assets_, "shaders", asset_name);
    if (source.empty()) {
        fail(std::string("Unable to load shader asset: shaders/") + asset_name);
        return 0;
    }
    std::string error;
    GLuint program = compileComputeProgram(source, defines, &error);
    if (!program) fail(std::string(asset_name) + ": " + error);
    return program;
}

bool SmplGpuPipeline::createBuffers() {
    glGenBuffers(kBufCount, smpl_buffers_);
    glGenBuffers(kMlpCount, mlp_buffers_);
    for (GLuint buffer : smpl_buffers_) {
        if (!buffer) {
            fail("Failed to allocate SMPL SSBO");
            return false;
        }
    }
    for (GLuint buffer : mlp_buffers_) {
        if (!buffer) {
            fail("Failed to allocate ZD-ECN SSBO");
            return false;
        }
    }
    return true;
}

bool SmplGpuPipeline::uploadAssetBuffer(GLuint buffer,
                                        const char* dir,
                                        const char* name,
                                        size_t expected_bytes,
                                        GLenum usage) {
    std::vector<uint8_t> bytes = readBinaryAsset(assets_, dir, name);
    if (bytes.size() != expected_bytes) {
        fail(std::string("Asset size mismatch for ") + dir + "/" + name +
             ": expected " + std::to_string(expected_bytes) +
             ", got " + std::to_string(bytes.size()));
        return false;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes.size()), bytes.data(), usage);
    return checkGlError(name);
}

bool SmplGpuPipeline::allocateBuffer(GLuint buffer, size_t bytes, GLenum usage) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, usage);
    return checkGlError("allocateBuffer");
}

bool SmplGpuPipeline::uploadSmplBuffers() {
    if (!uploadAssetBuffer(smpl_buffers_[kBufTemplate], "smpl_shader_data", "v_template_f32.bin", kVertexCount * 3 * sizeof(float), GL_STATIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufShapeDirs], "smpl_shader_data", "shapedirs_10_f32.bin", kNumBetas * kVertexCount * 3 * sizeof(float), GL_STATIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufBetas], "smpl_shader_data", "betas_zero_f32.bin", kNumBetas * sizeof(float), GL_DYNAMIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufJointRegressor], "smpl_shader_data", "joint_regressor_f32.bin", kJointCount * kVertexCount * sizeof(float), GL_STATIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufPose], "smpl_shader_data", "pose_axis_angle_zero_f32.bin", kJointCount * 3 * sizeof(float), GL_DYNAMIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufParents], "smpl_shader_data", "parents_i32.bin", kJointCount * sizeof(int32_t), GL_STATIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufPoseDirs], "smpl_shader_data", "posedirs_207_f32.bin", kPoseBasisCount * kVertexCount * 3 * sizeof(float), GL_STATIC_DRAW) ||
        !uploadAssetBuffer(smpl_buffers_[kBufWeights], "smpl_shader_data", "lbs_weights_f32.bin", kVertexCount * kJointCount * sizeof(float), GL_STATIC_DRAW)) {
        return false;
    }
    return allocateBuffer(smpl_buffers_[kBufShaped], kVertexCount * 3 * sizeof(float), GL_DYNAMIC_COPY) &&
           allocateBuffer(smpl_buffers_[kBufJoints], kJointCount * 3 * sizeof(float), GL_DYNAMIC_COPY) &&
           allocateBuffer(smpl_buffers_[kBufPoseFeatures], kPoseBasisCount * sizeof(float), GL_DYNAMIC_COPY) &&
           allocateBuffer(smpl_buffers_[kBufTransforms], kJointCount * 16 * sizeof(float), GL_DYNAMIC_COPY) &&
           allocateBuffer(smpl_buffers_[kBufOutput], kVertexCount * 3 * sizeof(float), GL_DYNAMIC_COPY);
}

bool SmplGpuPipeline::uploadMlpBuffers() {
    const char* asset_name = use_fp16_weights_ ? "zdecn_weights_fp16.bin" : "zdecn_weights_fp32.bin";
    const size_t scalar_size = use_fp16_weights_ ? sizeof(uint16_t) : sizeof(float);
    std::vector<uint8_t> raw = readBinaryAsset(assets_, "", asset_name);
    const size_t expected = (64*144 + 64 + 64*64 + 64 + 20670*64 + 20670) * scalar_size;
    if (raw.size() != expected) {
        fail(std::string(asset_name) + " size mismatch: expected " + std::to_string(expected) +
             ", got " + std::to_string(raw.size()));
        return false;
    }
    size_t cursor = 0;
    auto upload_segment = [&](MlpBuffer index, size_t scalar_count) -> bool {
        const size_t bytes = scalar_count * scalar_size;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mlp_buffers_[index]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), raw.data() + cursor, GL_STATIC_DRAW);
        cursor += bytes;
        return checkGlError("upload_mlp_segment");
    };
    if (!allocateBuffer(mlp_buffers_[kMlpInput], kInputSize * sizeof(float), GL_DYNAMIC_DRAW) ||
        !allocateBuffer(mlp_buffers_[kMlpHidden0], kHiddenSize * sizeof(float), GL_DYNAMIC_COPY) ||
        !allocateBuffer(mlp_buffers_[kMlpHidden1], kHiddenSize * sizeof(float), GL_DYNAMIC_COPY) ||
        !upload_segment(kMlpWeights0, 64*144) ||
        !upload_segment(kMlpBias0, 64) ||
        !upload_segment(kMlpWeights1, 64*64) ||
        !upload_segment(kMlpBias1, 64) ||
        !upload_segment(kMlpWeights2, 20670*64) ||
        !upload_segment(kMlpBias2, 20670) ||
        !allocateBuffer(mlp_buffers_[kMlpOutput], kOutputSize * sizeof(float), GL_DYNAMIC_COPY)) {
        return false;
    }
    return cursor == raw.size();
}

bool SmplGpuPipeline::dispatchLbs(const float* axis_angle_72) {
    if (!initialized_) return false;
    const auto start = std::chrono::steady_clock::now();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smpl_buffers_[kBufPose]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kJointCount * 3 * sizeof(float), axis_angle_72);

    const GLuint vertex_groups = static_cast<GLuint>((kVertexCount + kLocalSizeX - 1) / kLocalSizeX);
    glUseProgram(smpl_programs_[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, smpl_buffers_[kBufTemplate]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, smpl_buffers_[kBufShapeDirs]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, smpl_buffers_[kBufBetas]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, smpl_buffers_[kBufShaped]);
    glDispatchCompute(vertex_groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(smpl_programs_[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, smpl_buffers_[kBufShaped]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, smpl_buffers_[kBufJointRegressor]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, smpl_buffers_[kBufJoints]);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(smpl_programs_[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, smpl_buffers_[kBufPose]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, smpl_buffers_[kBufJoints]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, smpl_buffers_[kBufParents]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, smpl_buffers_[kBufPoseFeatures]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, smpl_buffers_[kBufTransforms]);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(smpl_programs_[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, smpl_buffers_[kBufShaped]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, smpl_buffers_[kBufPoseDirs]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, smpl_buffers_[kBufPoseFeatures]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, smpl_buffers_[kBufWeights]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, smpl_buffers_[kBufTransforms]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, smpl_buffers_[kBufOutput]);
    glDispatchCompute(vertex_groups, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    (void) start;
    return checkGlError("dispatchLbs");
}

bool SmplGpuPipeline::dispatchZdecn(const float* rot6d_144) {
    if (!initialized_) return false;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mlp_buffers_[kMlpInput]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kInputSize * sizeof(float), rot6d_144);

    if (timer_supported_ && zdecn_timer_query_ != 0) {
        glBeginQuery(GL_TIME_ELAPSED_EXT, zdecn_timer_query_);
    }

    glUseProgram(mlp_programs_[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mlp_buffers_[kMlpInput]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mlp_buffers_[kMlpHidden0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mlp_buffers_[kMlpWeights0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mlp_buffers_[kMlpBias0]);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(mlp_programs_[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mlp_buffers_[kMlpHidden0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mlp_buffers_[kMlpHidden1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mlp_buffers_[kMlpWeights1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mlp_buffers_[kMlpBias1]);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(mlp_programs_[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mlp_buffers_[kMlpHidden1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mlp_buffers_[kMlpOutput]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mlp_buffers_[kMlpWeights2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mlp_buffers_[kMlpBias2]);
    glDispatchCompute(static_cast<GLuint>((kOutputSize + kLocalSizeX - 1) / kLocalSizeX), 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    if (timer_supported_ && zdecn_timer_query_ != 0) {
        glEndQuery(GL_TIME_ELAPSED_EXT);
        glFinish();
        auto get_query_object_ui64v =
                reinterpret_cast<PFNGLGETQUERYOBJECTUI64VEXTPROC>(
                        eglGetProcAddress("glGetQueryObjectui64vEXT"));
        if (get_query_object_ui64v != nullptr) {
            GLuint64 elapsed_ns = 0;
            get_query_object_ui64v(zdecn_timer_query_, GL_QUERY_RESULT, &elapsed_ns);
            last_zdecn_gpu_ms_ = static_cast<double>(elapsed_ns) / 1.0e6;
            LOGI("zdecn_gpu_timer_ms=%.6f", last_zdecn_gpu_ms_);
        }
    }
    return checkGlError("dispatchZdecn");
}

bool SmplGpuPipeline::uploadExternalDelta(const float* delta_mm_20670) {
    if (!initialized_ || delta_mm_20670 == nullptr) return false;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mlp_buffers_[kMlpOutput]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kOutputSize * sizeof(float), delta_mm_20670);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return checkGlError("uploadExternalDelta");
}

bool SmplGpuPipeline::readCombinedVertices(bool apply_delta,
                                           std::vector<float>* vertices_xyz,
                                           SmplGpuFrameStats* stats) {
    if (!vertices_xyz) return false;
    const auto start = std::chrono::steady_clock::now();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smpl_buffers_[kBufOutput]);
    const void* lbs_mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, kOutputSize * sizeof(float), GL_MAP_READ_BIT);
    if (!lbs_mapped) {
        fail("glMapBufferRange failed for LBS output");
        return false;
    }
    std::memcpy(lbs_readback_.data(), lbs_mapped, kOutputSize * sizeof(float));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (apply_delta) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mlp_buffers_[kMlpOutput]);
        const void* delta_mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, kOutputSize * sizeof(float), GL_MAP_READ_BIT);
        if (!delta_mapped) {
            fail("glMapBufferRange failed for ZD-ECN output");
            return false;
        }
        std::memcpy(delta_readback_.data(), delta_mapped, kOutputSize * sizeof(float));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    vertices_xyz->resize(kOutputSize);
    double max_delta = 0.0;
    for (int i = 0; i < kOutputSize; ++i) {
        const float delta_m = apply_delta ? delta_readback_[i] * 0.001f : 0.0f;
        if (apply_delta) max_delta = std::max(max_delta, static_cast<double>(std::abs(delta_readback_[i])));
        (*vertices_xyz)[i] = lbs_readback_[i] + delta_m;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    const auto end = std::chrono::steady_clock::now();
    if (stats) {
        stats->readback_cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats->max_abs_delta_mm = max_delta;
    }
    return checkGlError("readCombinedVertices");
}

bool SmplGpuPipeline::checkGlError(const char* label) {
    GLenum err = glGetError();
    if (err == GL_NO_ERROR) return true;
    fail(std::string(label) + " GL error 0x" + std::to_string(err));
    return false;
}

void SmplGpuPipeline::fail(const std::string& message) {
    last_error_ = message;
    LOGE("%s", message.c_str());
}

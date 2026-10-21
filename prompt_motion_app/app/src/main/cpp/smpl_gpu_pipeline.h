#pragma once

#include <android/asset_manager.h>
#include <GLES3/gl31.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct SmplGpuFrameStats {
    double lbs_cpu_ms   = 0.0;
    double zdecn_cpu_ms = 0.0;
};

class SmplGpuPipeline {
public:
    bool init(AAssetManager* assets);
    void release();
    bool dispatchLbs(const float* axis_angle_72);
    bool dispatchZdecn(const float* rot6d_144);
    bool uploadExternalDelta(const float* delta_mm_20670);
    bool configureRenderTopology(const uint16_t* indices, size_t index_count);
    bool prepareRenderBuffer(bool apply_delta);
    void setInternalZdecnTimerEnabled(bool enabled) { internal_zdecn_timer_enabled_ = enabled; }
    GLuint renderBuffer() const { return smpl_buffers_[kBufRenderOutput]; }
    GLuint normalBuffer() const { return smpl_buffers_[kBufRenderNormals]; }
    const std::string& error() const { return last_error_; }
    bool   ready() const { return initialized_; }
    double lastZdecnGpuMs() const { return last_zdecn_gpu_ms_; }

private:
    enum SmplBuffer {
        kBufTemplate = 0,
        kBufShapeDirs,
        kBufBetas,
        kBufShaped,
        kBufJointRegressor,
        kBufJoints,
        kBufPose,
        kBufParents,
        kBufPoseFeatures,
        kBufTransforms,
        kBufPoseDirs,
        kBufWeights,
        kBufOutput,
        kBufRenderOutput,
        kBufFaceIndices,
        kBufVertexFaceOffsets,
        kBufVertexFaces,
        kBufRenderNormals,
        kBufCount
    };

    enum MlpBuffer {
        kMlpInput = 0,
        kMlpHidden0,
        kMlpHidden1,
        kMlpWeights0,
        kMlpBias0,
        kMlpWeights1,
        kMlpBias1,
        kMlpWeights2,
        kMlpBias2,
        kMlpOutput,
        kMlpCount
    };

    bool loadPrograms();
    bool createBuffers();
    bool uploadSmplBuffers();
    bool uploadMlpBuffers();
    bool uploadAssetBuffer(GLuint buffer, const char* dir, const char* name,
                           size_t expected_bytes, GLenum usage);
    bool allocateBuffer(GLuint buffer, size_t bytes, GLenum usage);
    GLuint compileComputeAsset(const char* asset_name,
                               const std::vector<std::pair<std::string, std::string>>& defines);
    bool checkGlError(const char* label);
    void fail(const std::string& message);

    AAssetManager* assets_    = nullptr;
    bool initialized_          = false;
    std::string last_error_;
    GLuint smpl_programs_[4]   = {};
    GLuint mlp_programs_[3]    = {};
    GLuint combine_program_    = 0;
    GLuint normals_program_    = 0;
    GLint  combine_apply_delta_location_ = -1;
    bool   render_topology_ready_ = false;
    GLuint smpl_buffers_[kBufCount] = {};
    GLuint mlp_buffers_[kMlpCount]  = {};
    bool   use_fp16_weights_        = true;
    bool   timer_supported_         = false;
    bool   internal_zdecn_timer_enabled_ = true;
    GLuint zdecn_timer_query_       = 0;
    double last_zdecn_gpu_ms_       = 0.0;
};

#include "benchmark_api.h"

#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <GLES3/gl31.h>
#include <GLES3/gl3ext.h>
#include <EGL/egl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32)
#include <direct.h>
#endif

#ifndef GL_TIME_ELAPSED_EXT
#define GL_TIME_ELAPSED_EXT 0x88BF
#endif

#ifndef GL_GPU_DISJOINT_EXT
#define GL_GPU_DISJOINT_EXT 0x8FBB
#endif

#ifndef GL_QUERY_RESULT_AVAILABLE
#define GL_QUERY_RESULT_AVAILABLE 0x8867
#endif

#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif

#ifndef PFNGLGENQUERIESEXTPROC
typedef void (GL_APIENTRYP PFNGLGENQUERIESEXTPROC)(GLsizei, GLuint*);
typedef void (GL_APIENTRYP PFNGLDELETEQUERIESEXTPROC)(GLsizei, const GLuint*);
typedef void (GL_APIENTRYP PFNGLBEGINQUERYEXTPROC)(GLenum, GLuint);
typedef void (GL_APIENTRYP PFNGLENDQUERYEXTPROC)(GLenum);
typedef void (GL_APIENTRYP PFNGLGETQUERYOBJECTUI64VEXTPROC)(GLuint, GLenum, GLuint64*);
typedef void (GL_APIENTRYP PFNGLGETQUERYOBJECTIVEXTPROC)(GLuint, GLenum, GLint*);
#endif

namespace {

constexpr int kDefaultInputSize = 144;
constexpr int kDefaultHiddenSize = 64;
constexpr int kDefaultOutputSize = 20670;
constexpr int kDefaultWarmupIterations = 100;
constexpr int kDefaultBenchmarkIterations = 1000;
constexpr int kLayerCount = 3;
constexpr int kSmplVertexCount = 6890;
constexpr int kSmplJointCount = 24;
constexpr int kSmplNumBetas = 10;
constexpr int kSmplPoseBasisCount = 207;
constexpr int kSmplProgramCount = 4;
constexpr GLuint kInputBinding = 0;
constexpr GLuint kOutputBinding = 1;
constexpr GLuint kWeightsBinding = 2;
constexpr GLuint kBiasBinding = 3;
constexpr int kLocalSizeX = 256;

struct LayerSpec {
    const char* asset_name;
    const char* weight_asset_name;
    const char* bias_asset_name;
    int input_size;
    int output_size;
    bool linear_output;
};

struct StatsAccumulator {
    int count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double min_value = 0.0;
    double max_value = 0.0;

    void add(double value) {
        if (count == 0) {
            min_value = max_value = value;
        } else {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    double stddev() const {
        return count > 1 ? std::sqrt(m2 / static_cast<double>(count - 1)) : 0.0;
    }
};

struct TimerQueryFns {
    PFNGLGENQUERIESEXTPROC gen_queries = nullptr;
    PFNGLDELETEQUERIESEXTPROC delete_queries = nullptr;
    PFNGLBEGINQUERYEXTPROC begin_query = nullptr;
    PFNGLENDQUERYEXTPROC end_query = nullptr;
    PFNGLGETQUERYOBJECTUI64VEXTPROC get_query_object_ui64v = nullptr;
    PFNGLGETQUERYOBJECTIVEXTPROC get_query_object_iv = nullptr;
    bool supported = false;
};

struct LayerRuntime {
    GLuint program = 0;
    GLuint weights_buffer = 0;
    GLuint bias_buffer = 0;
    int input_size = 0;
    int output_size = 0;
    bool linear_output = false;
};

static void log_android(int priority, const char* fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    __android_log_print(priority, "MlpBenchmark", "%s", buffer);
}

static void set_text(char* dst, size_t dst_size, const std::string& value) {
    if (dst_size == 0) {
        return;
    }
    std::snprintf(dst, dst_size, "%s", value.c_str());
}

static void set_text(char* dst, size_t dst_size, const char* value) {
    if (dst_size == 0) {
        return;
    }
    std::snprintf(dst, dst_size, "%s", value != nullptr ? value : "");
}

static std::string current_timestamp_string() {
    std::time_t now = std::time(nullptr);
    std::tm tm_value{};
#if defined(_WIN32)
    localtime_s(&tm_value, &now);
#else
    localtime_r(&now, &tm_value);
#endif
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d %02d:%02d:%02d",
                  tm_value.tm_year + 1900,
                  tm_value.tm_mon + 1,
                  tm_value.tm_mday,
                  tm_value.tm_hour,
                  tm_value.tm_min,
                  tm_value.tm_sec);
    return buffer;
}

static bool ensure_parent_directories(const std::string& file_path) {
    const size_t slash = file_path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return true;
    }
    std::string directory = file_path.substr(0, slash);
    if (directory.empty()) {
        return true;
    }

    std::string partial;
    size_t index = 0;
    if (directory.size() >= 2 && directory[1] == ':') {
        partial = directory.substr(0, 2);
        index = 2;
    } else if (directory[0] == '/' || directory[0] == '\\') {
        partial = "/";
        index = 1;
    }

    while (index < directory.size()) {
        while (index < directory.size() && (directory[index] == '/' || directory[index] == '\\')) {
            ++index;
        }
        const size_t start = index;
        while (index < directory.size() && directory[index] != '/' && directory[index] != '\\') {
            ++index;
        }
        if (start == index) {
            continue;
        }
        const std::string part = directory.substr(start, index - start);
        if (!partial.empty() && partial.back() != '/' && partial.back() != '\\' && partial.back() != ':') {
            partial.push_back('/');
        }
        partial += part;
#if defined(_WIN32)
        if (_mkdir(partial.c_str()) != 0 && errno != EEXIST) {
            return false;
        }
#else
        if (::mkdir(partial.c_str(), 0777) != 0 && errno != EEXIST) {
            return false;
        }
#endif
    }
    return true;
}

static std::string read_file_text(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return {};
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

static std::vector<uint8_t> read_file_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        return {};
    }

    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return {};
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file.good() && !file.eof()) {
        return {};
    }
    if (file.gcount() != size) {
        return {};
    }
    return bytes;
}

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    if (b.empty()) {
        return a;
    }
    if (a.back() == '/' || a.back() == '\\') {
        return a + b;
    }
    return a + '/' + b;
}

static std::string read_text_asset(AAssetManager* asset_manager,
                                   const std::string& shader_dir,
                                   const std::string& asset_name) {
    const std::string asset_path = join_path(shader_dir, asset_name);
    if (asset_manager != nullptr) {
        AAsset* asset = AAssetManager_open(asset_manager, asset_path.c_str(), AASSET_MODE_BUFFER);
        if (asset == nullptr) {
            return {};
        }
        const off_t size = AAsset_getLength(asset);
        std::string text;
        text.resize(static_cast<size_t>(size));
        const int64_t read_count = AAsset_read(asset, text.data(), static_cast<size_t>(size));
        AAsset_close(asset);
        if (read_count <= 0) {
            return {};
        }
        text.resize(static_cast<size_t>(read_count));
        return text;
    }
    return read_file_text(asset_path);
}

static std::vector<uint8_t> read_binary_asset(AAssetManager* asset_manager,
                                              const std::string& shader_dir,
                                              const std::string& asset_name) {
    const std::string asset_path = join_path(shader_dir, asset_name);
    if (asset_manager != nullptr) {
        AAsset* asset = AAssetManager_open(asset_manager, asset_path.c_str(), AASSET_MODE_BUFFER);
        if (asset == nullptr) {
            return {};
        }
        const off_t size = AAsset_getLength(asset);
        if (size <= 0) {
            AAsset_close(asset);
            return {};
        }

        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        const int64_t read_count = AAsset_read(asset, bytes.data(), static_cast<size_t>(size));
        AAsset_close(asset);
        if (read_count <= 0) {
            return {};
        }
        bytes.resize(static_cast<size_t>(read_count));
        return bytes;
    }
    return read_file_binary(asset_path);
}

static std::string gl_string(GLenum name) {
    const GLubyte* value = glGetString(name);
    if (value == nullptr) {
        return {};
    }
    return reinterpret_cast<const char*>(value);
}

static bool parse_es_version(const std::string& version, int* major, int* minor) {
    if (major == nullptr || minor == nullptr) {
        return false;
    }
    *major = 0;
    *minor = 0;
    return std::sscanf(version.c_str(), "OpenGL ES %d.%d", major, minor) == 2;
}

static TimerQueryFns load_timer_query_fns() {
    TimerQueryFns fns;
    const std::string extensions = gl_string(GL_EXTENSIONS);
    if (extensions.find("GL_EXT_disjoint_timer_query") == std::string::npos) {
        return fns;
    }
    fns.gen_queries = reinterpret_cast<PFNGLGENQUERIESEXTPROC>(eglGetProcAddress("glGenQueriesEXT"));
    fns.delete_queries = reinterpret_cast<PFNGLDELETEQUERIESEXTPROC>(eglGetProcAddress("glDeleteQueriesEXT"));
    fns.begin_query = reinterpret_cast<PFNGLBEGINQUERYEXTPROC>(eglGetProcAddress("glBeginQueryEXT"));
    fns.end_query = reinterpret_cast<PFNGLENDQUERYEXTPROC>(eglGetProcAddress("glEndQueryEXT"));
    fns.get_query_object_ui64v =
            reinterpret_cast<PFNGLGETQUERYOBJECTUI64VEXTPROC>(eglGetProcAddress("glGetQueryObjectui64vEXT"));
    fns.get_query_object_iv =
            reinterpret_cast<PFNGLGETQUERYOBJECTIVEXTPROC>(eglGetProcAddress("glGetQueryObjectivEXT"));
    fns.supported = fns.gen_queries != nullptr &&
                    fns.delete_queries != nullptr &&
                    fns.begin_query != nullptr &&
                    fns.end_query != nullptr &&
                    fns.get_query_object_ui64v != nullptr &&
                    fns.get_query_object_iv != nullptr;
    return fns;
}

static std::string prepend_defines(const std::string& source,
                                   const std::vector<std::pair<std::string, std::string>>& defines) {
    std::ostringstream out;
    out << "#version 310 es\n";
    out << "precision highp float;\n";
    out << "precision highp int;\n";
    for (const auto& def : defines) {
        out << "#define " << def.first << ' ' << def.second << '\n';
    }
    out << source;
    return out.str();
}

static std::string shader_error_log(GLuint object, bool is_program) {
    GLint log_length = 0;
    if (is_program) {
        glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
    } else {
        glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
    }
    if (log_length <= 1) {
        return {};
    }
    std::string log;
    log.resize(static_cast<size_t>(log_length));
    GLsizei written = 0;
    if (is_program) {
        glGetProgramInfoLog(object, log_length, &written, log.data());
    } else {
        glGetShaderInfoLog(object, log_length, &written, log.data());
    }
    log.resize(static_cast<size_t>(written));
    return log;
}

static GLuint compile_compute_program(const std::string& source,
                                      const std::vector<std::pair<std::string, std::string>>& defines,
                                      std::string* error_out) {
    const std::string final_source = prepend_defines(source, defines);
    const char* source_ptr = final_source.c_str();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    if (shader == 0) {
        if (error_out != nullptr) {
            *error_out = "glCreateShader failed";
        }
        return 0;
    }

    glShaderSource(shader, 1, &source_ptr, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
        const std::string log = shader_error_log(shader, false);
        if (error_out != nullptr) {
            *error_out = "Shader compile failed: " + log;
        }
        glDeleteShader(shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program == 0) {
        if (error_out != nullptr) {
            *error_out = "glCreateProgram failed";
        }
        glDeleteShader(shader);
        return 0;
    }

    glAttachShader(program, shader);
    glLinkProgram(program);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        const std::string log = shader_error_log(program, true);
        if (error_out != nullptr) {
            *error_out = "Program link failed: " + log;
        }
        glDeleteProgram(program);
        glDeleteShader(shader);
        return 0;
    }

    glDetachShader(program, shader);
    glDeleteShader(shader);
    return program;
}

static GLuint compile_graphics_program(const char* vertex_source,
                                       const char* fragment_source,
                                       std::string* error_out) {
    auto compile_shader = [&](GLenum type, const char* source) -> GLuint {
        GLuint shader = glCreateShader(type);
        if (shader == 0) {
            if (error_out != nullptr) {
                *error_out = "glCreateShader failed";
            }
            return 0;
        }
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint compiled = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != GL_TRUE) {
            const std::string log = shader_error_log(shader, false);
            if (error_out != nullptr) {
                *error_out = "Graphics shader compile failed: " + log;
            }
            glDeleteShader(shader);
            return 0;
        }
        return shader;
    };

    GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source);
    if (vertex_shader == 0) {
        return 0;
    }
    GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (fragment_shader == 0) {
        glDeleteShader(vertex_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program == 0) {
        if (error_out != nullptr) {
            *error_out = "glCreateProgram failed";
        }
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return 0;
    }

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
        const std::string log = shader_error_log(program, true);
        if (error_out != nullptr) {
            *error_out = "Graphics program link failed: " + log;
        }
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}

static void fill_random(std::vector<float>& values, float min_value, float max_value, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    for (float& value : values) {
        value = dist(rng);
    }
}

static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        return sign;
    }
    if (exp >= 31) {
        return sign | 0x7C00u;
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

static std::string default_summary_path(const std::string& sample_path) {
    if (sample_path.empty()) {
        return {};
    }
    return sample_path + ".summary.txt";
}

class BenchmarkCore {
public:
    BenchmarkCore() = default;
    ~BenchmarkCore() { destroy(); }

    bool init(AAssetManager* asset_manager, const MlpBenchmarkConfig& config) {
        destroy();

        asset_manager_ = asset_manager;
        config_ = config;
        if (config_.input_size <= 0) {
            config_.input_size = kDefaultInputSize;
        }
        if (config_.hidden_size <= 0) {
            config_.hidden_size = kDefaultHiddenSize;
        }
        if (config_.output_size <= 0) {
            config_.output_size = kDefaultOutputSize;
        }
        if (config_.warmup_iterations < 0) {
            config_.warmup_iterations = kDefaultWarmupIterations;
        }
        if (config_.benchmark_iterations <= 0) {
            config_.benchmark_iterations = kDefaultBenchmarkIterations;
        }
        if (config_.input_size != kDefaultInputSize ||
            config_.hidden_size != kDefaultHiddenSize ||
            config_.output_size != kDefaultOutputSize) {
            fail("This benchmark is fixed to 144 -> 64 -> 64 -> 20670.");
            return false;
        }
        if (config_.shader_asset_dir == nullptr || config_.shader_asset_dir[0] == '\0') {
            shader_asset_dir_ = "shaders";
        } else {
            shader_asset_dir_ = config_.shader_asset_dir;
        }
        csv_path_ = config_.csv_path != nullptr ? config_.csv_path : "";
        log_path_ = config_.log_path != nullptr ? config_.log_path : default_summary_path(csv_path_);
        last_error_.clear();
        reports_written_ = false;
        validation_mode_ = config_.validate_outputs != 0;
        validation_scalar_compare_ = config_.validate_outputs == 2;
        validation_samples_ = 0;
        validation_index_ = 0;
        validation_max_abs_ = 0.0;
        validation_mean_abs_ = 0.0;
        validation_abs_sum_ = 0.0;
        validation_value_count_ = 0;
        validation_tail_max_abs_ = 0.0;
        validation_tail_mean_abs_ = 0.0;
        validation_tail_abs_sum_ = 0.0;
        validation_tail_value_count_ = 0;
        validation_inputs_.clear();
        validation_outputs_.clear();
        validation_vector_output_.clear();
        cpu_stats_ = {};
        mlp_cpu_stats_ = {};
        gpu_stats_ = {};
        cpu_samples_.clear();
        gpu_samples_.clear();
        realtime_started_ = false;
        realtime_frame_count_ = 0;
        status_ = {};
        status_.input_size = config_.input_size;
        status_.hidden_size = config_.hidden_size;
        status_.output_size = config_.output_size;
        status_.warmup_iterations = config_.warmup_iterations;
        status_.benchmark_iterations = config_.benchmark_iterations;
        status_.dispatches_per_frame = kLayerCount;
        status_.use_fp16_weights = config_.use_fp16_weights;
        set_text(status_.phase, sizeof(status_.phase), "init");
        set_text(status_.last_error, sizeof(status_.last_error), "");

        const std::string version = gl_string(GL_VERSION);
        const std::string renderer = gl_string(GL_RENDERER);
        const std::string vendor = gl_string(GL_VENDOR);
        set_text(status_.renderer, sizeof(status_.renderer), renderer.empty() ? vendor : renderer);
        set_text(status_.version, sizeof(status_.version), version);
        if (version.empty()) {
            fail("Unable to query GL version");
            return false;
        }

        int major = 0;
        int minor = 0;
        if (!parse_es_version(version, &major, &minor) || major < 3 || (major == 3 && minor < 1)) {
            fail("OpenGL ES 3.1+ required");
            return false;
        }

        timer_query_fns_ = load_timer_query_fns();
        status_.gpu_timer_supported = timer_query_fns_.supported ? 1 : 0;
        if (timer_query_fns_.supported) {
            timer_query_fns_.gen_queries(1, &gpu_query_);
            if (gpu_query_ == 0) {
                timer_query_fns_ = {};
                status_.gpu_timer_supported = 0;
            }
        }

        if (!load_programs()) {
            return false;
        }
        if (!create_buffers()) {
            return false;
        }
        if (!upload_model_data()) {
            return false;
        }
        if (validation_mode_) {
            if (!load_validation_fixture()) {
                return false;
            }
            status_.warmup_iterations = 0;
            status_.benchmark_iterations = validation_samples_;
            set_text(status_.phase, sizeof(status_.phase), "validation");
        }

        status_.initialized = 1;
        status_.finished = 0;
        if (!validation_mode_) {
            set_text(status_.phase, sizeof(status_.phase), "warmup");
        }
        log_android(ANDROID_LOG_INFO,
                    "Initialized benchmark: renderer=%s, vendor=%s, version=%s, input=%d, hidden=%d, output=%d, warmup=%d, bench=%d, fp16_weights=%s, gpu_timer=%s",
                    status_.renderer,
                    vendor.c_str(),
                    status_.version,
                    status_.input_size,
                    status_.hidden_size,
                    status_.output_size,
                    status_.warmup_iterations,
                    status_.benchmark_iterations,
                    status_.use_fp16_weights ? "yes" : "no",
                    status_.gpu_timer_supported ? "yes" : "no");
        return true;
    }

    void resize(int width, int height) {
        if (width > 0 && height > 0) {
            glViewport(0, 0, width, height);
        }
    }

    bool draw() {
        if (!status_.initialized) {
            return false;
        }
        if (status_.finished) {
            if (!dispatch_frame(false)) {
                return false;
            }
            clear_frame();
            update_realtime_fps();
            return true;
        }

        const int validation_sample_index = validation_mode_ ? validation_index_ : -1;
        if (validation_mode_ && !upload_validation_input(validation_sample_index)) {
            return false;
        }

        // === MLP-only CPU timer: 3x glDispatchCompute + barriers + glFinish ===
        // Wraps ONLY the 3x glDispatchCompute + glMemoryBarrier calls, then glFinish.
        // This is the wall-clock timer used for runtime comparison with ShaderNN/ORT.
        const auto mlp_cpu_start = std::chrono::steady_clock::now();
        if (timer_query_fns_.supported) {
            timer_query_fns_.begin_query(GL_TIME_ELAPSED_EXT, gpu_query_);
        }

        if (!dispatch_frame(true)) {
            return false;
        }

        if (timer_query_fns_.supported) {
            timer_query_fns_.end_query(GL_TIME_ELAPSED_EXT);
        }
        glFinish();

        const auto mlp_cpu_end = std::chrono::steady_clock::now();
        const double mlp_cpu_ms = std::chrono::duration<double, std::milli>(mlp_cpu_end - mlp_cpu_start).count();
        status_.mlp_cpu_last_ms = mlp_cpu_ms;

        if (validation_mode_ &&
            !(validation_scalar_compare_
                      ? compare_scalar_validation_output(validation_sample_index)
                      : compare_validation_output(validation_sample_index))) {
            return false;
        }

        clear_frame();
        const double cpu_ms = mlp_cpu_ms;
        status_.cpu_last_ms = cpu_ms;

        const int total_seen = status_.warmup_completed + status_.benchmark_completed;
        if (total_seen < status_.warmup_iterations) {
            ++status_.warmup_completed;
            set_text(status_.phase, sizeof(status_.phase), "warmup");
        } else if (status_.benchmark_completed < status_.benchmark_iterations) {
            if (status_.benchmark_completed == 0) {
                benchmark_measurement_start_ = std::chrono::steady_clock::now();
            }
            ++status_.benchmark_completed;
            cpu_stats_.add(cpu_ms);
            cpu_samples_.push_back(cpu_ms);
            mlp_cpu_stats_.add(mlp_cpu_ms);
            mlp_cpu_samples_.push_back(mlp_cpu_ms);
            if (timer_query_fns_.supported) {
                record_gpu_sample();
            } else {
                gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            }
            update_stats_fields();
            set_text(status_.phase, sizeof(status_.phase), validation_mode_ ? "validation" : "benchmark");
            if (validation_mode_) {
                ++validation_index_;
            }
            if ((status_.benchmark_completed % 100) == 0 ||
                status_.benchmark_completed == status_.benchmark_iterations) {
                if (!csv_path_.empty()) {
                    write_csv_report();
                }
                if (!log_path_.empty()) {
                    write_log_report();
                }
            }

            const auto now = std::chrono::steady_clock::now();
            status_.elapsed_seconds =
                    std::chrono::duration<double>(now - benchmark_measurement_start_).count();
            if (status_.elapsed_seconds > 0.0) {
                status_.fps = static_cast<double>(status_.benchmark_completed) / status_.elapsed_seconds;
            }
        }

        if (status_.benchmark_completed >= status_.benchmark_iterations &&
            status_.benchmark_iterations > 0) {
            status_.finished = 1;
            set_text(status_.phase, sizeof(status_.phase), "realtime");
            const auto now = std::chrono::steady_clock::now();
            status_.elapsed_seconds =
                    std::chrono::duration<double>(now - benchmark_measurement_start_).count();
            if (status_.elapsed_seconds > 0.0) {
                status_.fps = static_cast<double>(status_.benchmark_completed) / status_.elapsed_seconds;
            }
            update_stats_fields();
            write_reports_if_needed();
        }

        return true;
    }

    bool run_fps_loop(int frames, double target_fps) {
        if (frames < 0) {
            return false;
        }
        const bool pace = target_fps > 0.0;
        const auto frame_step = std::chrono::duration<double>(pace ? (1.0 / target_fps) : 0.0);
        auto next_tick = std::chrono::steady_clock::now();
        for (int i = 0; i < frames; ++i) {
            if (!draw()) {
                return false;
            }
            if (status_.finished) {
                break;
            }
            if (pace) {
                next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_step);
                std::this_thread::sleep_until(next_tick);
            }
        }
        return true;
    }

    void destroy() {
        release_gl();
        asset_manager_ = nullptr;
        shader_asset_dir_.clear();
        csv_path_.clear();
        log_path_.clear();
        last_error_.clear();
        cpu_samples_.clear();
        mlp_cpu_samples_.clear();
        gpu_samples_.clear();
        status_ = {};
        cpu_stats_ = {};
        gpu_stats_ = {};
        reports_written_ = false;
        realtime_started_ = false;
        realtime_frame_count_ = 0;
    }

    const MlpBenchmarkStatus& status() const {
        return status_;
    }

    bool set_output_paths(const char* csv_path, const char* log_path) {
        csv_path_ = csv_path != nullptr ? csv_path : "";
        log_path_ = log_path != nullptr ? log_path : default_summary_path(csv_path_);
        if (status_.finished && !reports_written_) {
            write_reports_if_needed();
        }
        return true;
    }

private:
    bool load_programs() {
        const std::array<LayerSpec, kLayerCount> specs = {{
                {"mlp_144_64.comp", "weights_l0.bin", "biases_l0.bin", kDefaultInputSize, kDefaultHiddenSize, false},
                {"mlp_64_64.comp", "weights_l1.bin", "biases_l1.bin", kDefaultHiddenSize, kDefaultHiddenSize, false},
                {"mlp_64_out.comp", "weights_l2.bin", "biases_l2.bin", kDefaultHiddenSize, kDefaultOutputSize, true},
        }};

        for (size_t i = 0; i < specs.size(); ++i) {
            const std::string shader_source =
                    read_text_asset(asset_manager_, shader_asset_dir_, specs[i].asset_name);
            if (shader_source.empty()) {
                fail("Unable to load shader asset: " + join_path(shader_asset_dir_, specs[i].asset_name));
                return false;
            }

            std::vector<std::pair<std::string, std::string>> defines;
            if (config_.use_fp16_weights) {
                defines.emplace_back("USE_FP16_WEIGHTS", "1");
                defines.emplace_back("USE_FP16_BIASES", "1");
            }
            defines.emplace_back("INPUT_SIZE", std::to_string(specs[i].input_size));
            defines.emplace_back("OUTPUT_SIZE", std::to_string(specs[i].output_size));
            if (specs[i].linear_output) {
                defines.emplace_back("OUTPUT_LINEAR", "1");
            }

            std::string error;
            programs_[i] = compile_compute_program(shader_source, defines, &error);
            if (programs_[i] == 0) {
                fail(error);
                return false;
            }
            if (validation_scalar_compare_) {
                std::string scalar_asset_name = specs[i].asset_name;
                const size_t extension = scalar_asset_name.rfind(".comp");
                if (extension == std::string::npos) {
                    fail("Unexpected shader asset name without .comp extension: " + scalar_asset_name);
                    return false;
                }
                scalar_asset_name.replace(extension, 5, "_scalar.comp");
                const std::string scalar_source =
                        read_text_asset(asset_manager_, shader_asset_dir_, scalar_asset_name);
                if (scalar_source.empty()) {
                    fail("Unable to load scalar shader asset: " + join_path(shader_asset_dir_, scalar_asset_name));
                    return false;
                }
                scalar_programs_[i] = compile_compute_program(scalar_source, defines, &error);
                if (scalar_programs_[i] == 0) {
                    fail(error);
                    return false;
                }
            }
            layer_specs_[i] = specs[i];
        }
        return true;
    }

    bool create_buffers() {
        glGenBuffers(static_cast<GLsizei>(activation_buffers_.size()), activation_buffers_.data());
        for (GLuint buffer : activation_buffers_) {
            if (buffer == 0) {
                fail("Failed to create activation SSBOs");
                return false;
            }
        }

        for (size_t i = 0; i < kLayerCount; ++i) {
            glGenBuffers(1, &layers_[i].weights_buffer);
            glGenBuffers(1, &layers_[i].bias_buffer);
            if (layers_[i].weights_buffer == 0 || layers_[i].bias_buffer == 0) {
                fail("Failed to create weight/bias SSBOs");
                return false;
            }
        }
        return true;
    }

    bool upload_model_data() {
        std::mt19937 rng(config_.seed);
        const std::array<float, kLayerCount> fan_in = {
                static_cast<float>(kDefaultInputSize),
                static_cast<float>(kDefaultHiddenSize),
                static_cast<float>(kDefaultHiddenSize),
        };
        const std::array<float, kLayerCount> fan_out = {
                static_cast<float>(kDefaultHiddenSize),
                static_cast<float>(kDefaultHiddenSize),
                static_cast<float>(kDefaultOutputSize),
        };

        for (size_t i = 0; i < kLayerCount; ++i) {
            layers_[i].input_size = layer_specs_[i].input_size;
            layers_[i].output_size = layer_specs_[i].output_size;
            layers_[i].linear_output = layer_specs_[i].linear_output;

            const size_t weight_count = static_cast<size_t>(layers_[i].input_size) *
                                        static_cast<size_t>(layers_[i].output_size);
            const size_t bias_count = static_cast<size_t>(layers_[i].output_size);
            const size_t expected_weight_bytes =
                    weight_count * (config_.use_fp16_weights ? sizeof(uint16_t) : sizeof(float));
            const size_t expected_bias_bytes =
                    bias_count * (config_.use_fp16_weights ? sizeof(uint16_t) : sizeof(float));

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, layers_[i].weights_buffer);
            if (config_.load_weights_from_assets != 0) {
                const std::string weight_asset_path =
                        join_path(shader_asset_dir_, layer_specs_[i].weight_asset_name);
                std::vector<uint8_t> weight_bytes = read_binary_asset(
                        asset_manager_, shader_asset_dir_, layer_specs_[i].weight_asset_name);
                if (weight_bytes.size() != expected_weight_bytes) {
                    fail("Weight asset size mismatch for " + weight_asset_path + ": expected " +
                         std::to_string(expected_weight_bytes) + " bytes, got " +
                         std::to_string(weight_bytes.size()));
                    return false;
                }
                glBufferData(GL_SHADER_STORAGE_BUFFER,
                             static_cast<GLsizeiptr>(weight_bytes.size()),
                             weight_bytes.data(),
                             GL_STATIC_DRAW);
            } else {
                std::vector<float> weights(weight_count);
                const float scale = std::sqrt(6.0f / (fan_in[i] + fan_out[i]));
                fill_random(weights, -scale, scale, rng);
                if (config_.use_fp16_weights) {
                    std::vector<uint16_t> packed_weights(weight_count);
                    for (size_t j = 0; j < weight_count; ++j) {
                        packed_weights[j] = float_to_half(weights[j]);
                    }
                    glBufferData(GL_SHADER_STORAGE_BUFFER,
                                 static_cast<GLsizeiptr>(packed_weights.size() * sizeof(uint16_t)),
                                 packed_weights.data(),
                                 GL_STATIC_DRAW);
                } else {
                    glBufferData(GL_SHADER_STORAGE_BUFFER,
                                 static_cast<GLsizeiptr>(weights.size() * sizeof(float)),
                                 weights.data(),
                                 GL_STATIC_DRAW);
                }
            }

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, layers_[i].bias_buffer);
            if (config_.load_weights_from_assets != 0) {
                const std::string bias_asset_path =
                        join_path(shader_asset_dir_, layer_specs_[i].bias_asset_name);
                std::vector<uint8_t> bias_bytes = read_binary_asset(
                        asset_manager_, shader_asset_dir_, layer_specs_[i].bias_asset_name);
                if (bias_bytes.size() != expected_bias_bytes) {
                    fail("Bias asset size mismatch for " + bias_asset_path + ": expected " +
                         std::to_string(expected_bias_bytes) + " bytes, got " +
                         std::to_string(bias_bytes.size()));
                    return false;
                }
                glBufferData(GL_SHADER_STORAGE_BUFFER,
                             static_cast<GLsizeiptr>(bias_bytes.size()),
                             bias_bytes.data(),
                             GL_STATIC_DRAW);
            } else {
                std::vector<float> biases(bias_count);
                fill_random(biases, -0.05f, 0.05f, rng);
                if (config_.use_fp16_weights) {
                    std::vector<uint16_t> packed_biases(bias_count);
                    for (size_t j = 0; j < bias_count; ++j) {
                        packed_biases[j] = float_to_half(biases[j]);
                    }
                    glBufferData(GL_SHADER_STORAGE_BUFFER,
                                 static_cast<GLsizeiptr>(packed_biases.size() * sizeof(uint16_t)),
                                 packed_biases.data(),
                                 GL_STATIC_DRAW);
                } else {
                    glBufferData(GL_SHADER_STORAGE_BUFFER,
                                 static_cast<GLsizeiptr>(biases.size() * sizeof(float)),
                                 biases.data(),
                                 GL_STATIC_DRAW);
                }
            }
        }

        activation_a_.assign(static_cast<size_t>(kDefaultOutputSize), 0.0f);
        activation_b_.assign(static_cast<size_t>(kDefaultOutputSize), 0.0f);
        const std::vector<uint8_t> input_bytes =
                read_binary_asset(asset_manager_, shader_asset_dir_, "input_rot6d.bin");
        if (!input_bytes.empty()) {
            const size_t expected_input_bytes = static_cast<size_t>(kDefaultInputSize) * sizeof(float);
            if (input_bytes.size() != expected_input_bytes) {
                fail("Input asset size mismatch for " + join_path(shader_asset_dir_, "input_rot6d.bin") +
                     ": expected " + std::to_string(expected_input_bytes) + " bytes, got " +
                     std::to_string(input_bytes.size()));
                return false;
            }
            std::memcpy(activation_a_.data(), input_bytes.data(), input_bytes.size());
        } else {
            std::mt19937 input_rng(config_.seed ^ 0x9E3779B9u);
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            for (int i = 0; i < kDefaultInputSize; ++i) {
                activation_a_[static_cast<size_t>(i)] = input_dist(input_rng);
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, activation_buffers_[0]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(activation_a_.size() * sizeof(float)),
                     activation_a_.data(),
                     GL_DYNAMIC_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, activation_buffers_[1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(activation_b_.size() * sizeof(float)),
                     activation_b_.data(),
                     GL_DYNAMIC_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return true;
    }

    bool dispatch_frame(bool measure, const std::array<GLuint, kLayerCount>* program_set = nullptr) {
        const auto& active_programs = program_set != nullptr ? *program_set : programs_;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, activation_buffers_[0]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(config_.input_size * static_cast<int>(sizeof(float))),
                        activation_a_.data());

        GLuint input_buffer = activation_buffers_[0];
        GLuint output_buffer = activation_buffers_[1];
        for (int layer = 0; layer < kLayerCount; ++layer) {
            glUseProgram(active_programs[layer]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kInputBinding, input_buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kOutputBinding, output_buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kWeightsBinding, layers_[layer].weights_buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kBiasBinding, layers_[layer].bias_buffer);

            const GLuint groups = static_cast<GLuint>((layers_[layer].output_size + kLocalSizeX - 1) / kLocalSizeX);
            glDispatchCompute(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            std::swap(input_buffer, output_buffer);
        }
        final_output_buffer_ = input_buffer;
        if (measure) {
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        }
        return check_gl_error("dispatch_frame");
    }

    bool load_validation_fixture() {
        auto load_float_asset = [&](const char* name, std::vector<float>* out) -> bool {
            std::vector<uint8_t> bytes = read_binary_asset(asset_manager_, "zdecn_validation_fixture", name);
            if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
                fail(std::string("Invalid ZD-ECN validation asset: zdecn_validation_fixture/") + name);
                return false;
            }
            out->resize(bytes.size() / sizeof(float));
            std::memcpy(out->data(), bytes.data(), bytes.size());
            return true;
        };
        if (!load_float_asset("inputs_rot6d_f32.bin", &validation_inputs_) ||
            !load_float_asset("outputs_ref_f32.bin", &validation_outputs_)) {
            return false;
        }
        if ((validation_inputs_.size() % kDefaultInputSize) != 0 ||
            (validation_outputs_.size() % kDefaultOutputSize) != 0) {
            fail("ZD-ECN validation fixture shape mismatch");
            return false;
        }
        const int input_samples = static_cast<int>(validation_inputs_.size() / kDefaultInputSize);
        const int output_samples = static_cast<int>(validation_outputs_.size() / kDefaultOutputSize);
        validation_samples_ = std::min(input_samples, output_samples);
        if (validation_samples_ <= 0) {
            fail("ZD-ECN validation fixture has no samples");
            return false;
        }
        return true;
    }

    bool upload_validation_input(int sample_index) {
        if (sample_index < 0 || sample_index >= validation_samples_) {
            fail("ZD-ECN validation sample index out of range");
            return false;
        }
        std::memcpy(activation_a_.data(),
                    validation_inputs_.data() + static_cast<size_t>(sample_index) * kDefaultInputSize,
                    kDefaultInputSize * sizeof(float));
        return true;
    }

    bool compare_validation_output(int sample_index) {
        const size_t value_count = static_cast<size_t>(kDefaultOutputSize);
        const float* expected = validation_outputs_.data() + static_cast<size_t>(sample_index) * value_count;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, final_output_buffer_);
        const void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                              0,
                                              static_cast<GLsizeiptr>(value_count * sizeof(float)),
                                              GL_MAP_READ_BIT);
        if (mapped == nullptr) {
            fail("glMapBufferRange failed for ZD-ECN validation output");
            return false;
        }
        const float* actual = static_cast<const float*>(mapped);
        double sample_sum = 0.0;
        double sample_max = 0.0;
        double sample_tail_sum = 0.0;
        double sample_tail_max = 0.0;
        for (size_t i = 0; i < value_count; ++i) {
            const double abs_error = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
            sample_sum += abs_error;
            sample_max = std::max(sample_max, abs_error);
            if (i >= 20664u && i <= 20669u) {
                sample_tail_sum += abs_error;
                sample_tail_max = std::max(sample_tail_max, abs_error);
            }
        }
        if (sample_index == 0) {
            log_android(ANDROID_LOG_INFO,
                        "ZD-ECN validation sample0 raw actual_first10=[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g] expected_first10=[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",
                        actual[0], actual[1], actual[2], actual[3], actual[4],
                        actual[5], actual[6], actual[7], actual[8], actual[9],
                        expected[0], expected[1], expected[2], expected[3], expected[4],
                        expected[5], expected[6], expected[7], expected[8], expected[9]);
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (!check_gl_error("compare_zdecn_validation_output")) {
            return false;
        }
        validation_abs_sum_ += sample_sum;
        validation_value_count_ += value_count;
        validation_max_abs_ = std::max(validation_max_abs_, sample_max);
        validation_mean_abs_ = validation_abs_sum_ / static_cast<double>(validation_value_count_);
        validation_tail_abs_sum_ += sample_tail_sum;
        validation_tail_value_count_ += 6u;
        validation_tail_max_abs_ = std::max(validation_tail_max_abs_, sample_tail_max);
        validation_tail_mean_abs_ = validation_tail_abs_sum_ / static_cast<double>(validation_tail_value_count_);
        log_android(ANDROID_LOG_INFO,
                    "ZD-ECN validation sample=%d/%d max_abs=%.9g mean_abs=%.9g tail_max_abs=%.9g tail_mean_abs=%.9g",
                    sample_index + 1,
                    validation_samples_,
                    sample_max,
                    sample_sum / static_cast<double>(value_count),
                    sample_tail_max,
                    sample_tail_sum / 6.0);
        return true;
    }

    bool compare_scalar_validation_output(int sample_index) {
        const size_t value_count = static_cast<size_t>(kDefaultOutputSize);
        validation_vector_output_.resize(value_count);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, final_output_buffer_);
        const void* vector_mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                                     0,
                                                     static_cast<GLsizeiptr>(value_count * sizeof(float)),
                                                     GL_MAP_READ_BIT);
        if (vector_mapped == nullptr) {
            fail("glMapBufferRange failed for ZD-ECN vector output");
            return false;
        }
        std::memcpy(validation_vector_output_.data(), vector_mapped, value_count * sizeof(float));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        if (!dispatch_frame(false, &scalar_programs_)) {
            return false;
        }
        glFinish();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, final_output_buffer_);
        const void* scalar_mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                                     0,
                                                     static_cast<GLsizeiptr>(value_count * sizeof(float)),
                                                     GL_MAP_READ_BIT);
        if (scalar_mapped == nullptr) {
            fail("glMapBufferRange failed for ZD-ECN scalar output");
            return false;
        }
        const float* scalar_output = static_cast<const float*>(scalar_mapped);
        double sample_sum = 0.0;
        double sample_max = 0.0;
        for (size_t i = 0; i < value_count; ++i) {
            const double abs_error = std::abs(
                    static_cast<double>(validation_vector_output_[i]) -
                    static_cast<double>(scalar_output[i]));
            sample_sum += abs_error;
            sample_max = std::max(sample_max, abs_error);
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (!check_gl_error("compare_zdecn_scalar_validation_output")) {
            return false;
        }

        validation_abs_sum_ += sample_sum;
        validation_value_count_ += value_count;
        validation_max_abs_ = std::max(validation_max_abs_, sample_max);
        validation_mean_abs_ = validation_abs_sum_ / static_cast<double>(validation_value_count_);
        log_android(ANDROID_LOG_INFO,
                    "ZD-ECN scalar compare sample=%d/%d full_vector_max_abs=%.9g full_vector_mean_abs=%.9g running_max_abs=%.9g running_mean_abs=%.9g",
                    sample_index + 1,
                    validation_samples_,
                    sample_max,
                    sample_sum / static_cast<double>(value_count),
                    validation_max_abs_,
                    validation_mean_abs_);
        return true;
    }

    void record_gpu_sample() {
        if (!timer_query_fns_.supported) {
            return;
        }
        GLint available = GL_FALSE;
        timer_query_fns_.get_query_object_iv(gpu_query_, GL_QUERY_RESULT_AVAILABLE, &available);
        if (available != GL_TRUE) {
            gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            return;
        }
        GLuint64 elapsed_ns = 0;
        timer_query_fns_.get_query_object_ui64v(gpu_query_, GL_QUERY_RESULT, &elapsed_ns);
        GLint disjoint = GL_FALSE;
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
        if (disjoint == GL_TRUE) {
            gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            status_.gpu_timer_valid = 0;
            return;
        }
        const double elapsed_ms = static_cast<double>(elapsed_ns) / 1.0e6;
        gpu_stats_.add(elapsed_ms);
        gpu_samples_.push_back(elapsed_ms);
        status_.gpu_timer_valid = 1;
        status_.gpu_last_ms = elapsed_ms;
    }

    void clear_frame() {
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.03f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void update_realtime_fps() {
        const auto now = std::chrono::steady_clock::now();
        if (!realtime_started_) {
            realtime_started_ = true;
            realtime_frame_start_ = now;
            realtime_frame_count_ = 0;
        }

        ++realtime_frame_count_;
        const double seconds = std::chrono::duration<double>(now - realtime_frame_start_).count();
        if (seconds >= 1.0) {
            status_.fps = static_cast<double>(realtime_frame_count_) / seconds;
            log_android(ANDROID_LOG_INFO, "Realtime FPS: %.2f", status_.fps);
            realtime_frame_count_ = 0;
            realtime_frame_start_ = now;
        }
    }

    void update_stats_fields() {
        // MLP-only dispatch timer (apples-to-apples with ShaderNN/ORT)
        status_.mlp_cpu_mean_ms = mlp_cpu_stats_.mean;
        status_.mlp_cpu_stddev_ms = mlp_cpu_stats_.stddev();
        status_.cpu_mean_ms = cpu_stats_.mean;
        status_.cpu_stddev_ms = cpu_stats_.stddev();
        status_.cpu_min_ms = cpu_stats_.count > 0 ? cpu_stats_.min_value : 0.0;
        status_.cpu_max_ms = cpu_stats_.count > 0 ? cpu_stats_.max_value : 0.0;
        if (gpu_stats_.count > 0) {
            status_.gpu_timer_valid = 1;
            status_.gpu_mean_ms = gpu_stats_.mean;
            status_.gpu_stddev_ms = gpu_stats_.stddev();
            status_.gpu_min_ms = gpu_stats_.count > 0 ? gpu_stats_.min_value : 0.0;
            status_.gpu_max_ms = gpu_stats_.count > 0 ? gpu_stats_.max_value : 0.0;
        }
    }

    void write_reports_if_needed() {
        if (reports_written_) {
            return;
        }
        const bool csv_ok = csv_path_.empty() || write_csv_report();
        const bool log_ok = log_path_.empty() || write_log_report();
        reports_written_ = csv_ok && log_ok;
        if (!reports_written_) {
            log_android(ANDROID_LOG_ERROR, "Failed to write benchmark reports: %s", last_error_.c_str());
        }
    }

    bool write_csv_report() {
        if (!ensure_parent_directories(csv_path_)) {
            last_error_ = "Failed to create CSV parent directories";
            set_text(status_.last_error, sizeof(status_.last_error), last_error_);
            return false;
        }
        std::ofstream out(csv_path_, std::ios::binary | std::ios::trunc);
        if (!out.good()) {
            last_error_ = "Unable to open CSV path: " + csv_path_;
            set_text(status_.last_error, sizeof(status_.last_error), last_error_);
            return false;
        }
        // mlp_cpu_ms = dispatch+glFinish only (comparable to ShaderNN/ONNX wall-clock)
        // cpu_ms     = same boundary retained for backward-compatible parsers
        // gpu_ms     = GL_EXT_disjoint_timer_query on dispatch only (pure GPU kernel time)
        out << "run,mlp_cpu_ms,cpu_ms,gpu_ms\n";
        for (size_t i = 0; i < cpu_samples_.size(); ++i) {
            const double mlp_ms = i < mlp_cpu_samples_.size() ? mlp_cpu_samples_[i] : std::numeric_limits<double>::quiet_NaN();
            out << (i + 1) << ',';
            if (!std::isnan(mlp_ms)) {
                out << mlp_ms;
            } else {
                out << "NA";
            }
            out << ',' << cpu_samples_[i] << ',';
            if (i < gpu_samples_.size() && !std::isnan(gpu_samples_[i])) {
                out << gpu_samples_[i];
            } else {
                out << "NA";
            }
            out << '\n';
        }
        return out.good();
    }

    bool write_log_report() {
        if (!ensure_parent_directories(log_path_)) {
            last_error_ = "Failed to create log parent directories";
            set_text(status_.last_error, sizeof(status_.last_error), last_error_);
            return false;
        }
        std::ofstream out(log_path_, std::ios::binary | std::ios::trunc);
        if (!out.good()) {
            last_error_ = "Unable to open log path: " + log_path_;
            set_text(status_.last_error, sizeof(status_.last_error), last_error_);
            return false;
        }
        out << "===== Zero-Dependency Compute Shader Benchmark =====\n";
        out << "Timestamp: " << current_timestamp_string() << '\n';
        out << "Measured CPU path: 3x glDispatchCompute + glMemoryBarrier + glFinish\n";
        out << "Renderer: " << status_.renderer << '\n';
        out << "Version: " << status_.version << '\n';
        out << "Input size: " << config_.input_size << '\n';
        out << "Hidden size: " << config_.hidden_size << '\n';
        out << "Output size: " << config_.output_size << '\n';
        out << "Warmup iterations: " << config_.warmup_iterations << '\n';
        out << "Benchmark iterations: " << config_.benchmark_iterations << '\n';
        out << "CPU last/mean/stddev/min/max ms: "
            << status_.cpu_last_ms << " / " << status_.cpu_mean_ms << " / " << status_.cpu_stddev_ms
            << " / " << status_.cpu_min_ms << " / " << status_.cpu_max_ms << '\n';
        out << "GPU timer supported: " << status_.gpu_timer_supported
            << " valid: " << status_.gpu_timer_valid << '\n';
        out << "GPU last/mean/stddev/min/max ms: "
            << status_.gpu_last_ms << " / " << status_.gpu_mean_ms << " / " << status_.gpu_stddev_ms
            << " / " << status_.gpu_min_ms << " / " << status_.gpu_max_ms << '\n';
        out << "Elapsed seconds: " << status_.elapsed_seconds << '\n';
        out << "FPS: " << status_.fps << '\n';
        out << "Phase: " << status_.phase << '\n';
        if (validation_mode_) {
            out << "Validation samples: " << validation_samples_ << '\n';
            out << "Validation scalar_compare: " << (validation_scalar_compare_ ? "yes" : "no") << '\n';
            out << "Validation max_abs: " << validation_max_abs_ << '\n';
            out << "Validation mean_abs: " << validation_mean_abs_ << '\n';
            if (!validation_scalar_compare_) {
                out << "Validation tail_indices: 20664-20669\n";
                out << "Validation tail_max_abs: " << validation_tail_max_abs_ << '\n';
                out << "Validation tail_mean_abs: " << validation_tail_mean_abs_ << '\n';
            }
        }
        out << "Raw samples: " << csv_path_ << '\n';
        out << "===============================================\n";
        return out.good();
    }

    void release_gl() {
        for (GLuint program : programs_) {
            if (program != 0) {
                glDeleteProgram(program);
            }
        }
        for (GLuint program : scalar_programs_) {
            if (program != 0) {
                glDeleteProgram(program);
            }
        }
        for (GLuint buffer : activation_buffers_) {
            if (buffer != 0) {
                glDeleteBuffers(1, &buffer);
            }
        }
        for (auto& layer : layers_) {
            if (layer.weights_buffer != 0) {
                glDeleteBuffers(1, &layer.weights_buffer);
            }
            if (layer.bias_buffer != 0) {
                glDeleteBuffers(1, &layer.bias_buffer);
            }
        }
        if (gpu_query_ != 0 && timer_query_fns_.supported) {
            timer_query_fns_.delete_queries(1, &gpu_query_);
        }
        programs_ = {0, 0, 0};
        scalar_programs_ = {0, 0, 0};
        activation_buffers_ = {0, 0};
        layers_ = {};
        layer_specs_ = {};
        gpu_query_ = 0;
        timer_query_fns_ = {};
    }

    void fail(const std::string& message) {
        last_error_ = message;
        set_text(status_.last_error, sizeof(status_.last_error), message);
        set_text(status_.phase, sizeof(status_.phase), "error");
        log_android(ANDROID_LOG_ERROR, "%s", message.c_str());
    }

    bool check_gl_error(const char* tag) {
        const GLenum error = glGetError();
        if (error == GL_NO_ERROR) {
            return true;
        }
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "%s: glGetError 0x%x", tag, static_cast<unsigned int>(error));
        fail(buffer);
        return false;
    }

    MlpBenchmarkConfig config_{};
    AAssetManager* asset_manager_ = nullptr;
    std::string shader_asset_dir_ = "shaders";
    std::string csv_path_;
    std::string log_path_;
    std::string last_error_;
    std::array<GLuint, kLayerCount> programs_{{0, 0, 0}};
    std::array<GLuint, kLayerCount> scalar_programs_{{0, 0, 0}};
    std::array<GLuint, 2> activation_buffers_{{0, 0}};
    GLuint final_output_buffer_ = 0;
    std::array<LayerRuntime, kLayerCount> layers_{};
    std::array<LayerSpec, kLayerCount> layer_specs_{};
    std::vector<float> activation_a_;
    std::vector<float> activation_b_;
    bool validation_mode_ = false;
    bool validation_scalar_compare_ = false;
    int validation_samples_ = 0;
    int validation_index_ = 0;
    double validation_max_abs_ = 0.0;
    double validation_mean_abs_ = 0.0;
    double validation_abs_sum_ = 0.0;
    size_t validation_value_count_ = 0;
    double validation_tail_max_abs_ = 0.0;
    double validation_tail_mean_abs_ = 0.0;
    double validation_tail_abs_sum_ = 0.0;
    size_t validation_tail_value_count_ = 0;
    std::vector<float> validation_inputs_;
    std::vector<float> validation_outputs_;
    std::vector<float> validation_vector_output_;
    std::vector<double> cpu_samples_;
    std::vector<double> mlp_cpu_samples_;  // dispatch+glFinish only
    std::vector<double> gpu_samples_;
    TimerQueryFns timer_query_fns_{};
    GLuint gpu_query_ = 0;
    bool reports_written_ = false;
    StatsAccumulator cpu_stats_{};
    StatsAccumulator mlp_cpu_stats_{};  // dispatch+glFinish only
    StatsAccumulator gpu_stats_{};
    std::chrono::steady_clock::time_point benchmark_measurement_start_{};
    std::chrono::steady_clock::time_point realtime_frame_start_{};
    bool realtime_started_ = false;
    int realtime_frame_count_ = 0;
    MlpBenchmarkStatus status_{};
};

class SmplLbsBenchmarkCore {
public:
    ~SmplLbsBenchmarkCore() {
        release_gl();
    }

    bool init(AAssetManager* asset_manager,
              const std::string& csv_path,
              const std::string& log_path,
              bool validation_mode = false,
              bool app_mode = false,
              bool force_glsl_pose = false) {
        release_gl();
        asset_manager_ = asset_manager;
        csv_path_ = csv_path;
        log_path_ = log_path;
        validation_mode_ = validation_mode;
        app_mode_ = app_mode;
        validation_force_glsl_pose_ = force_glsl_pose;
        validation_samples_ = 0;
        validation_index_ = 0;
        animation_samples_ = 0;
        animation_index_ = 0;
        validation_max_abs_mm_ = 0.0;
        validation_mean_abs_mm_ = 0.0;
        validation_abs_sum_mm_ = 0.0;
        validation_value_count_ = 0;
        validation_precomputed_pose_ = false;
        validation_betas_.clear();
        validation_poses_.clear();
        validation_vertices_.clear();
        validation_pose_features_.clear();
        validation_transforms_.clear();
        cpu_samples_.clear();
        gpu_samples_.clear();
        cpu_stats_ = {};
        gpu_stats_ = {};
        reports_written_ = false;
        status_ = {};
        status_.input_size = 72;
        status_.hidden_size = 0;
        status_.output_size = kSmplVertexCount * 3;
        status_.warmup_iterations = app_mode_ ? 0 : kDefaultWarmupIterations;
        status_.benchmark_iterations = app_mode_ ? 0 : kDefaultBenchmarkIterations;
        status_.dispatches_per_frame = kSmplProgramCount;
        status_.use_fp16_weights = false;
        set_text(status_.phase, sizeof(status_.phase), "init");
        set_text(status_.last_error, sizeof(status_.last_error), "");

        const std::string version = gl_string(GL_VERSION);
        const std::string renderer = gl_string(GL_RENDERER);
        const std::string vendor = gl_string(GL_VENDOR);
        set_text(status_.renderer, sizeof(status_.renderer), renderer.empty() ? vendor : renderer);
        set_text(status_.version, sizeof(status_.version), version);
        if (version.empty()) {
            fail("Unable to query GL version");
            return false;
        }

        int major = 0;
        int minor = 0;
        if (!parse_es_version(version, &major, &minor) || major < 3 || (major == 3 && minor < 1)) {
            fail("OpenGL ES 3.1+ required");
            return false;
        }

        timer_query_fns_ = load_timer_query_fns();
        status_.gpu_timer_supported = timer_query_fns_.supported ? 1 : 0;
        if (timer_query_fns_.supported) {
            timer_query_fns_.gen_queries(1, &gpu_query_);
            if (gpu_query_ == 0) {
                timer_query_fns_ = {};
                status_.gpu_timer_supported = 0;
            }
        }

        if (!load_programs() || !create_buffers() || !upload_buffers()) {
            return false;
        }
        if (validation_mode_) {
            if (!load_validation_fixture()) {
                return false;
            }
            status_.warmup_iterations = 0;
            status_.benchmark_iterations = validation_samples_;
            set_text(status_.phase, sizeof(status_.phase), "validation");
        }
        if (app_mode_) {
            if (!load_animation_fixture()) {
                return false;
            }
            set_text(status_.phase, sizeof(status_.phase), "realtime");
        }

        status_.initialized = 1;
        if (!validation_mode_ && !app_mode_) {
            set_text(status_.phase, sizeof(status_.phase), "warmup");
        }
        log_android(ANDROID_LOG_INFO,
                    "Initialized SMPL LBS benchmark: mode=%s, renderer=%s, vendor=%s, version=%s, vertices=%d, joints=%d, warmup=%d, bench=%d, gpu_timer=%s",
                    app_mode_ ? "app" : (validation_mode_ ? "validation" : "benchmark"),
                    status_.renderer,
                    vendor.c_str(),
                    status_.version,
                    kSmplVertexCount,
                    kSmplJointCount,
                    status_.warmup_iterations,
                    status_.benchmark_iterations,
                    status_.gpu_timer_supported ? "yes" : "no");
        return true;
    }

    void resize(int width, int height) {
        if (width > 0 && height > 0) {
            viewport_width_ = width;
            viewport_height_ = height;
            glViewport(0, 0, width, height);
        }
    }

    bool draw() {
        if (!status_.initialized) {
            return false;
        }
        if (app_mode_) {
            if (!upload_animation_sample()) {
                return false;
            }
            const auto frame_start = std::chrono::steady_clock::now();
            if (!dispatch_frame()) {
                return false;
            }
            glFinish();
            const auto frame_end = std::chrono::steady_clock::now();
            const double cpu_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            status_.cpu_last_ms = cpu_ms;
            status_.mlp_cpu_last_ms = cpu_ms;
            cpu_stats_.add(cpu_ms);
            update_stats_fields();
            render_mesh();
            ++realtime_frame_count_;
            update_realtime_fps();
            set_text(status_.phase, sizeof(status_.phase), "realtime");
            return true;
        }
        if (status_.finished) {
            render_mesh();
            return true;
        }

        const int sample_index = validation_mode_ ? validation_index_ : -1;
        if (validation_mode_ && !upload_validation_sample(sample_index)) {
            return false;
        }

        const auto cpu_start = std::chrono::steady_clock::now();
        if (timer_query_fns_.supported) {
            timer_query_fns_.begin_query(GL_TIME_ELAPSED_EXT, gpu_query_);
        }
        if (!dispatch_frame()) {
            return false;
        }
        if (timer_query_fns_.supported) {
            timer_query_fns_.end_query(GL_TIME_ELAPSED_EXT);
        }
        glFinish();
        const auto cpu_end = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        status_.cpu_last_ms = cpu_ms;
        status_.mlp_cpu_last_ms = cpu_ms;

        if (validation_mode_ && !compare_validation_sample(sample_index)) {
            return false;
        }

        render_mesh();
        const int total_seen = status_.warmup_completed + status_.benchmark_completed;
        if (total_seen < status_.warmup_iterations) {
            ++status_.warmup_completed;
            set_text(status_.phase, sizeof(status_.phase), "warmup");
        } else if (status_.benchmark_completed < status_.benchmark_iterations) {
            if (status_.benchmark_completed == 0) {
                benchmark_measurement_start_ = std::chrono::steady_clock::now();
            }
            ++status_.benchmark_completed;
            cpu_stats_.add(cpu_ms);
            cpu_samples_.push_back(cpu_ms);
            if (timer_query_fns_.supported) {
                record_gpu_sample();
            } else {
                gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            }
            update_stats_fields();
            set_text(status_.phase, sizeof(status_.phase), validation_mode_ ? "validation" : "benchmark");
            if (validation_mode_) {
                ++validation_index_;
            }
            if ((status_.benchmark_completed % 100) == 0 ||
                status_.benchmark_completed == status_.benchmark_iterations) {
                if (!csv_path_.empty()) {
                    write_csv_report();
                }
                if (!log_path_.empty()) {
                    write_log_report();
                }
            }
        }

        if (status_.benchmark_completed >= status_.benchmark_iterations) {
            const auto now = std::chrono::steady_clock::now();
            status_.elapsed_seconds = std::chrono::duration<double>(now - benchmark_measurement_start_).count();
            status_.fps = status_.elapsed_seconds > 0.0
                                  ? static_cast<double>(status_.benchmark_completed) / status_.elapsed_seconds
                                  : 0.0;
            status_.finished = 1;
            set_text(status_.phase, sizeof(status_.phase), "realtime");
            write_reports_if_needed();
        }
        return true;
    }

    const MlpBenchmarkStatus& status() const {
        return status_;
    }

private:
    enum BufferIndex {
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
        kBufCount
    };

    bool load_programs() {
        const std::array<const char*, kSmplProgramCount> shader_names = {{
                "smpl_shape_blend.comp",
                "smpl_joint_regression.comp",
                "smpl_pose_kinematics.comp",
                "smpl_skinning.comp",
        }};
        for (size_t i = 0; i < shader_names.size(); ++i) {
            const std::string shader_source = read_text_asset(asset_manager_, "shaders", shader_names[i]);
            if (shader_source.empty()) {
                fail(std::string("Unable to load shader asset: shaders/") + shader_names[i]);
                return false;
            }
            std::string error;
            programs_[i] = compile_compute_program(shader_source, {}, &error);
            if (programs_[i] == 0) {
                fail(error);
                return false;
            }
        }
        static constexpr const char* kVertexShader = R"(#version 310 es
precision highp float;
precision highp int;
layout(location = 0) in vec3 aPosition;
uniform float uTime;
uniform float uAspect;
out vec3 vColor;
void main() {
    vec3 p = aPosition;
    float angle = 0.35 * sin(uTime * 0.65);
    float c = cos(angle);
    float s = sin(angle);
    vec3 q = vec3(c * p.x + s * p.y, -s * p.x + c * p.y, p.z);
    q.z -= 0.10;
    float scale = 1.45;
    gl_Position = vec4((q.x * scale) / max(uAspect, 0.1), q.z * scale - 0.05, q.y * 0.35, 1.0);
    vColor = mix(vec3(0.10, 0.55, 0.95), vec3(0.85, 0.95, 1.00), clamp(q.z + 0.55, 0.0, 1.0));
}
)";
        static constexpr const char* kFragmentShader = R"(#version 310 es
precision highp float;
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}
)";
        std::string error;
        render_program_ = compile_graphics_program(kVertexShader, kFragmentShader, &error);
        if (render_program_ == 0) {
            fail(error);
            return false;
        }
        return true;
    }

    bool create_buffers() {
        glGenBuffers(kBufCount, buffers_.data());
        for (GLuint buffer : buffers_) {
            if (buffer == 0) {
                fail("Failed to create SMPL LBS SSBOs");
                return false;
            }
        }
        glGenBuffers(1, &face_index_buffer_);
        glGenBuffers(1, &render_vertex_buffer_);
        glGenVertexArrays(1, &render_vao_);
        if (face_index_buffer_ == 0 || render_vertex_buffer_ == 0 || render_vao_ == 0) {
            fail("Failed to create SMPL render buffers");
            return false;
        }
        return true;
    }

    bool upload_asset_buffer(BufferIndex index,
                             const char* name,
                             size_t expected_bytes,
                             GLenum usage = GL_STATIC_DRAW) {
        std::vector<uint8_t> bytes = read_binary_asset(asset_manager_, "smpl_shader_data", name);
        if (bytes.size() != expected_bytes) {
            fail(std::string("SMPL asset size mismatch for smpl_shader_data/") + name +
                 ": expected " + std::to_string(expected_bytes) +
                 " bytes, got " + std::to_string(bytes.size()));
            return false;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[index]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(bytes.size()),
                     bytes.data(),
                     usage);
        return true;
    }

    bool allocate_buffer(BufferIndex index, size_t bytes, GLenum usage = GL_DYNAMIC_COPY) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[index]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     static_cast<GLsizeiptr>(bytes),
                     nullptr,
                     usage);
        return check_gl_error("allocate_buffer");
    }

    bool upload_buffers() {
        if (!upload_asset_buffer(kBufTemplate, "v_template_f32.bin", kSmplVertexCount * 3 * sizeof(float)) ||
            !upload_asset_buffer(kBufShapeDirs, "shapedirs_10_f32.bin", kSmplNumBetas * kSmplVertexCount * 3 * sizeof(float)) ||
            !upload_asset_buffer(kBufBetas, "betas_zero_f32.bin", kSmplNumBetas * sizeof(float)) ||
            !upload_asset_buffer(kBufJointRegressor, "joint_regressor_f32.bin", kSmplJointCount * kSmplVertexCount * sizeof(float)) ||
            !upload_asset_buffer(kBufPose, "pose_axis_angle_zero_f32.bin", kSmplJointCount * 3 * sizeof(float)) ||
            !upload_asset_buffer(kBufParents, "parents_i32.bin", kSmplJointCount * sizeof(int32_t)) ||
            !upload_asset_buffer(kBufPoseDirs, "posedirs_207_f32.bin", kSmplPoseBasisCount * kSmplVertexCount * 3 * sizeof(float)) ||
            !upload_asset_buffer(kBufWeights, "lbs_weights_f32.bin", kSmplVertexCount * kSmplJointCount * sizeof(float))) {
            return false;
        }
        if (!allocate_buffer(kBufShaped, kSmplVertexCount * 3 * sizeof(float)) ||
            !allocate_buffer(kBufJoints, kSmplJointCount * 3 * sizeof(float)) ||
            !allocate_buffer(kBufPoseFeatures, kSmplPoseBasisCount * sizeof(float)) ||
            !allocate_buffer(kBufTransforms, kSmplJointCount * 16 * sizeof(float)) ||
            !allocate_buffer(kBufOutput, kSmplVertexCount * 3 * sizeof(float))) {
            return false;
        }
        std::vector<uint8_t> face_bytes = read_binary_asset(asset_manager_, "smpl_shader_data", "faces_u16.bin");
        if (face_bytes.empty() || (face_bytes.size() % sizeof(uint16_t)) != 0) {
            fail("Invalid SMPL face asset: smpl_shader_data/faces_u16.bin");
            return false;
        }
        face_index_count_ = static_cast<int>(face_bytes.size() / sizeof(uint16_t));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, face_index_buffer_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(face_bytes.size()),
                     face_bytes.data(),
                     GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, render_vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(kSmplVertexCount * 3 * sizeof(float)),
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return check_gl_error("upload_buffers");
    }

    bool load_validation_fixture() {
        auto load_float_asset = [&](const char* name, std::vector<float>* out) -> bool {
            std::vector<uint8_t> bytes = read_binary_asset(asset_manager_, "smpl_lbs_validation_fixture", name);
            if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
                fail(std::string("Invalid SMPL validation asset: smpl_lbs_validation_fixture/") + name);
                return false;
            }
            out->resize(bytes.size() / sizeof(float));
            std::memcpy(out->data(), bytes.data(), bytes.size());
            return true;
        };

        auto try_load_float_asset = [&](const char* name, std::vector<float>* out) -> bool {
            std::vector<uint8_t> bytes = read_binary_asset(asset_manager_, "smpl_lbs_validation_fixture", name);
            if (bytes.empty()) {
                return false;
            }
            if ((bytes.size() % sizeof(float)) != 0) {
                fail(std::string("Invalid SMPL validation asset: smpl_lbs_validation_fixture/") + name);
                return false;
            }
            out->resize(bytes.size() / sizeof(float));
            std::memcpy(out->data(), bytes.data(), bytes.size());
            return true;
        };

        if (!load_float_asset("betas_f32.bin", &validation_betas_) ||
            !load_float_asset("poses_axis_angle_f32.bin", &validation_poses_) ||
            !load_float_asset("vertices_ref_f32.bin", &validation_vertices_)) {
            return false;
        }
        const bool has_pose_features = try_load_float_asset("pose_features_f32.bin", &validation_pose_features_);
        const bool has_transforms = try_load_float_asset("joint_transforms_f32.bin", &validation_transforms_);
        validation_precomputed_pose_ = has_pose_features && has_transforms && !validation_force_glsl_pose_;
        if ((validation_betas_.size() % kSmplNumBetas) != 0 ||
            (validation_poses_.size() % (kSmplJointCount * 3)) != 0 ||
            (validation_vertices_.size() % (kSmplVertexCount * 3)) != 0) {
            fail("SMPL validation fixture shape mismatch");
            return false;
        }
        if (validation_precomputed_pose_ &&
            ((validation_pose_features_.size() % kSmplPoseBasisCount) != 0 ||
             (validation_transforms_.size() % (kSmplJointCount * 16)) != 0)) {
            fail("SMPL validation precomputed pose fixture shape mismatch");
            return false;
        }
        const int beta_samples = static_cast<int>(validation_betas_.size() / kSmplNumBetas);
        const int pose_samples = static_cast<int>(validation_poses_.size() / (kSmplJointCount * 3));
        const int vertex_samples = static_cast<int>(validation_vertices_.size() / (kSmplVertexCount * 3));
        validation_samples_ = std::min(beta_samples, std::min(pose_samples, vertex_samples));
        if (validation_precomputed_pose_) {
            const int pose_feature_samples = static_cast<int>(validation_pose_features_.size() / kSmplPoseBasisCount);
            const int transform_samples = static_cast<int>(validation_transforms_.size() / (kSmplJointCount * 16));
            validation_samples_ = std::min(validation_samples_, std::min(pose_feature_samples, transform_samples));
        }
        if (validation_samples_ <= 0) {
            fail("SMPL validation fixture has no samples");
            return false;
        }
        return true;
    }

    bool load_animation_fixture() {
        auto load_float_asset = [&](const char* name, std::vector<float>* out) -> bool {
            std::vector<uint8_t> bytes = read_binary_asset(asset_manager_, "smpl_lbs_validation_fixture", name);
            if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
                fail(std::string("Invalid SMPL app animation asset: smpl_lbs_validation_fixture/") + name);
                return false;
            }
            out->resize(bytes.size() / sizeof(float));
            std::memcpy(out->data(), bytes.data(), bytes.size());
            return true;
        };

        if (!load_float_asset("betas_f32.bin", &validation_betas_) ||
            !load_float_asset("poses_axis_angle_f32.bin", &validation_poses_) ||
            !load_float_asset("pose_features_f32.bin", &validation_pose_features_) ||
            !load_float_asset("joint_transforms_f32.bin", &validation_transforms_)) {
            return false;
        }
        const int beta_samples = static_cast<int>(validation_betas_.size() / kSmplNumBetas);
        const int pose_samples = static_cast<int>(validation_poses_.size() / (kSmplJointCount * 3));
        const int feature_samples = static_cast<int>(validation_pose_features_.size() / kSmplPoseBasisCount);
        const int transform_samples = static_cast<int>(validation_transforms_.size() / (kSmplJointCount * 16));
        animation_samples_ = std::min(std::min(beta_samples, pose_samples), std::min(feature_samples, transform_samples));
        if (animation_samples_ <= 0) {
            fail("SMPL app animation fixture has no samples");
            return false;
        }
        validation_precomputed_pose_ = true;
        return upload_animation_sample();
    }

    bool upload_validation_sample(int sample_index) {
        if (sample_index < 0 || sample_index >= validation_samples_) {
            fail("SMPL validation sample index out of range");
            return false;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufBetas]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        kSmplNumBetas * sizeof(float),
                        validation_betas_.data() + sample_index * kSmplNumBetas);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufPose]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        kSmplJointCount * 3 * sizeof(float),
                        validation_poses_.data() + sample_index * kSmplJointCount * 3);
        if (validation_precomputed_pose_) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufPoseFeatures]);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            0,
                            kSmplPoseBasisCount * sizeof(float),
                            validation_pose_features_.data() + sample_index * kSmplPoseBasisCount);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufTransforms]);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            0,
                            kSmplJointCount * 16 * sizeof(float),
                            validation_transforms_.data() + sample_index * kSmplJointCount * 16);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return check_gl_error("upload_validation_sample");
    }

    bool upload_animation_sample() {
        if (animation_samples_ <= 0) {
            fail("SMPL app animation fixture has no samples");
            return false;
        }
        const int sample_index = animation_index_ % animation_samples_;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufBetas]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        kSmplNumBetas * sizeof(float),
                        validation_betas_.data() + sample_index * kSmplNumBetas);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufPose]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        kSmplJointCount * 3 * sizeof(float),
                        validation_poses_.data() + sample_index * kSmplJointCount * 3);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufPoseFeatures]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        kSmplPoseBasisCount * sizeof(float),
                        validation_pose_features_.data() + sample_index * kSmplPoseBasisCount);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufTransforms]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        kSmplJointCount * 16 * sizeof(float),
                        validation_transforms_.data() + sample_index * kSmplJointCount * 16);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        animation_index_ = (animation_index_ + 1) % animation_samples_;
        return check_gl_error("upload_animation_sample");
    }

    bool dispatch_frame() {
        const GLuint vertex_groups = static_cast<GLuint>((kSmplVertexCount + kLocalSizeX - 1) / kLocalSizeX);

        glUseProgram(programs_[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers_[kBufTemplate]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers_[kBufShapeDirs]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers_[kBufBetas]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers_[kBufShaped]);
        glDispatchCompute(vertex_groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        if (!validation_precomputed_pose_) {
            glUseProgram(programs_[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers_[kBufShaped]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers_[kBufJointRegressor]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers_[kBufJoints]);
            glDispatchCompute(1, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            glUseProgram(programs_[2]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers_[kBufPose]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers_[kBufJoints]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers_[kBufParents]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers_[kBufPoseFeatures]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, buffers_[kBufTransforms]);
            glDispatchCompute(1, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        glUseProgram(programs_[3]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers_[kBufShaped]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers_[kBufPoseDirs]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers_[kBufPoseFeatures]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers_[kBufWeights]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, buffers_[kBufTransforms]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, buffers_[kBufOutput]);
        glDispatchCompute(vertex_groups, 1, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        return check_gl_error("smpl_dispatch_frame");
    }

    bool compare_validation_sample(int sample_index) {
        const size_t value_count = static_cast<size_t>(kSmplVertexCount) * 3u;
        const float* expected = validation_vertices_.data() + static_cast<size_t>(sample_index) * value_count;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufOutput]);
        const void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                              0,
                                              static_cast<GLsizeiptr>(value_count * sizeof(float)),
                                              GL_MAP_READ_BIT);
        if (mapped == nullptr) {
            fail("glMapBufferRange failed for SMPL validation output");
            return false;
        }
        const float* actual = static_cast<const float*>(mapped);
        double sample_sum_mm = 0.0;
        double sample_max_mm = 0.0;
        for (size_t i = 0; i < value_count; ++i) {
            const double abs_mm = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])) * 1000.0;
            sample_sum_mm += abs_mm;
            if (abs_mm > sample_max_mm) {
                sample_max_mm = abs_mm;
            }
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (!check_gl_error("compare_validation_sample")) {
            return false;
        }

        validation_max_abs_mm_ = std::max(validation_max_abs_mm_, sample_max_mm);
        validation_abs_sum_mm_ += sample_sum_mm;
        validation_value_count_ += value_count;
        validation_mean_abs_mm_ = validation_value_count_ > 0
                                          ? validation_abs_sum_mm_ / static_cast<double>(validation_value_count_)
                                          : 0.0;
        log_android(ANDROID_LOG_INFO,
                    "SMPL validation sample=%d/%d max_abs_mm=%.6f running_max_abs_mm=%.6f mean_abs_mm=%.6f",
                    sample_index + 1,
                    validation_samples_,
                    sample_max_mm,
                    validation_max_abs_mm_,
                    validation_mean_abs_mm_);
        return true;
    }

    void record_gpu_sample() {
        GLint available = GL_FALSE;
        timer_query_fns_.get_query_object_iv(gpu_query_, GL_QUERY_RESULT_AVAILABLE, &available);
        if (available != GL_TRUE) {
            gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            return;
        }
        GLuint64 elapsed_ns = 0;
        timer_query_fns_.get_query_object_ui64v(gpu_query_, GL_QUERY_RESULT, &elapsed_ns);
        GLint disjoint = GL_FALSE;
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
        if (disjoint == GL_TRUE) {
            gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            status_.gpu_timer_valid = 0;
            return;
        }
        const double elapsed_ms = static_cast<double>(elapsed_ns) / 1.0e6;
        gpu_stats_.add(elapsed_ms);
        gpu_samples_.push_back(elapsed_ms);
        status_.gpu_timer_valid = 1;
        status_.gpu_last_ms = elapsed_ms;
    }

    void clear_frame() {
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.01f, 0.01f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void render_mesh() {
        glViewport(0, 0, viewport_width_, viewport_height_);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDisable(GL_CULL_FACE);
        glClearColor(0.015f, 0.020f, 0.030f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (render_program_ == 0 ||
            face_index_buffer_ == 0 ||
            render_vertex_buffer_ == 0 ||
            render_vao_ == 0 ||
            face_index_count_ <= 0) {
            return;
        }
        if (!upload_render_vertices()) {
            return;
        }
        glUseProgram(render_program_);
        const float aspect = viewport_height_ > 0
                                     ? static_cast<float>(viewport_width_) / static_cast<float>(viewport_height_)
                                     : 1.0f;
        const GLint time_location = glGetUniformLocation(render_program_, "uTime");
        const GLint aspect_location = glGetUniformLocation(render_program_, "uAspect");
        if (time_location >= 0) {
            glUniform1f(time_location, static_cast<float>(realtime_frame_count_) * 0.0167f);
        }
        if (aspect_location >= 0) {
            glUniform1f(aspect_location, aspect);
        }
        glBindVertexArray(render_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, render_vertex_buffer_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, face_index_buffer_);
        glDrawElements(GL_TRIANGLES, face_index_count_, GL_UNSIGNED_SHORT, nullptr);
        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    bool upload_render_vertices() {
        constexpr size_t kVertexBytes = static_cast<size_t>(kSmplVertexCount) * 3u * sizeof(float);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[kBufOutput]);
        const void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                              0,
                                              static_cast<GLsizeiptr>(kVertexBytes),
                                              GL_MAP_READ_BIT);
        if (mapped == nullptr) {
            fail("glMapBufferRange failed for SMPL render vertices");
            return false;
        }
        glBindBuffer(GL_ARRAY_BUFFER, render_vertex_buffer_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(kVertexBytes), mapped);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return check_gl_error("upload_render_vertices");
    }

    void update_realtime_fps() {
        if (!realtime_started_) {
            realtime_started_ = true;
            realtime_frame_start_ = std::chrono::steady_clock::now();
            realtime_frame_count_ = 0;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - realtime_frame_start_).count();
        if (seconds > 0.0) {
            status_.elapsed_seconds = seconds;
            status_.fps = static_cast<double>(realtime_frame_count_) / seconds;
        }
    }

    void update_stats_fields() {
        status_.cpu_mean_ms = cpu_stats_.mean;
        status_.cpu_stddev_ms = cpu_stats_.stddev();
        status_.cpu_min_ms = cpu_stats_.count > 0 ? cpu_stats_.min_value : 0.0;
        status_.cpu_max_ms = cpu_stats_.count > 0 ? cpu_stats_.max_value : 0.0;
        status_.mlp_cpu_mean_ms = status_.cpu_mean_ms;
        status_.mlp_cpu_stddev_ms = status_.cpu_stddev_ms;
        if (gpu_stats_.count > 0) {
            status_.gpu_timer_valid = 1;
            status_.gpu_mean_ms = gpu_stats_.mean;
            status_.gpu_stddev_ms = gpu_stats_.stddev();
            status_.gpu_min_ms = gpu_stats_.min_value;
            status_.gpu_max_ms = gpu_stats_.max_value;
        }
    }

    void write_reports_if_needed() {
        if (reports_written_) {
            return;
        }
        const bool csv_ok = csv_path_.empty() || write_csv_report();
        const bool log_ok = log_path_.empty() || write_log_report();
        reports_written_ = csv_ok && log_ok;
    }

    bool write_csv_report() {
        if (!ensure_parent_directories(csv_path_)) {
            fail("Failed to create SMPL CSV parent directories");
            return false;
        }
        std::ofstream out(csv_path_, std::ios::binary | std::ios::trunc);
        if (!out.good()) {
            fail("Unable to open SMPL CSV path: " + csv_path_);
            return false;
        }
        out << "run,mlp_cpu_ms,cpu_ms,gpu_ms\n";
        for (size_t i = 0; i < cpu_samples_.size(); ++i) {
            out << (i + 1) << ',' << cpu_samples_[i] << ',' << cpu_samples_[i] << ',';
            if (i < gpu_samples_.size() && !std::isnan(gpu_samples_[i])) {
                out << gpu_samples_[i];
            } else {
                out << "NA";
            }
            out << '\n';
        }
        return out.good();
    }

    bool write_log_report() {
        if (!ensure_parent_directories(log_path_)) {
            fail("Failed to create SMPL log parent directories");
            return false;
        }
        std::ofstream out(log_path_, std::ios::binary | std::ios::trunc);
        if (!out.good()) {
            fail("Unable to open SMPL log path: " + log_path_);
            return false;
        }
        out << "===== SMPL GLSL LBS Compute Shader Benchmark =====\n";
        out << "Timestamp: " << current_timestamp_string() << '\n';
        out << "Measured CPU path: 4x glDispatchCompute + glMemoryBarrier + glFinish\n";
        out << "Mode: " << (validation_mode_ ? "validation" : "benchmark") << '\n';
        out << "Renderer: " << status_.renderer << '\n';
        out << "Version: " << status_.version << '\n';
        out << "Vertices: " << kSmplVertexCount << '\n';
        out << "Joints: " << kSmplJointCount << '\n';
        out << "Pose basis: " << kSmplPoseBasisCount << '\n';
        out << "Warmup iterations: " << status_.warmup_iterations << '\n';
        out << "Benchmark iterations: " << status_.benchmark_iterations << '\n';
        out << "CPU last/mean/stddev/min/max ms: "
            << status_.cpu_last_ms << " / " << status_.cpu_mean_ms << " / " << status_.cpu_stddev_ms
            << " / " << status_.cpu_min_ms << " / " << status_.cpu_max_ms << '\n';
        out << "GPU timer supported: " << status_.gpu_timer_supported
            << " valid: " << status_.gpu_timer_valid << '\n';
        out << "GPU last/mean/stddev/min/max ms: "
            << status_.gpu_last_ms << " / " << status_.gpu_mean_ms << " / " << status_.gpu_stddev_ms
            << " / " << status_.gpu_min_ms << " / " << status_.gpu_max_ms << '\n';
        if (validation_mode_) {
            out << "Validation samples: " << validation_samples_ << '\n';
            out << "Validation precomputed pose/kinematics: " << (validation_precomputed_pose_ ? "yes" : "no") << '\n';
            out << "Validation max_abs_mm: " << validation_max_abs_mm_ << '\n';
            out << "Validation mean_abs_mm: " << validation_mean_abs_mm_ << '\n';
            out << "Validation threshold_mm: 0.01\n";
            out << "Validation pass: " << (validation_max_abs_mm_ < 0.01 ? "yes" : "no") << '\n';
        }
        out << "Raw samples: " << csv_path_ << '\n';
        out << "===============================================\n";
        return out.good();
    }

    void release_gl() {
        for (GLuint program : programs_) {
            if (program != 0) {
                glDeleteProgram(program);
            }
        }
        if (render_program_ != 0) {
            glDeleteProgram(render_program_);
        }
        for (GLuint buffer : buffers_) {
            if (buffer != 0) {
                glDeleteBuffers(1, &buffer);
            }
        }
        if (face_index_buffer_ != 0) {
            glDeleteBuffers(1, &face_index_buffer_);
        }
        if (render_vertex_buffer_ != 0) {
            glDeleteBuffers(1, &render_vertex_buffer_);
        }
        if (render_vao_ != 0) {
            glDeleteVertexArrays(1, &render_vao_);
        }
        if (gpu_query_ != 0 && timer_query_fns_.supported) {
            timer_query_fns_.delete_queries(1, &gpu_query_);
        }
        programs_.fill(0);
        buffers_.fill(0);
        render_program_ = 0;
        face_index_buffer_ = 0;
        render_vertex_buffer_ = 0;
        render_vao_ = 0;
        face_index_count_ = 0;
        gpu_query_ = 0;
        timer_query_fns_ = {};
    }

    void fail(const std::string& message) {
        last_error_ = message;
        set_text(status_.last_error, sizeof(status_.last_error), message);
        set_text(status_.phase, sizeof(status_.phase), "error");
        log_android(ANDROID_LOG_ERROR, "%s", message.c_str());
    }

    bool check_gl_error(const char* tag) {
        const GLenum error = glGetError();
        if (error == GL_NO_ERROR) {
            return true;
        }
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "%s: glGetError 0x%x", tag, static_cast<unsigned int>(error));
        fail(buffer);
        return false;
    }

    AAssetManager* asset_manager_ = nullptr;
    std::string csv_path_;
    std::string log_path_;
    std::string last_error_;
    std::array<GLuint, kSmplProgramCount> programs_{{0, 0, 0, 0}};
    std::array<GLuint, kBufCount> buffers_{};
    GLuint render_program_ = 0;
    GLuint face_index_buffer_ = 0;
    GLuint render_vertex_buffer_ = 0;
    GLuint render_vao_ = 0;
    int face_index_count_ = 0;
    int viewport_width_ = 1;
    int viewport_height_ = 1;
    std::vector<double> cpu_samples_;
    std::vector<double> gpu_samples_;
    TimerQueryFns timer_query_fns_{};
    GLuint gpu_query_ = 0;
    bool reports_written_ = false;
    bool validation_mode_ = false;
    bool app_mode_ = false;
    bool validation_force_glsl_pose_ = false;
    bool validation_precomputed_pose_ = false;
    int validation_samples_ = 0;
    int validation_index_ = 0;
    int animation_samples_ = 0;
    int animation_index_ = 0;
    double validation_max_abs_mm_ = 0.0;
    double validation_mean_abs_mm_ = 0.0;
    double validation_abs_sum_mm_ = 0.0;
    size_t validation_value_count_ = 0;
    std::vector<float> validation_betas_;
    std::vector<float> validation_poses_;
    std::vector<float> validation_vertices_;
    std::vector<float> validation_pose_features_;
    std::vector<float> validation_transforms_;
    StatsAccumulator cpu_stats_{};
    StatsAccumulator gpu_stats_{};
    std::chrono::steady_clock::time_point benchmark_measurement_start_{};
    std::chrono::steady_clock::time_point realtime_frame_start_{};
    bool realtime_started_ = false;
    int realtime_frame_count_ = 0;
    MlpBenchmarkStatus status_{};
};

}  // namespace

struct MlpBenchmarkHandle {
    BenchmarkCore core;
};

MLP_BENCHMARK_API MlpBenchmarkConfig benchmark_default_config(void) {
    MlpBenchmarkConfig config{};
    config.input_size = kDefaultInputSize;
    config.hidden_size = kDefaultHiddenSize;
    config.output_size = kDefaultOutputSize;
    config.warmup_iterations = kDefaultWarmupIterations;
    config.benchmark_iterations = kDefaultBenchmarkIterations;
    config.use_fp16_weights = true;
    config.load_weights_from_assets = 1;
    config.validate_outputs = 0;
    config.seed = 0x5EED1234u;
    config.shader_asset_dir = "shaders";
    config.csv_path = nullptr;
    config.log_path = nullptr;
    return config;
}

MLP_BENCHMARK_API MlpBenchmarkHandle* benchmark_create(void) {
    return new (std::nothrow) MlpBenchmarkHandle();
}

MLP_BENCHMARK_API int benchmark_init(MlpBenchmarkHandle* handle,
                                     AAssetManager* asset_manager,
                                     const MlpBenchmarkConfig* config) {
    if (handle == nullptr) {
        return 0;
    }
    MlpBenchmarkConfig resolved = benchmark_default_config();
    if (config != nullptr) {
        resolved = *config;
        if (resolved.input_size <= 0) {
            resolved.input_size = kDefaultInputSize;
        }
        if (resolved.hidden_size <= 0) {
            resolved.hidden_size = kDefaultHiddenSize;
        }
        if (resolved.output_size <= 0) {
            resolved.output_size = kDefaultOutputSize;
        }
        if (resolved.warmup_iterations < 0) {
            resolved.warmup_iterations = kDefaultWarmupIterations;
        }
        if (resolved.benchmark_iterations <= 0) {
            resolved.benchmark_iterations = kDefaultBenchmarkIterations;
        }
        if (resolved.shader_asset_dir == nullptr || resolved.shader_asset_dir[0] == '\0') {
            resolved.shader_asset_dir = "shaders";
        }
    }
    return handle->core.init(asset_manager, resolved) ? 1 : 0;
}

MLP_BENCHMARK_API void benchmark_resize(MlpBenchmarkHandle* handle, int width, int height) {
    if (handle == nullptr) {
        return;
    }
    handle->core.resize(width, height);
}

MLP_BENCHMARK_API int benchmark_draw(MlpBenchmarkHandle* handle) {
    if (handle == nullptr) {
        return 0;
    }
    return handle->core.draw() ? 1 : 0;
}

MLP_BENCHMARK_API int benchmark_run_fps_loop(MlpBenchmarkHandle* handle, int frames, double target_fps) {
    if (handle == nullptr) {
        return 0;
    }
    return handle->core.run_fps_loop(frames, target_fps) ? 1 : 0;
}

MLP_BENCHMARK_API const MlpBenchmarkStatus* benchmark_status(const MlpBenchmarkHandle* handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return &handle->core.status();
}

MLP_BENCHMARK_API int benchmark_set_output_paths(MlpBenchmarkHandle* handle, const char* csv_path, const char* log_path) {
    if (handle == nullptr) {
        return 0;
    }
    return handle->core.set_output_paths(csv_path, log_path) ? 1 : 0;
}

MLP_BENCHMARK_API void benchmark_destroy(MlpBenchmarkHandle* handle) {
    delete handle;
}

namespace {

std::mutex g_benchmark_mutex;
MlpBenchmarkHandle* g_benchmark_handle = nullptr;
SmplLbsBenchmarkCore* g_smpl_lbs_handle = nullptr;
int g_last_logged_warmup = -1;
int g_last_logged_benchmark = -1;

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeInitialize(
        JNIEnv* env,
        jobject /* this */,
        jobject asset_manager_obj,
        jstring log_path_obj,
        jstring benchmark_mode_obj) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);

    if (g_benchmark_handle != nullptr) {
        benchmark_destroy(g_benchmark_handle);
        g_benchmark_handle = nullptr;
    }
    if (g_smpl_lbs_handle != nullptr) {
        delete g_smpl_lbs_handle;
        g_smpl_lbs_handle = nullptr;
    }

    AAssetManager* asset_manager = AAssetManager_fromJava(env, asset_manager_obj);
    if (asset_manager == nullptr) {
        log_android(ANDROID_LOG_ERROR, "AAssetManager_fromJava failed");
        return JNI_FALSE;
    }

    const char* path_chars = env->GetStringUTFChars(log_path_obj, nullptr);
    const std::string sample_path = path_chars != nullptr ? path_chars : "";
    if (path_chars != nullptr) {
        env->ReleaseStringUTFChars(log_path_obj, path_chars);
    }

    const char* mode_chars = env->GetStringUTFChars(benchmark_mode_obj, nullptr);
    const std::string benchmark_mode = mode_chars != nullptr ? mode_chars : "zdecn";
    if (mode_chars != nullptr) {
        env->ReleaseStringUTFChars(benchmark_mode_obj, mode_chars);
    }

    const std::string summary_path = default_summary_path(sample_path);
    g_last_logged_warmup = -1;
    g_last_logged_benchmark = -1;

    if (benchmark_mode == "smpl_lbs" ||
        benchmark_mode == "smpl_lbs_validate" ||
        benchmark_mode == "smpl_lbs_validate_glsl" ||
        benchmark_mode == "smpl_app") {
        g_smpl_lbs_handle = new (std::nothrow) SmplLbsBenchmarkCore();
        if (g_smpl_lbs_handle == nullptr) {
            log_android(ANDROID_LOG_ERROR, "SmplLbsBenchmarkCore allocation failed");
            return JNI_FALSE;
        }
        if (!g_smpl_lbs_handle->init(asset_manager,
                                     sample_path,
                                     summary_path,
                                     benchmark_mode == "smpl_lbs_validate" ||
                                             benchmark_mode == "smpl_lbs_validate_glsl",
                                     benchmark_mode == "smpl_app",
                                     benchmark_mode == "smpl_lbs_validate_glsl")) {
            const MlpBenchmarkStatus& status = g_smpl_lbs_handle->status();
            log_android(ANDROID_LOG_ERROR, "SMPL LBS benchmark init failed: %s", status.last_error);
            delete g_smpl_lbs_handle;
            g_smpl_lbs_handle = nullptr;
            return JNI_FALSE;
        }
        return JNI_TRUE;
    }

    g_benchmark_handle = benchmark_create();
    if (g_benchmark_handle == nullptr) {
        log_android(ANDROID_LOG_ERROR, "benchmark_create failed");
        return JNI_FALSE;
    }

    MlpBenchmarkConfig config = benchmark_default_config();
    config.warmup_iterations = kDefaultWarmupIterations;
    config.benchmark_iterations = kDefaultBenchmarkIterations;
    if (benchmark_mode == "zdecn_validate") {
        config.warmup_iterations = 0;
        config.benchmark_iterations = 200;
        config.validate_outputs = 1;
    } else if (benchmark_mode == "zdecn_scalar_compare") {
        config.warmup_iterations = 0;
        config.benchmark_iterations = 200;
        config.validate_outputs = 2;
    }
    config.csv_path = sample_path.c_str();
    config.log_path = summary_path.c_str();

    if (!benchmark_init(g_benchmark_handle, asset_manager, &config)) {
        const MlpBenchmarkStatus* status = benchmark_status(g_benchmark_handle);
        log_android(ANDROID_LOG_ERROR,
                    "benchmark_init failed: %s",
                    status != nullptr ? status->last_error : "unknown");
        benchmark_destroy(g_benchmark_handle);
        g_benchmark_handle = nullptr;
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeResize(
        JNIEnv* /* env */,
        jobject /* this */,
        jint width,
        jint height) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);
    if (g_smpl_lbs_handle != nullptr) {
        g_smpl_lbs_handle->resize(static_cast<int>(width), static_cast<int>(height));
    } else if (g_benchmark_handle != nullptr) {
        benchmark_resize(g_benchmark_handle, static_cast<int>(width), static_cast<int>(height));
    } else {
        glViewport(0, 0, width, height);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeRenderFrame(
        JNIEnv* /* env */,
        jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);
    if (g_smpl_lbs_handle != nullptr) {
        if (!g_smpl_lbs_handle->draw()) {
            const MlpBenchmarkStatus& status = g_smpl_lbs_handle->status();
            log_android(ANDROID_LOG_ERROR, "SMPL LBS draw failed: %s", status.last_error);
            return;
        }

        const MlpBenchmarkStatus& status = g_smpl_lbs_handle->status();
        bool should_log = false;
        if (status.warmup_completed != g_last_logged_warmup &&
            (status.warmup_completed == status.warmup_iterations ||
             (status.warmup_completed > 0 && (status.warmup_completed % 25) == 0))) {
            g_last_logged_warmup = status.warmup_completed;
            should_log = true;
        }
        if (status.benchmark_completed != g_last_logged_benchmark &&
            (status.benchmark_completed == status.benchmark_iterations ||
             (status.benchmark_completed > 0 && (status.benchmark_completed % 100) == 0))) {
            g_last_logged_benchmark = status.benchmark_completed;
            should_log = true;
        }
        if (should_log) {
            log_android(ANDROID_LOG_INFO,
                        "SMPL LBS progress phase=%s warmup=%d/%d benchmark=%d/%d cpu_ms=%.3f gpu_ms=%.3f",
                        status.phase,
                        status.warmup_completed,
                        status.warmup_iterations,
                        status.benchmark_completed,
                        status.benchmark_iterations,
                        status.cpu_last_ms,
                        status.gpu_last_ms);
        }
    } else if (g_benchmark_handle != nullptr) {
        if (!benchmark_draw(g_benchmark_handle)) {
            const MlpBenchmarkStatus* status = benchmark_status(g_benchmark_handle);
            log_android(ANDROID_LOG_ERROR,
                        "benchmark_draw failed: %s",
                        status != nullptr ? status->last_error : "unknown");
            return;
        }

        const MlpBenchmarkStatus* status = benchmark_status(g_benchmark_handle);
        if (status != nullptr) {
            bool should_log = false;
            if (status->warmup_completed != g_last_logged_warmup &&
                (status->warmup_completed == status->warmup_iterations ||
                 (status->warmup_completed > 0 && (status->warmup_completed % 25) == 0))) {
                g_last_logged_warmup = status->warmup_completed;
                should_log = true;
            }
            if (status->benchmark_completed != g_last_logged_benchmark &&
                (status->benchmark_completed == status->benchmark_iterations ||
                 (status->benchmark_completed > 0 && (status->benchmark_completed % 100) == 0))) {
                g_last_logged_benchmark = status->benchmark_completed;
                should_log = true;
            }
            if (should_log) {
                log_android(ANDROID_LOG_INFO,
                            "Progress phase=%s warmup=%d/%d benchmark=%d/%d cpu_ms=%.3f gpu_ms=%.3f",
                            status->phase,
                            status->warmup_completed,
                            status->warmup_iterations,
                            status->benchmark_completed,
                            status->benchmark_iterations,
                            status->cpu_last_ms,
                            status->gpu_last_ms);
            }
        }
    } else {
        glClearColor(0.03f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeStatusText(
        JNIEnv* env,
        jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);
    if (g_benchmark_handle == nullptr && g_smpl_lbs_handle == nullptr) {
        return env->NewStringUTF("Native backend is not initialized yet.");
    }

    const MlpBenchmarkStatus* status = nullptr;
    if (g_smpl_lbs_handle != nullptr) {
        status = &g_smpl_lbs_handle->status();
    } else {
        status = benchmark_status(g_benchmark_handle);
        if (status == nullptr) {
            return env->NewStringUTF("Native benchmark status unavailable.");
        }
    }

    char buffer[2048];
    std::snprintf(
            buffer,
            sizeof(buffer),
            "Phase: %s%s\n"
            "Warmup: %d / %d\n"
            "Benchmark: %d / %d\n"
            "Renderer: %s\n"
            "OpenGL: %s\n"
            "GPU timer query: %s%s\n"
            "Weights: %s\n"
            "CPU ms last/mean/max: %.3f / %.3f / %.3f\n"
            "GPU ms last/mean: %.3f / %.3f\n"
            "FPS: %.1f\n"
            "Last error: %s",
            status->phase,
            status->finished ? " (complete)" : "",
            status->warmup_completed,
            status->warmup_iterations,
            status->benchmark_completed,
            status->benchmark_iterations,
            status->renderer,
            status->version,
            status->gpu_timer_supported ? "available" : "unavailable",
            status->gpu_timer_valid ? ", valid" : "",
            status->use_fp16_weights ? "fp16" : "fp32",
            status->cpu_last_ms,
            status->cpu_mean_ms,
            status->cpu_max_ms,
            status->gpu_last_ms,
            status->gpu_mean_ms,
            status->fps,
            status->last_error);
    return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeShutdown(
        JNIEnv* /* env */,
        jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);
    if (g_benchmark_handle != nullptr) {
        benchmark_destroy(g_benchmark_handle);
        g_benchmark_handle = nullptr;
    }
    if (g_smpl_lbs_handle != nullptr) {
        delete g_smpl_lbs_handle;
        g_smpl_lbs_handle = nullptr;
    }
    g_last_logged_warmup = -1;
    g_last_logged_benchmark = -1;
}

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
            fail("This benchmark is fixed to 72 -> 256 -> 256 -> 20670.");
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
        cpu_stats_ = {};
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

        status_.initialized = 1;
        status_.finished = 0;
        set_text(status_.phase, sizeof(status_.phase), "warmup");
        log_android(ANDROID_LOG_INFO,
                    "Initialized benchmark: renderer=%s, version=%s, output=%d, warmup=%d, bench=%d, gpu_timer=%s",
                    status_.renderer,
                    status_.version,
                    status_.output_size,
                    status_.warmup_iterations,
                    status_.benchmark_iterations,
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

        const auto cpu_start = std::chrono::steady_clock::now();
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

        const auto cpu_end = std::chrono::steady_clock::now();
        clear_frame();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
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
            if (timer_query_fns_.supported) {
                record_gpu_sample();
            } else {
                gpu_samples_.push_back(std::numeric_limits<double>::quiet_NaN());
            }
            update_stats_fields();
            set_text(status_.phase, sizeof(status_.phase), "benchmark");
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
            const size_t expected_bias_bytes = bias_count * sizeof(float);

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
                glBufferData(GL_SHADER_STORAGE_BUFFER,
                             static_cast<GLsizeiptr>(biases.size() * sizeof(float)),
                             biases.data(),
                             GL_STATIC_DRAW);
            }
        }

        activation_a_.assign(static_cast<size_t>(kDefaultOutputSize), 0.0f);
        activation_b_.assign(static_cast<size_t>(kDefaultOutputSize), 0.0f);
        std::mt19937 input_rng(config_.seed ^ 0x9E3779B9u);
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        for (int i = 0; i < kDefaultInputSize; ++i) {
            activation_a_[static_cast<size_t>(i)] = input_dist(input_rng);
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

    bool dispatch_frame(bool measure) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, activation_buffers_[0]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(config_.input_size * static_cast<int>(sizeof(float))),
                        activation_a_.data());

        GLuint input_buffer = activation_buffers_[0];
        GLuint output_buffer = activation_buffers_[1];
        for (int layer = 0; layer < kLayerCount; ++layer) {
            glUseProgram(programs_[layer]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kInputBinding, input_buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kOutputBinding, output_buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kWeightsBinding, layers_[layer].weights_buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, kBiasBinding, layers_[layer].bias_buffer);

            const GLuint groups = static_cast<GLuint>((layers_[layer].output_size + kLocalSizeX - 1) / kLocalSizeX);
            glDispatchCompute(groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            std::swap(input_buffer, output_buffer);
        }
        if (measure) {
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
        }
        return check_gl_error("dispatch_frame");
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
        out << "run,cpu_ms,gpu_ms\n";
        for (size_t i = 0; i < cpu_samples_.size(); ++i) {
            out << (i + 1) << ',' << cpu_samples_[i] << ',';
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
    std::array<GLuint, 2> activation_buffers_{{0, 0}};
    std::array<LayerRuntime, kLayerCount> layers_{};
    std::array<LayerSpec, kLayerCount> layer_specs_{};
    std::vector<float> activation_a_;
    std::vector<float> activation_b_;
    std::vector<double> cpu_samples_;
    std::vector<double> gpu_samples_;
    TimerQueryFns timer_query_fns_{};
    GLuint gpu_query_ = 0;
    bool reports_written_ = false;
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
    config.use_fp16_weights = false;
    config.load_weights_from_assets = 0;
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
int g_last_logged_warmup = -1;
int g_last_logged_benchmark = -1;

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeInitialize(
        JNIEnv* env,
        jobject /* this */,
        jobject asset_manager_obj,
        jstring log_path_obj) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);

    if (g_benchmark_handle != nullptr) {
        benchmark_destroy(g_benchmark_handle);
        g_benchmark_handle = nullptr;
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

    g_benchmark_handle = benchmark_create();
    if (g_benchmark_handle == nullptr) {
        log_android(ANDROID_LOG_ERROR, "benchmark_create failed");
        return JNI_FALSE;
    }

    MlpBenchmarkConfig config = benchmark_default_config();
    config.warmup_iterations = kDefaultWarmupIterations;
    config.benchmark_iterations = kDefaultBenchmarkIterations;
    config.csv_path = sample_path.c_str();
    const std::string summary_path = default_summary_path(sample_path);
    config.log_path = summary_path.c_str();
    g_last_logged_warmup = -1;
    g_last_logged_benchmark = -1;

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
    if (g_benchmark_handle != nullptr) {
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
    if (g_benchmark_handle != nullptr) {
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

extern "C" JNIEXPORT void JNICALL
Java_com_graphicsresearch_androidcomputebenchmark_BenchmarkNative_nativeShutdown(
        JNIEnv* /* env */,
        jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_benchmark_mutex);
    if (g_benchmark_handle != nullptr) {
        benchmark_destroy(g_benchmark_handle);
        g_benchmark_handle = nullptr;
    }
    g_last_logged_warmup = -1;
    g_last_logged_benchmark = -1;
}

#pragma once

#include <android/asset_manager.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MLP_BENCHMARK_API __attribute__((visibility("default")))
#else
#define MLP_BENCHMARK_API
#endif

typedef struct MlpBenchmarkHandle MlpBenchmarkHandle;

typedef struct MlpBenchmarkConfig {
    int input_size;
    int hidden_size;
    int output_size;
    int warmup_iterations;
    int benchmark_iterations;
    bool use_fp16_weights;
    int load_weights_from_assets;
    int validate_outputs;
    uint32_t seed;
    const char* shader_asset_dir;
    const char* csv_path;
    const char* log_path;
} MlpBenchmarkConfig;

typedef struct MlpBenchmarkStatus {
    int initialized;
    int finished;
    int input_size;
    int hidden_size;
    int output_size;
    int warmup_iterations;
    int warmup_completed;
    int benchmark_iterations;
    int benchmark_completed;
    int dispatches_per_frame;
    int gpu_timer_supported;
    int gpu_timer_valid;
    bool use_fp16_weights;
    // MLP-dispatch-only CPU timer (dispatch_frame + glFinish)
    // Comparable to ShaderNN std::chrono and ORT System.nanoTime boundaries.
    double mlp_cpu_last_ms;
    double mlp_cpu_mean_ms;
    double mlp_cpu_stddev_ms;
    // Backward-compatible CPU timer with the same boundary as mlp_cpu_*.
    double cpu_last_ms;
    double cpu_mean_ms;
    double cpu_stddev_ms;
    double cpu_min_ms;
    double cpu_max_ms;
    double gpu_last_ms;
    double gpu_mean_ms;
    double gpu_stddev_ms;
    double gpu_min_ms;
    double gpu_max_ms;
    double elapsed_seconds;
    double fps;
    char phase[32];
    char renderer[128];
    char version[64];
    char last_error[256];
} MlpBenchmarkStatus;

MLP_BENCHMARK_API MlpBenchmarkConfig benchmark_default_config(void);
MLP_BENCHMARK_API MlpBenchmarkHandle* benchmark_create(void);
MLP_BENCHMARK_API int benchmark_init(
        MlpBenchmarkHandle* handle,
        AAssetManager* asset_manager,
        const MlpBenchmarkConfig* config);
MLP_BENCHMARK_API void benchmark_resize(
        MlpBenchmarkHandle* handle,
        int width,
        int height);
MLP_BENCHMARK_API int benchmark_draw(MlpBenchmarkHandle* handle);
MLP_BENCHMARK_API int benchmark_run_fps_loop(
        MlpBenchmarkHandle* handle,
        int frames,
        double target_fps);
MLP_BENCHMARK_API const MlpBenchmarkStatus* benchmark_status(const MlpBenchmarkHandle* handle);
MLP_BENCHMARK_API int benchmark_set_output_paths(
        MlpBenchmarkHandle* handle,
        const char* csv_path,
        const char* log_path);
MLP_BENCHMARK_API void benchmark_destroy(MlpBenchmarkHandle* handle);

#ifdef __cplusplus
}  // extern "C"
#endif

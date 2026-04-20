# Android Compute Benchmark

This project is a standalone Android NDK sample for benchmarking a three-layer MLP executed entirely in OpenGL ES compute shaders with SSBO-backed buffers.

## Project Structure

- `app/src/main/cpp/main.cpp`: native benchmark core, JNI bridge, timing, and logging.
- `app/src/main/cpp/benchmark_api.h`: opaque C API for the benchmark runtime.
- `app/src/main/assets/shaders/*.comp`: compute shader stages for `72 -> 256`, `256 -> 256`, and `256 -> 20670`.
- `app/src/main/java/com/graphicsresearch/androidcomputebenchmark/*`: Kotlin activity, `GLSurfaceView`, renderer, and native loader.
- `app/build.gradle.kts`: Android app and CMake/NDK wiring.
- `tools/analyze_latency.py`: offline latency analysis and plotting script.
- `RESEARCH_PURPOSE.md`: paper-ready introduction-purpose section.

## What It Does

- Creates an OpenGL ES 3.1+ context through `GLSurfaceView`.
- Allocates SSBOs for input activations, intermediate activations, output activations, weights, and biases.
- Dispatches three compute shader layers per inference.
- Runs `100` warmup iterations and `1000` measured iterations.
- Measures CPU latency with `std::chrono` plus `glFinish()`.
- Uses `GL_EXT_disjoint_timer_query` when available for GPU-only timing.
- Writes raw per-run latency samples to `/sdcard/inference_log.txt`.
- Writes a summary report to `/sdcard/inference_log.txt.summary.txt`.
- Continues running a real-time dispatch loop and prints FPS to `logcat` once per second.

## Build

Open the `android_compute_benchmark` folder in Android Studio and let it sync, or build with your local Gradle installation:

```powershell
cd D:\GraphicsResearch\android_compute_benchmark
gradle assembleDebug
```

This project targets:

- `compileSdk = 34`
- `targetSdk = 28`
- `minSdk = 24`
- `ABI = arm64-v8a`
- `NDK = 27.0.12077973`

`targetSdk = 28` is intentional for research-side direct logging to `/sdcard/inference_log.txt` on sideloaded builds.

## Install And Run

```powershell
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
adb shell am start -n com.graphicsresearch.androidcomputebenchmark/.BenchmarkActivity
adb logcat | findstr ZD_ECN_Benchmark
```

If the device asks for storage access on older Android versions, grant it so the app can write to `/sdcard/inference_log.txt`.

## Pull Logs

```powershell
adb pull /sdcard/inference_log.txt
adb pull /sdcard/inference_log.txt.summary.txt
```

## Analyze Results

```powershell
cd D:\GraphicsResearch\android_compute_benchmark
py -3.12 tools\analyze_latency.py D:\path\to\inference_log.txt --output-dir plots
```

The script prints summary statistics and saves:

- `plots/latency_over_time.png`
- `plots/latency_histogram.png`

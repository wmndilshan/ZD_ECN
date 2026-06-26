cmake_minimum_required(VERSION 3.22.1)
project(smpl_renderer)
add_library(smpl_renderer SHARED
    smpl_renderer.cpp
    smpl_gpu_pipeline.cpp
    smpl_lbs.cpp
    gl_context.cpp
    backend_zdecn.cpp
    backend_shadernn_style.cpp
    backend_onnx.cpp)
target_compile_features(smpl_renderer PRIVATE cxx_std_17)
target_link_libraries(smpl_renderer android log GLESv3 EGL)

cmake_minimum_required(VERSION 3.22.1)
project(skin_renderer)

add_library(skin_renderer SHARED
    skin_renderer.cpp
    smpl_gpu_pipeline.cpp
    gl_context.cpp)

target_compile_features(skin_renderer PRIVATE cxx_std_17)

target_link_libraries(skin_renderer
    android
    log
    GLESv3
    EGL)

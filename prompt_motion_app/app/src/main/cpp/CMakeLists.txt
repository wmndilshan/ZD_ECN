cmake_minimum_required(VERSION 3.22.1)
project(prompt_motion)

add_library(prompt_motion SHARED
    prompt_motion_renderer.cpp
    smpl_gpu_pipeline.cpp)

target_compile_features(prompt_motion PRIVATE cxx_std_17)

target_link_libraries(prompt_motion
    android
    log
    GLESv3
    EGL)

cmake_minimum_required(VERSION 3.22)

project(android_compute_benchmark LANGUAGES C CXX)

add_library(android_compute_benchmark SHARED
    main.cpp
)

target_compile_features(android_compute_benchmark PRIVATE cxx_std_17)

target_include_directories(android_compute_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library(android-lib android)
find_library(egl-lib EGL)
find_library(gles-lib GLESv3)
find_library(log-lib log)

target_link_libraries(android_compute_benchmark
    ${android-lib}
    ${egl-lib}
    ${gles-lib}
    ${log-lib}
)

target_compile_options(android_compute_benchmark PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

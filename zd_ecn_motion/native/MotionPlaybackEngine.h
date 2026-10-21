#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct Keyframe {
    float time_seconds;
    float root_translation[3];
    float joint_transforms[24][16];
    bool left_foot_planted;
    bool right_foot_planted;
};

class MotionPlaybackEngine {
public:
    MotionPlaybackEngine();
    ~MotionPlaybackEngine();

    bool load_from_json(const std::string& json_str);
    void update(float delta_time);
    void seek(float target_time);
    void play() { m_is_playing = true; }
    void pause() { m_is_playing = false; }
    void set_looping(bool loop) { m_is_looping = loop; }

    float get_current_time() const { return m_current_time; }
    float get_total_duration() const { return m_total_duration; }
    int get_frame_count() const { return static_cast<int>(m_keyframes.size()); }
    bool is_playing() const { return m_is_playing; }

    void get_current_transforms(float* out_matrices) const;
    void update_gpu_ssbo_buffer(uint32_t ssbo_buffer_id);

private:
    std::vector<Keyframe> m_keyframes;
    float m_current_time;
    float m_total_duration;
    int m_current_frame_index;
    bool m_is_playing;
    bool m_is_looping;
    mutable float m_cached_gpu_buffer[24 * 16];

    void interpolate_transforms(int idx_a, int idx_b, float alpha, float* out_matrices) const;
};

extern "C" {
    void* motion_engine_create();
    void motion_engine_destroy(void* handle);
    bool motion_engine_load_json(void* handle, const char* json_str);
    void motion_engine_update(void* handle, float delta_time);
    void motion_engine_seek(void* handle, float target_time);
    void motion_engine_play(void* handle);
    void motion_engine_pause(void* handle);
    float motion_engine_get_time(void* handle);
    float motion_engine_get_duration(void* handle);
    int motion_engine_get_frame_count(void* handle);
    void motion_engine_get_transforms(void* handle, float* out_buffer);
}

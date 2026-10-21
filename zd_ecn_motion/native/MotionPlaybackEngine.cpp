#include "MotionPlaybackEngine.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <cctype>

MotionPlaybackEngine::MotionPlaybackEngine()
    : m_current_time(0.0f),
      m_total_duration(0.0f),
      m_current_frame_index(0),
      m_is_playing(true),
      m_is_looping(true) {
    std::memset(m_cached_gpu_buffer, 0, sizeof(m_cached_gpu_buffer));
}

MotionPlaybackEngine::~MotionPlaybackEngine() {
    m_keyframes.clear();
}

static float extract_float(const std::string& str, size_t start, size_t* out_end = nullptr) {
    while (start < str.size() && (std::isspace(str[start]) || str[start] == ':' || str[start] == ',' || str[start] == '[' || str[start] == ']')) {
        start++;
    }
    size_t end = start;
    while (end < str.size() && (std::isdigit(str[end]) || str[end] == '-' || str[end] == '.' || str[end] == 'e' || str[end] == 'E' || str[end] == '+')) {
        end++;
    }
    if (out_end) *out_end = end;
    if (start == end) return 0.0f;
    try {
        return std::stof(str.substr(start, end - start));
    } catch (...) {
        return 0.0f;
    }
}

bool MotionPlaybackEngine::load_from_json(const std::string& json_str) {
    m_keyframes.clear();
    m_current_time = 0.0f;
    m_current_frame_index = 0;

    size_t frames_pos = json_str.find("\"frames\"");
    if (frames_pos == std::string::npos) return false;

    size_t cur = frames_pos;
    while (true) {
        size_t frame_start = json_str.find("\"frame\"", cur);
        if (frame_start == std::string::npos) break;

        Keyframe kf;
        std::memset(&kf, 0, sizeof(Keyframe));

        // Read time_seconds
        size_t time_pos = json_str.find("\"time_seconds\"", frame_start);
        if (time_pos != std::string::npos) {
            size_t colon = json_str.find(':', time_pos);
            kf.time_seconds = extract_float(json_str, colon + 1);
        }

        // Read root_translation
        size_t trans_pos = json_str.find("\"root_translation\"", frame_start);
        if (trans_pos != std::string::npos) {
            size_t bracket = json_str.find('[', trans_pos);
            size_t p = bracket + 1;
            for (int i = 0; i < 3; ++i) {
                kf.root_translation[i] = extract_float(json_str, p, &p);
            }
        }

        // Read joint_transforms_4x4
        size_t transforms_pos = json_str.find("\"joint_transforms_4x4\"", frame_start);
        if (transforms_pos != std::string::npos) {
            size_t p = json_str.find('[', transforms_pos);
            for (int j = 0; j < 24; ++j) {
                for (int k = 0; k < 16; ++k) {
                    kf.joint_transforms[j][k] = extract_float(json_str, p, &p);
                }
            }
        }

        m_keyframes.push_back(kf);
        cur = frame_start + 7;
    }

    if (m_keyframes.empty()) return false;
    m_total_duration = m_keyframes.back().time_seconds;
    return true;
}

void MotionPlaybackEngine::update(float delta_time) {
    if (!m_is_playing || m_keyframes.empty()) return;

    m_current_time += delta_time;

    if (m_current_time > m_total_duration) {
        if (m_is_looping && m_total_duration > 0.0f) {
            m_current_time = std::fmod(m_current_time, m_total_duration);
        } else {
            m_current_time = m_total_duration;
            m_is_playing = false;
        }
    } else if (m_current_time < 0.0f) {
        m_current_time = 0.0f;
    }

    for (size_t i = 0; i < m_keyframes.size() - 1; ++i) {
        if (m_current_time >= m_keyframes[i].time_seconds && m_current_time <= m_keyframes[i + 1].time_seconds) {
            m_current_frame_index = static_cast<int>(i);
            return;
        }
    }
    m_current_frame_index = static_cast<int>(m_keyframes.size() - 1);
}

void MotionPlaybackEngine::seek(float target_time) {
    m_current_time = std::max(0.0f, std::min(target_time, m_total_duration));
    update(0.0f);
}

void MotionPlaybackEngine::interpolate_transforms(int idx_a, int idx_b, float alpha, float* out_matrices) const {
    const Keyframe& kf_a = m_keyframes[idx_a];
    const Keyframe& kf_b = m_keyframes[idx_b];

    for (int j = 0; j < 24; ++j) {
        for (int k = 0; k < 16; ++k) {
            float val_a = kf_a.joint_transforms[j][k];
            float val_b = kf_b.joint_transforms[j][k];
            out_matrices[j * 16 + k] = (1.0f - alpha) * val_a + alpha * val_b;
        }
    }
}

void MotionPlaybackEngine::get_current_transforms(float* out_matrices) const {
    if (m_keyframes.empty()) {
        std::memset(out_matrices, 0, 24 * 16 * sizeof(float));
        for (int j = 0; j < 24; ++j) {
            out_matrices[j * 16 + 0] = 1.0f;
            out_matrices[j * 16 + 5] = 1.0f;
            out_matrices[j * 16 + 10] = 1.0f;
            out_matrices[j * 16 + 15] = 1.0f;
        }
        return;
    }

    if (m_current_frame_index >= static_cast<int>(m_keyframes.size()) - 1) {
        const Keyframe& kf = m_keyframes.back();
        for (int j = 0; j < 24; ++j) {
            std::memcpy(&out_matrices[j * 16], kf.joint_transforms[j], 16 * sizeof(float));
        }
        return;
    }

    const Keyframe& kf_a = m_keyframes[m_current_frame_index];
    const Keyframe& kf_b = m_keyframes[m_current_frame_index + 1];
    float denom = kf_b.time_seconds - kf_a.time_seconds;
    float alpha = (denom > 1e-5f) ? (m_current_time - kf_a.time_seconds) / denom : 0.0f;
    alpha = std::max(0.0f, std::min(1.0f, alpha));

    interpolate_transforms(m_current_frame_index, m_current_frame_index + 1, alpha, out_matrices);
}

void MotionPlaybackEngine::update_gpu_ssbo_buffer(uint32_t ssbo_buffer_id) {
    get_current_transforms(m_cached_gpu_buffer);
}

extern "C" {
    void* motion_engine_create() {
        return new MotionPlaybackEngine();
    }

    void motion_engine_destroy(void* handle) {
        delete static_cast<MotionPlaybackEngine*>(handle);
    }

    bool motion_engine_load_json(void* handle, const char* json_str) {
        if (!handle || !json_str) return false;
        return static_cast<MotionPlaybackEngine*>(handle)->load_from_json(json_str);
    }

    void motion_engine_update(void* handle, float delta_time) {
        if (handle) static_cast<MotionPlaybackEngine*>(handle)->update(delta_time);
    }

    void motion_engine_seek(void* handle, float target_time) {
        if (handle) static_cast<MotionPlaybackEngine*>(handle)->seek(target_time);
    }

    void motion_engine_play(void* handle) {
        if (handle) static_cast<MotionPlaybackEngine*>(handle)->play();
    }

    void motion_engine_pause(void* handle) {
        if (handle) static_cast<MotionPlaybackEngine*>(handle)->pause();
    }

    float motion_engine_get_time(void* handle) {
        return handle ? static_cast<MotionPlaybackEngine*>(handle)->get_current_time() : 0.0f;
    }

    float motion_engine_get_duration(void* handle) {
        return handle ? static_cast<MotionPlaybackEngine*>(handle)->get_total_duration() : 0.0f;
    }

    int motion_engine_get_frame_count(void* handle) {
        return handle ? static_cast<MotionPlaybackEngine*>(handle)->get_frame_count() : 0;
    }

    void motion_engine_get_transforms(void* handle, float* out_buffer) {
        if (handle && out_buffer) {
            static_cast<MotionPlaybackEngine*>(handle)->get_current_transforms(out_buffer);
        }
    }
}

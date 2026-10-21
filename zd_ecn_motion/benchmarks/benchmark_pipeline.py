"""
Performance and Quality Benchmarking Suite for ZD-ECN Motion Pipeline.
Measures latency, throughput, memory overhead, and biomechanical quality constraints.
"""
import time
import json
import resource
import numpy as np
from zd_ecn_motion.planner.mock_planner import MockMotionPlanner
from zd_ecn_motion.generator.motion_generator import DeterministicMotionGenerator
from zd_ecn_motion.native.native_bridge import NativeMotionPlaybackBridge


def run_benchmark():
    print("=" * 70)
    print(" ZD-ECN TEXT-TO-MOTION OPTIMIZED PERFORMANCE & QUALITY BENCHMARK ")
    print("=" * 70)

    prompt = "Walk forward, turn left, jump, land, and return to idle"
    dance_prompt = "Perform traditional dance sequence with mandiya stance, rhythmic stepping, vattam turn, and samapthiya freeze"
    planner = MockMotionPlanner(default_fps=20)
    generator = DeterministicMotionGenerator()
    bridge = NativeMotionPlaybackBridge()

    # 1. Motion Plan Generation Latency
    plan_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        plan = planner.plan(prompt)
        t1 = time.perf_counter()
        plan_times.append((t1 - t0) * 1000.0)

    plan_mean = float(np.mean(plan_times))
    plan_p95 = float(np.percentile(plan_times, 95))
    print(f"[1. Motion Planner Latency (100 runs)]")
    print(f"    Mean: {plan_mean:.3f} ms | P95: {plan_p95:.3f} ms | Min: {min(plan_times):.3f} ms")

    # 2. Frame Generation Latency & Throughput
    frame_gen_times = []
    total_frames = 0
    motion_payload = None

    for _ in range(30):
        t0 = time.perf_counter()
        motion_payload = generator.generate(plan)
        t1 = time.perf_counter()
        frame_gen_times.append((t1 - t0) * 1000.0)
        total_frames = motion_payload["metadata"]["total_frames"]

    gen_mean = float(np.mean(frame_gen_times))
    per_frame_ms = gen_mean / total_frames
    fps_throughput = 1000.0 / per_frame_ms

    print(f"\n[2. Closed-Loop IK Frame Generator (30 runs, {total_frames} frames/run)]")
    print(f"    Total Gen Latency: {gen_mean:.2f} ms")
    print(f"    Per-Frame Latency: {per_frame_ms:.4f} ms/frame")
    print(f"    Generation Speed:  {fps_throughput:.1f} frames/sec ({fps_throughput / 20.0:.1f}x real-time)")

    # 3. Memory Footprint
    mem_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json_bytes = len(json.dumps(motion_payload).encode('utf-8'))
    print(f"\n[3. Memory Utilization & Payload Size]")
    print(f"    Process Resident Memory (RSS): {mem_rss_kb / 1024.0:.2f} MB")
    print(f"    Motion JSON Payload ({total_frames} frames): {json_bytes / 1024.0:.1f} KB")

    # 4. Native C++ Playback Throughput
    json_str = json.dumps(motion_payload)
    bridge.load_json(json_str)

    update_times_us = []
    for _ in range(1000):
        t0 = time.perf_counter()
        bridge.update(0.01667)
        _ = bridge.get_current_transforms()
        t1 = time.perf_counter()
        update_times_us.append((t1 - t0) * 1e6)

    update_mean_us = float(np.mean(update_times_us))
    native_fps = 1e6 / update_mean_us
    print(f"\n[4. Native C++ Playback Throughput (1000 steps)]")
    print(f"    Per-Step Interpolation & SSBO Latency: {update_mean_us:.2f} µs ({update_mean_us / 1000.0:.4f} ms)")
    print(f"    Native Timeline Throughput:            {native_fps:.1f} updates/sec")

    # 5. Biomechanical & Geometric Quality Verification
    dance_plan = planner.plan(dance_prompt)
    dance_payload = generator.generate(dance_plan)

    frames = motion_payload["frames"]
    root_vels = []
    for i in range(1, len(frames)):
        dt = frames[i]["time_seconds"] - frames[i - 1]["time_seconds"]
        p0 = np.array(frames[i - 1]["root_translation"])
        p1 = np.array(frames[i]["root_translation"])
        root_vels.append(float(np.linalg.norm(p1 - p0)) / dt)

    max_vel = max(root_vels)
    mean_vel = float(np.mean(root_vels))
    qm_loco = motion_payload["quality_metrics"]
    qm_dance = dance_payload["quality_metrics"]

    print(f"\n[5. Biomechanical & Geometric Quality Verification]")
    print(f"    Locomotion Max Root Velocity:      {max_vel:.2f} m/s (expected <= 4.5 m/s)")
    print(f"    Locomotion Mean Root Velocity:     {mean_vel:.2f} m/s")
    print(f"    Locomotion Ground Penetration:     {qm_loco['max_ground_penetration_m']:.4f} m (0.0000 m ideal)")
    print(f"    Locomotion Stance Foot Slip:       {qm_loco['max_foot_slip_velocity_m_per_s']:.4f} m/s (0.0000 m/s ideal)")
    print(f"    Locomotion Joint Limit Clampings:  {qm_loco['joint_limit_clampings']}")
    print(f"    Trad. Dance Ground Penetration:    {qm_dance['max_ground_penetration_m']:.4f} m (0.0000 m ideal)")
    print(f"    Trad. Dance Stance Foot Slip:      {qm_dance['max_foot_slip_velocity_m_per_s']:.4f} m/s (0.0000 m/s ideal)")
    print(f"    Trad. Dance Joint Limit Clampings: {qm_dance['joint_limit_clampings']}")
    print("=" * 70)


if __name__ == '__main__':
    run_benchmark()

"""
Python ctypes wrapper for libzd_ecn_playback.so native C++ playback engine.
"""
import ctypes
import os
import numpy as np
from typing import Optional


class NativeMotionPlaybackBridge:
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__), "libzd_ecn_playback.so")

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native library not found at: {lib_path}. Run 'make' in native/ first.")

        self.lib = ctypes.CDLL(lib_path)

        # Setup ctypes signatures
        self.lib.motion_engine_create.restype = ctypes.c_void_p
        self.lib.motion_engine_create.argtypes = []

        self.lib.motion_engine_destroy.restype = None
        self.lib.motion_engine_destroy.argtypes = [ctypes.c_void_p]

        self.lib.motion_engine_load_json.restype = ctypes.c_bool
        self.lib.motion_engine_load_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.lib.motion_engine_update.restype = None
        self.lib.motion_engine_update.argtypes = [ctypes.c_void_p, ctypes.c_float]

        self.lib.motion_engine_seek.restype = None
        self.lib.motion_engine_seek.argtypes = [ctypes.c_void_p, ctypes.c_float]

        self.lib.motion_engine_play.restype = None
        self.lib.motion_engine_play.argtypes = [ctypes.c_void_p]

        self.lib.motion_engine_pause.restype = None
        self.lib.motion_engine_pause.argtypes = [ctypes.c_void_p]

        self.lib.motion_engine_get_time.restype = ctypes.c_float
        self.lib.motion_engine_get_time.argtypes = [ctypes.c_void_p]

        self.lib.motion_engine_get_duration.restype = ctypes.c_float
        self.lib.motion_engine_get_duration.argtypes = [ctypes.c_void_p]

        self.lib.motion_engine_get_frame_count.restype = ctypes.c_int
        self.lib.motion_engine_get_frame_count.argtypes = [ctypes.c_void_p]

        self.lib.motion_engine_get_transforms.restype = None
        self.lib.motion_engine_get_transforms.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]

        self.handle = self.lib.motion_engine_create()

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.motion_engine_destroy(self.handle)
            self.handle = None

    def load_json(self, json_string: str) -> bool:
        b_str = json_string.encode('utf-8')
        return bool(self.lib.motion_engine_load_json(self.handle, b_str))

    def update(self, delta_time: float):
        self.lib.motion_engine_update(self.handle, ctypes.c_float(delta_time))

    def seek(self, target_time: float):
        self.lib.motion_engine_seek(self.handle, ctypes.c_float(target_time))

    def play(self):
        self.lib.motion_engine_play(self.handle)

    def pause(self):
        self.lib.motion_engine_pause(self.handle)

    def get_time(self) -> float:
        return float(self.lib.motion_engine_get_time(self.handle))

    def get_duration(self) -> float:
        return float(self.lib.motion_engine_get_duration(self.handle))

    def get_frame_count(self) -> int:
        return int(self.lib.motion_engine_get_frame_count(self.handle))

    def get_current_transforms(self) -> np.ndarray:
        buf = (ctypes.c_float * (24 * 16))()
        self.lib.motion_engine_get_transforms(self.handle, buf)
        arr = np.frombuffer(buf, dtype=np.float32).reshape((24, 4, 4))
        return np.copy(arr)

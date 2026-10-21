package com.graphicsresearch.promptmotion

import android.content.res.AssetManager

/** JNI surface of libprompt_motion.so. All calls must run on the GL thread. */
object MotionRendererNative {
    init { System.loadLibrary("prompt_motion") }

    // Lifecycle
    external fun nativeInit(assetManager: AssetManager)
    external fun nativeResize(w: Int, h: Int)
    external fun nativeDraw()

    // Motion clip + playback
    external fun nativeLoadMotion(aa: FloatArray, rot6d: FloatArray, trans: FloatArray,
                                  restPelvis: FloatArray, fps: Int): Boolean
    external fun nativeSetPlaying(playing: Boolean)
    external fun nativeSetSpeed(speed: Float)
    external fun nativeSeek(seconds: Float)
    external fun nativeSetFixedFrame(frame: Int)         // -1 = follow playback clock
    external fun nativeGetPlaybackTime(): Float

    // Rendering controls
    external fun nativeSetBackend(backend: Int)         // 0=ZD-ECN 1=ONNX CPU 2=LBS only
    external fun nativeSetCamera(yaw: Float, pitch: Float, zoom: Float)
    external fun nativeUploadTexture(index: Int, rgbaBytes: ByteArray, width: Int, height: Int)
    external fun nativeSetTexture(index: Int)
    external fun nativeSetUseTexture(useTexture: Boolean)
    external fun nativeSetLighting(wrap: Float, spec: Float, lightAngle: Float)

    // ONNX CPU backend sync
    external fun nativeUploadOnnxDelta(delta: FloatArray)
    external fun nativeGetCurrentFrame(): Int
    external fun nativeGetCurrentRot6d(): FloatArray?

    // Benchmark / status
    external fun nativeRunFullPipelineBenchmark(warmup: Int, iterations: Int): String
    external fun nativeGetStatus(): String
    external fun nativeGetZdecnMs(): Float
}

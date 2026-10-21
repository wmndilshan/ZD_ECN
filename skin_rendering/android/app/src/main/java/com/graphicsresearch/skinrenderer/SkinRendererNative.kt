package com.graphicsresearch.skinrenderer

import android.content.res.AssetManager

object SkinRendererNative {
    init { System.loadLibrary("skin_renderer") }

    // Lifecycle
    external fun nativeInit(assetManager: AssetManager)
    external fun nativeResize(w: Int, h: Int)
    external fun nativeDraw()

    // Per-frame control
    external fun nativeSetBackend(backend: Int)         // 0=ZD-ECN 1=ONNX CPU 2=LBS only
    external fun nativeSetTier(tier: Int)               // 0=natural 1=moderate 2=extreme
    external fun nativeSetFixedFrame(frame: Int, enabled: Boolean)
    external fun nativeSetCamera(yaw: Float, pitch: Float, zoom: Float)

    // Texture
    external fun nativeUploadTexture(index: Int, rgbaBytes: ByteArray, width: Int, height: Int)
    external fun nativeSetTexture(index: Int)           // 0=procedural 1=Meshcapade crop
    external fun nativeSetUseTexture(useTexture: Boolean)

    // Lighting
    external fun nativeSetLighting(wrap: Float, spec: Float, lightAngle: Float)

    // ONNX (called from OnnxSkinBackend)
    external fun nativeUploadOnnxDelta(delta: FloatArray)
    external fun nativeGetCurrentFrame(): Int           // last rendered pose frame (-1 before first draw)
    external fun nativeGetCurrentRot6d(): FloatArray?   // rot6d[144] of the current pose frame

    // Benchmark (must run on the GL thread)
    external fun nativeRunFullPipelineBenchmark(warmup: Int, iterations: Int): String

    // Status
    external fun nativeGetStatus(): String
    external fun nativeGetZdecnMs(): Float
}

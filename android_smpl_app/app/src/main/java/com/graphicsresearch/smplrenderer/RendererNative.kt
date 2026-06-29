package com.graphicsresearch.smplrenderer

import android.content.res.AssetManager

object RendererNative {
    init { System.loadLibrary("smpl_renderer") }
    external fun nativeInit(assetManager: AssetManager)
    external fun nativeResize(width: Int, height: Int)
    external fun nativeDraw()
    external fun nativeSetBackend(backend: Int, tier: Int)
    external fun nativeSetFixedFrame(frame: Int, enabled: Boolean)
    external fun nativeSetCamera(yaw: Float, pitch: Float, zoom: Float)
    external fun nativeUploadExternalDelta(delta: FloatArray): Boolean
    external fun nativeRunCombinedBenchmarkSession(session: Int, warmup: Int, iterations: Int): String
    external fun nativeStatus(): String
    external fun nativeRelease()
}

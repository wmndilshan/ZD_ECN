package com.graphicsresearch.androidcomputebenchmark

import android.content.Context
import android.opengl.GLSurfaceView

class BenchmarkSurfaceView(
    context: Context,
    logPath: String,
) : GLSurfaceView(context) {
    private val benchmarkRenderer = BenchmarkRenderer(context.applicationContext, logPath)

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 0, 0)
        preserveEGLContextOnPause = true
        setRenderer(benchmarkRenderer)
        renderMode = RENDERMODE_CONTINUOUSLY
        isFocusable = true
        isFocusableInTouchMode = true
        requestFocus()
    }

    fun releaseNativeResources() {
        queueEvent { BenchmarkNative.shutdown() }
    }
}

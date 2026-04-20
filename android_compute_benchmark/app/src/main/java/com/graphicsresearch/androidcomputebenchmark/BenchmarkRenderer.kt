package com.graphicsresearch.androidcomputebenchmark

import android.content.Context
import android.opengl.GLES31
import android.opengl.GLSurfaceView
import android.util.Log
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class BenchmarkRenderer(
    private val context: Context,
    private val logPath: String,
) : GLSurfaceView.Renderer {
    private var nativeReady = false

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        nativeReady = BenchmarkNative.initialize(context.assets, logPath)
        GLES31.glClearColor(0f, 0f, 0f, 1f)
        if (!nativeReady) {
            Log.w(TAG, "Native benchmark backend is unavailable; rendering stays on the Java side.")
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES31.glViewport(0, 0, width, height)
        if (nativeReady) {
            BenchmarkNative.resize(width, height)
        }
    }

    override fun onDrawFrame(gl: GL10?) {
        if (nativeReady) {
            BenchmarkNative.renderFrame()
        } else {
            GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT)
        }
    }

    private companion object {
        private const val TAG = "BenchmarkRenderer"
    }
}

package com.graphicsresearch.androidcomputebenchmark

import android.content.res.AssetManager
import android.util.Log

object BenchmarkNative {
    private const val TAG = "BenchmarkNative"
    private const val LIBRARY_NAME = "android_compute_benchmark"

    @Volatile
    private var libraryLoaded = false

    @Volatile
    private var initialized = false

    init {
        libraryLoaded = runCatching {
            System.loadLibrary(LIBRARY_NAME)
            true
        }.getOrElse { throwable ->
            Log.e(TAG, "Failed to load native library $LIBRARY_NAME", throwable)
            false
        }
    }

    fun initialize(assetManager: AssetManager, logPath: String): Boolean {
        if (!libraryLoaded) {
            return false
        }
        if (initialized) {
            return true
        }

        return runCatching {
            val ok = nativeInitialize(assetManager, logPath)
            initialized = ok
            ok
        }.getOrElse { throwable ->
            Log.e(TAG, "Failed to initialize native benchmark backend", throwable)
            false
        }
    }

    fun resize(width: Int, height: Int) {
        if (!initialized) {
            return
        }

        runCatching {
            nativeResize(width, height)
        }.onFailure { throwable ->
            Log.e(TAG, "Failed to resize native benchmark backend", throwable)
        }
    }

    fun renderFrame() {
        if (!initialized) {
            return
        }

        runCatching {
            nativeRenderFrame()
        }.onFailure { throwable ->
            Log.e(TAG, "Failed to render native benchmark frame", throwable)
        }
    }

    fun shutdown() {
        if (!initialized) {
            return
        }

        runCatching {
            nativeShutdown()
        }.onFailure { throwable ->
            Log.e(TAG, "Failed to shut down native benchmark backend", throwable)
        }
        initialized = false
    }

    private external fun nativeInitialize(assetManager: AssetManager, logPath: String): Boolean
    private external fun nativeResize(width: Int, height: Int)
    private external fun nativeRenderFrame()
    private external fun nativeShutdown()
}

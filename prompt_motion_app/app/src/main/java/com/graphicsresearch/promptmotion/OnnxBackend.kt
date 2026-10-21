package com.graphicsresearch.promptmotion

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/** ZD-ECN fp16 ONNX model on ONNX Runtime CPU (reference backend 1). */
class OnnxBackend(context: Context) {

    companion object {
        private const val TAG = "OnnxBackend"
        private const val ASSET_NAME = "zdecn_fp16.onnx"
        private const val INPUT_SIZE = 144
        private const val OUTPUT_SIZE = 20670
    }

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    init {
        try {
            val bytes = context.assets.open(ASSET_NAME).readBytes()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            session = env.createSession(bytes, opts)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX init failed: ${e.message}")
        }
    }

    fun infer(rot6d144: FloatArray): FloatArray? {
        val sess = session ?: return null
        return try {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(rot6d144), longArrayOf(1, INPUT_SIZE.toLong())).use { t ->
                sess.run(mapOf(sess.inputNames.first() to t)).use { r ->
                    // r.first().value is the OnnxTensor itself (Map.Entry value)
                    val out = FloatArray(OUTPUT_SIZE)
                    (r.first().value as OnnxTensor).floatBuffer.get(out)
                    out
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "ONNX infer failed: ${e.message}")
            null
        }
    }

    fun close() { session?.close(); session = null }
}

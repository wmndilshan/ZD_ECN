package com.graphicsresearch.skinrenderer

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

class OnnxSkinBackend(context: Context) {

    companion object {
        private const val TAG = "OnnxSkinBackend"
        private const val ASSET_NAME = "zdecn_fp16.onnx"
        private const val INPUT_SIZE  = 144
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
            Log.i(TAG, "ONNX session created from $ASSET_NAME")
        } catch (e: Exception) {
            Log.e(TAG, "ONNX init failed: ${e.message}")
        }
    }

    fun infer(rot6d144: FloatArray): FloatArray? {
        val sess = session ?: return null
        return try {
            val buf = FloatBuffer.wrap(rot6d144)
            val shape = longArrayOf(1, INPUT_SIZE.toLong())
            val tensor = OnnxTensor.createTensor(env, buf, shape)
            tensor.use { t ->
                val results = sess.run(mapOf(sess.inputNames.first() to t))
                results.use { r ->
                    // r.first().value is the OnnxTensor itself (Map.Entry value)
                    val outTensor = r.first().value as OnnxTensor
                    val fb = outTensor.floatBuffer
                    val out = FloatArray(OUTPUT_SIZE)
                    fb.get(out)
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

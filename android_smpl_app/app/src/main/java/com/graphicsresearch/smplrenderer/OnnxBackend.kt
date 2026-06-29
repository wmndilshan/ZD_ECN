package com.graphicsresearch.smplrenderer

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class OnnxBackend(context: Context) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val sessionNnapi: OrtSession?
    private val inputName: String
    val nnapiAvailable: Boolean get() = sessionNnapi != null

    init {
        val bytes = context.assets.open("zdecn_fp16.onnx").use { it.readBytes() }
        session = env.createSession(bytes, OrtSession.SessionOptions())
        sessionNnapi = try {
            val options = OrtSession.SessionOptions()
            options.addNnapi()
            env.createSession(bytes, options)
        } catch (error: Throwable) {
            Log.w("OnnxBackend", "NNAPI session unavailable; CPU fallback will be used", error)
            null
        }
        inputName = session.inputNames.firstOrNull() ?: "input"
        Log.i("OnnxBackend", "sessions initialized: cpu=true nnapi=${sessionNnapi != null}")
    }

    fun infer(rot6d: FloatArray, useNnapi: Boolean): FloatArray {
        require(rot6d.size == 144) { "rot6d must have 144 floats" }
        val selected = if (useNnapi && sessionNnapi != null) sessionNnapi else session
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(rot6d), longArrayOf(1, 144))
        tensor.use {
            selected.run(mapOf(inputName to tensor)).use { result ->
                val value = result[0].value
                return when (value) {
                    is Array<*> -> (value[0] as FloatArray).copyOf()
                    is FloatArray -> value.copyOf()
                    else -> error("Unexpected ONNX output type: ${value::class.java.name}")
                }
            }
        }
    }

    override fun close() {
        session.close()
        sessionNnapi?.close()
    }
}

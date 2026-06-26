package com.graphicsresearch.smplrenderer

import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class RenderSurfaceView(context: Context) : GLSurfaceView(context) {
    private val renderer: SmplRenderer
    private val scaleDetector: ScaleGestureDetector
    private var cameraYaw = 0.0f
    private var cameraPitch = -0.10f
    private var cameraZoom = 1.0f
    private var lastX = 0.0f
    private var lastY = 0.0f
    private var dragging = false

    init {
        setEGLContextClientVersion(3)
        renderer = SmplRenderer(context)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
        scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                cameraZoom = (cameraZoom / detector.scaleFactor).coerceIn(0.70f, 1.55f)
                pushCamera()
                return true
            }
        })
        setOnTouchListener { _, event -> handleTouch(event) }
    }

    fun setBackend(backend: Backend, tier: PoseTier) {
        renderer.backend = backend
        renderer.tier = tier
        queueEvent { RendererNative.nativeSetBackend(backend.ordinal, tier.ordinal) }
    }

    fun setFixedFrame(frame: Int?, enabled: Boolean) {
        renderer.fixedFrame = if (enabled) frame else null
        queueEvent { RendererNative.nativeSetFixedFrame(frame ?: 0, enabled) }
    }

    fun statusText(): String = renderer.statusText

    fun resetCamera() {
        cameraYaw = 0.0f
        cameraPitch = -0.10f
        cameraZoom = 1.0f
        pushCamera()
    }

    fun releaseNativeResources() {
        queueEvent {
            renderer.closeOnGlThread()
            RendererNative.nativeRelease()
        }
    }

    private fun handleTouch(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                dragging = true
                lastX = event.x
                lastY = event.y
            }
            MotionEvent.ACTION_POINTER_DOWN -> dragging = false
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount == 1 && dragging && !scaleDetector.isInProgress) {
                    val dx = event.x - lastX
                    val dy = event.y - lastY
                    cameraYaw += dx * 0.008f
                    cameraPitch = (cameraPitch + dy * 0.004f).coerceIn(-0.55f, 0.35f)
                    lastX = event.x
                    lastY = event.y
                    pushCamera()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> dragging = false
        }
        return true
    }

    private fun pushCamera() {
        queueEvent { RendererNative.nativeSetCamera(cameraYaw, cameraPitch, cameraZoom) }
    }
}

class SmplRenderer(private val context: Context) : GLSurfaceView.Renderer {
    @Volatile var backend: Backend = Backend.ZDECN_NATIVE
    @Volatile var tier: PoseTier = PoseTier.NATURAL
    @Volatile var fixedFrame: Int? = null
    @Volatile var statusText: String = "starting"
    private var frames = 0
    private var lastNs = System.nanoTime()
    private var fps = 0.0
    private var startNs = System.nanoTime()
    private var onnxBackend: OnnxBackend? = null
    private lateinit var rot6dFrames: FloatArray
    private var lastOnnxFrame = -1
    private var lastOnnxBackend = Backend.ZDECN_NATIVE

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        rot6dFrames = loadFloatAsset("animation/animation_poses_rot6d_f32.bin")
        onnxBackend = OnnxBackend(context)
        RendererNative.nativeInit(context.assets)
        RendererNative.nativeSetBackend(backend.ordinal, tier.ordinal)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        RendererNative.nativeResize(width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        RendererNative.nativeSetBackend(backend.ordinal, tier.ordinal)
        if (backend == Backend.ONNX_CPU || backend == Backend.ONNX_NNAPI) {
            val frame = currentFrame()
            if (fixedFrame == null) {
                RendererNative.nativeSetFixedFrame(frame, true)
            }
            if (frame != lastOnnxFrame || backend != lastOnnxBackend) {
                val input = FloatArray(144)
                System.arraycopy(rot6dFrames, frame * 144, input, 0, 144)
                val delta = onnxBackend?.infer(input, backend == Backend.ONNX_NNAPI)
                if (delta != null) {
                    RendererNative.nativeUploadExternalDelta(delta)
                }
                lastOnnxFrame = frame
                lastOnnxBackend = backend
            }
        } else if (fixedFrame == null) {
            RendererNative.nativeSetFixedFrame(0, false)
        }
        RendererNative.nativeDraw()
        frames++
        val now = System.nanoTime()
        val dt = now - lastNs
        if (dt > 1_000_000_000L) {
            fps = frames * 1e9 / dt.toDouble()
            frames = 0
            lastNs = now
            statusText = "FPS: %.1f | GPU: pending | %s".format(fps, RendererNative.nativeStatus())
        }
    }

    fun closeOnGlThread() {
        onnxBackend?.close()
        onnxBackend = null
    }

    private fun currentFrame(): Int {
        fixedFrame?.let { return it.coerceIn(0, 599) }
        val elapsed = (System.nanoTime() - startNs) / 1_000_000_000.0
        val base = when (tier) {
            PoseTier.NATURAL -> 0
            PoseTier.MODERATE -> 200
            PoseTier.EXTREME -> 400
        }
        return base + ((elapsed * 10.0).toInt() % 200)
    }

    private fun loadFloatAsset(path: String): FloatArray {
        val bytes = context.assets.open(path).use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }
}

enum class Backend(val label: String) {
    ZDECN_NATIVE("ZD-ECN native"),
    SHADERNN_STYLE("ShaderNN-style"),
    ONNX_CPU("ONNX CPU"),
    ONNX_NNAPI("ONNX NNAPI"),
    LBS_ONLY("LBS only"),
}

enum class PoseTier(val label: String) {
    NATURAL("Natural"), MODERATE("Moderate"), EXTREME("Extreme")
}

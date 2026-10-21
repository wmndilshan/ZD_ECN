package com.graphicsresearch.skinrenderer

import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.hypot

class SkinRenderSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: android.util.AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    private val renderer = SkinGLRenderer()

    /** Frames rendered since the last [takeFrameCount] call — drives the FPS overlay. */
    private val frameCounter = AtomicInteger(0)

    /** Invoked on the GL thread after every rendered frame (used for ONNX pose sync). */
    @Volatile var onFrameDrawn: (() -> Unit)? = null

    fun takeFrameCount(): Int = frameCounter.getAndSet(0)

    // Touch tracking for orbit + pinch-zoom
    private var prevX = 0f
    private var prevY = 0f
    private var prevSpan = 0f
    private var pointerCount = 0

    private var yaw   = 0.0f
    private var pitch = -0.10f
    private var zoom  = 1.0f

    init {
        setEGLContextClientVersion(3)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setBackend(b: Int)           = queueEvent { SkinRendererNative.nativeSetBackend(b) }
    fun setTier(t: Int)              = queueEvent { SkinRendererNative.nativeSetTier(t) }
    fun setTexture(idx: Int)         = queueEvent { SkinRendererNative.nativeSetTexture(idx) }
    fun setUseTexture(use: Boolean)  = queueEvent { SkinRendererNative.nativeSetUseTexture(use) }
    fun setLighting(w: Float, s: Float, angle: Float) =
        queueEvent { SkinRendererNative.nativeSetLighting(w, s, angle) }
    fun setFixedFrame(f: Int, on: Boolean) =
        queueEvent { SkinRendererNative.nativeSetFixedFrame(f, on) }

    override fun onTouchEvent(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                prevX = ev.x; prevY = ev.y; pointerCount = 1
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                pointerCount = ev.pointerCount
                if (ev.pointerCount == 2)
                    prevSpan = hypot(ev.getX(0)-ev.getX(1), ev.getY(0)-ev.getY(1))
            }
            MotionEvent.ACTION_MOVE -> {
                if (pointerCount == 1) {
                    yaw   += (ev.x - prevX) * 0.005f
                    pitch += (ev.y - prevY) * 0.005f
                    prevX = ev.x; prevY = ev.y
                    updateCamera()
                } else if (ev.pointerCount >= 2) {
                    val span = hypot(ev.getX(0)-ev.getX(1), ev.getY(0)-ev.getY(1))
                    zoom *= prevSpan / span.coerceAtLeast(1f)
                    prevSpan = span
                    updateCamera()
                }
            }
            MotionEvent.ACTION_UP,
            MotionEvent.ACTION_POINTER_UP -> { pointerCount = maxOf(0, pointerCount - 1) }
        }
        return true
    }

    private fun updateCamera() {
        val y = yaw; val p = pitch; val z = zoom
        queueEvent { SkinRendererNative.nativeSetCamera(y, p, z) }
    }

    inner class SkinGLRenderer : Renderer {
        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            SkinRendererNative.nativeInit(context.assets)
        }
        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            SkinRendererNative.nativeResize(width, height)
        }
        override fun onDrawFrame(gl: GL10?) {
            SkinRendererNative.nativeDraw()
            frameCounter.incrementAndGet()
            onFrameDrawn?.invoke()
        }
    }
}

package com.graphicsresearch.promptmotion

import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.hypot

class MotionSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: android.util.AttributeSet? = null,
) : GLSurfaceView(context, attrs) {

    private val frameCounter = AtomicInteger(0)

    /** Invoked on the GL thread after every rendered frame (ONNX pose sync). */
    @Volatile var onFrameDrawn: (() -> Unit)? = null

    /** Invoked on the GL thread after (re)initialisation — clip and textures must be re-sent. */
    @Volatile var onContextCreated: (() -> Unit)? = null

    fun takeFrameCount(): Int = frameCounter.getAndSet(0)

    private var prevX = 0f
    private var prevY = 0f
    private var prevSpan = 0f
    private var pointerCount = 0
    private var yaw = 0.35f
    private var pitch = 0.0f
    private var zoom = 1.0f

    init {
        setEGLContextClientVersion(3)
        setRenderer(object : Renderer {
            override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
                MotionRendererNative.nativeInit(context.assets)
                MotionRendererNative.nativeSetCamera(yaw, pitch, zoom)
                onContextCreated?.invoke()
            }
            override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
                MotionRendererNative.nativeResize(width, height)
            }
            override fun onDrawFrame(gl: GL10?) {
                MotionRendererNative.nativeDraw()
                frameCounter.incrementAndGet()
                onFrameDrawn?.invoke()
            }
        })
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    override fun onTouchEvent(ev: MotionEvent): Boolean {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> { prevX = ev.x; prevY = ev.y; pointerCount = 1 }
            MotionEvent.ACTION_POINTER_DOWN -> {
                pointerCount = ev.pointerCount
                if (ev.pointerCount == 2) prevSpan = hypot(ev.getX(0) - ev.getX(1), ev.getY(0) - ev.getY(1))
            }
            MotionEvent.ACTION_MOVE -> {
                if (pointerCount == 1) {
                    yaw -= (ev.x - prevX) * 0.006f
                    pitch = (pitch + (ev.y - prevY) * 0.004f).coerceIn(-0.10f, 0.9f)
                    prevX = ev.x; prevY = ev.y
                } else if (ev.pointerCount >= 2) {
                    val span = hypot(ev.getX(0) - ev.getX(1), ev.getY(0) - ev.getY(1))
                    zoom = (zoom * prevSpan / span.coerceAtLeast(1f)).coerceIn(0.45f, 2.2f)
                    prevSpan = span
                }
                val y = yaw; val p = pitch; val z = zoom
                queueEvent { MotionRendererNative.nativeSetCamera(y, p, z) }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> pointerCount = maxOf(0, pointerCount - 1)
        }
        return true
    }
}

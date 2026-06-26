package com.graphicsresearch.smplrenderer

import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

open class MainActivity : AppCompatActivity() {
    private lateinit var surface: RenderSurfaceView
    private lateinit var overlay: TextView
    private lateinit var poseLabel: TextView
    private lateinit var playbackButton: Button
    private var backend = Backend.ZDECN_NATIVE
    private var tier = PoseTier.NATURAL
    private var fixedFrame: Int? = null
    private var paused = false
    private val handler = Handler(Looper.getMainLooper())
    private val updater = object : Runnable {
        override fun run() {
            overlay.text = buildString {
                appendLine("${backend.label} · ${tier.label} · ${if (paused) "paused" else "animated"}")
                append(conciseStatus(surface.statusText()))
            }
            handler.postDelayed(this, 1000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        hideBars()

        backend = Backend.entries.getOrElse(intent.getIntExtra("backend", Backend.ZDECN_NATIVE.ordinal)) { Backend.ZDECN_NATIVE }
        tier = PoseTier.entries.getOrElse(intent.getIntExtra("tier", PoseTier.NATURAL.ordinal)) { PoseTier.NATURAL }
        if (intent.hasExtra("frame")) {
            fixedFrame = intent.getIntExtra("frame", 0).coerceIn(0, 599)
            paused = true
        }

        surface = RenderSurfaceView(this)
        overlay = statusOverlay()
        poseLabel = smallText("Pose tier: ${tier.label}")
        playbackButton = actionButton(if (paused) "Play" else "Pause")

        val root = FrameLayout(this)
        root.addView(surface, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(topHero(), FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        root.addView(bottomPanel(), FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
        setContentView(root)

        surface.setBackend(backend, tier)
        fixedFrame?.let { surface.setFixedFrame(it, true) }
    }

    private fun topHero(): View = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER_HORIZONTAL
        setPadding(dp(18), dp(28), dp(18), dp(6))
        addView(TextView(context).apply {
            text = "ZD-ECN Character"
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 22f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            letterSpacing = 0.02f
        })
        addView(TextView(context).apply {
            text = "Real SMPL mesh · GPU LBS · shader correction"
            setTextColor(0xCCD7E6FF.toInt())
            textSize = 12f
            gravity = Gravity.CENTER
            setPadding(0, dp(3), 0, 0)
        })
    }

    private fun bottomPanel(): View = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(16), dp(10), dp(16), dp(14))
        background = rounded(0xE60B1220.toInt(), dp(24).toFloat())
        elevation = dp(12).toFloat()

        addView(sectionLabel("Correction Backend"))
        addView(radioGroup(listOf("Native", "ShaderNN", "ONNX CPU", "NNAPI", "LBS"), backend.ordinal) { index ->
            backend = Backend.entries[index]
            surface.setBackend(backend, tier)
            applyPausedFrame()
        })

        addView(sectionLabel("Character Motion"))
        addView(poseLabel)
        addView(radioGroup(PoseTier.entries.map { it.label }, tier.ordinal) { index ->
            tier = PoseTier.entries[index]
            poseLabel.text = "Pose tier: ${tier.label}"
            surface.setBackend(backend, tier)
            applyPausedFrame(resetToTier = true)
        })

        addView(controlRow())
        addView(this@MainActivity.overlay)
    }

    private fun controlRow(): View = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER
        setPadding(0, dp(8), 0, dp(8))
        addView(playbackButton.apply {
            setOnClickListener { togglePlayback() }
        }, LinearLayout.LayoutParams(0, dp(42), 1f).apply { marginEnd = dp(8) })
        addView(actionButton("Reset View").apply {
            setOnClickListener { surface.resetCamera() }
        }, LinearLayout.LayoutParams(0, dp(42), 1f).apply { marginStart = dp(8) })
    }

    private fun togglePlayback() {
        paused = !paused
        playbackButton.text = if (paused) "Play" else "Pause"
        applyPausedFrame(resetToTier = paused)
        if (!paused) {
            fixedFrame = null
            surface.setFixedFrame(null, false)
        }
    }

    private fun applyPausedFrame(resetToTier: Boolean = false) {
        if (!paused) return
        if (fixedFrame == null || resetToTier) fixedFrame = representativeFrame(tier)
        surface.setFixedFrame(fixedFrame, true)
    }

    private fun representativeFrame(tier: PoseTier): Int = when (tier) {
        PoseTier.NATURAL -> 80
        PoseTier.MODERATE -> 280
        PoseTier.EXTREME -> 470
    }

    private fun conciseStatus(status: String): String {
        val fps = Regex("FPS: ([0-9.]+)").find(status)?.groupValues?.getOrNull(1)
        val gpu = Regex("ZD GPU ([0-9.]+) ms").find(status)?.groupValues?.getOrNull(1)
        val delta = Regex("maxΔ ([0-9.]+) mm").find(status)?.groupValues?.getOrNull(1)
        return buildString {
            if (fps != null) append("FPS $fps")
            if (gpu != null) append(" · ZD GPU ${gpu}ms")
            if (delta != null) append(" · Δ ${delta}mm")
            if (isEmpty()) append("renderer starting")
            appendLine()
            append("Drag to rotate · pinch to zoom")
        }
    }

    private fun radioGroup(labels: List<String>, checked: Int, onChange: (Int) -> Unit): View {
        val group = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            setPadding(0, dp(4), 0, dp(8))
            labels.forEachIndexed { index, label ->
                addView(RadioButton(context).apply {
                    id = View.generateViewId()
                    text = label
                    setTextColor(0xFFFFFFFF.toInt())
                    textSize = 13f
                    tag = index
                    isChecked = index == checked
                    buttonTintList = android.content.res.ColorStateList.valueOf(0xFF78B7FF.toInt())
                    setPadding(dp(4), dp(4), dp(14), dp(4))
                })
            }
            setOnCheckedChangeListener { group, checkedId ->
                val rb = group.findViewById<RadioButton>(checkedId)
                onChange((rb.tag as Int))
            }
        }
        return HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(group, ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
    }

    private fun sectionLabel(textValue: String): TextView = TextView(this).apply {
        text = textValue
        setTextColor(0xFF78B7FF.toInt())
        textSize = 11f
        typeface = Typeface.DEFAULT_BOLD
        setPadding(0, dp(8), 0, dp(2))
        letterSpacing = 0.08f
    }

    private fun smallText(textValue: String): TextView = TextView(this).apply {
        text = textValue
        setTextColor(0xCCD7E6FF.toInt())
        textSize = 12f
        setPadding(dp(6), 0, dp(6), dp(2))
    }

    private fun actionButton(textValue: String): Button = Button(this).apply {
        text = textValue
        isAllCaps = false
        setTextColor(0xFFFFFFFF.toInt())
        textSize = 13f
        typeface = Typeface.DEFAULT_BOLD
        background = rounded(0xFF1D6ED0.toInt(), dp(18).toFloat())
    }

    private fun statusOverlay(): TextView = TextView(this).apply {
        setTextColor(0xFFEAF2FF.toInt())
        textSize = 10f
        setLineSpacing(0f, 1.08f)
        maxLines = 2
        setPadding(dp(12), dp(10), dp(12), dp(10))
        background = rounded(0x99000000.toInt(), dp(16).toFloat())
    }

    private fun rounded(color: Int, radius: Float): GradientDrawable = GradientDrawable().apply {
        setColor(color)
        cornerRadius = radius
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    override fun onResume() { super.onResume(); hideBars(); surface.onResume(); handler.post(updater) }
    override fun onPause() { handler.removeCallbacks(updater); surface.onPause(); super.onPause() }
    override fun onDestroy() { surface.releaseNativeResources(); super.onDestroy() }

    private fun hideBars() {
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }
}

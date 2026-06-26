package com.graphicsresearch.androidcomputebenchmark

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import java.io.File

class BenchmarkActivity : AppCompatActivity() {
    private var surfaceView: BenchmarkSurfaceView? = null
    private var statusView: TextView? = null
    private var activeLogFile: File? = null
    private var benchmarkMode: String = "smpl_app"
    private var benchmarkStarted = false
    private val statusHandler = Handler(Looper.getMainLooper())
    private val statusUpdater = object : Runnable {
        override fun run() {
            refreshStatusText()
            statusHandler.postDelayed(this, STATUS_UPDATE_INTERVAL_MS)
        }
    }

    private val storagePermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            startBenchmark(granted)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        keepTheScreenOn()
        hideSystemBars()

        if (needsLegacyExternalStoragePermission() && !hasLegacyStoragePermission()) {
            setContentView(View(this).apply { setBackgroundColor(0xFF000000.toInt()) })
            storagePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        } else {
            startBenchmark(hasLegacyStoragePermission())
        }
    }

    override fun onResume() {
        super.onResume()
        hideSystemBars()
        surfaceView?.onResume()
        statusHandler.post(statusUpdater)
    }

    override fun onPause() {
        statusHandler.removeCallbacks(statusUpdater)
        surfaceView?.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        statusHandler.removeCallbacks(statusUpdater)
        surfaceView?.releaseNativeResources()
        surfaceView = null
        statusView = null
        super.onDestroy()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            hideSystemBars()
        }
    }

    private fun startBenchmark(permissionGranted: Boolean) {
        if (benchmarkStarted) {
            return
        }
        benchmarkStarted = true

        if (!permissionGranted && needsLegacyExternalStoragePermission()) {
            Toast.makeText(
                this,
                "Using app-specific external storage for benchmark logs.",
                Toast.LENGTH_SHORT,
            ).show()
        }

        benchmarkMode = intent?.getStringExtra("benchmark_mode") ?: "smpl_app"
        val logFile = resolveLogFile(permissionGranted)
        activeLogFile = logFile
        val benchmarkView = BenchmarkSurfaceView(this, logFile.absolutePath, benchmarkMode)
        val overlay = createStatusView(logFile)
        val root = FrameLayout(this).apply {
            setBackgroundColor(0xFF030508.toInt())
            addView(
                benchmarkView,
                FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT,
                ),
            )
            addView(overlay)
        }
        surfaceView = benchmarkView
        statusView = overlay
        setContentView(root)
        refreshStatusText()
        statusHandler.post(statusUpdater)
        hideSystemBars()
    }

    private fun createStatusView(logFile: File): TextView {
        return TextView(this).apply {
            text = initialStatusText(logFile)
            setTextColor(0xFFE8F0FF.toInt())
            setBackgroundColor(0xCC08111F.toInt())
            textSize = 14f
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(28, 24, 28, 24)
            gravity = Gravity.START
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP,
            )
        }
    }

    private fun initialStatusText(logFile: File): String {
        return buildString {
            appendLine("ZD-ECN / SMPL Real-Time Demo")
            appendLine("Mode: $benchmarkMode")
            appendLine("Starting OpenGL ES compute shader pipeline…")
            appendLine("Architecture: 144 → 64 → 64 → 20,670")
            appendLine("Log: ${logFile.absolutePath}")
        }
    }

    private fun refreshStatusText() {
        val logPath = activeLogFile?.absolutePath ?: "unavailable"
        statusView?.text = buildString {
            appendLine("ZD-ECN / SMPL Real-Time Demo")
            appendLine("Mode: $benchmarkMode")
            appendLine(BenchmarkNative.statusText())
            appendLine("Log: $logPath")
            appendLine("Keep this screen open until phase becomes realtime.")
        }
    }

    private fun keepTheScreenOn() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun hideSystemBars() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.hide(WindowInsetsCompat.Type.systemBars())
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    private fun needsLegacyExternalStoragePermission(): Boolean {
        return Build.VERSION.SDK_INT <= Build.VERSION_CODES.Q
    }

    private fun hasLegacyStoragePermission(): Boolean {
        if (!needsLegacyExternalStoragePermission()) {
            return true
        }
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
        ) == PackageManager.PERMISSION_GRANTED
    }

    @Suppress("DEPRECATION")
    private fun resolveLogFile(permissionGranted: Boolean): File {
        val outputFile = if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.Q && permissionGranted) {
            File("/sdcard/inference_log.txt")
        } else {
            File(filesDir, "inference_log.txt")
        }
        outputFile.parentFile?.mkdirs()
        return outputFile
    }

    private companion object {
        private const val STATUS_UPDATE_INTERVAL_MS = 500L
    }
}

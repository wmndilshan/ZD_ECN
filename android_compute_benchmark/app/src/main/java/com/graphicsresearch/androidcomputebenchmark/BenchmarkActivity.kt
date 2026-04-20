package com.graphicsresearch.androidcomputebenchmark

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowManager
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
    private var benchmarkStarted = false

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
    }

    override fun onPause() {
        surfaceView?.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        surfaceView?.releaseNativeResources()
        surfaceView = null
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

        val logFile = resolveLogFile(permissionGranted)
        val benchmarkView = BenchmarkSurfaceView(this, logFile.absolutePath)
        surfaceView = benchmarkView
        setContentView(benchmarkView)
        hideSystemBars()
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
}

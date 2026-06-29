package com.graphicsresearch.smplrenderer

import android.content.Context
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Build
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class BenchmarkActivity : AppCompatActivity() {
    private lateinit var surface: BenchmarkSurfaceView
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        status = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xCC081220.toInt())
            textSize = 15f
            gravity = Gravity.CENTER
            setPadding(24, 24, 24, 24)
            text = "Preparing combined SMPL + ZD-ECN benchmark…"
        }
        val sessions = intent.getIntExtra("sessions", 5).coerceAtLeast(1)
        val warmup = intent.getIntExtra("warmup", 100).coerceAtLeast(0)
        val iterations = intent.getIntExtra("iterations", 1000).coerceAtLeast(1)
        val cooldownMs = intent.getLongExtra("cooldown_ms", 30_000L).coerceAtLeast(0L)
        surface = BenchmarkSurfaceView(this,
            sessions = sessions,
            warmup = warmup,
            iterations = iterations,
            cooldownMs = cooldownMs,
            onProgress = { message -> runOnUiThread { status.text = message } },
            onComplete = { message -> runOnUiThread { status.text = message } },
        )
        val root = FrameLayout(this)
        root.addView(surface, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(status, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER))
        setContentView(root)
    }

    override fun onResume() { super.onResume(); surface.onResume() }
    override fun onPause() { surface.onPause(); super.onPause() }
    override fun onDestroy() { surface.release(); super.onDestroy() }
}

private class BenchmarkSurfaceView(
    context: Context,
    sessions: Int,
    warmup: Int,
    iterations: Int,
    cooldownMs: Long,
    onProgress: (String) -> Unit,
    onComplete: (String) -> Unit,
) : GLSurfaceView(context) {
    private val benchmarkRenderer = CombinedBenchmarkRenderer(
        context, sessions, warmup, iterations, cooldownMs, onProgress, onComplete
    )

    init {
        setEGLContextClientVersion(3)
        setRenderer(benchmarkRenderer)
        renderMode = RENDERMODE_WHEN_DIRTY
    }

    fun release() {
        queueEvent { RendererNative.nativeRelease() }
    }
}

private class CombinedBenchmarkRenderer(
    private val context: Context,
    private val sessions: Int,
    private val warmup: Int,
    private val iterations: Int,
    private val cooldownMs: Long,
    private val onProgress: (String) -> Unit,
    private val onComplete: (String) -> Unit,
) : GLSurfaceView.Renderer {
    private var started = false

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        RendererNative.nativeInit(context.assets)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        RendererNative.nativeResize(width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        if (started) return
        started = true
        var progressFile: File? = null
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val outputRoot = File(context.getExternalFilesDir(null), "benchmark_results/$timestamp")
            val rawDirectory = File(outputRoot, "raw").apply { mkdirs() }
            val rawCsv = File(rawDirectory, "benchmark_raw.csv").apply {
                writeText("session,iteration,frame,gpu_ms,cpu_ms,disjoint\n")
            }
            val activeProgressFile = File(outputRoot, "benchmark_progress.txt")
            progressFile = activeProgressFile
            val samples = mutableListOf<CombinedSample>()
            for (session in 1..sessions) {
                val runningMessage = "Session $session/$sessions: $warmup warmup + $iterations measured"
                Log.i("CombinedBenchmark", runningMessage)
                activeProgressFile.writeText(runningMessage + "\n")
                onProgress(runningMessage.replace(": ", "\n"))
                val rows = RendererNative.nativeRunCombinedBenchmarkSession(session, warmup, iterations)
                check(!rows.startsWith("ERROR,")) { rows }
                rawCsv.appendText(rows)
                rows.lineSequence().filter { it.isNotBlank() }.forEach { line ->
                    val columns = line.split(',')
                    check(columns.size == 6) { "Malformed native CSV row: $line" }
                    samples += CombinedSample(
                        session = columns[0].toInt(),
                        iteration = columns[1].toInt(),
                        frame = columns[2].toInt(),
                        gpuMs = columns[3].toDouble(),
                        cpuMs = columns[4].toDouble(),
                        disjoint = columns[5].toInt() != 0,
                    )
                }
                activeProgressFile.writeText("completed_session=$session/$sessions samples=${samples.size}\n")
                if (session < sessions) {
                    onProgress("Session $session/$sessions complete\nCooling device for ${cooldownMs / 1000}s")
                    Thread.sleep(cooldownMs)
                }
            }
            val output = saveOutputs(outputRoot, samples)
            activeProgressFile.writeText("complete samples=${samples.size}\n")
            onComplete("Benchmark complete\nN=${samples.size}\n${output.absolutePath}")
        } catch (error: Throwable) {
            progressFile?.writeText("failed: ${error.message ?: error.javaClass.simpleName}\n")
            onComplete("Benchmark failed\n${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun saveOutputs(root: File, samples: List<CombinedSample>): File {
        val raw = File(root, "raw").apply { mkdirs() }
        val summary = File(root, "summary").apply { mkdirs() }
        File(raw, "device_info.txt").writeText(
            "manufacturer=${Build.MANUFACTURER}\n" +
            "model=${Build.MODEL}\n" +
            "device=${Build.DEVICE}\n" +
            "android=${Build.VERSION.RELEASE}\n" +
            "api=${Build.VERSION.SDK_INT}\n" +
            "abi=${Build.SUPPORTED_ABIS.joinToString()}\n" +
            "protocol=sessions=$sessions,warmup=$warmup,iterations=$iterations,cooldown_ms=$cooldownMs\n" +
            "boundary=GPU query and CPU wall timer wrap LBS+ZD-ECN+combine+smooth_normals+indexed_draw; CPU ends after glFinish\n"
        )
        val valid = samples.filterNot { it.disjoint }
        check(valid.size == samples.size) { "GPU disjoint detected in ${samples.size - valid.size} samples" }
        val gpu = statistics(valid.map { it.gpuMs })
        val cpu = statistics(valid.map { it.cpuMs })
        val json = JSONObject()
            .put("protocol", JSONObject()
                .put("sessions", sessions)
                .put("warmup_per_session", warmup)
                .put("iterations_per_session", iterations)
                .put("cooldown_ms", cooldownMs)
                .put("measured_samples", valid.size)
                .put("timer_boundary", "LBS+ZD-ECN+combine+smooth_normals+indexed_draw; glFinish inside CPU window"))
            .put("gpu_ms", gpu.toJson())
            .put("cpu_ms", cpu.toJson())
        File(summary, "benchmark_summary.json").writeText(json.toString(2) + "\n")
        File(summary, "benchmark_summary.md").writeText(
            "# Combined SMPL + ZD-ECN App Benchmark\n\n" +
            "| Timer | Mean ms | Std | p50 | p95 | p99 | CV % | 95% CI | FPS mean | FPS p99 |\n" +
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
            gpu.markdownRow("GPU") + "\n" + cpu.markdownRow("CPU wall") + "\n\n" +
            "Protocol: $sessions sessions × $iterations measured iterations, $warmup warmups discarded per session, ${cooldownMs / 1000}s cooldown between sessions.\n"
        )
        return root
    }
}

private data class CombinedSample(
    val session: Int,
    val iteration: Int,
    val frame: Int,
    val gpuMs: Double,
    val cpuMs: Double,
    val disjoint: Boolean,
)

private data class Statistics(
    val n: Int,
    val mean: Double,
    val std: Double,
    val p50: Double,
    val p95: Double,
    val p99: Double,
    val cvPercent: Double,
    val ciLow: Double,
    val ciHigh: Double,
    val fpsMean: Double,
    val fpsP99: Double,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("n", n).put("mean", mean).put("std", std)
        .put("p50", p50).put("p95", p95).put("p99", p99)
        .put("cv_percent", cvPercent)
        .put("ci95_low", ciLow).put("ci95_high", ciHigh)
        .put("fps_from_mean", fpsMean).put("fps_from_p99", fpsP99)

    fun markdownRow(label: String): String =
        "| $label | %.6f | %.6f | %.6f | %.6f | %.6f | %.3f | [%.6f, %.6f] | %.2f | %.2f |".format(
            Locale.US, mean, std, p50, p95, p99, cvPercent, ciLow, ciHigh, fpsMean, fpsP99
        )
}

private fun statistics(values: List<Double>): Statistics {
    require(values.isNotEmpty())
    val sorted = values.sorted()
    val mean = values.average()
    val variance = if (values.size > 1) values.sumOf { (it - mean) * (it - mean) } / (values.size - 1) else 0.0
    val std = sqrt(variance)
    val margin = 1.96 * std / sqrt(values.size.toDouble())
    val p50 = percentile(sorted, 0.50)
    val p95 = percentile(sorted, 0.95)
    val p99 = percentile(sorted, 0.99)
    return Statistics(
        n = values.size, mean = mean, std = std, p50 = p50, p95 = p95, p99 = p99,
        cvPercent = if (mean == 0.0) 0.0 else std / mean * 100.0,
        ciLow = mean - margin, ciHigh = mean + margin,
        fpsMean = 1000.0 / mean, fpsP99 = 1000.0 / p99,
    )
}

private fun percentile(sorted: List<Double>, probability: Double): Double {
    if (sorted.size == 1) return sorted[0]
    val position = probability * (sorted.size - 1)
    val lower = position.toInt()
    val upper = minOf(lower + 1, sorted.lastIndex)
    val fraction = position - lower
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

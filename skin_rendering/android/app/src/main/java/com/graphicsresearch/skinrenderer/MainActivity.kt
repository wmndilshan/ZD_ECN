package com.graphicsresearch.skinrenderer

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.PixelCopy
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var glView: SkinRenderSurfaceView
    private lateinit var onnxBackend: OnnxSkinBackend
    private val handler = Handler(Looper.getMainLooper())
    private var fpsRunnable: Runnable? = null
    private val onnxExecutor = Executors.newSingleThreadExecutor()
    private val onnxInFlight = AtomicBoolean(false)
    @Volatile private var lastOnnxFrame = -1

    // Current state
    private var backend = 0        // 0=ZD-ECN, 1=ONNX CPU, 2=LBS only
    private var tier    = 0        // 0=natural, 1=moderate, 2=extreme
    private var texIdx  = 0        // 0=procedural, 1=Meshcapade crop
    private var useTexture = true
    private var wrapFactor  = 0.3f
    private var specStrength = 0.05f
    private var lightAngle  = 0.5f
    private var benchmarkRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        glView = findViewById(R.id.gl_surface_view)
        onnxBackend = OnnxSkinBackend(this)

        // Upload textures once GL context is ready (done in GL thread)
        glView.post { uploadTextureAssets() }

        // ── ONNX pose sync: whenever the rendered pose frame advances and the
        //    ONNX CPU backend is active, run inference off the GL thread and
        //    upload the fresh delta (upload forces a recombine next draw).
        glView.onFrameDrawn = {
            if (backend == 1) {
                val frame = SkinRendererNative.nativeGetCurrentFrame()
                if (frame >= 0 && frame != lastOnnxFrame &&
                    onnxInFlight.compareAndSet(false, true)) {
                    val rot6d = SkinRendererNative.nativeGetCurrentRot6d()
                    if (rot6d != null) {
                        onnxExecutor.execute {
                            try {
                                // Attempt once per pose frame even on failure —
                                // avoids an error-log flood at render rate.
                                lastOnnxFrame = frame
                                val delta = onnxBackend.infer(rot6d)
                                if (delta != null) {
                                    glView.queueEvent {
                                        SkinRendererNative.nativeUploadOnnxDelta(delta)
                                    }
                                }
                            } finally {
                                onnxInFlight.set(false)
                            }
                        }
                    } else {
                        onnxInFlight.set(false)
                    }
                }
            }
        }

        // ── Backend selector ───────────────────────────────────────────────
        val backendGroup = findViewById<RadioGroup>(R.id.rg_backend)
        backendGroup.setOnCheckedChangeListener { _, id ->
            backend = when (id) {
                R.id.rb_zdecn    -> 0
                R.id.rb_onnx_cpu -> 1
                R.id.rb_lbs_only -> 2
                else             -> 0
            }
            lastOnnxFrame = -1   // re-infer current pose when entering ONNX mode
            glView.setBackend(backend)
        }

        // ── Texture selector ───────────────────────────────────────────────
        val texGroup = findViewById<RadioGroup>(R.id.rg_texture)
        texGroup.setOnCheckedChangeListener { _, id ->
            texIdx = when (id) {
                R.id.rb_tex0        -> 0
                R.id.rb_tex1        -> 1
                R.id.rb_tex_none    -> { glView.setUseTexture(false); return@setOnCheckedChangeListener }
                else -> 0
            }
            useTexture = true
            glView.setUseTexture(true)
            glView.setTexture(texIdx)
        }

        // ── Pose tier selector ─────────────────────────────────────────────
        val tierGroup = findViewById<RadioGroup>(R.id.rg_tier)
        tierGroup.setOnCheckedChangeListener { _, id ->
            tier = when (id) {
                R.id.rb_natural  -> 0
                R.id.rb_moderate -> 1
                R.id.rb_extreme  -> 2
                else             -> 0
            }
            glView.setTier(tier)
        }

        // ── Lighting controls ──────────────────────────────────────────────
        fun updateLighting() {
            glView.setLighting(wrapFactor, specStrength, lightAngle)
        }

        val seekWrap = findViewById<SeekBar>(R.id.seek_wrap)
        seekWrap.max = 80
        seekWrap.progress = (wrapFactor * 100).toInt()
        seekWrap.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, u: Boolean) {
                wrapFactor = p / 100.0f
                updateLighting()
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })

        val seekSpec = findViewById<SeekBar>(R.id.seek_spec)
        seekSpec.max = 50
        seekSpec.progress = (specStrength * 100).toInt()
        seekSpec.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, u: Boolean) {
                specStrength = p / 100.0f
                updateLighting()
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })

        val seekLight = findViewById<SeekBar>(R.id.seek_light_angle)
        seekLight.max = 360
        seekLight.progress = (lightAngle * 180 / Math.PI).toInt()
        seekLight.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, u: Boolean) {
                lightAngle = p * Math.PI.toFloat() / 180.0f
                updateLighting()
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })

        // ── Screenshot button ──────────────────────────────────────────────
        findViewById<Button>(R.id.btn_screenshot).setOnClickListener {
            takeScreenshot()
        }

        // ── Benchmark button ───────────────────────────────────────────────
        findViewById<Button>(R.id.btn_benchmark).setOnClickListener {
            runFullPipelineBenchmark()
        }

        // ── FPS overlay ────────────────────────────────────────────────────
        startFpsTimer()

        // Headless trigger: adb shell am start ... --ez run_benchmark true
        if (intent.getBooleanExtra("run_benchmark", false)) {
            handler.postDelayed({ runFullPipelineBenchmark() }, 4000)
        }

        // Fixed-frame lock for reproducible comparison screenshots:
        //   adb shell am start ... --ei fixed_frame 500   (-1 = auto-advance)
        val fixedFrame = intent.getIntExtra("fixed_frame", -1)
        if (fixedFrame >= 0) {
            glView.setFixedFrame(fixedFrame, true)
        }
    }

    private fun runFullPipelineBenchmark(warmup: Int = 50, iterations: Int = 300) {
        if (benchmarkRunning) return
        benchmarkRunning = true
        Toast.makeText(this, "Benchmark: $warmup warmup + $iterations measured…",
            Toast.LENGTH_SHORT).show()
        glView.queueEvent {
            val json = SkinRendererNative.nativeRunFullPipelineBenchmark(warmup, iterations)
            runOnUiThread {
                benchmarkRunning = false
                try {
                    File(filesDir, "full_pipeline_benchmark.json").writeText(json)
                } catch (e: Exception) {
                    android.util.Log.e("MainActivity", "benchmark write failed: ${e.message}")
                }
                android.util.Log.i("MainActivity", "FULL_PIPELINE_BENCHMARK $json")
                AlertDialog.Builder(this)
                    .setTitle("Full pipeline benchmark")
                    .setMessage(json)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun uploadTextureAssets() {
        glView.queueEvent {
            for (i in 0..1) {
                val name = "texture_skin_$i.png"
                try {
                    val bmp = BitmapFactory.decodeStream(assets.open(name))
                    val w = bmp.width; val h = bmp.height
                    // Extract RGBA bytes
                    val pixels = IntArray(w * h)
                    bmp.getPixels(pixels, 0, w, 0, 0, w, h)
                    val rgba = ByteArray(w * h * 4)
                    for (j in pixels.indices) {
                        val p = pixels[j]
                        rgba[j*4+0] = ((p shr 16) and 0xFF).toByte()  // R
                        rgba[j*4+1] = ((p shr  8) and 0xFF).toByte()  // G
                        rgba[j*4+2] = ( p         and 0xFF).toByte()  // B
                        rgba[j*4+3] = ((p shr 24) and 0xFF).toByte()  // A
                    }
                    SkinRendererNative.nativeUploadTexture(i, rgba, w, h)
                    bmp.recycle()
                } catch (e: Exception) {
                    android.util.Log.e("MainActivity", "texture $name load failed: ${e.message}")
                }
            }
        }
    }

    private fun startFpsTimer() {
        val tv = findViewById<TextView>(R.id.tv_fps)
        var lastMs = System.currentTimeMillis()

        fpsRunnable = object : Runnable {
            override fun run() {
                val now = System.currentTimeMillis()
                val dtSec = (now - lastMs).coerceAtLeast(1) / 1000.0f
                lastMs = now
                val fps = glView.takeFrameCount() / dtSec
                val zdecnMs = SkinRendererNative.nativeGetZdecnMs()
                val backendName = when (backend) {
                    0 -> "ZD-ECN"; 1 -> "ONNX CPU"; else -> "LBS only"
                }
                tv.text = String.format("FPS: %.0f | ECN: %.2f ms | %s",
                    fps, zdecnMs, backendName)
                handler.postDelayed(this, 1000)
            }
        }
        handler.postDelayed(fpsRunnable!!, 1000)
    }

    private fun takeScreenshot() {
        val bitmap = Bitmap.createBitmap(glView.width, glView.height,
            Bitmap.Config.ARGB_8888)
        PixelCopy.request(glView, bitmap, { result ->
            if (result == PixelCopy.SUCCESS) {
                try {
                    val dir = File(getExternalFilesDir(null) ?: filesDir, "screenshots")
                    dir.mkdirs()
                    val file = File(dir, "skin_${System.currentTimeMillis()}.png")
                    FileOutputStream(file).use { out ->
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }
                    Toast.makeText(this, "Saved: ${file.absolutePath}",
                        Toast.LENGTH_LONG).show()
                    android.util.Log.i("MainActivity", "screenshot saved: ${file.absolutePath}")
                } catch (e: Exception) {
                    Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } else {
                Toast.makeText(this, "PixelCopy failed: $result", Toast.LENGTH_LONG).show()
            }
            bitmap.takeIf { result != PixelCopy.SUCCESS }?.recycle()
        }, handler)
    }

    override fun onResume()  { super.onResume();  glView.onResume() }
    override fun onPause()   { super.onPause();   glView.onPause() }
    override fun onDestroy() {
        fpsRunnable?.let { handler.removeCallbacks(it) }
        glView.onFrameDrawn = null
        onnxExecutor.shutdown()
        onnxBackend.close()
        super.onDestroy()
    }
}

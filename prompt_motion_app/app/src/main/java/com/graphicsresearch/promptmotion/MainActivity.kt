package com.graphicsresearch.promptmotion

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.PixelCopy
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.graphicsresearch.promptmotion.motion.MotionClip
import com.graphicsresearch.promptmotion.motion.MotionGenerator
import com.graphicsresearch.promptmotion.motion.SmplClip
import com.graphicsresearch.promptmotion.motion.SmplMotionConverter
import com.graphicsresearch.promptmotion.planner.MotionPlan
import com.graphicsresearch.promptmotion.planner.MotionPlannerAdapter
import com.graphicsresearch.promptmotion.planner.OpenAiMotionPlanner
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Prompt -> motion plan (OpenAI or rule-based, zd_ecn_motion schema) -> deterministic
 * biomechanical generator (foot-locked IK) -> SMPL pose stream -> GPU LBS + ZD-ECN
 * correction -> textured skin rendering.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "PromptMotion"
        private const val FPS = 30
        private val PRESETS = listOf(
            "Walk forward, turn left, jump, land, and return to idle",
            "Perform traditional dance",
            "Michael Jackson style dance",
            "Squat deeply then jump high then land",
            "Walk forward for 3 seconds, turn around, walk forward",
            "A cheerful person jogs a few steps, spins around and celebrates",
            "Bow politely, then walk slowly to the right and wait",
        )
    }

    private lateinit var glView: MotionSurfaceView
    private lateinit var onnx: OnnxBackend
    private val handler = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val onnxExecutor = Executors.newSingleThreadExecutor()
    private val onnxInFlight = AtomicBoolean(false)
    @Volatile private var lastOnnxFrame = -1

    private val generator = MotionGenerator()
    private var converter: SmplMotionConverter? = null
    private lateinit var planner: MotionPlannerAdapter

    // Current clip (kept for GL context re-creation)
    @Volatile private var smplClip: SmplClip? = null
    private var plan: MotionPlan? = null
    private var motion: MotionClip? = null
    @Volatile private var restPelvis = FloatArray(3)

    private var backend = 0
    private var playing = true
    private var benchmarkRunning = false
    private var userSeeking = false

    private lateinit var etPrompt: EditText
    private lateinit var btnGenerate: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvPhase: TextView
    private lateinit var tvTime: TextView
    private lateinit var seekTime: SeekBar
    private lateinit var progress: ProgressBar
    private lateinit var swLlm: Switch

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        glView = findViewById(R.id.gl_surface_view)
        etPrompt = findViewById(R.id.et_prompt)
        btnGenerate = findViewById(R.id.btn_generate)
        tvStatus = findViewById(R.id.tv_status)
        tvPhase = findViewById(R.id.tv_phase)
        tvTime = findViewById(R.id.tv_time)
        seekTime = findViewById(R.id.seek_time)
        progress = findViewById(R.id.progress)
        swLlm = findViewById(R.id.sw_llm)
        onnx = OnnxBackend(this)

        val key = BuildConfig.OPENAI_API_KEY
        planner = MotionPlannerAdapter(
            if (key.isNotBlank()) OpenAiMotionPlanner(key, BuildConfig.OPENAI_MODEL, FPS) else null, FPS)
        swLlm.isChecked = key.isNotBlank() && intent.getBooleanExtra("use_llm", true)
        swLlm.text = if (key.isNotBlank()) "OpenAI (${BuildConfig.OPENAI_MODEL}) " else "OpenAI (no key) "

        glView.onContextCreated = {
            uploadTextures()
            smplClip?.let { loadClipOnGlThread(it) }
            MotionRendererNative.nativeSetBackend(backend)
        }
        glView.onFrameDrawn = { syncOnnx() }

        setupControls()
        startStatusTimer()

        // Headless hooks:  am start ... --es prompt "..." --ez use_llm false --ei backend 2
        //                  --ei fixed_frame 40 --ez run_benchmark true
        backend = intent.getIntExtra("backend", 0)
        findViewById<RadioGroup>(R.id.rg_backend).check(
            when (backend) { 1 -> R.id.rb_onnx_cpu; 2 -> R.id.rb_lbs_only; else -> R.id.rb_zdecn })
        val fixedFrame = intent.getIntExtra("fixed_frame", -1)
        if (fixedFrame >= 0) glView.queueEvent { MotionRendererNative.nativeSetFixedFrame(fixedFrame) }

        val initialPrompt = intent.getStringExtra("prompt") ?: PRESETS[0]
        etPrompt.setText(initialPrompt)
        worker.execute {
            converter = loadConverter()
            runOnUiThread {
                // First clip uses the rule-based planner unless a prompt was given explicitly
                animate(initialPrompt, useLlm = intent.hasExtra("prompt") && swLlm.isChecked)
            }
        }
        if (intent.getBooleanExtra("run_benchmark", false)) {
            handler.postDelayed({ runBenchmark() }, 6000)
        }
    }

    private fun loadConverter(): SmplMotionConverter {
        val v = SmplMotionConverter.floatsLE(assets.open("smpl_shader_data/v_template_f32.bin").readBytes())
        val jr = SmplMotionConverter.floatsLE(assets.open("smpl_shader_data/joint_regressor_f32.bin").readBytes())
        return SmplMotionConverter(v, jr).also { c -> restPelvis = FloatArray(3) { c.restPelvis()[it].toFloat() } }
    }

    // ── prompt -> motion ──────────────────────────────────────────────────────
    private fun animate(prompt: String, useLlm: Boolean = swLlm.isChecked) {
        if (prompt.isBlank()) return
        val conv = converter ?: return
        btnGenerate.isEnabled = false
        progress.visibility = View.VISIBLE
        tvStatus.text = if (useLlm) "Planning with OpenAI…" else "Planning…"
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(etPrompt.windowToken, 0)

        worker.execute {
            val t0 = System.nanoTime()
            val result = runCatching { planner.plan(prompt, useLlm) }
            val t1 = System.nanoTime()
            val r = result.getOrElse { e ->
                Log.w(TAG, "PLAN_FAILED ${e.message}")
                // Keep something on screen: the very first clip falls back to idle
                if (smplClip == null) {
                    val idle = planner.plan("idle for 3 seconds", useLlm = false).plan
                    val clip = generator.generate(idle)
                    val smpl = conv.convert(clip)
                    smplClip = smpl
                    glView.queueEvent { loadClipOnGlThread(smpl) }
                    runOnUiThread { plan = idle; motion = clip; seekTime.max = clip.frameCount - 1 }
                }
                runOnUiThread {
                    progress.visibility = View.GONE
                    btnGenerate.isEnabled = true
                    tvStatus.text = "Plan failed:\n${e.message}"
                }
                return@execute
            }
            val clip = generator.generate(r.plan)
            val t2 = System.nanoTime()
            val smpl = conv.convert(clip)
            val t3 = System.nanoTime()
            Log.i(TAG, "PLAN ${r.source} ${r.plan.toJson()}")
            if (intent.getBooleanExtra("dump_clip", false)) dumpClip(r.plan, smpl)

            smplClip = smpl
            glView.queueEvent { loadClipOnGlThread(smpl) }
            runOnUiThread {
                plan = r.plan; motion = clip
                progress.visibility = View.GONE
                btnGenerate.isEnabled = true
                playing = true
                findViewById<Button>(R.id.btn_play).text = "Pause"
                seekTime.max = clip.frameCount - 1
                val phases = r.plan.phases.joinToString(" → ") { "${it.action}(${"%.1f".format(it.duration)}s)" }
                tvStatus.text = buildString {
                    append("Planner: ${r.source}  ${ms(t0, t1)} ms | gen ${ms(t1, t2)} ms | smpl ${ms(t2, t3)} ms\n")
                    append("$phases\n")
                    append("${clip.frameCount} frames @ ${clip.fps} fps · ${"%.2f".format(clip.durationSeconds)} s · ")
                    append("clamps ${clip.jointLimitClampings} · penetration ${"%.4f".format(clip.maxGroundPenetrationM)} m · ")
                    append("slip ${"%.4f".format(clip.maxFootSlipVelocity)} m/s")
                    r.note?.let { append("\n$it") }
                }
                Log.i(TAG, "STATUS ${tvStatus.text.toString().replace("\n", " | ")}")
            }
        }
    }

    /** Parity hook: writes plan + SMPL arrays for tools/check_device_parity.py. */
    private fun dumpClip(p: MotionPlan, c: SmplClip) {
        val dir = File(filesDir, "dump").apply { mkdirs() }
        File(dir, "plan.json").writeText(p.toJson().toString())
        fun write(name: String, a: FloatArray) {
            val bb = java.nio.ByteBuffer.allocate(a.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            bb.asFloatBuffer().put(a)
            File(dir, name).writeBytes(bb.array())
        }
        write("aa_f32.bin", c.axisAngle); write("rot6d_f32.bin", c.rot6d); write("trans_f32.bin", c.trans)
        Log.i(TAG, "DUMP_DONE ${dir.absolutePath}")
    }

    private fun ms(a: Long, b: Long) = "%.1f".format((b - a) / 1e6)

    private fun loadClipOnGlThread(c: SmplClip) {
        lastOnnxFrame = -1
        MotionRendererNative.nativeLoadMotion(c.axisAngle, c.rot6d, c.trans, restPelvis, c.fps)
        MotionRendererNative.nativeSetPlaying(playing)
    }

    // ── ONNX CPU backend: infer when the displayed pose frame changes ─────────
    private fun syncOnnx() {
        if (backend != 1) return
        val frame = MotionRendererNative.nativeGetCurrentFrame()
        if (frame < 0 || frame == lastOnnxFrame || !onnxInFlight.compareAndSet(false, true)) return
        val rot6d = MotionRendererNative.nativeGetCurrentRot6d()
        if (rot6d == null) { onnxInFlight.set(false); return }
        onnxExecutor.execute {
            try {
                lastOnnxFrame = frame
                onnx.infer(rot6d)?.let { d -> glView.queueEvent { MotionRendererNative.nativeUploadOnnxDelta(d) } }
            } finally {
                onnxInFlight.set(false)
            }
        }
    }

    // ── UI ────────────────────────────────────────────────────────────────────
    private fun setupControls() {
        btnGenerate.setOnClickListener { animate(etPrompt.text.toString()) }
        etPrompt.setOnEditorActionListener { _, id, _ ->
            if (id == EditorInfo.IME_ACTION_DONE) { animate(etPrompt.text.toString()); true } else false
        }

        val row = findViewById<LinearLayout>(R.id.preset_row)
        PRESETS.forEach { p ->
            row.addView(Button(this).apply {
                text = if (p.length > 28) p.take(26) + "…" else p
                isAllCaps = false
                textSize = 11f
                setOnClickListener { etPrompt.setText(p); animate(p) }
            })
        }

        findViewById<Button>(R.id.btn_play).setOnClickListener { b ->
            playing = !playing
            (b as Button).text = if (playing) "Pause" else "Play"
            val p = playing
            glView.queueEvent { MotionRendererNative.nativeSetPlaying(p) }
        }
        seekTime.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, v: Int, fromUser: Boolean) {
                if (!fromUser) return
                val t = v / FPS.toFloat()
                glView.queueEvent { MotionRendererNative.nativeSeek(t) }
            }
            override fun onStartTrackingTouch(sb: SeekBar) { userSeeking = true }
            override fun onStopTrackingTouch(sb: SeekBar) { userSeeking = false }
        })

        val speeds = listOf(0.25f, 0.5f, 1.0f, 1.5f)
        findViewById<Spinner>(R.id.sp_speed).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                speeds.map { "${it}×" })
            setSelection(2)
            onItemSelectedListener = simpleSelect { i ->
                val s = speeds[i]; glView.queueEvent { MotionRendererNative.nativeSetSpeed(s) }
            }
        }
        findViewById<Spinner>(R.id.sp_texture).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                listOf("Procedural skin", "Meshcapade skin", "Grey"))
            onItemSelectedListener = simpleSelect { i ->
                glView.queueEvent {
                    MotionRendererNative.nativeSetUseTexture(i < 2)
                    if (i < 2) MotionRendererNative.nativeSetTexture(i)
                }
            }
        }

        findViewById<RadioGroup>(R.id.rg_backend).setOnCheckedChangeListener { _, id ->
            backend = when (id) { R.id.rb_onnx_cpu -> 1; R.id.rb_lbs_only -> 2; else -> 0 }
            lastOnnxFrame = -1
            val b = backend
            glView.queueEvent { MotionRendererNative.nativeSetBackend(b) }
        }
        findViewById<Button>(R.id.btn_screenshot).setOnClickListener { takeScreenshot() }
        findViewById<Button>(R.id.btn_benchmark).setOnClickListener { runBenchmark() }
    }

    private fun simpleSelect(f: (Int) -> Unit) = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) = f(pos)
        override fun onNothingSelected(p: AdapterView<*>?) {}
    }

    private fun startStatusTimer() {
        val tvFps = findViewById<TextView>(R.id.tv_fps)
        var lastMs = System.currentTimeMillis()
        var frames = 0
        handler.post(object : Runnable {
            override fun run() {
                val now = System.currentTimeMillis()
                frames += glView.takeFrameCount()
                if (now - lastMs >= 1000) {
                    val fps = frames * 1000f / (now - lastMs)
                    frames = 0; lastMs = now
                    val name = when (backend) { 0 -> "ZD-ECN GPU"; 1 -> "ONNX CPU"; else -> "LBS only" }
                    tvFps.text = "FPS %.0f | ECN %.2f ms | %s".format(fps, MotionRendererNative.nativeGetZdecnMs(), name)
                }
                val p = plan; val m = motion
                if (p != null && m != null) {
                    val t = MotionRendererNative.nativeGetPlaybackTime()
                    // The frame on screen (also correct when pinned with fixed_frame)
                    val shown = MotionRendererNative.nativeGetCurrentFrame()
                    val f = (if (shown >= 0) shown else (t * m.fps).toInt()).coerceIn(0, m.frameCount - 1)
                    val phase = p.phases[m.phaseIndex[f]]
                    tvPhase.text = "${phase.phaseId}/${p.phases.size}  ${phase.action.replace('_', ' ')}"
                    tvTime.text = "%.1f / %.1f s".format(t, m.durationSeconds)
                    if (!userSeeking) seekTime.progress = f
                }
                handler.postDelayed(this, 100)
            }
        })
    }

    private fun uploadTextures() {
        for (i in 0..1) {
            try {
                val bmp = BitmapFactory.decodeStream(assets.open("texture_skin_$i.png"))
                val w = bmp.width; val h = bmp.height
                val px = IntArray(w * h)
                bmp.getPixels(px, 0, w, 0, 0, w, h)
                val rgba = ByteArray(w * h * 4)
                for (j in px.indices) {
                    val p = px[j]
                    rgba[j * 4] = ((p shr 16) and 0xFF).toByte()
                    rgba[j * 4 + 1] = ((p shr 8) and 0xFF).toByte()
                    rgba[j * 4 + 2] = (p and 0xFF).toByte()
                    rgba[j * 4 + 3] = ((p shr 24) and 0xFF).toByte()
                }
                MotionRendererNative.nativeUploadTexture(i, rgba, w, h)
                bmp.recycle()
            } catch (e: Exception) {
                Log.e(TAG, "texture $i load failed: ${e.message}")
            }
        }
    }

    private fun runBenchmark(warmup: Int = 50, iterations: Int = 300) {
        if (benchmarkRunning || smplClip == null) return
        benchmarkRunning = true
        Toast.makeText(this, "Benchmark: $warmup warmup + $iterations measured…", Toast.LENGTH_SHORT).show()
        glView.queueEvent {
            val json = MotionRendererNative.nativeRunFullPipelineBenchmark(warmup, iterations)
            runOnUiThread {
                benchmarkRunning = false
                runCatching { File(filesDir, "prompt_pipeline_benchmark.json").writeText(json) }
                Log.i(TAG, "FULL_PIPELINE_BENCHMARK $json")
                AlertDialog.Builder(this).setTitle("Full pipeline benchmark").setMessage(json)
                    .setPositiveButton("OK", null).show()
            }
        }
    }

    private fun takeScreenshot() {
        val bitmap = Bitmap.createBitmap(glView.width, glView.height, Bitmap.Config.ARGB_8888)
        PixelCopy.request(glView, bitmap, { result ->
            if (result == PixelCopy.SUCCESS) {
                val dir = File(getExternalFilesDir(null) ?: filesDir, "screenshots").apply { mkdirs() }
                val file = File(dir, "motion_${System.currentTimeMillis()}.png")
                FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                Log.i(TAG, "screenshot saved: ${file.absolutePath}")
                Toast.makeText(this, "Saved: ${file.name}", Toast.LENGTH_SHORT).show()
            }
            bitmap.recycle()
        }, handler)
    }

    override fun onResume() { super.onResume(); glView.onResume() }
    override fun onPause() { super.onPause(); glView.onPause() }
    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        glView.onFrameDrawn = null
        glView.onContextCreated = null
        worker.shutdown()
        onnxExecutor.shutdown()
        onnx.close()
        super.onDestroy()
    }
}

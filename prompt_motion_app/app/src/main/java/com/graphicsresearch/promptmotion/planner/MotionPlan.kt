package com.graphicsresearch.promptmotion.planner

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

/** Port of zd_ecn_motion/planner/schema.py — ZDECNMotionPlan v1.0. */
val SUPPORTED_ACTIONS = listOf(
    "walk", "turn", "squat", "jump", "land", "idle",
    "dance_mandiya", "dance_step", "dance_turn", "dance_freeze",
    "moonwalk", "spin", "toe_stand", "pop_hits", "pose_point",
)

class PhaseParams(val raw: JSONObject) {
    fun num(key: String, default: Double): Double =
        if (raw.has(key) && raw.get(key) is Number) raw.getDouble(key) else default
    fun str(key: String, default: String): String = raw.optString(key, default)
}

data class MotionPhase(val phaseId: Int, val action: String, val duration: Double, val parameters: PhaseParams)

data class MotionPlan(
    val version: String,
    val prompt: String,
    val targetFps: Int,
    val totalDuration: Double,
    val phases: List<MotionPhase>,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("version", version)
        put("prompt", prompt)
        put("target_fps", targetFps)
        put("total_duration", totalDuration)
        put("phases", JSONArray().apply {
            phases.forEach { p ->
                put(JSONObject().apply {
                    put("phase_id", p.phaseId)
                    put("action", p.action)
                    put("duration", p.duration)
                    put("parameters", p.parameters.raw)
                })
            }
        })
    }

    companion object {
        private val TOP_KEYS = setOf("version", "prompt", "target_fps", "total_duration", "phases")
        private val PHASE_KEYS = setOf("phase_id", "action", "duration", "parameters")

        /**
         * Strict structural + semantic validation mirroring MOTION_PLAN_SCHEMA and
         * validate_motion_plan(). Throws IllegalArgumentException with the reason.
         */
        fun parseAndValidate(obj: JSONObject): MotionPlan {
            fun fail(msg: String): Nothing = throw IllegalArgumentException(msg)

            for (k in TOP_KEYS) if (!obj.has(k)) fail("Schema validation error: '$k' is a required property")
            obj.keys().forEach { if (it !in TOP_KEYS) fail("Schema validation error: additional property '$it'") }

            val version = obj.get("version") as? String ?: fail("'version' must be a string")
            if (version != "1.0" && version != "1.0.0") fail("'version' must be one of [1.0, 1.0.0]")
            val prompt = obj.get("prompt") as? String ?: fail("'prompt' must be a string")
            if (prompt.isEmpty()) fail("'prompt' must be non-empty")
            val fpsAny = obj.get("target_fps")
            if (fpsAny !is Int && !(fpsAny is Long)) fail("'target_fps' must be an integer")
            val fps = (fpsAny as Number).toInt()
            if (fps < 5 || fps > 120) fail("'target_fps' must be in [5, 120]")
            val total = (obj.get("total_duration") as? Number)?.toDouble() ?: fail("'total_duration' must be a number")
            if (total <= 0.0 || total > 300.0) fail("'total_duration' must be in (0, 300]")
            val arr = obj.get("phases") as? JSONArray ?: fail("'phases' must be an array")
            if (arr.length() < 1) fail("'phases' must have at least 1 item")

            val phases = ArrayList<MotionPhase>()
            for (i in 0 until arr.length()) {
                val p = arr.get(i) as? JSONObject ?: fail("phase $i must be an object")
                for (k in PHASE_KEYS) if (!p.has(k)) fail("Schema validation error: '$k' is a required property at path [phases, $i]")
                p.keys().forEach { if (it !in PHASE_KEYS) fail("additional property '$it' at path [phases, $i]") }
                val idAny = p.get("phase_id")
                if (idAny !is Int && idAny !is Long) fail("'phase_id' must be an integer at [phases, $i]")
                val id = (idAny as Number).toInt()
                if (id < 1) fail("'phase_id' must be >= 1 at [phases, $i]")
                val action = p.get("action") as? String ?: fail("'action' must be a string at [phases, $i]")
                if (action !in SUPPORTED_ACTIONS) fail("'$action' is not one of $SUPPORTED_ACTIONS")
                val dur = (p.get("duration") as? Number)?.toDouble() ?: fail("'duration' must be a number at [phases, $i]")
                if (dur <= 0.0 || dur > 60.0) fail("'duration' must be in (0, 60] at [phases, $i]")
                val params = p.get("parameters") as? JSONObject ?: fail("'parameters' must be an object at [phases, $i]")
                phases.add(MotionPhase(id, action, dur, PhaseParams(params)))
            }

            val sum = phases.sumOf { it.duration }
            if (abs(sum - total) > 0.05) {
                fail("Semantic error: Sum of phase durations (%.3fs) does not match total_duration (%.3fs)".format(sum, total))
            }
            phases.forEachIndexed { idx, p ->
                val i = idx + 1
                if (p.phaseId != i) fail("Semantic error: Phase at index $idx has phase_id ${p.phaseId}, expected $i")
                when (p.action) {
                    "turn" -> if (!p.parameters.raw.has("angle_degrees")) fail("Semantic error: 'turn' action in phase $i missing 'angle_degrees'")
                    "walk" -> if (p.parameters.num("speed", 1.0) <= 0) fail("Semantic error: 'walk' action in phase $i speed must be > 0")
                    "jump" -> if (p.parameters.num("height", 0.5) <= 0) fail("Semantic error: 'jump' action in phase $i height must be > 0")
                    "squat" -> {
                        val d = p.parameters.num("depth", 0.5)
                        if (d < 0.0 || d > 1.0) fail("Semantic error: 'squat' depth must be between 0.0 and 1.0")
                    }
                }
            }
            return MotionPlan(version, prompt, fps, total, phases)
        }
    }
}

package com.graphicsresearch.promptmotion.planner

import org.json.JSONObject

/** Port of zd_ecn_motion/planner/mock_planner.py — deterministic rule-based planner. */
class MockMotionPlanner(private val defaultFps: Int = 20) {

    private val actionPatterns = listOf(
        Regex("""\b(moonwalk|moonwalking|backslide|moon walk)\b""") to "moonwalk",
        Regex("""\b(spin|spins|spinning|twirl)\b""") to "spin",
        Regex("""\b(toe stand|on (?:his |her |their )?toes|tiptoe|tip toe)\b""") to "toe_stand",
        Regex("""\b(walk|walking|stride|step forward|walk forward)\b""") to "walk",
        Regex("""\b(turn left|turn right|turn around|turn)\b""") to "turn",
        Regex("""\b(squat|squatting|crouch|crouching|knee bend)\b""") to "squat",
        Regex("""\b(jump|jumping|leap|hop)\b""") to "jump",
        Regex("""\b(land|landing|absorb impact|ground contact)\b""") to "land",
        Regex("""\b(idle|return to idle|stand|pause|wait|rest)\b""") to "idle",
        Regex("""\b(dance mandiya|mandiya|namaskaraya|salutation|mandiya stance)\b""") to "dance_mandiya",
        Regex("""\b(dance step|stepping|kastiram|thei-tha|footwork|rhythmic stamp)\b""") to "dance_step",
        Regex("""\b(dance turn|vattam|pirouette|circular turn)\b""") to "dance_turn",
        Regex("""\b(dance freeze|samapthiya|heroic freeze|statue)\b""") to "dance_freeze",
        Regex("""\b(pop hits|popping|pop and lock|pop|hits|isolations)\b""") to "pop_hits",
        Regex("""\b(point|pointing|signature pose|final pose|pose)\b""") to "pose_point",
    )

    private val unsupported = listOf("fly", "swim", "somersault", "backflip", "cartwheel", "crawl", "teleport", "climb")
    private val fillers = setOf("please", "now", "forward", "cleanly", "smoothly", "again", "for traditional dancer")
    private val clauseSplit = Regex(""",|;|\band\s+then\b|\bthen\b|\bafter\s+that\b|\band\s+return\s+to\b|\band\b""")

    fun plan(prompt: String): MotionPlan {
        require(prompt.isNotBlank()) { "Prompt cannot be empty." }
        val clean = prompt.trim().lowercase()

        for (u in unsupported) {
            if (Regex("""\b$u\b""").containsMatchIn(clean)) {
                throw IllegalArgumentException(
                    "Unsupported action '$u' detected in prompt. Supported actions are: ${SUPPORTED_ACTIONS.joinToString()}")
            }
        }

        if (Regex("""\b(traditional dance|dance traditionally|traditional dancer|kandyan dance|uva dance)\b""").containsMatchIn(clean)) {
            if (!Regex(""",|;|\band\s+then\b|\bthen\b""").containsMatchIn(clean) ||
                "perform" in clean || "generation" in clean || "again" in clean) {
                return traditionalDance(prompt)
            }
        }

        // High-level macro for pop / Michael Jackson style dance
        if (Regex("""\b(michael jackson|mj style|king of pop|pop dance|pop star dance)\b""").containsMatchIn(clean)) {
            if (!Regex(""",|;|\band\s+then\b|\bthen\b""").containsMatchIn(clean) ||
                Regex("""\b(perform|style|like)\b""").containsMatchIn(clean)) {
                return popDance(prompt)
            }
        }

        val clauses = clean.split(clauseSplit).map { it.trim() }.filter { it.isNotEmpty() }
        val phases = ArrayList<JSONObject>()
        for (clause in clauses) {
            val action = actionPatterns.firstOrNull { it.first.containsMatchIn(clause) }?.second
            if (action == null) {
                if (clause in fillers) continue
                throw IllegalArgumentException(
                    "Could not parse action from clause: '$clause'. Supported actions are: ${SUPPORTED_ACTIONS.joinToString()}")
            }
            val (duration, params) = buildParameters(action, clause)
            phases.add(JSONObject().apply {
                put("phase_id", phases.size + 1)
                put("action", action)
                put("duration", round3(duration))
                put("parameters", params)
            })
        }
        require(phases.isNotEmpty()) { "No valid motion phases could be extracted from prompt." }
        return assemble(prompt, phases)
    }

    private fun round3(x: Double) = Math.rint(x * 1000.0) / 1000.0

    private fun assemble(prompt: String, phases: List<JSONObject>): MotionPlan {
        val total = round3(phases.sumOf { it.getDouble("duration") })
        val obj = JSONObject().apply {
            put("version", "1.0")
            put("prompt", prompt.trim())
            put("target_fps", defaultFps)
            put("total_duration", total)
            put("phases", org.json.JSONArray(phases))
        }
        return MotionPlan.parseAndValidate(obj)
    }

    private fun traditionalDance(prompt: String): MotionPlan {
        fun ph(id: Int, a: String, d: Double, p: JSONObject) = JSONObject().apply {
            put("phase_id", id); put("action", a); put("duration", d); put("parameters", p)
        }
        return assemble(prompt, listOf(
            ph(1, "dance_mandiya", 2.5, JSONObject().put("depth", 0.75).put("arm_mudra", "anjali_to_open")),
            ph(2, "dance_step", 4.0, JSONObject().put("cadence", "thei_tha").put("tempo_bpm", 80).put("step_lift_m", 0.12)),
            ph(3, "dance_turn", 2.5, JSONObject().put("turn_type", "vattam").put("angle_degrees", 360.0)),
            ph(4, "dance_freeze", 2.0, JSONObject().put("posture", "samapthiya").put("chin_elevation", true)),
        ))
    }

    /** Five-phase pop routine: hits -> moonwalk -> spin -> toe stand -> point pose. */
    private fun popDance(prompt: String): MotionPlan {
        fun ph(id: Int, a: String, d: Double, p: JSONObject) = JSONObject().apply {
            put("phase_id", id); put("action", a); put("duration", d); put("parameters", p)
        }
        return assemble(prompt, listOf(
            ph(1, "pop_hits", 2.4, JSONObject().put("tempo_bpm", 117)),
            ph(2, "moonwalk", 3.3, JSONObject().put("stride", 0.18)),
            ph(3, "spin", 1.4, JSONObject().put("angle_degrees", 360.0)),
            ph(4, "toe_stand", 1.5, JSONObject()),
            ph(5, "pose_point", 2.4, JSONObject()),
        ))
    }

    private fun buildParameters(action: String, clause: String): Pair<Double, JSONObject> {
        val custom = Regex("""for\s+([0-9]+(?:\.[0-9]+)?)\s*(?:seconds|second|s|sec)\b""")
            .find(clause)?.groupValues?.get(1)?.toDouble()
        fun dur(d: Double) = if (custom != null && custom != 0.0) custom else d

        return when (action) {
            "walk" -> {
                val speed = Regex("""speed\s+([0-9]+(?:\.[0-9]+)?)\b""").find(clause)?.groupValues?.get(1)?.toDouble() ?: 1.2
                dur(2.0) to JSONObject().put("speed", speed).put("step_frequency", 1.8).put("direction", "forward")
            }
            "turn" -> {
                var angle = 90.0
                var direction = "left"
                if ("right" in clause) { angle = -90.0; direction = "right" }
                else if ("around" in clause || "180" in clause) { angle = 180.0; direction = "around" }
                Regex("""([0-9]+(?:\.[0-9]+)?)\s*(?:deg|degrees)\b""").find(clause)?.let {
                    val v = it.groupValues[1].toDouble()
                    angle = if ("right" in clause) -v else v
                }
                dur(1.0) to JSONObject().put("angle_degrees", angle).put("direction", direction)
            }
            "squat" -> {
                val depth = when { "deep" in clause -> 0.8; "shallow" in clause -> 0.35; else -> 0.6 }
                dur(1.5) to JSONObject().put("depth", depth).put("hold_duration", 0.3)
            }
            "jump" -> {
                val h = when { "high" in clause -> 0.65; "small" in clause || "low" in clause -> 0.25; else -> 0.45 }
                dur(1.0) to JSONObject().put("height", h).put("forward_distance", 0.2)
            }
            "land" -> dur(0.8) to JSONObject().put("recovery_duration", 0.5).put("damping", 0.85)
            "idle" -> {
                val posture = if ("mandiya" in clause) "mandiya" else "neutral"
                dur(1.5) to JSONObject().put("posture", posture).put("breathing", true)
            }
            "dance_mandiya" -> dur(2.5) to JSONObject().put("depth", 0.75).put("arm_mudra", "anjali_to_open")
            "dance_step" -> dur(3.5) to JSONObject().put("cadence", "thei_tha").put("tempo_bpm", 80).put("step_lift_m", 0.12)
            "dance_turn" -> dur(2.5) to JSONObject().put("turn_type", "vattam").put("angle_degrees", 360.0)
            "dance_freeze" -> dur(2.0) to JSONObject().put("posture", "samapthiya").put("chin_elevation", true)
            "moonwalk" -> dur(3.3) to JSONObject().put("stride", 0.18)
            "spin" -> {
                var angle = if (Regex("""\b(twice|double|two)\b""").containsMatchIn(clause)) 720.0 else 360.0
                if (Regex("""\bright\b""").containsMatchIn(clause)) angle = -angle
                dur(1.4) to JSONObject().put("angle_degrees", angle)
            }
            "toe_stand" -> dur(1.5) to JSONObject()
            "pop_hits" -> dur(2.4) to JSONObject().put("tempo_bpm", 117)
            "pose_point" -> dur(2.4) to JSONObject()
            else -> 1.0 to JSONObject()
        }
    }
}

package com.graphicsresearch.promptmotion.planner

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * LLM backend for the motion planner: asks an OpenAI chat model to decompose a free-form
 * prompt into a ZDECNMotionPlan restricted to the generator's supported actions.
 * The reply is validated with the same strict schema as the mock planner; one repair
 * round-trip feeds the validation error back before giving up.
 */
class OpenAiMotionPlanner(
    private val apiKey: String,
    private val model: String,
    private val fps: Int,
    private val endpoint: String = "https://api.openai.com/v1/chat/completions",
) {
    private val systemPrompt = """
You are the motion planner of ZD-ECN, a mobile SMPL human-body animation system.
Convert the user's description into a JSON motion plan. Output ONLY a JSON object:

{"version":"1.0","prompt":<user prompt verbatim>,"target_fps":$fps,"total_duration":<sum of phase durations>,
 "phases":[{"phase_id":1,"action":<action>,"duration":<seconds>,"parameters":{...}}, ...]}

Supported actions and their parameters (use ONLY these actions):
- "idle":   {"posture":"neutral","breathing":true}                 default 1.5 s
- "walk":   {"speed":0.4-1.15 m/s,"step_frequency":1.4-2.2 Hz,"direction":"forward"}  default 2.0 s
- "turn":   {"angle_degrees":<+left / -right, e.g. 90, -90, 180>,"direction":"left|right|around"}  REQUIRED angle_degrees, default 1.0 s
- "squat":  {"depth":0.0-1.0,"hold_duration":0.3}                   default 1.5 s
- "jump":   {"height":0.15-0.7 m,"forward_distance":0.0-0.4 m}       default 1.0 s; always follow with "land"
- "land":   {"recovery_duration":0.5,"damping":0.6-1.0}              default 0.8 s
- "dance_mandiya": Kandyan/Uva salutation into deep mandiya stance   {"depth":0.75,"arm_mudra":"anjali_to_open"}  default 2.5 s
- "dance_step":    rhythmic Kandyan stamping footwork               {"cadence":"thei_tha","tempo_bpm":80,"step_lift_m":0.12}  default 3.5 s
- "dance_turn":    vattam circular turn                              {"turn_type":"vattam","angle_degrees":360}  default 2.5 s
- "dance_freeze":  samapthiya heroic final pose                       {"posture":"samapthiya","chin_elevation":true}  default 2.0 s
- "pop_hits":   staccato pop/popping upper-body hits on the beat, feet planted  {"tempo_bpm":90-140}  default 2.4 s
- "moonwalk":   backslide gliding backward (moonwalk)                {"stride":0.12-0.25 m}  default 3.3 s
- "spin":       fast pivot on one foot with knee tucked              {"angle_degrees":360 or 720, negative = clockwise}  default 1.4 s
- "toe_stand":  rise up onto both toes and hold                      {}  default 1.5 s
- "pose_point": closing freeze pointing one arm diagonally upward    {}  default 2.4 s; use only as the LAST phase

Rules:
- phase_id starts at 1 and increments by 1. Each duration in (0, 60]. total_duration is the sum of durations (<=300 s).
- Approximate unsupported verbs with the closest supported actions (e.g. "run" -> fast walk, "spin" -> turn 360,
  "bow" -> squat shallow, "celebrate" -> jump + land, "look around" -> idle). Never invent new action names.
- Dance phases (dance_*) should be grouped together; start a dance with dance_mandiya when it begins from standing.
- Pop / Michael-Jackson-style dancing uses pop_hits, moonwalk, spin, toe_stand and pose_point (e.g. hits -> moonwalk ->
  spin -> toe_stand -> pose_point). Do not mix them with the Kandyan dance_* actions unless asked.
- Start from a standing pose; end with "idle" or "dance_freeze" unless the user says otherwise.
""".trim()

    fun plan(prompt: String): MotionPlan {
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", systemPrompt))
            .put(JSONObject().put("role", "user").put("content", prompt))

        var lastError = "no response"
        repeat(2) {
            val content = complete(messages)
            try {
                val obj = JSONObject(content)
                obj.put("prompt", prompt.trim())  // keep the user's exact prompt
                deriveBookkeeping(obj)
                return MotionPlan.parseAndValidate(obj)
            } catch (e: Exception) {
                lastError = e.message ?: e.toString()
                messages.put(JSONObject().put("role", "assistant").put("content", content))
                messages.put(JSONObject().put("role", "user").put("content",
                    "That plan failed validation: $lastError. Return a corrected JSON plan only."))
            }
        }
        throw IllegalStateException("LLM plan invalid after repair: $lastError")
    }

    /**
     * total_duration and phase_id are derived bookkeeping that small models often get
     * arithmetically wrong; recompute them from the phase list. Everything the motion
     * depends on (actions, durations, parameters, fps) is still strictly validated.
     */
    private fun deriveBookkeeping(obj: JSONObject) {
        val phases = obj.optJSONArray("phases") ?: return
        var sum = 0.0
        for (i in 0 until phases.length()) {
            val p = phases.optJSONObject(i) ?: return
            p.put("phase_id", i + 1)
            sum += (p.opt("duration") as? Number)?.toDouble() ?: return
        }
        obj.put("total_duration", Math.rint(sum * 1000.0) / 1000.0)
    }

    private fun complete(messages: JSONArray): String {
        val body = JSONObject()
            .put("model", model)
            .put("messages", messages)
            .put("temperature", 0.2)
            .put("response_format", JSONObject().put("type", "json_object"))
        val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15_000
            readTimeout = 45_000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Authorization", "Bearer $apiKey")
        }
        try {
            conn.outputStream.use { it.write(body.toString().toByteArray()) }
            val code = conn.responseCode
            val text = (if (code in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()?.use { it.readText() } ?: ""
            if (code !in 200..299) {
                val msg = runCatching { JSONObject(text).getJSONObject("error").getString("message") }.getOrNull()
                throw IllegalStateException("OpenAI HTTP $code: ${msg ?: text.take(200)}")
            }
            return JSONObject(text).getJSONArray("choices").getJSONObject(0)
                .getJSONObject("message").getString("content")
        } finally {
            conn.disconnect()
        }
    }
}

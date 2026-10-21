package com.graphicsresearch.promptmotion.planner

/**
 * Port of zd_ecn_motion/planner/llm_adapter.py: try the LLM backend, fall back to the
 * deterministic mock planner. Also reports which planner produced the plan and why.
 */
class MotionPlannerAdapter(
    private val llm: OpenAiMotionPlanner?,
    fps: Int,
) {
    private val mock = MockMotionPlanner(defaultFps = fps)

    data class Result(val plan: MotionPlan, val source: String, val note: String?)

    fun plan(prompt: String, useLlm: Boolean): Result {
        var llmError: String? = null
        if (useLlm && llm != null) {
            try {
                return Result(llm.plan(prompt), "OpenAI", null)
            } catch (e: Exception) {
                llmError = "LLM failed: ${e.message}"
            }
        } else if (useLlm) {
            llmError = "No OPENAI_API_KEY in build"
        }
        try {
            return Result(mock.plan(prompt), "rule-based", llmError?.let { "$it; used rule-based planner" })
        } catch (e: Exception) {
            // Neither planner could handle it: report both reasons
            throw IllegalArgumentException(listOfNotNull(llmError, "Rule-based: ${e.message}").joinToString("\n"))
        }
    }
}

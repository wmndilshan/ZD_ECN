package com.graphicsresearch.promptmotion.motion

/** Port of zd_ecn_motion/generator/joint_limits.py (radians; pitch, roll, yaw). */
object JointLimits {
    private fun r(deg: Double) = Math.toRadians(deg)
    private val PI = Math.PI

    private val LIMITS: Array<DoubleArray> = arrayOf(
        // Root yaw is the accumulated heading, not an anatomical angle: unbounded
        doubleArrayOf(-PI, PI, -PI, PI, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY),
        doubleArrayOf(-r(135.0), r(90.0), -r(75.0), r(75.0), -r(90.0), r(90.0)),
        doubleArrayOf(-r(135.0), r(90.0), -r(75.0), r(75.0), -r(90.0), r(90.0)),
        doubleArrayOf(-r(35.0), r(35.0), -r(30.0), r(30.0), -r(35.0), r(35.0)),
        doubleArrayOf(0.0, r(150.0), -r(10.0), r(10.0), -r(15.0), r(15.0)),
        doubleArrayOf(0.0, r(150.0), -r(10.0), r(10.0), -r(15.0), r(15.0)),
        doubleArrayOf(-r(30.0), r(30.0), -r(25.0), r(25.0), -r(30.0), r(30.0)),
        doubleArrayOf(-r(95.0), r(95.0), -r(65.0), r(65.0), -r(85.0), r(85.0)),
        doubleArrayOf(-r(95.0), r(95.0), -r(65.0), r(65.0), -r(85.0), r(85.0)),
        doubleArrayOf(-r(30.0), r(30.0), -r(25.0), r(25.0), -r(30.0), r(30.0)),
        doubleArrayOf(-r(70.0), r(45.0), -r(25.0), r(25.0), -r(30.0), r(30.0)),  // toe dorsiflexion to 70°
        doubleArrayOf(-r(70.0), r(45.0), -r(25.0), r(25.0), -r(30.0), r(30.0)),
        doubleArrayOf(-r(45.0), r(45.0), -r(40.0), r(40.0), -r(60.0), r(60.0)),
        doubleArrayOf(-r(25.0), r(25.0), -r(25.0), r(25.0), -r(25.0), r(25.0)),
        doubleArrayOf(-r(25.0), r(25.0), -r(25.0), r(25.0), -r(25.0), r(25.0)),
        doubleArrayOf(-r(50.0), r(50.0), -r(40.0), r(40.0), -r(70.0), r(70.0)),
        doubleArrayOf(-r(120.0), r(180.0), -r(110.0), r(110.0), -r(110.0), r(110.0)),
        doubleArrayOf(-r(120.0), r(180.0), -r(110.0), r(110.0), -r(110.0), r(110.0)),
        doubleArrayOf(0.0, r(150.0), -r(20.0), r(20.0), -r(20.0), r(20.0)),
        doubleArrayOf(0.0, r(150.0), -r(20.0), r(20.0), -r(20.0), r(20.0)),
        doubleArrayOf(-r(80.0), r(80.0), -r(50.0), r(50.0), -r(45.0), r(45.0)),
        doubleArrayOf(-r(80.0), r(80.0), -r(50.0), r(50.0), -r(45.0), r(45.0)),
        doubleArrayOf(-r(40.0), r(40.0), -r(30.0), r(30.0), -r(30.0), r(30.0)),
        doubleArrayOf(-r(40.0), r(40.0), -r(30.0), r(30.0), -r(30.0), r(30.0)),
    )

    /** Clamps in place; returns the number of (frame, joint) pairs that were clamped. */
    fun clamp(rotations: Array<Array<DoubleArray>>): Int {
        var violations = 0
        for (frame in rotations) {
            for (j in 0 until 24) {
                val lim = LIMITS[j]
                var clamped = false
                for (a in 0 until 3) {
                    val old = frame[j][a]
                    val c = old.coerceIn(lim[a * 2], lim[a * 2 + 1])
                    if (kotlin.math.abs(c - old) > 1e-5) clamped = true
                    frame[j][a] = c
                }
                if (clamped) violations++
            }
        }
        return violations
    }
}

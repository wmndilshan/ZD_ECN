package com.graphicsresearch.promptmotion.motion

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/**
 * Port of zd_ecn_motion/generator/actions.py — procedural action synthesizers.
 * Each produces desired root trajectory, joint Euler rotations, foot contact
 * schedule and swing clearances; legs are re-solved later by the foot-lock IK.
 */
class PhaseResult(
    val rotations: Array<Array<DoubleArray>>,  // [n][24][3] pitch, roll, yaw
    val translations: Array<DoubleArray>,      // [n][3]
    val leftPlanted: BooleanArray,
    val rightPlanted: BooleanArray,
    val footLifts: Array<DoubleArray>,         // [n][2]
    val style: String,
    /** [n][2][5] explicit (x, y, z, yaw, pitch) per foot, NaN = regular foot lock. */
    val footOverride: Array<Array<DoubleArray>>? = null,
) {
    val endRoot: DoubleArray get() = translations.last().copyOf()
    var endHeading: Double = 0.0
}

object Actions {

    fun minJerk(s0: Double): Double {
        val s = max(0.0, min(1.0, s0))
        return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
    }

    /** Python round() is half-to-even; Math.rint matches it. */
    private fun frameCount(duration: Double, fps: Int) = max(1, Math.rint(duration * fps).toInt())

    private class Buf(val n: Int, duration: Double, style: String, endpoint: Boolean = false) {
        // np.linspace(0, duration, n, endpoint=...)
        val times = DoubleArray(n) { if (endpoint) (if (n > 1) it * duration / (n - 1) else 0.0) else it * duration / n }
        val override: Array<Array<DoubleArray>> by lazy { Array(n) { Array(2) { DoubleArray(5) { Double.NaN } } } }
        val rot = Array(n) { Array(24) { DoubleArray(3) } }
        val trans = Array(n) { DoubleArray(3) }
        val lp = BooleanArray(n)
        val rp = BooleanArray(n)
        val lifts = Array(n) { DoubleArray(2) }
        val style = style
        fun set(i: Int, j: Int, p: Double, r: Double, y: Double) {
            rot[i][j][0] = p; rot[i][j][1] = r; rot[i][j][2] = y
        }
        fun result(endHeading: Double, withOverride: Boolean = false) =
            PhaseResult(rot, trans, lp, rp, lifts, style, if (withOverride) override else null)
                .also { it.endHeading = endHeading }
        fun setOverride(i: Int, foot: Int, x: Double, y: Double, z: Double, yaw: Double, pitch: Double) {
            val o = override[i][foot]
            o[0] = x; o[1] = y; o[2] = max(0.0, z); o[3] = yaw; o[4] = pitch
        }
    }

    private fun fwd(h: Double) = doubleArrayOf(-sin(h), cos(h), 0.0)
    private fun lat(h: Double) = doubleArrayOf(cos(h), sin(h), 0.0)

    /** Shared stance/swing schedule used by walk, turn, dance_step, dance_turn. */
    private fun schedule(b: Buf, i: Int, sinP: Double, threshold: Double, liftHeight: Double,
                         liftFn: (Double) -> Double = { sin(it * PI) }) {
        val lSwing = max(0.0, sinP - threshold) / (1.0 - threshold)
        val rSwing = max(0.0, -sinP - threshold) / (1.0 - threshold)
        val lPlanted = (lSwing <= 0.0) || i == 0 || i == b.n - 1
        val rPlanted = (rSwing <= 0.0) || i == 0 || i == b.n - 1
        b.lifts[i][0] = if (!lPlanted) liftHeight * liftFn(lSwing) else 0.0
        b.lifts[i][1] = if (!rPlanted) liftHeight * liftFn(rSwing) else 0.0
        b.lp[i] = lPlanted; b.rp[i] = rPlanted
    }

    fun idle(duration: Double, fps: Int, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard")
        val baseZ = 0.78
        for (i in 0 until b.n) {
            val t = b.times[i]
            val breath = 0.015 * sin(2.0 * PI * 0.28 * t)
            val swayX = 0.006 * sin(2.0 * PI * 0.15 * t)
            val sBlend = minJerk(min(1.0, t / max(0.25, duration * 0.25)))
            val zCurr = (1.0 - sBlend) * startRoot[2] + sBlend * (baseZ + breath * 0.25)

            b.trans[i][0] = startRoot[0] + swayX; b.trans[i][1] = startRoot[1]; b.trans[i][2] = zCurr
            b.set(i, 0, 0.02, -swayX * 0.5, startHeading)
            b.set(i, 3, breath * 0.6, swayX * 0.3, 0.0)
            b.set(i, 6, breath * 0.4, swayX * 0.2, 0.0)
            b.set(i, 9, breath * 0.3, 0.0, 0.0)
            b.set(i, 12, -breath * 0.3, 0.0, 0.0)
            b.set(i, 15, -breath * 0.2, 0.0, 0.02 * sin(2.0 * PI * 0.1 * t))
            b.set(i, 16, 0.04, -0.18, 0.05)
            b.set(i, 17, 0.04, 0.18, -0.05)
            b.set(i, 18, 0.18 + 0.02 * breath, 0.0, 0.0)
            b.set(i, 19, 0.18 + 0.02 * breath, 0.0, 0.0)
            b.lp[i] = true; b.rp[i] = true
        }
        return b.result(startHeading)
    }

    fun walk(duration: Double, fps: Int, speed: Double, stepFreq: Double,
             startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard")
        val fwdDir = fwd(startHeading)
        val latDir = lat(startHeading)
        val effectiveSpeed = min(speed, 1.15)
        val baseZ = 0.77
        for (i in 0 until b.n) {
            val t = b.times[i]
            val env = minJerk(min(1.0, t / 0.35)) * minJerk(min(1.0, (duration - t) / 0.35))
            val disp = effectiveSpeed * t * (0.85 + 0.15 * env)
            val phase = 2.0 * PI * stepFreq * t
            val sinP = sin(phase)
            val cosP = cos(phase)
            val bounce = 0.022 * abs(cosP) * env
            val swayLat = -0.035 * sinP * env

            for (k in 0 until 3) b.trans[i][k] = startRoot[k] + fwdDir[k] * disp + latDir[k] * swayLat
            b.trans[i][2] = baseZ + bounce

            b.set(i, 0, 0.05 * env, 0.05 * sinP * env, startHeading + 0.08 * sinP * env)
            b.set(i, 3, 0.03 * env, -0.03 * sinP * env, -0.04 * sinP * env)
            b.set(i, 6, 0.02 * env, -0.02 * sinP * env, -0.05 * sinP * env)
            b.set(i, 9, 0.01 * env, -0.01 * sinP * env, -0.04 * sinP * env)
            b.set(i, 12, -0.03 * env, 0.0, 0.03 * sinP * env)
            b.set(i, 15, -0.02 * env, 0.0, 0.02 * sinP * env)

            val armLag = sin(phase - 0.25)
            val wristLag = sin(phase - 0.45)
            b.set(i, 16, 0.38 * sinP * env, -0.14, 0.0)
            b.set(i, 17, -0.38 * sinP * env, 0.14, 0.0)
            b.set(i, 18, 0.28 + 0.22 * max(0.0, armLag) * env, 0.0, 0.0)
            b.set(i, 19, 0.28 + 0.22 * max(0.0, -armLag) * env, 0.0, 0.0)
            b.set(i, 20, 0.10 * wristLag * env, 0.0, 0.0)
            b.set(i, 21, -0.10 * wristLag * env, 0.0, 0.0)

            schedule(b, i, sinP, 0.15, 0.09)
        }
        return b.result(startHeading)
    }

    fun turn(duration: Double, fps: Int, angleDegrees: Double,
             startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard")
        val angleRad = Math.toRadians(angleDegrees)
        val baseZ = 0.77
        for (i in 0 until b.n) {
            val t = b.times[i]
            val s = t / duration
            val currentYaw = startHeading + minJerk(s) * angleRad
            val stepPhase = 2.0 * PI * (t / max(0.5, duration * 0.5))
            val sinP = sin(stepPhase)

            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]
            b.trans[i][2] = baseZ + 0.015 * abs(sinP)
            b.set(i, 0, 0.03, 0.03 * sinP, currentYaw)

            val lead = sin(s * PI)
            val sign = if (angleRad >= 0) 1.0 else -1.0
            b.rot[i][3][2] = 0.10 * sign * lead
            b.rot[i][6][2] = 0.12 * sign * lead
            b.rot[i][12][2] = 0.18 * sign * lead
            b.rot[i][15][2] = 0.25 * sign * lead

            b.set(i, 16, 0.15 * sinP, -0.18, 0.0)
            b.set(i, 17, -0.15 * sinP, 0.18, 0.0)
            b.set(i, 18, 0.30, 0.0, 0.0)
            b.set(i, 19, 0.30, 0.0, 0.0)

            schedule(b, i, sinP, 0.1, 0.07)
        }
        return b.result(startHeading + angleRad)
    }

    fun squat(duration: Double, fps: Int, depth: Double, @Suppress("UNUSED_PARAMETER") holdDuration: Double,
              startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard")
        val baseZ = 0.78
        val maxDrop = 0.28 * depth
        for (i in 0 until b.n) {
            val s = b.times[i] / duration
            val curve = when {
                s < 0.42 -> minJerk(s / 0.42)
                s < 0.58 -> 1.0
                else -> 1.0 - minJerk((s - 0.58) / 0.42)
            }
            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]; b.trans[i][2] = baseZ - maxDrop * curve
            b.set(i, 0, 0.18 * curve, 0.0, startHeading)
            b.set(i, 3, 0.22 * curve, 0.0, 0.0)
            b.set(i, 6, 0.15 * curve, 0.0, 0.0)
            b.set(i, 9, 0.08 * curve, 0.0, 0.0)
            b.set(i, 12, -0.25 * curve, 0.0, 0.0)
            b.set(i, 15, -0.15 * curve, 0.0, 0.0)
            b.set(i, 16, -0.85 * curve, -0.12, 0.0)
            b.set(i, 17, -0.85 * curve, 0.12, 0.0)
            b.set(i, 18, 0.35 * curve + 0.15, 0.0, 0.0)
            b.set(i, 19, 0.35 * curve + 0.15, 0.0, 0.0)
            b.lp[i] = true; b.rp[i] = true
        }
        return b.result(startHeading)
    }

    fun jump(duration: Double, fps: Int, height: Double, forwardDist: Double,
             startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard")
        val fwdDir = fwd(startHeading)
        val baseZ = 0.78
        for (i in 0 until b.n) {
            val s = b.times[i] / duration
            val zPos: Double
            val fwdPos: Double
            if (s < 0.30) {
                val pc = minJerk(s / 0.30)
                zPos = baseZ - 0.16 * pc
                fwdPos = 0.0
                b.set(i, 0, 0.20 * pc, 0.0, startHeading)
                b.set(i, 3, 0.22 * pc, 0.0, 0.0)
                b.set(i, 16, 0.55 * pc, -0.15, 0.0)
                b.set(i, 17, 0.55 * pc, 0.15, 0.0)
                b.set(i, 18, 0.45 * pc + 0.15, 0.0, 0.0)
                b.set(i, 19, 0.45 * pc + 0.15, 0.0, 0.0)
                b.lp[i] = true; b.rp[i] = true
            } else {
                val pAir = (s - 0.30) / 0.70
                val parabola = 4.0 * pAir * (1.0 - pAir)
                zPos = (baseZ - 0.16 * (1.0 - pAir)) + height * parabola
                fwdPos = forwardDist * minJerk(pAir)
                b.set(i, 0, -0.06 * parabola, 0.0, startHeading)
                b.set(i, 3, -0.08 * parabola, 0.0, 0.0)
                b.set(i, 16, -1.15 * parabola, -0.20, 0.0)
                b.set(i, 17, -1.15 * parabola, 0.20, 0.0)
                b.set(i, 18, 0.35, 0.0, 0.0)
                b.set(i, 19, 0.35, 0.0, 0.0)
                val airLift = max(0.02, height * parabola * 0.88)
                b.lifts[i][0] = airLift; b.lifts[i][1] = airLift
                b.lp[i] = false; b.rp[i] = false
            }
            for (k in 0 until 3) b.trans[i][k] = startRoot[k] + fwdDir[k] * fwdPos
            b.trans[i][2] = zPos
        }
        return b.result(startHeading)
    }

    fun land(duration: Double, fps: Int, @Suppress("UNUSED_PARAMETER") recoveryDuration: Double, damping: Double,
             startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard")
        val baseZ = 0.78
        val impactDrop = 0.18 * damping
        for (i in 0 until b.n) {
            val s = b.times[i] / duration
            val shock = max(0.0, exp(-3.8 * s) * sin(PI * min(1.0, s * 1.4)))
            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]; b.trans[i][2] = baseZ - impactDrop * shock
            b.set(i, 0, 0.16 * shock, 0.0, startHeading)
            b.set(i, 3, 0.22 * shock, 0.0, 0.0)
            b.set(i, 6, 0.12 * shock, 0.0, 0.0)
            b.set(i, 12, -0.20 * shock, 0.0, 0.0)
            b.set(i, 16, -0.45 * shock, -0.18, 0.0)
            b.set(i, 17, -0.45 * shock, 0.18, 0.0)
            b.set(i, 18, 0.40 * shock + 0.15, 0.0, 0.0)
            b.set(i, 19, 0.40 * shock + 0.15, 0.0, 0.0)
            b.lp[i] = true; b.rp[i] = true
        }
        return b.result(startHeading)
    }

    /** Kandyan/Uva opening salutation (Namaskaraya) sinking into deep Mandiya. */
    fun danceMandiya(duration: Double, fps: Int, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "mandiya")
        val targetZ = 0.64
        for (i in 0 until b.n) {
            val prog = minJerk(b.times[i] / duration)
            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]
            b.trans[i][2] = startRoot[2] - (startRoot[2] - targetZ) * prog
            b.set(i, 0, 0.06 * prog, 0.0, startHeading)
            b.set(i, 3, 0.08 * prog, 0.0, 0.0)
            b.set(i, 6, -0.06 * prog, 0.0, 0.0)
            b.set(i, 9, -0.05 * prog, 0.0, 0.0)
            b.set(i, 12, 0.04 * prog, 0.0, 0.0)
            b.set(i, 15, 0.05 * (1.0 - prog), 0.0, 0.0)
            if (prog < 0.5) {
                val pa = minJerk(prog / 0.5)
                b.set(i, 16, -0.35 * pa, -0.35 * pa, 0.35 * pa)
                b.set(i, 17, -0.35 * pa, 0.35 * pa, -0.35 * pa)
                b.set(i, 18, 1.10 * pa, 0.0, 0.0)
                b.set(i, 19, 1.10 * pa, 0.0, 0.0)
            } else {
                val po = minJerk((prog - 0.5) / 0.5)
                b.set(i, 16, -0.35 + 0.45 * po, -0.35 - 0.45 * po, 0.35)
                b.set(i, 17, -0.35 + 0.45 * po, 0.35 + 0.45 * po, -0.35)
                b.set(i, 18, 1.10 - 0.15 * po, -0.18 * po, 0.0)
                b.set(i, 19, 1.10 - 0.15 * po, 0.18 * po, 0.0)
                b.set(i, 20, 0.35 * po, 0.25 * po, -0.30 * po)
                b.set(i, 21, 0.35 * po, -0.25 * po, 0.30 * po)
            }
            b.lp[i] = true; b.rp[i] = true
        }
        return b.result(startHeading)
    }

    /** Pa Saramba & Kastiram rhythmic stamping with CoM weight transfer. */
    fun danceStep(duration: Double, fps: Int, @Suppress("UNUSED_PARAMETER") cadence: String,
                  startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "mandiya")
        val tempoHz = 0.75
        val baseZ = 0.64
        val latDir = lat(startHeading)
        for (i in 0 until b.n) {
            val t = b.times[i]
            val env = minJerk(min(1.0, t / 0.25)) * minJerk(min(1.0, (duration - t) / 0.25))
            val phase = 2.0 * PI * tempoHz * t
            val sinP = sin(phase)
            val cosP = cos(phase)
            val bounce = 0.032 * sin(2.0 * phase).pow(2) * env
            val weightShift = -0.045 * sinP * env

            for (k in 0 until 3) b.trans[i][k] = startRoot[k] + latDir[k] * weightShift
            b.trans[i][2] = baseZ - bounce

            b.set(i, 0, 0.06, 0.06 * sinP * env, startHeading + 0.05 * cosP * env)
            val shAccent = 0.06 * sin(2.0 * phase) * env
            b.set(i, 3, 0.08, -0.07 * sinP * env, 0.06 * sinP * env)
            b.set(i, 6, -0.05, -0.05 * sinP * env, 0.05 * sinP * env)
            b.set(i, 9, -0.04, -0.04 * sinP * env, 0.04 * sinP * env)
            b.set(i, 13, 0.0, shAccent, 0.04 * sinP * env)
            b.set(i, 14, 0.0, -shAccent, -0.04 * sinP * env)
            b.set(i, 12, 0.03, 0.06 * cosP * env, 0.12 * sinP * env)
            b.set(i, 15, 0.05, 0.12 * cosP * env, 0.28 * sinP * env)

            val armLag = phase - 0.22
            val wristLag = phase - 0.45
            b.set(i, 16, 0.28 * sin(armLag) * env, -0.78 - 0.16 * cos(armLag) * env, 0.35)
            b.set(i, 18, 0.92 + 0.28 * sin(armLag) * env, -0.18, 0.15)
            b.set(i, 20, 0.48 * cos(wristLag) * env, 0.38 * sin(wristLag) * env, -0.35)
            b.set(i, 17, 0.28 * sin(armLag + PI) * env, 0.78 + 0.16 * cos(armLag) * env, -0.35)
            b.set(i, 19, 0.92 + 0.28 * sin(armLag + PI) * env, 0.18, -0.15)
            b.set(i, 21, -0.48 * cos(wristLag) * env, -0.38 * sin(wristLag) * env, 0.35)

            schedule(b, i, sinP, 0.20, 0.12) { sw -> sin(sw * PI).pow(1.4) * env }
        }
        return b.result(startHeading)
    }

    /** Vattam circular turn holding deep Mandiya with flowing arm sweeps. */
    fun danceTurn(duration: Double, fps: Int, angleDegrees: Double,
                  startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "mandiya")
        val angleRad = Math.toRadians(angleDegrees)
        val baseZ = 0.65
        for (i in 0 until b.n) {
            val t = b.times[i]
            val s = t / duration
            val turnProg = minJerk(s) * angleRad
            val currentYaw = startHeading + turnProg
            val sinP = sin(2.0 * PI * 1.6 * t)

            b.trans[i][0] = startRoot[0] + 0.04 * sin(turnProg)
            b.trans[i][1] = startRoot[1] + 0.04 * (1.0 - cos(turnProg))
            b.trans[i][2] = baseZ + 0.02 * abs(sinP)
            b.set(i, 0, 0.06, 0.04 * sinP, currentYaw)

            val vp = turnProg * 2.0
            b.set(i, 3, 0.08, -0.05 * sin(vp), 0.08 * cos(vp))
            b.set(i, 6, -0.04, -0.04 * sin(vp), 0.06 * cos(vp))
            b.set(i, 15, 0.05, 0.10 * cos(vp), 0.30 * sin(vp))
            b.set(i, 16, 0.32 * cos(vp), -0.72, 0.35)
            b.set(i, 17, -0.32 * cos(vp), 0.72, -0.35)
            b.set(i, 18, 0.95 + 0.22 * sin(vp), -0.18, 0.12)
            b.set(i, 19, 0.95 - 0.22 * sin(vp), 0.18, -0.12)
            b.set(i, 20, 0.40 * sin(vp), 0.30 * cos(vp), -0.25)
            b.set(i, 21, -0.40 * sin(vp), -0.30 * cos(vp), 0.25)

            schedule(b, i, sinP, 0.15, 0.09)
        }
        return b.result(startHeading + angleRad)
    }

    /** Samapthiya heroic final statue posture with subtle breathing. */
    fun danceFreeze(duration: Double, fps: Int, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "mandiya")
        val baseZ = 0.63
        for (i in 0 until b.n) {
            val t = b.times[i]
            val settle = minJerk(min(1.0, t / 0.4))
            val breath = 0.008 * sin(2.0 * PI * 0.3 * t)
            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]
            b.trans[i][2] = (1.0 - settle) * startRoot[2] + settle * baseZ + breath * 0.2
            b.set(i, 0, 0.06, 0.0, startHeading)
            b.set(i, 3, 0.10 + breath * 0.4, 0.0, 0.0)
            b.set(i, 6, -0.08 + breath * 0.3, 0.0, 0.0)
            b.set(i, 9, -0.06, 0.0, 0.0)
            b.set(i, 12, 0.05, 0.0, 0.0)
            b.set(i, 15, 0.12, 0.0, 0.0)
            b.set(i, 16, 0.05, -0.68, 0.32)
            b.set(i, 17, 0.05, 0.68, -0.32)
            b.set(i, 18, 1.22, -0.16, 0.0)
            b.set(i, 19, 1.22, 0.16, 0.0)
            b.set(i, 20, 0.32, 0.0, 0.0)
            b.set(i, 21, 0.32, 0.0, 0.0)
            b.lp[i] = true; b.rp[i] = true
        }
        return b.result(startHeading)
    }

    // ── Pop-dance vocabulary (see zd_ecn_motion/generator/actions.py) ──────────
    // Moves needing sliding / pivoting / toe-standing feet emit explicit foot
    // overrides; arm angles use the generator's hanging-arm convention.

    private const val STANCE_LAT = 0.13
    private const val STANCE_FWD = 0.10

    /** (left, right) standard stance foot positions for a root and heading. */
    private fun stanceFeet(root: DoubleArray, h: Double): Pair<DoubleArray, DoubleArray> {
        val f = fwd(h); val l = lat(h)
        val left = DoubleArray(3) { k -> (if (k < 2) root[k] else 0.0) + l[k] * STANCE_LAT + f[k] * STANCE_FWD }
        val right = DoubleArray(3) { k -> (if (k < 2) root[k] else 0.0) - l[k] * STANCE_LAT + f[k] * STANCE_FWD }
        return left to right
    }

    /** Backslide: loaded flat foot glides back past the toe-standing foot, then they swap. */
    fun moonwalk(duration: Double, fps: Int, stride: Double, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard", endpoint = true)
        val h = startHeading
        val f = fwd(h); val l = lat(h)
        val steps = max(2, Math.rint(duration / 0.55).toInt())
        val half = duration / steps
        val toePitch = 0.95
        val swap = 0.18

        for (i in 0 until b.n) {
            val t = b.times[i]
            val k = min(steps - 1, (t / half).toInt())
            val u = (t - k * half) / half
            val slide = 0.5 - 0.5 * cos(PI * min(1.0, max(0.0, (u - swap) / (1.0 - swap))))
            val rollIn = minJerk(u / swap)
            val offsets = DoubleArray(2)
            for (foot in 0..1) {
                val sliding = (k % 2 == 0) == (foot == 1)
                val a = if (sliding) max(0, k - 1).toDouble() else k.toDouble()
                val e = if (!sliding) k.toDouble() else if (k < steps - 1) (k + 1).toDouble() else k.toDouble()
                val back = (a + (e - a) * slide) * stride
                val moving = a != e
                var pitch = when {
                    k == 0 -> if (moving) 0.0 else toePitch * rollIn
                    moving -> toePitch * (1.0 - rollIn)
                    else -> toePitch * rollIn
                }
                if (k == steps - 1) pitch *= 1.0 - minJerk((u - 0.7) / 0.3)
                val side = if (foot == 0) STANCE_LAT else -STANCE_LAT
                b.setOverride(i, foot,
                    startRoot[0] + l[0] * side + f[0] * (STANCE_FWD - back),
                    startRoot[1] + l[1] * side + f[1] * (STANCE_FWD - back), 0.0, h, pitch)
                offsets[foot] = back
            }
            val meanBack = 0.5 * (offsets[0] + offsets[1])
            val phase = PI * t / half
            val bob = 0.012 * sin(phase).pow(2)
            b.trans[i][0] = startRoot[0] - f[0] * meanBack
            b.trans[i][1] = startRoot[1] - f[1] * meanBack
            b.trans[i][2] = 0.765 - bob

            b.set(i, 0, 0.05, 0.03 * sin(phase), h)
            b.set(i, 3, 0.03, 0.0, 0.0)
            b.set(i, 9, -0.03, 0.0, 0.0)
            b.set(i, 15, -0.06, 0.0, 0.0)
            b.set(i, 16, 0.28 * cos(phase), -0.12, 0.0)
            b.set(i, 17, -0.28 * cos(phase), 0.12, 0.0)
            b.set(i, 18, 0.55 + 0.15 * max(0.0, cos(phase)), 0.0, 0.0)
            b.set(i, 19, 0.55 + 0.15 * max(0.0, -cos(phase)), 0.0, 0.0)
            b.set(i, 20, 0.25, 0.0, 0.0)
            b.set(i, 21, 0.25, 0.0, 0.0)
        }
        // Planted (locked, weight-bearing) only while the foot is not gliding
        for (i in 0 until b.n) {
            val nx = b.override[min(b.n - 1, i + 1)]
            val pv = b.override[max(0, i - 1)]
            for (foot in 0..1) {
                val moving = kotlin.math.hypot(nx[foot][0] - pv[foot][0], nx[foot][1] - pv[foot][1]) > 1e-6
                if (foot == 0) b.lp[i] = !moving else b.rp[i] = !moving
            }
        }
        return b.result(startHeading, withOverride = true)
    }

    /** Pivot on the ball of the right foot, left foot tucked at the knee, arms drawn in. */
    fun spin(duration: Double, fps: Int, angleDegrees: Double, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard", endpoint = true)
        val angle = Math.toRadians(angleDegrees)
        val h0 = startHeading
        val f0 = fwd(h0); val l0 = lat(h0)
        val pivot = doubleArrayOf(startRoot[0] - l0[0] * STANCE_LAT + f0[0] * STANCE_FWD,
            startRoot[1] - l0[1] * STANCE_LAT + f0[1] * STANCE_FWD, 0.0)

        for (i in 0 until b.n) {
            val s = b.times[i] / duration
            val tuck = minJerk(s / 0.2) * (1.0 - minJerk((s - 0.8) / 0.2))
            val h = h0 + angle * minJerk((s - 0.2) / 0.6)
            val f = fwd(h); val l = lat(h)
            val latOff = STANCE_LAT - 0.07 * tuck
            for (k in 0..1) b.trans[i][k] = pivot[k] + l[k] * latOff - f[k] * STANCE_FWD
            b.trans[i][2] = 0.78 + 0.05 * tuck
            val r = b.trans[i]
            val nx = r[0] + l[0] * STANCE_LAT + f[0] * STANCE_FWD
            val ny = r[1] + l[1] * STANCE_LAT + f[1] * STANCE_FWD
            val tx = r[0] + l[0] * 0.04 + f[0] * 0.14
            val ty = r[1] + l[1] * 0.04 + f[1] * 0.14
            b.setOverride(i, 0, nx + (tx - nx) * tuck, ny + (ty - ny) * tuck, 0.32 * tuck, h, 0.7 * tuck)
            b.setOverride(i, 1, pivot[0], pivot[1], 0.0, h, 0.55 * tuck)
            b.lp[i] = tuck <= 0.0; b.rp[i] = true

            b.set(i, 0, 0.03 * tuck, 0.0, h)
            b.set(i, 3, 0.04 * tuck, 0.0, 0.0)
            b.set(i, 15, -0.05 * tuck, 0.0, 0.0)
            b.set(i, 16, -0.55 * tuck + 0.04, 0.15 * tuck - 0.12, 0.0)
            b.set(i, 17, -0.55 * tuck + 0.04, -0.15 * tuck + 0.12, 0.0)
            b.set(i, 18, 0.25 + 1.75 * tuck, 0.0, 0.0)
            b.set(i, 19, 0.25 + 1.75 * tuck, 0.0, 0.0)
        }
        return b.result(h0 + angle, withOverride = true)
    }

    /** Rise onto both toes, hold with one arm lifted, drop back. */
    fun toeStand(duration: Double, fps: Int, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard", endpoint = true)
        val h = startHeading
        val (left, right) = stanceFeet(startRoot, h)
        for (i in 0 until b.n) {
            val s = b.times[i] / duration
            val rise = minJerk(s / 0.3) * (1.0 - minJerk((s - 0.75) / 0.25))
            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]; b.trans[i][2] = 0.78 + 0.075 * rise
            b.setOverride(i, 0, left[0], left[1], 0.0, h, 1.05 * rise)
            b.setOverride(i, 1, right[0], right[1], 0.0, h, 1.05 * rise)
            b.lp[i] = true; b.rp[i] = true

            b.set(i, 0, 0.0, 0.0, h)
            b.set(i, 3, -0.05 * rise, 0.0, 0.0)
            b.set(i, 6, -0.04 * rise, 0.0, 0.0)
            b.set(i, 12, -0.08 * rise, 0.0, 0.0)
            b.set(i, 15, -0.12 * rise, 0.0, 0.0)
            b.set(i, 16, 0.15 * rise, -0.20, 0.0)
            b.set(i, 17, -1.05 * rise, 0.12 + 0.2 * rise, -0.3 * rise)
            b.set(i, 18, 0.20, 0.0, 0.0)
            b.set(i, 19, 0.30 + 1.10 * rise, 0.0, 0.0)
            b.set(i, 21, -0.30 * rise, 0.0, 0.0)
        }
        return b.result(h, withOverride = true)
    }

    // Staccato hit poses: joint -> (pitch, roll, yaw); key 24 = pelvis (pitch, roll, yaw offset)
    private val POP_NEUTRAL = mapOf(16 to doubleArrayOf(0.04, -0.18, 0.05), 17 to doubleArrayOf(0.04, 0.18, -0.05),
        18 to doubleArrayOf(0.25, 0.0, 0.0), 19 to doubleArrayOf(0.25, 0.0, 0.0))
    private val POP_POSES: List<Map<Int, DoubleArray>> = listOf(
        mapOf(24 to doubleArrayOf(0.02, 0.0, 0.15), 3 to doubleArrayOf(0.0, 0.0, 0.10), 15 to doubleArrayOf(0.0, 0.0, 0.45),
            16 to doubleArrayOf(0.10, -0.15, 0.0), 17 to doubleArrayOf(-0.95, 0.20, -0.20), 18 to doubleArrayOf(0.20, 0.0, 0.0),
            19 to doubleArrayOf(1.45, 0.0, 0.0), 21 to doubleArrayOf(-0.40, 0.0, 0.0)),
        mapOf(24 to doubleArrayOf(0.0, 0.08, -0.10), 13 to doubleArrayOf(0.0, 0.0, 0.20), 14 to doubleArrayOf(0.0, 0.0, -0.20),
            15 to doubleArrayOf(-0.05, 0.0, -0.45), 16 to doubleArrayOf(0.20, -0.45, 0.0), 17 to doubleArrayOf(0.20, 0.45, 0.0),
            18 to doubleArrayOf(1.60, 0.0, 0.0), 19 to doubleArrayOf(1.60, 0.0, 0.0)),
        mapOf(24 to doubleArrayOf(0.0, -0.06, -0.20), 3 to doubleArrayOf(0.0, 0.0, -0.12), 9 to doubleArrayOf(0.0, 0.0, -0.08),
            15 to doubleArrayOf(-0.08, 0.0, -0.30), 16 to doubleArrayOf(-1.40, -0.10, 0.20), 17 to doubleArrayOf(-0.50, 0.15, 0.30),
            18 to doubleArrayOf(0.15, 0.0, 0.0), 19 to doubleArrayOf(2.00, 0.0, 0.0)),
        mapOf(24 to doubleArrayOf(-0.10, 0.0, 0.0), 3 to doubleArrayOf(0.10, 0.0, 0.0), 12 to doubleArrayOf(0.15, 0.0, 0.0),
            15 to doubleArrayOf(0.20, 0.0, 0.0), 16 to doubleArrayOf(0.25, -0.30, 0.0), 17 to doubleArrayOf(0.25, 0.30, 0.0),
            18 to doubleArrayOf(1.00, 0.0, 0.0), 19 to doubleArrayOf(1.00, 0.0, 0.0)),
    )

    private fun popPose(pose: Map<Int, DoubleArray>): Array<DoubleArray> {
        val arr = Array(25) { DoubleArray(3) }
        for ((j, v) in POP_NEUTRAL) v.copyInto(arr[j])
        for ((j, v) in pose) v.copyInto(arr[j])
        return arr
    }

    /** Staccato upper-body hits on each beat, knee dip on the accent, feet pinned. */
    fun popHits(duration: Double, fps: Int, tempoBpm: Double, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard", endpoint = true)
        val beat = 60.0 / max(40.0, min(200.0, tempoBpm))
        val poses = POP_POSES.map { popPose(it) }
        val neutral = popPose(emptyMap())
        val (left, right) = stanceFeet(startRoot, startHeading)
        for (i in 0 until b.n) {
            val t = b.times[i]
            val k = (t / beat).toInt()
            val u = (t - k * beat) / beat
            val cur = poses[k % poses.size]
            val prev = if (k == 0) neutral else poses[(k - 1) % poses.size]
            val w = minJerk(u / 0.2)
            val out = minJerk((t - (duration - 0.35)) / 0.35)
            val pose = Array(25) { j -> DoubleArray(3) { a ->
                val p = prev[j][a] + (cur[j][a] - prev[j][a]) * w
                p + (neutral[j][a] - p) * out
            } }
            val dip = 0.028 * sin(PI * min(1.0, u / 0.45)) * (1.0 - out)
            b.trans[i][0] = startRoot[0]; b.trans[i][1] = startRoot[1]; b.trans[i][2] = 0.77 - dip
            for (j in 1 until 24) pose[j].copyInto(b.rot[i][j])
            b.set(i, 0, 0.02 + pose[24][0], pose[24][1], startHeading + pose[24][2])
            b.setOverride(i, 0, left[0], left[1], 0.0, startHeading, 0.0)
            b.setOverride(i, 1, right[0], right[1], 0.0, startHeading, 0.0)
            b.lp[i] = true; b.rp[i] = true
        }
        return b.result(startHeading, withOverride = true)
    }

    /** Closing freeze: left arm points up and out, gaze follows, left foot forward on its toe. */
    fun posePoint(duration: Double, fps: Int, startRoot: DoubleArray, startHeading: Double): PhaseResult {
        val b = Buf(frameCount(duration, fps), duration, "standard", endpoint = true)
        val h = startHeading
        val f = fwd(h); val l = lat(h)
        val (left0, right0) = stanceFeet(startRoot, h)
        val leftPose = DoubleArray(3) { k -> left0[k] + f[k] * 0.10 - l[k] * 0.03 }
        val settleTime = min(0.5, duration * 0.4)
        for (i in 0 until b.n) {
            val t = b.times[i]
            val s = minJerk(t / settleTime)
            val breath = 0.006 * sin(2.0 * PI * 0.3 * t)
            b.trans[i][0] = startRoot[0] - l[0] * 0.04 * s
            b.trans[i][1] = startRoot[1] - l[1] * 0.04 * s
            b.trans[i][2] = 0.78 - 0.02 * s + breath
            b.setOverride(i, 0, left0[0] + (leftPose[0] - left0[0]) * s, left0[1] + (leftPose[1] - left0[1]) * s,
                0.0, h - 0.25 * s, 0.85 * s)
            b.setOverride(i, 1, right0[0], right0[1], 0.0, h + 0.15 * s, 0.0)
            b.lp[i] = t == 0.0 || t >= settleTime; b.rp[i] = true

            b.set(i, 0, 0.0, -0.06 * s, h - 0.10 * s)
            b.set(i, 3, -0.04 * s, 0.04 * s, -0.08 * s)
            b.set(i, 9, -0.04 * s, 0.0, -0.10 * s)
            b.set(i, 12, -0.08 * s, 0.0, -0.20 * s)
            b.set(i, 15, -0.12 * s, 0.0, -0.30 * s)
            b.set(i, 16, -2.05 * s, -0.18 + 0.48 * s, -0.5 * s)
            b.set(i, 18, 0.25 - 0.15 * s, 0.0, 0.0)
            b.set(i, 17, 0.04, 0.18 + 0.20 * s, 0.0)
            b.set(i, 19, 0.25, 0.0, 0.0)
            b.set(i, 21, -0.35 * s, 0.0, 0.0)
        }
        return b.result(h, withOverride = true)
    }
}

package com.graphicsresearch.promptmotion.motion

import com.graphicsresearch.promptmotion.planner.MotionPlan
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/** Generator output on the generator's own skeleton (see motion_generator.py). */
class MotionClip(
    val fps: Int,
    val rotations: Array<Array<DoubleArray>>,  // [n][24][3] clamped local Euler
    val translations: Array<DoubleArray>,      // [n][3] pelvis, generator world
    val leftPlanted: BooleanArray,
    val rightPlanted: BooleanArray,
    val phaseIndex: IntArray,                  // [n] index into plan.phases
    val jointLimitClampings: Int,
    val maxGroundPenetrationM: Double,
    val maxFootSlipVelocity: Double,
) {
    val frameCount: Int get() = rotations.size
    val durationSeconds: Double get() = frameCount / fps.toDouble()
}

/**
 * Port of zd_ecn_motion/generator/motion_generator.py (DeterministicMotionGenerator):
 * per-phase synthesis -> min-jerk boundary blending -> world-space foot locking
 * with analytical leg IK -> anatomical joint limits -> FK quality metrics.
 */
class MotionGenerator {
    private val defaultRoot = doubleArrayOf(0.0, 0.0, 0.78)

    fun generate(plan: MotionPlan): MotionClip {
        val fps = plan.targetFps
        var root = defaultRoot.copyOf()
        var heading = 0.0
        val results = ArrayList<PhaseResult>()

        for (phase in plan.phases) {
            val d = phase.duration
            val p = phase.parameters
            val res = when (phase.action) {
                "walk" -> Actions.walk(d, fps, p.num("speed", 1.1), p.num("step_frequency", 1.8), root, heading)
                "turn" -> Actions.turn(d, fps, p.num("angle_degrees", 90.0), root, heading)
                "squat" -> Actions.squat(d, fps, p.num("depth", 0.6), p.num("hold_duration", 0.3), root, heading)
                "jump" -> Actions.jump(d, fps, p.num("height", 0.45), p.num("forward_distance", 0.2), root, heading)
                "land" -> Actions.land(d, fps, p.num("recovery_duration", 0.5), p.num("damping", 0.85), root, heading)
                "dance_mandiya" -> Actions.danceMandiya(d, fps, root, heading)
                "dance_step" -> Actions.danceStep(d, fps, p.str("cadence", "thei_tha"), root, heading)
                "dance_turn" -> Actions.danceTurn(d, fps, p.num("angle_degrees", 360.0), root, heading)
                "dance_freeze" -> Actions.danceFreeze(d, fps, root, heading)
                "moonwalk" -> Actions.moonwalk(d, fps, p.num("stride", 0.18), root, heading)
                "spin" -> Actions.spin(d, fps, p.num("angle_degrees", 360.0), root, heading)
                "toe_stand" -> Actions.toeStand(d, fps, root, heading)
                "pop_hits" -> Actions.popHits(d, fps, p.num("tempo_bpm", 117.0), root, heading)
                "pose_point" -> Actions.posePoint(d, fps, root, heading)
                else -> Actions.idle(d, fps, root, heading)
            }
            results.add(res)
            root = res.endRoot
            heading = res.endHeading
        }

        // 1. Stitch & blend across phase boundaries
        val blendWindow = 4
        val rots = ArrayList<Array<DoubleArray>>()
        val trans = ArrayList<DoubleArray>()
        val lp = ArrayList<Boolean>()
        val rp = ArrayList<Boolean>()
        val lifts = ArrayList<DoubleArray>()
        val styles = ArrayList<String>()
        val tags = ArrayList<Int>()
        val overrides = ArrayList<Array<DoubleArray>>()
        for ((pIdx, res) in results.withIndex()) {
            val r = Array(res.rotations.size) { f -> Array(24) { j -> res.rotations[f][j].copyOf() } }
            val t = Array(res.translations.size) { res.translations[it].copyOf() }
            val n = r.size
            if (pIdx > 0 && rots.size >= blendWindow && n >= blendWindow) {
                val prevRot = rots.last()
                val prevTrans = trans.last()
                for (b in 0 until blendWindow) {
                    val w = Actions.minJerk((b + 1) / (blendWindow + 1).toDouble())
                    for (k in 0 until 3) t[b][k] = (1.0 - w) * prevTrans[k] + w * t[b][k]
                    for (j in 0 until 24) for (k in 0 until 3) {
                        r[b][j][k] = (1.0 - w) * prevRot[j][k] + w * r[b][j][k]
                    }
                }
            }
            for (i in 0 until n) {
                rots.add(r[i]); trans.add(t[i])
                lp.add(res.leftPlanted[i]); rp.add(res.rightPlanted[i])
                lifts.add(res.footLifts[i]); styles.add(res.style); tags.add(pIdx)
                overrides.add(res.footOverride?.get(i) ?: Array(2) { DoubleArray(5) { Double.NaN } })
            }
        }
        val rotations = rots.toTypedArray()
        val translations = trans.toTypedArray()
        val lPlanted = lp.toBooleanArray()
        val rPlanted = rp.toBooleanArray()

        // 2. Closed-loop foot locking + leg IK (in place)
        applyFootIk(rotations, translations, lPlanted, rPlanted, lifts.toTypedArray(), styles, overrides)

        // 3. Anatomical joint limits
        val clampings = JointLimits.clamp(rotations)

        // 4. FK quality metrics
        var maxPen = 0.0
        var maxSlip = 0.0
        var prevL: DoubleArray? = null
        var prevR: DoubleArray? = null
        var prevLp = false
        var prevRp = false
        for (f in rotations.indices) {
            val pos = Kinematics.forwardPositions(rotations[f], translations[f])
            val lf = pos[10]; val rf = pos[11]
            if (lf[2] < -1e-4) maxPen = max(maxPen, -lf[2])
            if (rf[2] < -1e-4) maxPen = max(maxPen, -rf[2])
            if (prevL != null && lPlanted[f] && prevLp) {
                val s = kotlin.math.hypot(lf[0] - prevL[0], lf[1] - prevL[1]) * fps
                if (s > 1e-3) maxSlip = max(maxSlip, s)
            }
            if (prevR != null && rPlanted[f] && prevRp) {
                val s = kotlin.math.hypot(rf[0] - prevR[0], rf[1] - prevR[1]) * fps
                if (s > 1e-3) maxSlip = max(maxSlip, s)
            }
            prevL = lf; prevR = rf; prevLp = lPlanted[f]; prevRp = rPlanted[f]
        }

        return MotionClip(fps, rotations, translations, lPlanted, rPlanted, tags.toIntArray(),
            clampings, maxPen, maxSlip)
    }

    private fun applyFootIk(
        rotations: Array<Array<DoubleArray>>,
        translations: Array<DoubleArray>,
        lPlanted: BooleanArray,
        rPlanted: BooleanArray,
        lifts: Array<DoubleArray>,
        styles: List<String>,
        overrides: List<Array<DoubleArray>>,
    ) {
        val n = rotations.size
        val nomL = Array(n) { DoubleArray(3) }
        val nomR = Array(n) { DoubleArray(3) }
        val nomYawL = DoubleArray(n)
        val nomYawR = DoubleArray(n)

        for (f in 0 until n) {
            val heading = rotations[f][0][2]
            val fwd = doubleArrayOf(-sin(heading), cos(heading), 0.0)
            val lat = doubleArrayOf(cos(heading), sin(heading), 0.0)
            val rootXy = doubleArrayOf(translations[f][0], translations[f][1], 0.0)
            val mandiya = styles[f] == "mandiya"
            val wLat = if (mandiya) 0.28 else 0.13
            val fwdOff = if (mandiya) 0.06 else 0.10
            val turnout = if (mandiya) Math.toRadians(42.0) else 0.0
            for (k in 0 until 3) {
                nomL[f][k] = rootXy[k] + lat[k] * wLat + fwd[k] * fwdOff
                nomR[f][k] = rootXy[k] - lat[k] * wLat + fwd[k] * fwdOff
            }
            nomYawL[f] = heading - turnout
            nomYawR[f] = heading + turnout
        }

        val (footL, yawL, pitchL) = solveFootChannel(nomL, nomYawL, lPlanted, DoubleArray(n) { lifts[it][0] }, Array(n) { overrides[it][0] })
        val (footR, yawR, pitchR) = solveFootChannel(nomR, nomYawR, rPlanted, DoubleArray(n) { lifts[it][1] }, Array(n) { overrides[it][1] })

        val footOffL = Kinematics.REST_OFFSETS[10]
        val footOffR = Kinematics.REST_OFFSETS[11]
        val maxLegReach = 0.742

        for (f in 0 until n) {
            val rPelvis = Kinematics.eulerToMatrix(rotations[f][0][0], rotations[f][0][1], rotations[f][0][2])
            var pelvis = translations[f]

            val pAnkL = Kinematics.sub(footL[f], Kinematics.matVec(Kinematics.eulerToMatrix(pitchL[f], 0.0, yawL[f]), footOffL))
            val pAnkR = Kinematics.sub(footR[f], Kinematics.matVec(Kinematics.eulerToMatrix(pitchR[f], 0.0, yawR[f]), footOffR))

            // Pull pelvis inside the reach sphere of planted ankles so planted feet never slip
            repeat(2) {
                if (lPlanted[f]) {
                    val vL = Kinematics.sub(Kinematics.add(pelvis, Kinematics.matVec(rPelvis, Kinematics.REST_OFFSETS[1])), pAnkL)
                    val dL = Kinematics.norm(vL)
                    if (dL > maxLegReach) pelvis = Kinematics.sub(pelvis, Kinematics.scale(vL, (dL - maxLegReach) / dL))
                }
                if (rPlanted[f]) {
                    val vR = Kinematics.sub(Kinematics.add(pelvis, Kinematics.matVec(rPelvis, Kinematics.REST_OFFSETS[2])), pAnkR)
                    val dR = Kinematics.norm(vR)
                    if (dR > maxLegReach) pelvis = Kinematics.sub(pelvis, Kinematics.scale(vR, (dR - maxLegReach) / dR))
                }
            }
            translations[f] = pelvis

            val mandiya = styles[f] == "mandiya"
            val poleL = if (mandiya) doubleArrayOf(1.0, 0.38, 0.0) else doubleArrayOf(0.12, 1.0, 0.0)
            val poleR = if (mandiya) doubleArrayOf(-1.0, 0.38, 0.0) else doubleArrayOf(-0.12, 1.0, 0.0)

            val legL = Kinematics.solveLegIk(pelvis, rPelvis, footL[f], yawL[f], pitchL[f], poleL, true)
            val legR = Kinematics.solveLegIk(pelvis, rPelvis, footR[f], yawR[f], pitchR[f], poleR, false)
            rotations[f][1] = legL[0]; rotations[f][4] = legL[1]; rotations[f][7] = legL[2]
            rotations[f][2] = legR[0]; rotations[f][5] = legR[1]; rotations[f][8] = legR[2]

            // Toes stay flat on the floor when a grounded foot raises its heel (MTP dorsiflexion)
            if (footL[f][2] <= 1e-6 && pitchL[f] > 0.0) rotations[f][10][0] = -pitchL[f]
            if (footR[f][2] <= 1e-6 && pitchR[f] > 0.0) rotations[f][11][0] = -pitchR[f]
        }
    }

    /**
     * Locks XY + yaw during each contiguous planted interval (Z = 0) and fills swing
     * intervals with minimum-jerk arcs between the neighbouring locked positions.
     * Frames with an explicit override (x, y, z, yaw, pitch; non-NaN) take it verbatim
     * and form their own intervals; a stance touching an override keeps that foot spot.
     */
    private fun solveFootChannel(
        nomPos: Array<DoubleArray>,
        nomYaw: DoubleArray,
        planted: BooleanArray,
        liftZ: DoubleArray,
        override: Array<DoubleArray>,
    ): Triple<Array<DoubleArray>, DoubleArray, DoubleArray> {
        val n = planted.size
        val outPos = Array(n) { nomPos[it].copyOf() }
        val outYaw = nomYaw.copyOf()
        val outPitch = DoubleArray(n)

        // 0 = swing, 1 = planted, 2 = explicit override
        val stateOf = IntArray(n) { f -> if (!override[f][0].isNaN()) 2 else if (planted[f]) 1 else 0 }
        data class Interval(val start: Int, val end: Int, val state: Int)
        val intervals = ArrayList<Interval>()
        var idx = 0
        while (idx < n) {
            val state = stateOf[idx]
            var end = idx
            while (end + 1 < n && stateOf[end + 1] == state) end++
            intervals.add(Interval(idx, end, state))
            idx = end + 1
        }

        for (iv in intervals) if (iv.state == 2) {
            for (f in iv.start..iv.end) {
                outPos[f][0] = override[f][0]; outPos[f][1] = override[f][1]; outPos[f][2] = override[f][2]
                outYaw[f] = override[f][3]; outPitch[f] = override[f][4]
            }
        }
        for ((k, iv) in intervals.withIndex()) if (iv.state == 1) {
            val mid = (iv.start + iv.end) / 2
            var lx = nomPos[mid][0]; var ly = nomPos[mid][1]; var lyaw = nomYaw[mid]
            if (k > 0 && intervals[k - 1].state == 2) {
                lx = outPos[iv.start - 1][0]; ly = outPos[iv.start - 1][1]; lyaw = outYaw[iv.start - 1]
            } else if (k + 1 < intervals.size && intervals[k + 1].state == 2) {
                lx = outPos[iv.end + 1][0]; ly = outPos[iv.end + 1][1]; lyaw = outYaw[iv.end + 1]
            }
            for (f in iv.start..iv.end) {
                outPos[f][0] = lx; outPos[f][1] = ly; outPos[f][2] = 0.0
                outYaw[f] = lyaw; outPitch[f] = 0.0
            }
        }

        for (iv in intervals) if (iv.state == 0) {
            val s0 = iv.start; val e0 = iv.end
            val pStart = if (s0 > 0) outPos[s0 - 1] else nomPos[s0]
            val yStart = if (s0 > 0) outYaw[s0 - 1] else nomYaw[s0]
            val pEnd = if (e0 + 1 < n) outPos[e0 + 1] else nomPos[e0]
            val yEnd = if (e0 + 1 < n) outYaw[e0 + 1] else nomYaw[e0]
            val sx = pStart[0]; val sy = pStart[1]; val ex = pEnd[0]; val ey = pEnd[1]
            val span = max(1, e0 - s0 + 1)
            for (f in s0..e0) {
                val tau = (f - s0 + 1) / (span + 1).toDouble()
                val s = Actions.minJerk(tau)
                outPos[f][0] = (1.0 - s) * sx + s * ex
                outPos[f][1] = (1.0 - s) * sy + s * ey
                outPos[f][2] = max(0.005, liftZ[f])
                outYaw[f] = (1.0 - s) * yStart + s * yEnd
                outPitch[f] = 0.18 * sin(2.0 * PI * tau) * sin(PI * tau)
            }
        }
        return Triple(outPos, outYaw, outPitch)
    }
}

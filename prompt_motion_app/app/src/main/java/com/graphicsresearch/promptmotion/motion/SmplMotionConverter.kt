package com.graphicsresearch.promptmotion.motion

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.atan2
import kotlin.math.sqrt

/** Per-frame SMPL pose stream consumed by the native GPU pipeline. */
class SmplClip(
    val fps: Int,
    val frameCount: Int,
    val axisAngle: FloatArray,  // [n*72]  SMPL local axis-angle (root includes Z-up world orientation)
    val rot6d: FloatArray,      // [n*144] first two rotation-matrix columns per joint (ZD-ECN input)
    val trans: FloatArray,      // [n*3]   Z-up world offset added to the LBS output (pelvis rest at J0)
)

/**
 * Kotlin mirror of prompt_motion_app/tools/motion_to_smpl.py — see that file for the
 * frame derivation. Summary:
 *   M = swap(y,z) (SMPL body <-> generator body), W = diag(1,-1,1) (generator world ->
 *   right-handed Z-up world, the frame ZD-ECN was trained in).
 *   L_smpl_0 = W L_gen_0 M,  L_smpl_j = M L_gen_j M,  p_world = W p_gen.
 */
class SmplMotionConverter(vTemplate: FloatArray, jointRegressor: FloatArray) {

    companion object {
        const val V = 6890
        const val J = 24
        /** Generator straight-leg pelvis height: hip 0.06 + femur 0.38 + tibia 0.38 + foot 0.05 */
        const val GEN_STANDING_PELVIS_HEIGHT = 0.87

        fun floatsLE(bytes: ByteArray): FloatArray {
            val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(fb.remaining()).also { fb.get(it) }
        }

        /** Shepperd quaternion extraction -> axis-angle; stable at theta ~ pi. */
        fun rotmatToAxisAngle(r: DoubleArray, out: FloatArray, off: Int) {
            val tr = r[0] + r[4] + r[8]
            var w: Double; var x: Double; var y: Double; var z: Double
            if (tr > 0.0) {
                val s = sqrt(tr + 1.0) * 2.0
                w = 0.25 * s; x = (r[7] - r[5]) / s; y = (r[2] - r[6]) / s; z = (r[3] - r[1]) / s
            } else if (r[0] > r[4] && r[0] > r[8]) {
                val s = sqrt(1.0 + r[0] - r[4] - r[8]) * 2.0
                w = (r[7] - r[5]) / s; x = 0.25 * s; y = (r[1] + r[3]) / s; z = (r[2] + r[6]) / s
            } else if (r[4] > r[8]) {
                val s = sqrt(1.0 + r[4] - r[0] - r[8]) * 2.0
                w = (r[2] - r[6]) / s; x = (r[1] + r[3]) / s; y = 0.25 * s; z = (r[5] + r[7]) / s
            } else {
                val s = sqrt(1.0 + r[8] - r[0] - r[4]) * 2.0
                w = (r[3] - r[1]) / s; x = (r[2] + r[6]) / s; y = (r[5] + r[7]) / s; z = 0.25 * s
            }
            if (w < 0.0) { w = -w; x = -x; y = -y; z = -z }
            val vn = sqrt(x * x + y * y + z * z)
            if (vn < 1e-12) { out[off] = 0f; out[off + 1] = 0f; out[off + 2] = 0f; return }
            val k = 2.0 * atan2(vn, w) / vn
            out[off] = (x * k).toFloat(); out[off + 1] = (y * k).toFloat(); out[off + 2] = (z * k).toFloat()
        }
    }

    private val restJoints: Array<DoubleArray>
    private val scale: Double
    private val soleHeight = DoubleArray(J)

    init {
        require(vTemplate.size == V * 3 && jointRegressor.size == J * V) { "SMPL rest asset shape mismatch" }
        restJoints = Array(J) { j ->
            val p = DoubleArray(3)
            for (i in 0 until V) {
                val w = jointRegressor[j * V + i].toDouble()
                if (w != 0.0) for (k in 0 until 3) p[k] += w * vTemplate[i * 3 + k]
            }
            p
        }
        var floorY = Double.MAX_VALUE
        for (i in 0 until V) floorY = minOf(floorY, vTemplate[i * 3 + 1].toDouble())
        scale = (restJoints[0][1] - floorY) / GEN_STANDING_PELVIS_HEIGHT
        for (j in intArrayOf(7, 8, 10, 11)) soleHeight[j] = restJoints[j][1] - floorY
    }

    /** Rest pelvis J0 (Z-up world == LBS output frame). */
    fun restPelvis(): DoubleArray = restJoints[0].copyOf()

    // Arm retargeting (see motion_to_smpl.py): the generator authors arm rotations for
    // arms hanging along -Z, its rest skeleton is a T-pose. B maps T-pose bone -> hanging.
    //   shoulder L = R·B,  elbow/wrist/hand L = Bᵀ·R·B,  elbow pitch negated (flexion forward)
    private val bLeft = Kinematics.eulerToMatrix(0.0, Math.PI / 2, 0.0)    // Ry(+90°): +X -> -Z
    private val bRight = Kinematics.eulerToMatrix(0.0, -Math.PI / 2, 0.0)  // Ry(-90°): -X -> -Z

    private fun genLocal(j: Int, e: DoubleArray): DoubleArray {
        val pitch = if (j == 18 || j == 19) -e[0] else e[0]
        val r = Kinematics.eulerToMatrix(pitch, e[1], e[2])
        if (j < 16) return r
        val b = if (j % 2 == 0) bLeft else bRight
        return if (j <= 17) Kinematics.matMul(r, b)
               else Kinematics.matMul(Kinematics.transpose(b), Kinematics.matMul(r, b))
    }

    private val LEGS = arrayOf(intArrayOf(1, 4, 7, 10), intArrayOf(2, 5, 8, 11))  // hip, knee, ankle, foot

    private fun smplFk(local: Array<DoubleArray>, root: DoubleArray, g: Array<DoubleArray>, p: Array<DoubleArray>) {
        g[0] = local[0]; p[0] = root
        for (j in 1 until J) {
            val par = Kinematics.PARENTS[j]
            g[j] = Kinematics.matMul(g[par], local[j])
            p[j] = Kinematics.add(p[par], Kinematics.matVec(g[par], Kinematics.sub(restJoints[j], restJoints[par])))
        }
    }

    /** Smallest rotation taking direction a onto direction b (row-major 3×3). */
    private fun alignRotation(a0: DoubleArray, b0: DoubleArray): DoubleArray {
        val a = Kinematics.scale(a0, 1.0 / Kinematics.norm(a0))
        val b = Kinematics.scale(b0, 1.0 / Kinematics.norm(b0))
        val v = Kinematics.cross(a, b)
        val c = Kinematics.dot(a, b)
        if (c < -1.0 + 1e-9) return doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        val vx = doubleArrayOf(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
        val vx2 = Kinematics.matMul(vx, vx)
        val k = 1.0 / (1.0 + c)
        return DoubleArray(9) { i -> (if (i % 4 == 0) 1.0 else 0.0) + vx[i] + vx2[i] * k }
    }

    /**
     * Two-bone IK on the SMPL skeleton so each SMPL foot joint lands on the generator's
     * foot target; knees keep the generator's bend plane, feet keep their orientation.
     * Absorbs the SMPL / generator bone-length mismatch. Mirrors retarget_legs().
     */
    private fun retargetLegs(local: Array<DoubleArray>, root: DoubleArray, footTargets: Array<DoubleArray>) {
        val g = Array(J) { DoubleArray(9) }
        val p = Array(J) { DoubleArray(3) }
        smplFk(local, root, g, p)
        for ((leg, idx) in LEGS.withIndex()) {
            val (h, k, a, ft) = idx.toList()
            val targetFoot = doubleArrayOf(footTargets[leg][0], footTargets[leg][1], footTargets[leg][2] + soleHeight[ft])
            val targetAnkle = Kinematics.sub(targetFoot, Kinematics.matVec(g[a], Kinematics.sub(restJoints[ft], restJoints[a])))
            val l1 = Kinematics.norm(Kinematics.sub(restJoints[k], restJoints[h]))
            val l2 = Kinematics.norm(Kinematics.sub(restJoints[a], restJoints[k]))
            val dVec = Kinematics.sub(targetAnkle, p[h])
            val d = Kinematics.norm(dVec).coerceIn(kotlin.math.abs(l1 - l2) + 1e-4, (l1 + l2) * 0.9995)
            val u = Kinematics.scale(dVec, 1.0 / Kinematics.norm(dVec))
            val ankleNew = Kinematics.add(p[h], Kinematics.scale(u, d))
            val thigh = Kinematics.sub(p[k], p[h])
            var bend = Kinematics.sub(thigh, Kinematics.scale(u, Kinematics.dot(thigh, u)))
            if (Kinematics.norm(bend) < 1e-6) {
                bend = Kinematics.matVec(g[h], doubleArrayOf(0.0, 0.0, 1.0))  // SMPL forward
                bend = Kinematics.sub(bend, Kinematics.scale(u, Kinematics.dot(bend, u)))
            }
            bend = Kinematics.scale(bend, 1.0 / Kinematics.norm(bend))
            val cosA = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
            val kneeNew = Kinematics.add(p[h], Kinematics.scale(
                Kinematics.add(Kinematics.scale(u, cosA), Kinematics.scale(bend, sqrt(maxOf(0.0, 1.0 - cosA * cosA)))), l1))

            val gAnkle = g[a]
            val gHip = Kinematics.matMul(alignRotation(thigh, Kinematics.sub(kneeNew, p[h])), g[h])
            val kneeLocalNow = Kinematics.matMul(gHip, local[k])
            val shinNow = Kinematics.matVec(kneeLocalNow, Kinematics.sub(restJoints[a], restJoints[k]))
            val gKnee = Kinematics.matMul(alignRotation(shinNow, Kinematics.sub(ankleNew, kneeNew)), kneeLocalNow)
            local[h] = Kinematics.matMul(Kinematics.transpose(g[0]), gHip)  // hips are children of the pelvis
            local[k] = Kinematics.matMul(Kinematics.transpose(gHip), gKnee)
            local[a] = Kinematics.matMul(Kinematics.transpose(gKnee), gAnkle)
        }
    }

    // M·R·M (swap rows y,z then columns y,z) and W·R·M (negate row y, then swap columns y,z)
    private fun mrm(r: DoubleArray) = doubleArrayOf(r[0], r[2], r[1], r[6], r[8], r[7], r[3], r[5], r[4])
    private fun wrm(r: DoubleArray) = doubleArrayOf(r[0], r[2], r[1], -r[3], -r[5], -r[4], r[6], r[8], r[7])

    fun convert(clip: MotionClip): SmplClip {
        val n = clip.frameCount
        val aa = FloatArray(n * 72)
        val r6 = FloatArray(n * 144)
        val pelvis = Array(n) { DoubleArray(3) }
        val ground = DoubleArray(n) { Double.NaN }
        val local = Array(J) { DoubleArray(9) }
        val g = Array(J) { DoubleArray(9) }
        val p = Array(J) { DoubleArray(3) }

        for (f in 0 until n) {
            for (j in 0 until J) {
                val rg = genLocal(j, clip.rotations[f][j])
                local[j] = if (j == 0) wrm(rg) else mrm(rg)
            }
            val t = clip.translations[f]
            val root = doubleArrayOf(t[0] * scale, -t[1] * scale, t[2] * scale)
            pelvis[f] = root

            // Leg IK on the SMPL skeleton towards the generator's (scaled) foot joints
            val gen = Kinematics.forwardPositions(clip.rotations[f], clip.translations[f])
            val feet = arrayOf(
                doubleArrayOf(gen[10][0] * scale, -gen[10][1] * scale, gen[10][2] * scale),
                doubleArrayOf(gen[11][0] * scale, -gen[11][1] * scale, gen[11][2] * scale),
            )
            retargetLegs(local, root, feet)

            for (j in 0 until J) {
                rotmatToAxisAngle(local[j], aa, f * 72 + j * 3)
                val o = f * 144 + j * 6
                r6[o] = local[j][0].toFloat(); r6[o + 1] = local[j][3].toFloat(); r6[o + 2] = local[j][6].toFloat()
                r6[o + 3] = local[j][1].toFloat(); r6[o + 4] = local[j][4].toFloat(); r6[o + 5] = local[j][7].toFloat()
            }

            // Ground contact: every foot the generator has on the floor (planted or gliding)
            smplFk(local, root, g, p)
            var lo = Double.NaN
            for ((leg, ft) in intArrayOf(10, 11).withIndex()) {
                if (gen[ft][2] <= 1e-3) {
                    val v = p[LEGS[leg][3]][2] - soleHeight[ft]
                    lo = if (lo.isNaN()) v else minOf(lo, v)
                }
            }
            ground[f] = lo
        }

        // Airborne frames: linear blend of neighbouring contact offsets (np.interp semantics)
        val known = (0 until n).filter { !ground[it].isNaN() }
        if (known.isEmpty()) ground.fill(0.0) else {
            for (f in 0 until n) if (ground[f].isNaN()) {
                val prev = known.lastOrNull { it < f }
                val next = known.firstOrNull { it > f }
                ground[f] = when {
                    prev == null -> ground[next!!]
                    next == null -> ground[prev]
                    else -> ground[prev] + (ground[next] - ground[prev]) * (f - prev) / (next - prev).toDouble()
                }
            }
        }
        // 5-tap [1,2,3,2,1]/9 smoothing with edge padding
        val kernel = doubleArrayOf(1.0, 2.0, 3.0, 2.0, 1.0)
        val smooth = DoubleArray(n) { f ->
            var acc = 0.0
            for (k in -2..2) acc += kernel[k + 2] * ground[(f + k).coerceIn(0, n - 1)]
            acc / 9.0
        }

        val trans = FloatArray(n * 3)
        val j0 = restJoints[0]
        for (f in 0 until n) {
            trans[f * 3] = (pelvis[f][0] - j0[0]).toFloat()
            trans[f * 3 + 1] = (pelvis[f][1] - j0[1]).toFloat()
            trans[f * 3 + 2] = (pelvis[f][2] - smooth[f] - j0[2]).toFloat()
        }
        return SmplClip(clip.fps, n, aa, r6, trans)
    }
}

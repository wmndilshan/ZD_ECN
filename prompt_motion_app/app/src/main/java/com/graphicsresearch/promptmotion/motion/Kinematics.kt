package com.graphicsresearch.promptmotion.motion

import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Port of zd_ecn_motion/generator/kinematics.py — SMPL-24 forward kinematics and
 * closed-form two-bone leg IK on the generator's own (Z-up) skeleton.
 *
 * Matrices are row-major DoubleArray(9); vectors are DoubleArray(3).
 */
object Kinematics {

    val PARENTS = intArrayOf(-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)

    // Canonical generator skeleton rest offsets (metres)
    val REST_OFFSETS: Array<DoubleArray> = arrayOf(
        v(0.0, 0.0, 0.84),      // 0: Pelvis
        v(0.12, -0.02, -0.06),  // 1: L_Hip
        v(-0.12, -0.02, -0.06), // 2: R_Hip
        v(0.0, 0.02, 0.14),     // 3: Spine1
        v(0.0, 0.0, -0.38),     // 4: L_Knee
        v(0.0, 0.0, -0.38),     // 5: R_Knee
        v(0.0, 0.0, 0.14),      // 6: Spine2
        v(0.0, 0.0, -0.38),     // 7: L_Ankle
        v(0.0, 0.0, -0.38),     // 8: R_Ankle
        v(0.0, 0.0, 0.14),      // 9: Spine3
        v(0.0, 0.12, -0.05),    // 10: L_Foot
        v(0.0, 0.12, -0.05),    // 11: R_Foot
        v(0.0, 0.0, 0.12),      // 12: Neck
        v(0.08, -0.02, 0.08),   // 13: L_Collar
        v(-0.08, -0.02, 0.08),  // 14: R_Collar
        v(0.0, 0.0, 0.14),      // 15: Head
        v(0.15, 0.0, 0.0),      // 16: L_Shoulder
        v(-0.15, 0.0, 0.0),     // 17: R_Shoulder
        v(0.24, 0.0, 0.0),      // 18: L_Elbow
        v(-0.24, 0.0, 0.0),     // 19: R_Elbow
        v(0.22, 0.0, 0.0),      // 20: L_Wrist
        v(-0.22, 0.0, 0.0),     // 21: R_Wrist
        v(0.08, 0.0, 0.0),      // 22: L_Hand
        v(-0.08, 0.0, 0.0),     // 23: R_Hand
    )

    // ── small vector / matrix helpers ───────────────────────────────────────
    fun v(x: Double, y: Double, z: Double) = doubleArrayOf(x, y, z)
    fun add(a: DoubleArray, b: DoubleArray) = v(a[0] + b[0], a[1] + b[1], a[2] + b[2])
    fun sub(a: DoubleArray, b: DoubleArray) = v(a[0] - b[0], a[1] - b[1], a[2] - b[2])
    fun scale(a: DoubleArray, s: Double) = v(a[0] * s, a[1] * s, a[2] * s)
    fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    fun norm(a: DoubleArray) = sqrt(dot(a, a))
    fun cross(a: DoubleArray, b: DoubleArray) =
        v(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    fun matVec(m: DoubleArray, a: DoubleArray) = v(
        m[0] * a[0] + m[1] * a[1] + m[2] * a[2],
        m[3] * a[0] + m[4] * a[1] + m[5] * a[2],
        m[6] * a[0] + m[7] * a[1] + m[8] * a[2],
    )

    fun matMul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val r = DoubleArray(9)
        for (i in 0 until 3) for (j in 0 until 3) {
            r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
        }
        return r
    }

    fun transpose(a: DoubleArray) = doubleArrayOf(a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8])

    /** Matrix whose columns are x, y, z (np.column_stack). */
    fun columns(x: DoubleArray, y: DoubleArray, z: DoubleArray) =
        doubleArrayOf(x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2])

    /** R = Rz(yaw) · Ry(roll) · Rx(-pitch). */
    fun eulerToMatrix(pitch: Double, roll: Double, yaw: Double): DoubleArray {
        val cx = cos(-pitch); val sx = sin(-pitch)
        val cy = cos(roll); val sy = sin(roll)
        val cz = cos(yaw); val sz = sin(yaw)
        return doubleArrayOf(
            cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
            -sy, cy * sx, cy * cx,
        )
    }

    fun matrixToEuler(r: DoubleArray): DoubleArray {
        val sy = -max(-1.0, min(1.0, r[6]))
        val roll = asin(sy)
        return if (kotlin.math.abs(r[6]) < 0.99999) {
            v(-atan2(r[7], r[8]), roll, atan2(r[3], r[0]))
        } else {
            v(-atan2(-r[5], r[4]), roll, 0.0)
        }
    }

    /**
     * Closed-form 3-DOF two-bone leg IK. Returns (hipEuler, kneeEuler, ankleEuler)
     * so FK places the foot joint (10/11) at [footTargetWorld] (Z >= 0).
     */
    fun solveLegIk(
        pelvisPos: DoubleArray,
        rPelvis: DoubleArray,
        footTargetWorld: DoubleArray,
        footYaw: Double,
        footPitch: Double,
        poleVectorLocal: DoubleArray,
        isLeft: Boolean,
    ): Array<DoubleArray> {
        val hipIdx = if (isLeft) 1 else 2
        val footIdx = if (isLeft) 10 else 11
        val l1 = 0.38
        val l2 = 0.38

        val pHip = add(pelvisPos, matVec(rPelvis, REST_OFFSETS[hipIdx]))
        val rAnkleGlob = eulerToMatrix(footPitch, 0.0, footYaw)
        val footOffset = REST_OFFSETS[footIdx]

        val safeFoot = v(footTargetWorld[0], footTargetWorld[1], max(0.0, footTargetWorld[2]))
        val pAnkleTarget = sub(safeFoot, matVec(rAnkleGlob, footOffset))
        if (pAnkleTarget[2] < 0.05) pAnkleTarget[2] = 0.05

        val vec = sub(pAnkleTarget, pHip)
        val d = norm(vec)
        val maxReach = (l1 + l2) * 0.998
        val uT: DoubleArray
        val dC: Double
        val pAnkleReach: DoubleArray
        if (d > maxReach) {
            uT = scale(vec, 1.0 / d); dC = maxReach; pAnkleReach = add(pHip, scale(uT, dC))
        } else if (d < 1e-4) {
            uT = v(0.0, 0.0, -1.0); dC = 1e-4; pAnkleReach = add(pHip, scale(uT, dC))
        } else {
            uT = scale(vec, 1.0 / d); dC = d; pAnkleReach = pAnkleTarget
        }

        val cosA = max(-1.0, min(1.0, (l1 * l1 + dC * dC - l2 * l2) / (2.0 * l1 * dC)))
        val sinA = sqrt(max(0.0, 1.0 - cosA * cosA))

        val poleWorld = matVec(rPelvis, poleVectorLocal)
        var normal = cross(uT, poleWorld)
        val nLen = norm(normal)
        normal = if (nLen < 1e-5) v(1.0, 0.0, 0.0) else scale(normal, 1.0 / nLen)

        var bendDir = cross(normal, uT)
        bendDir = scale(bendDir, 1.0 / norm(bendDir))

        val pKnee = add(pHip, scale(add(scale(uT, cosA), scale(bendDir, sinA)), l1))
        val uFemur = scale(sub(pKnee, pHip), 1.0 / l1)
        val uTibia = scale(sub(pAnkleReach, pKnee), 1.0 / l2)

        val zHip = scale(uFemur, -1.0)
        val crossFt = cross(uFemur, uTibia)
        val cLen = norm(crossFt)
        val xHip = if (cLen < 1e-5) normal else scale(crossFt, -1.0 / cLen)
        val yHip = cross(zHip, xHip)
        val rHipGlob = columns(xHip, yHip, zHip)

        val kneePitch = acos(max(-1.0, min(1.0, dot(uFemur, uTibia))))
        val rKneeGlob = matMul(rHipGlob, eulerToMatrix(kneePitch, 0.0, 0.0))

        val rHipLocal = matMul(transpose(rPelvis), rHipGlob)
        val rAnkleLocal = matMul(transpose(rKneeGlob), rAnkleGlob)

        return arrayOf(matrixToEuler(rHipLocal), v(kneePitch, 0.0, 0.0), matrixToEuler(rAnkleLocal))
    }

    /** Global joint positions (24) for local Euler rotations + root translation. */
    fun forwardPositions(rot: Array<DoubleArray>, rootTranslation: DoubleArray): Array<DoubleArray> {
        val g = arrayOfNulls<DoubleArray>(24)
        val p = arrayOfNulls<DoubleArray>(24)
        g[0] = eulerToMatrix(rot[0][0], rot[0][1], rot[0][2])
        p[0] = rootTranslation.copyOf()
        for (j in 1 until 24) {
            val parent = PARENTS[j]
            val local = eulerToMatrix(rot[j][0], rot[j][1], rot[j][2])
            g[j] = matMul(g[parent]!!, local)
            p[j] = add(p[parent]!!, matVec(g[parent]!!, REST_OFFSETS[j]))
        }
        @Suppress("UNCHECKED_CAST")
        return p as Array<DoubleArray>
    }
}

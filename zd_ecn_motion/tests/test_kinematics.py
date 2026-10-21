"""
Unit tests for SMPL-24 Kinematics Engine.
"""
import unittest
import numpy as np
from zd_ecn_motion.generator.kinematics import (
    compute_forward_kinematics, verify_transformation_dynamic_response,
    PARENT_MAP, REST_OFFSETS
)


class TestKinematics(unittest.TestCase):
    def test_forward_kinematics_shapes(self):
        rot = np.zeros((24, 3), dtype=np.float32)
        trans = np.array([0.0, 0.0, 0.85], dtype=np.float32)
        transforms, positions = compute_forward_kinematics(rot, trans)

        self.assertEqual(transforms.shape, (24, 4, 4))
        self.assertEqual(positions.shape, (24, 3))
        # Root position must match root translation
        np.testing.assert_allclose(positions[0], trans, rtol=1e-5)

    def test_transformation_dynamic_response(self):
        # Verification that rotations induce real geometric changes
        has_response = verify_transformation_dynamic_response()
        self.assertTrue(has_response, "Forward kinematics did not exhibit dynamic response!")

    def test_no_identity_substitutions(self):
        rot = np.zeros((24, 3), dtype=np.float32)
        trans = np.array([0.0, 0.0, 0.85], dtype=np.float32)
        transforms, _ = compute_forward_kinematics(rot, trans)

        identity = np.eye(4, dtype=np.float32)
        for j in range(1, 24):
            # All non-root joints must incorporate anatomical offsets and differ from identity
            is_ident = np.allclose(transforms[j], identity, atol=1e-3)
            self.assertFalse(is_ident, f"Joint {j} is an invalid identity substitution!")

    def test_rotation_matrix_orthogonality(self):
        # Generate arbitrary valid pose
        rot = np.random.uniform(-0.5, 0.5, size=(24, 3)).astype(np.float32)
        trans = np.array([0.1, 0.2, 0.85], dtype=np.float32)
        transforms, _ = compute_forward_kinematics(rot, trans)

        for j in range(24):
            R = transforms[j, :3, :3]
            # Orthogonality: R @ R.T == I
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)
            # Determinant must be +1 (proper SO(3) rotation)
            det = np.linalg.det(R)
            self.assertAlmostEqual(det, 1.0, places=4, msg=f"Joint {j} rotation det is not 1.0")


if __name__ == '__main__':
    unittest.main()

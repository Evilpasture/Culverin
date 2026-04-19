import unittest
import math
import struct
from culverin import MathService
from typing import cast


class TestMathService(unittest.TestCase):
    def setUp(self):
        self.math = MathService()

    

    def assertIsMatrix4x4(self, obj: object) -> None:
        # 1. Narrow to tuple
        self.assertIsInstance(obj, tuple)
        
        # 2. Tell Pylance exactly what is in that tuple
        # We use cast because we've already verified it's a tuple at runtime
        items = cast(tuple[float, ...], obj)
        
        self.assertEqual(len(items), 16)
        for val in items:
            # Now Pylance knows 'val' is a float
            self.assertIsInstance(val, float)

    def test_get_perspective(self):
        # fovy=45 deg, aspect=1.0, near=0.1, far=100.0
        fovy = math.radians(45.0)
        assert (mat := self.math.get_perspective(fovy, 1.0, 0.1, 100.0)) is not None
        self.assertIsMatrix4x4(mat)
        
        # In column-major, [0][0] is focal length (1 / tan(fovy/2))
        # For 45 deg, it's ~2.414
        self.assertAlmostEqual(mat[0], 2.4142135, places=5)
        # [2][3] should be -1.0 for a standard GL-style projection
        self.assertEqual(mat[11], -1.0)

    def test_get_ortho(self):
        assert (mat := self.math.get_ortho(-1, 1, -1, 1, 0.1, 100.0)) is not None
        self.assertIsMatrix4x4(mat)
        # Center of ortho should have identity-like scale for these bounds
        self.assertEqual(mat[0], 1.0)
        self.assertEqual(mat[5], 1.0)

    def test_get_look_at(self):
        eye = (0.0, 0.0, 5.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        assert (mat := self.math.get_look_at(eye, target, up)) is not None
        self.assertIsMatrix4x4(mat)
        # Translation part of view matrix should be at [12], [13], [14]
        # LookAt from (0,0,5) looking at origin results in Z translation of -5
        self.assertEqual(mat[14], -5.0)

    def test_get_trs(self):
        t = (10.0, 20.0, 30.0)
        r = (0.0, 0.0, 0.0, 1.0) # Identity quat
        s = (1.0, 1.0, 1.0)
        
        assert (mat := self.math.get_trs(t, r, s)) is not None
        self.assertIsMatrix4x4(mat)
        # Column-major translation check
        self.assertEqual(mat[12], 10.0)
        self.assertEqual(mat[13], 20.0)
        self.assertEqual(mat[14], 30.0)

    def test_get_trs_batch(self):
        # Create a batch of 2 entities
        # Use 'f' for float32 to match your C-API buffer expectations
        translations = struct.pack('6f', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        rotations = struct.pack('8f', 0, 0, 0, 1, 0, 0, 0, 1)
        scales = struct.pack('6f', 1, 1, 1, 1, 1, 1)

        result = self.math.get_trs_batch(translations, rotations, scales)
        
        self.assertIsInstance(result, bytes)
        # 2 matrices * 16 floats * 4 bytes = 128 bytes
        self.assertEqual(len(result), 128)
        
        # Verify first matrix translation (indices 12, 13, 14 in floats)
        # 12 * 4 = 48 byte offset
        res_floats = struct.unpack('32f', result)
        self.assertEqual(res_floats[12], 1.0)
        self.assertEqual(res_floats[13], 2.0)
        self.assertEqual(res_floats[14], 3.0)
        
        # Verify second matrix translation (16 + 12 = 28 index)
        self.assertEqual(res_floats[28], 4.0)

    def test_inverse(self):
        # Create a translation matrix
        t = (1.0, 2.0, 3.0)
        r = (0.0, 0.0, 0.0, 1.0)
        s = (1.0, 1.0, 1.0)
        mat = self.math.get_trs(t, r, s)
        
        inv = self.math.inverse(mat)
        self.assertIsMatrix4x4(inv)
        
        # Inverting a translation of (1, 2, 3) should give (-1, -2, -3)
        self.assertAlmostEqual(inv[12], -1.0)
        self.assertAlmostEqual(inv[13], -2.0)
        self.assertAlmostEqual(inv[14], -3.0)

    def test_matmul(self):
        # Identity matrices
        eye_quat = (0.0, 0.0, 0.0, 1.0)
        one_scale = (1.0, 1.0, 1.0)
        
        m1 = self.math.get_trs((10.0, 0.0, 0.0), eye_quat, one_scale)
        m2 = self.math.get_trs((5.0, 0.0, 0.0), eye_quat, one_scale)
        
        # (Translate 10) * (Translate 5) = Translate 15
        res = self.math.matmul(m1, m2)
        self.assertIsMatrix4x4(res)
        self.assertAlmostEqual(res[12], 15.0)

    def test_transform_vec3(self):
        t = (10.0, 20.0, 30.0)
        r = (0.0, 0.0, 0.0, 1.0)
        s = (1.0, 1.0, 1.0)
        mat = self.math.get_trs(t, r, s)
        
        point = (1.0, 1.0, 1.0)
        # (1,1,1) + (10, 20, 30) = (11, 21, 31)
        res = self.math.transform_vec3(mat, point)
        
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertAlmostEqual(res[0], 11.0)
        self.assertAlmostEqual(res[1], 21.0)
        self.assertAlmostEqual(res[2], 31.0)

    def test_matmul_batch(self):
        eye_quat = (0.0, 0.0, 0.0, 1.0)
        one_scale = (1.0, 1.0, 1.0)
        
        # Matrix to multiply by (e.g., a View-Projection)
        vp = self.math.get_trs((100.0, 0.0, 0.0), eye_quat, one_scale)
        
        # Batch of 2 model matrices
        m1 = self.math.get_trs((1.0, 0.0, 0.0), eye_quat, one_scale)
        m2 = self.math.get_trs((2.0, 0.0, 0.0), eye_quat, one_scale)
        batch = struct.pack('32f', *m1, *m2)
        
        res_bytes = self.math.matmul_batch(vp, batch)
        self.assertIsInstance(res_bytes, bytes)
        self.assertEqual(len(res_bytes), 128)
        
        res_floats = struct.unpack('32f', res_bytes)
        # First matrix translation: 100 + 1 = 101
        self.assertAlmostEqual(res_floats[12], 101.0)
        # Second matrix translation: 100 + 2 = 102
        self.assertAlmostEqual(res_floats[28], 102.0)

    def test_cull_aabb(self):
        # Create a basic perspective projection
        vp = self.math.get_perspective(math.radians(45.0), 1.0, 0.1, 100.0)
        
        # Box inside frustum (directly in front of camera)
        box_min = (-1.0, -1.0, -10.0)
        box_max = (1.0, 1.0, -5.0)
        self.assertTrue(self.math.cull_aabb(vp, box_min, box_max))
        
        # Box far behind camera
        box_min_off = (-1.0, -1.0, 10.0)
        box_max_off = (1.0, 1.0, 15.0)
        self.assertFalse(self.math.cull_aabb(vp, box_min_off, box_max_off))

    def test_cull_aabb_batch(self):
        vp = self.math.get_perspective(math.radians(45.0), 1.0, 0.1, 100.0)
        
        # 1. Inside, 2. Outside
        aabbs = struct.pack(
            '12f',
            -1.0, -1.0, -10.0, 1.0, 1.0, -5.0,  # Box 1
            -1.0, -1.0, 10.0, 1.0, 1.0, 15.0    # Box 2
        )
        
        result = self.math.cull_aabb_batch(vp, aabbs)
        self.assertIsInstance(result, bytearray)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 1) # Visible
        self.assertEqual(result[1], 0) # Culled

if __name__ == '__main__':
    unittest.main()
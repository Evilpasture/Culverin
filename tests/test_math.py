import math
import struct
import unittest
from typing import cast

from culverin import MathService


class TestMathService(unittest.TestCase):
    def setUp(self) -> None:
        self.math = MathService()

    def assertIsMatrix4x4(self, obj: object) -> None:
        # 1. Narrow to tuple
        self.assertIsInstance(obj, tuple)

        # 2. Tell Pylance exactly what is in that tuple
        items = cast(tuple[float, ...], obj)

        self.assertEqual(len(items), 16)
        for val in items:
            self.assertIsInstance(val, float)

    def assertTupleAlmostEqual(
        self, t1: tuple[float, ...] | None, t2: tuple[float, ...], places: int = 5
    ) -> None:
        # First, satisfy the type checker (and the test logic)
        if t1 is None:
            self.fail("Received None, but expected a tuple of floats.")

        self.assertEqual(len(t1), len(t2), f"Tuple lengths differ: {len(t1)} != {len(t2)}")
        for a, b in zip(t1, t2, strict=False):
            self.assertAlmostEqual(a, b, places=places)

    # =========================================================================
    # ORIGINAL TESTS
    # =========================================================================

    def test_get_perspective(self) -> None:
        fovy = math.radians(45.0)
        assert (mat := self.math.get_perspective(fovy, 1.0, 0.1, 100.0)) is not None
        self.assertIsMatrix4x4(mat)
        self.assertAlmostEqual(mat[0], 2.4142135, places=5)
        self.assertEqual(mat[11], -1.0)

    def test_get_ortho(self) -> None:
        assert (mat := self.math.get_ortho(-1, 1, -1, 1, 0.1, 100.0)) is not None
        self.assertIsMatrix4x4(mat)
        self.assertAlmostEqual(mat[0], 1.0, places=6)
        self.assertAlmostEqual(mat[5], 1.0, places=6)

    def test_get_look_at(self) -> None:
        eye, target, up = (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        assert (mat := self.math.get_look_at(eye, target, up)) is not None
        self.assertIsMatrix4x4(mat)
        self.assertEqual(mat[14], -5.0)

    def test_get_trs(self) -> None:
        t, r, s = (10.0, 20.0, 30.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)
        assert (mat := self.math.get_trs(t, r, s)) is not None
        self.assertIsMatrix4x4(mat)
        self.assertEqual(mat[12], 10.0)
        self.assertEqual(mat[13], 20.0)
        self.assertEqual(mat[14], 30.0)

    def test_get_trs_batch(self) -> None:
        translations = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        rotations = struct.pack("8f", 0, 0, 0, 1, 0, 0, 0, 1)
        scales = struct.pack("6f", 1, 1, 1, 1, 1, 1)
        result = self.math.get_trs_batch(translations, rotations, scales)
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), 128)
        res_floats = struct.unpack("32f", result)
        self.assertEqual(res_floats[12], 1.0)
        self.assertEqual(res_floats[13], 2.0)
        self.assertEqual(res_floats[14], 3.0)
        self.assertEqual(res_floats[28], 4.0)

    def test_inverse(self) -> None:
        t, r, s = (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)
        mat = self.math.get_trs(t, r, s)
        inv = self.math.inverse(mat)
        self.assertIsMatrix4x4(inv)
        self.assertAlmostEqual(inv[12], -1.0)
        self.assertAlmostEqual(inv[13], -2.0)
        self.assertAlmostEqual(inv[14], -3.0)

    def test_matmul(self) -> None:
        eye_quat, one_scale = (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)
        m1 = self.math.get_trs((10.0, 0.0, 0.0), eye_quat, one_scale)
        m2 = self.math.get_trs((5.0, 0.0, 0.0), eye_quat, one_scale)
        res = self.math.matmul(m1, m2)
        self.assertIsMatrix4x4(res)
        self.assertAlmostEqual(res[12], 15.0)

    def test_transform_vec3(self) -> None:
        t, r, s = (10.0, 20.0, 30.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)
        mat = self.math.get_trs(t, r, s)
        res = self.math.transform_vec3(mat, (1.0, 1.0, 1.0))
        self.assertTupleAlmostEqual(res, (11.0, 21.0, 31.0))

    def test_matmul_batch(self) -> None:
        eye_quat, one_scale = (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)
        vp = self.math.get_trs((100.0, 0.0, 0.0), eye_quat, one_scale)
        m1 = self.math.get_trs((1.0, 0.0, 0.0), eye_quat, one_scale)
        m2 = self.math.get_trs((2.0, 0.0, 0.0), eye_quat, one_scale)
        batch = struct.pack("32f", *m1, *m2)
        res_bytes = self.math.matmul_batch(vp, batch)
        res_floats = struct.unpack("32f", res_bytes)
        self.assertAlmostEqual(res_floats[12], 101.0)
        self.assertAlmostEqual(res_floats[28], 102.0)

    def test_cull_aabb(self) -> None:
        vp = self.math.get_perspective(math.radians(45.0), 1.0, 0.1, 100.0)
        self.assertTrue(self.math.cull_aabb(vp, (-1.0, -1.0, -10.0), (1.0, 1.0, -5.0)))
        self.assertFalse(self.math.cull_aabb(vp, (-1.0, -1.0, 10.0), (1.0, 1.0, 15.0)))

    def test_cull_aabb_batch(self) -> None:
        vp = self.math.get_perspective(math.radians(45.0), 1.0, 0.1, 100.0)
        aabbs = struct.pack(
            "12f", -1.0, -1.0, -10.0, 1.0, 1.0, -5.0, -1.0, -1.0, 10.0, 1.0, 1.0, 15.0
        )
        result = self.math.cull_aabb_batch(vp, aabbs)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 0)

    # =========================================================================
    # NEW TESTS: VECTORS
    # =========================================================================

    def test_vec3_normalize(self) -> None:
        # 3-4-5 triangle
        norm = self.math.vec3_normalize((3.0, 0.0, 4.0))
        self.assertTupleAlmostEqual(norm, (0.6, 0.0, 0.8))

        # Degenerate case should return zero safely
        zero = self.math.vec3_normalize((0.0, 0.0, 0.0))
        self.assertTupleAlmostEqual(zero, (0.0, 0.0, 0.0))

    def test_vec3_normalize_batch(self) -> None:
        vecs = struct.pack("6f", 3.0, 0.0, 4.0, 0.0, 3.0, 4.0)
        res_bytes = self.math.vec3_normalize_batch(vecs)
        res_floats = struct.unpack("6f", res_bytes)
        self.assertAlmostEqual(res_floats[0], 0.6)
        self.assertAlmostEqual(res_floats[2], 0.8)
        self.assertAlmostEqual(res_floats[4], 0.6)

    def test_vec3_dot(self) -> None:
        dot = self.math.vec3_dot((1.0, 2.0, 3.0), (4.0, -5.0, 6.0))
        self.assertAlmostEqual(dot, 12.0)  # 4 - 10 + 18 = 12

    def test_vec3_cross(self) -> None:
        cross = self.math.vec3_cross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        self.assertTupleAlmostEqual(cross, (0.0, 0.0, 1.0))

    def test_vec3_distance(self) -> None:
        dist = self.math.vec3_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        self.assertAlmostEqual(dist, 5.0)

    def test_vec3_distance_batch(self) -> None:
        a = struct.pack("6f", 0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        b = struct.pack("6f", 3.0, 4.0, 0.0, 10.0, 10.0, 10.0)
        dist_bytes = self.math.vec3_distance_batch(a, b)
        dist_floats = struct.unpack("2f", dist_bytes)
        self.assertAlmostEqual(dist_floats[0], 5.0)
        self.assertAlmostEqual(dist_floats[1], 0.0)

    def test_vec3_lerp_batch(self) -> None:
        a = struct.pack("3f", 0.0, 0.0, 0.0)
        b = struct.pack("3f", 10.0, 20.0, 30.0)
        res_bytes = self.math.vec3_lerp_batch(a, b, 0.5)
        res_floats = struct.unpack("3f", res_bytes)
        self.assertAlmostEqual(res_floats[0], 5.0)
        self.assertAlmostEqual(res_floats[1], 10.0)
        self.assertAlmostEqual(res_floats[2], 15.0)

    def test_vec3_reflect(self) -> None:
        vel = (1.0, -1.0, 0.0)
        normal = (0.0, 1.0, 0.0)  # Floor normal
        reflected = self.math.vec3_reflect(vel, normal)
        self.assertTupleAlmostEqual(reflected, (1.0, 1.0, 0.0))

    # =========================================================================
    # NEW TESTS: QUATERNIONS
    # =========================================================================

    def test_euler_quat_conversions(self) -> None:
        # Avoid 90 degrees exactly to prevent ambiguous equivalent representations (e.g. pi vs 0)
        euler_in = (0.1, 0.2, 0.3)
        q = self.math.quat_from_euler(*euler_in)

        # Convert back
        euler_out = self.math.quat_to_euler(*q)
        self.assertTupleAlmostEqual(euler_out, euler_in, places=5)

    def test_quat_slerp(self) -> None:
        q1 = (0.0, 0.0, 0.0, 1.0)  # Identity
        q2 = self.math.quat_from_euler(0.0, math.pi / 2.0, 0.0)  # 90 deg Y

        # Slerp halfway -> 45 deg Y
        res = self.math.quat_slerp(q1, q2, 0.5)
        expected = self.math.quat_from_euler(0.0, math.pi / 4.0, 0.0)
        self.assertTupleAlmostEqual(res, expected, places=5)

    def test_quat_mul(self) -> None:
        q90 = self.math.quat_from_euler(0.0, math.pi / 2.0, 0.0)
        # 90 deg * 90 deg = 180 deg
        res = self.math.quat_mul(q90, q90)
        expected = self.math.quat_from_euler(0.0, math.pi, 0.0)
        self.assertTupleAlmostEqual(res, expected)

    def test_quat_inverse(self) -> None:
        q = self.math.quat_from_euler(math.pi / 4.0, math.pi / 3.0, 0.0)
        inv = self.math.quat_inverse(q)

        # Multiply q by its inverse should yield Identity
        identity = self.math.quat_mul(q, inv)
        self.assertTupleAlmostEqual(identity, (0.0, 0.0, 0.0, 1.0), places=6)

    def test_quat_from_to(self) -> None:
        # Rotation from +X to +Y is a 90 deg rotation around +Z
        q = self.math.quat_from_to((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        # Rotate X vector by q
        res = self.math.quat_rotate_vec3(q, (1.0, 0.0, 0.0))
        self.assertTupleAlmostEqual(res, (0.0, 1.0, 0.0))

    def test_quat_axis_angle(self) -> None:
        axis = (0.0, 1.0, 0.0)
        angle = math.pi / 2.0
        q = self.math.quat_from_axis_angle(axis, angle)

        out_axis, out_angle = self.math.quat_get_axis_angle(q)
        self.assertTupleAlmostEqual(out_axis, axis, places=5)
        self.assertAlmostEqual(out_angle, angle, places=5)

    def test_quat_rotate_vec3(self) -> None:
        q = self.math.quat_from_euler(0.0, math.pi / 2.0, 0.0)  # 90 deg Y
        v = (1.0, 0.0, 0.0)  # Right
        # Right rotated 90 deg Left (Y-up) is Forward (-Z in right-handed)
        res = self.math.quat_rotate_vec3(q, v)
        self.assertTupleAlmostEqual(res, (0.0, 0.0, -1.0))

    def test_quat_rotate_vec3_batch(self) -> None:
        q = self.math.quat_from_euler(0.0, math.pi / 2.0, 0.0)
        vecs = struct.pack("6f", 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        res_bytes = self.math.quat_rotate_vec3_batch(q, vecs)
        res_floats = struct.unpack("6f", res_bytes)

        # (1,0,0) -> (0,0,-1)
        self.assertAlmostEqual(res_floats[0], 0.0, places=5)
        self.assertAlmostEqual(res_floats[2], -1.0, places=5)
        # (0,0,-1) -> (-1,0,0)
        self.assertAlmostEqual(res_floats[3], -1.0, places=5)
        self.assertAlmostEqual(res_floats[5], 0.0, places=5)

    def test_quat_rotate_vec3_inverse(self) -> None:
        q = self.math.quat_from_euler(0.0, math.pi / 2.0, 0.0)  # 90 deg Y
        v = (0.0, 0.0, -1.0)  # Forward
        # Inverse rotate Forward by 90 deg Y -> Right
        res = self.math.quat_rotate_vec3_inverse(q, v)
        self.assertTupleAlmostEqual(res, (1.0, 0.0, 0.0))

    # =========================================================================
    # NEW TESTS: MATRICES & PROJECTION
    # =========================================================================

    def test_mat44_identity(self) -> None:
        m = self.math.mat44_identity()
        self.assertIsMatrix4x4(m)
        self.assertEqual(m[0], 1.0)
        self.assertEqual(m[5], 1.0)
        self.assertEqual(m[10], 1.0)
        self.assertEqual(m[15], 1.0)
        self.assertEqual(m[12], 0.0)  # Translation X

    def test_mat44_get_components(self) -> None:
        pos = (10.0, -5.0, 42.0)
        rot = self.math.quat_from_euler(0.0, math.pi, 0.0)
        mat = self.math.get_trs(pos, rot, (1.0, 1.0, 1.0))

        out_pos = self.math.mat44_get_translation(mat)
        self.assertTupleAlmostEqual(out_pos, pos)

        out_rot = self.math.mat44_get_rotation(mat)
        self.assertTupleAlmostEqual(out_rot, rot)

    def test_project_unproject(self) -> None:
        # Setup camera looking down -Z
        view = self.math.get_look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
        proj = self.math.get_perspective(math.radians(90), 1.0, 0.1, 100.0)
        mvp = self.math.matmul(proj, view)
        vp = (0, 0, 800, 600)

        world_pt = (0.0, 0.0, 0.0)
        # Project center of world
        screen_pt = self.math.project(world_pt, mvp, vp)

        # It should be perfectly in the middle of the 800x600 screen
        self.assertAlmostEqual(screen_pt[0], 400.0)
        self.assertAlmostEqual(screen_pt[1], 300.0)

        # Unproject back to world space (using the exact depth from projection)
        unproj_pt = self.math.unproject(screen_pt, mvp, vp)
        self.assertTupleAlmostEqual(unproj_pt, world_pt)

    def test_intersect_ray_plane(self) -> None:
        # Ray straight down from (0, 10, 0)
        ro = (0.0, 10.0, 0.0)
        rd = (0.0, -1.0, 0.0)

        # Plane at origin, pointing up
        po = (0.0, 0.0, 0.0)
        pn = (0.0, 1.0, 0.0)

        hit, dist, point = self.math.intersect_ray_plane(ro, rd, po, pn)

        self.assertTrue(hit)
        self.assertAlmostEqual(dist, 10.0, delta=1e-5)
        self.assertTupleAlmostEqual(point, (0.0, 0.0, 0.0))

        # Test parallel miss
        rd_miss = (1.0, 0.0, 0.0)  # Shooting sideways
        hit_miss, _, _ = self.math.intersect_ray_plane(ro, rd_miss, po, pn)
        self.assertFalse(hit_miss)

    # =========================================================================
    # NEW TESTS: QUATERNIONS (Continued)
    # =========================================================================

    def test_euler_to_quat_single(self) -> None:
        """Tests the tuple-to-tuple (Vec3 -> Quat) conversion."""
        euler_vec = (0.1, 0.2, 0.3)

        # Test single vec version
        q = self.math.euler_to_quat(euler_vec)

        # Verify it matches the component-based version
        q_expected = self.math.quat_from_euler(0.1, 0.2, 0.3)
        self.assertIsInstance(q, tuple)
        self.assertEqual(len(q), 4)
        self.assertTupleAlmostEqual(q, q_expected)

    def test_euler_to_quat_batch(self) -> None:
        """Tests the high-performance Buffer-to-Bytes conversion."""
        # Define two distinct rotations
        e1 = (math.radians(45), 0.0, 0.0)
        e2 = (0.0, math.radians(-90), 0.0)

        # Pack into float32 buffer (12 bytes per rotation)
        eulers_buf = struct.pack("6f", *e1, *e2)

        # Perform batch conversion
        res_bytes = self.math.euler_to_quat_batch(eulers_buf)

        # Verify output size (16 bytes per quaternion)
        self.assertIsInstance(res_bytes, bytes)
        self.assertEqual(len(res_bytes), 32)  # 2 rotations * 4 floats * 4 bytes

        # Unpack and verify against single-call logic
        res_floats = struct.unpack("8f", res_bytes)

        q1_expected = self.math.euler_to_quat(e1)
        q2_expected = self.math.euler_to_quat(e2)

        self.assertTupleAlmostEqual(res_floats[0:4], q1_expected)
        self.assertTupleAlmostEqual(res_floats[4:8], q2_expected)

    def test_euler_to_quat_batch_invalid(self) -> None:
        """Ensures the batch conversion catches malformed buffers."""
        # Only 2 floats (8 bytes) when 3 (12 bytes) are required per element
        bad_buf = struct.pack("2f", 1.0, 2.0)
        with self.assertRaises(ValueError):
            self.math.euler_to_quat_batch(bad_buf)


if __name__ == "__main__":
    unittest.main()

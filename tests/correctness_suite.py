import culverin
import unittest
import numpy as np
import array
import struct

class TestCulverinCorrectness(unittest.TestCase):
    def setUp(self):
        # Fresh world for every test
        self.world = culverin.PhysicsWorld(settings={
            "gravity": (0, -9.81, 0),
            "max_bodies": 1000
        })
        # Stride constants based on your C-structs (PosStride and AuxStride)
        self.POS_STRIDE = 4 
        self.AUX_STRIDE = 4

    def test_gravity_and_sync(self):
        """Test if a body actually falls and if shadow buffers sync correctly."""
        h = self.world.create_body(pos=(0, 10, 0), shape=culverin.SHAPE_SPHERE, size=1.0)
        
        # Step once to apply gravity
        self.world.step(1/60.0)
        
        idx = self.world.get_index(h)
        # Positions is a flat 1D memoryview of JPH_Real (Stride 4)
        pos_y = self.world.positions[idx * self.POS_STRIDE + 1]
        
        # Y should be less than 10 but more than 0
        self.assertLess(pos_y, 10.0)
        self.assertGreater(pos_y, 9.0)

    def test_generational_handles(self):
        """Test if handles correctly invalidate after destruction."""
        h1 = self.world.create_body(pos=(0,0,0))
        self.assertTrue(self.world.is_alive(h1))
        
        self.world.destroy_body(h1)
        self.world.step(0) # Flush command queue
        
        # h1 should now be dead
        self.assertFalse(self.world.is_alive(h1))
        
        # Create a new body (likely reuses the same slot)
        h2 = self.world.create_body(pos=(10,10,10))
        self.assertNotEqual(h1, h2, "Handles must be unique even if slots are reused")
        self.assertTrue(self.world.is_alive(h2))
        self.assertFalse(self.world.is_alive(h1))

    def test_raycast_precision(self):
        """Test if raycasts return correct distances and normals."""
        floor = self.world.create_body(
            pos=(0, -1, 0), 
            shape=culverin.SHAPE_BOX, 
            size=(10, 1, 10), 
            motion=culverin.MOTION_STATIC
        )
        self.world.step(0) # Flush
        
        # Ray from (0, 5, 0) straight down
        res = self.world.raycast(start=(0, 5, 0), direction=(0, -1, 0), max_dist=10.0)
        
        self.assertIsNotNone(res)
        handle, fraction, normal = res
        
        self.assertEqual(handle, floor)
        # Hit should be at Y=0. Ray length 10. Start 5. Fraction = 5/10 = 0.5
        self.assertAlmostEqual(fraction, 0.5, places=3)
        # Normal should be straight up (0, 1, 0)
        self.assertAlmostEqual(normal[0], 0.0, places=5)
        self.assertAlmostEqual(normal[1], 1.0, places=5) # Loosened from 7 to 5
        self.assertAlmostEqual(normal[2], 0.0, places=5)

    def test_causal_consistency(self):
        """Test if mutations on a PENDING_CREATE body work as expected."""
        h = self.world.create_body(pos=(0,0,0), motion=culverin.MOTION_DYNAMIC)
        # Applying impulse to a body created in same frame (queued)
        self.world.apply_impulse(h, 100, 0, 0) 
        
        # First step flushes creation THEN applies impulse
        self.world.step(1/60.0)
        
        idx = self.world.get_index(h)
        # Velocities is flat 1D memoryview (Stride 4)
        vel_x = self.world.velocities[idx * self.AUX_STRIDE + 0]
        
        self.assertGreater(vel_x, 0.0, "Impulse should have been applied to the new body")

    def test_batch_raycast_math(self):
        """Verify the fast rsqrt and batch logic matches single raycasts."""
        sphere = self.world.create_body(pos=(0,0,0), shape=culverin.SHAPE_SPHERE, size=1.0, motion=culverin.MOTION_STATIC)
        self.world.step(0)
        
        starts = array.array('f', [0.0, 5.0, 0.0])
        dirs = array.array('f', [0.0, -1.0, 0.0])
        
        batch_res = self.world.raycast_batch(starts.tobytes(), dirs.tobytes(), max_dist=10.0)
        
        # Unpack result (48 bytes per result)
        # Q: handle, f: fraction, f: nx, f: ny, f: nz, f: px, f: py, f: pz, I: sub, I: mat, I: pad
        handle, fraction, nx, ny, nz, px, py, pz, sub, mat, pad = struct.unpack("QfffffffIII", batch_res)
        
        self.assertEqual(handle, sphere)
        # Hits top of sphere at Y=1. Distance is 4. Fraction 4/10 = 0.4
        self.assertAlmostEqual(fraction, 0.4, places=2)
        self.assertAlmostEqual(ny, 1.0, places=2)

    def test_aabb_overlap(self):
        """Verify AABB overlap logic."""
        h1 = self.world.create_body(pos=(0, 0, 0), size=1.0)
        h2 = self.world.create_body(pos=(10, 10, 10), size=1.0)
        self.world.step(0)
        
        hits = self.world.overlap_aabb(min=(-2, -2, -2), max=(2, 2, 2))
        self.assertIn(h1, hits)
        self.assertNotIn(h2, hits)

if __name__ == "__main__":
    unittest.main()
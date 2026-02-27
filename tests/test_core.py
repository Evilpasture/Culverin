import unittest
import math
import time
import struct
import threading
import array
import numpy as np
import culverin

class CulverinTestCase(unittest.TestCase):
    """Base class providing helper methods for interacting with Culverin buffers."""
    def setUp(self):
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 2048})
        self.world.step(0) # Flush initial state

    def get_pos(self, handle):
        idx = self.world.get_index(handle)
        return self.world.positions[idx * 4 : idx * 4 + 3]

    def get_vel(self, handle):
        idx = self.world.get_index(handle)
        return self.world.velocities[idx * 4 : idx * 4 + 3]


class TestCoreMechanics(CulverinTestCase):
    def test_activation_and_gravity(self):
        h = self.world.create_body(pos=(0, 10, 0), shape=culverin.SHAPE_SPHERE, size=1.0)
        self.world.step(0)
        
        self.assertEqual(self.get_pos(h)[1], 10.0)
        self.world.step(1/60.0)
        self.assertLess(self.get_pos(h)[1], 10.0, "Body did not fall")

    def test_generational_handles(self):
        h1 = self.world.create_body(pos=(0, 0, 0))
        self.assertTrue(self.world.is_alive(h1))
        
        self.world.destroy_body(h1)
        self.world.step(0)
        self.assertFalse(self.world.is_alive(h1))
        
        h2 = self.world.create_body(pos=(10, 10, 10))
        self.assertNotEqual(h1, h2, "Handles must be unique across generations")
        self.assertTrue(self.world.is_alive(h2))

    def test_causal_consistency(self):
        # Apply impulse to a body created in the same un-stepped frame
        h = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.apply_impulse(h, 100, 0, 0)
        self.world.step(1/60.0)
        self.assertGreater(self.get_vel(h)[0], 0.0)

    def test_buoyancy(self):
        ball = self.world.create_body(pos=(0, 10, 0), shape=culverin.SHAPE_SPHERE, size=(0.5,), mass=10.0)
        self.world.step(0)
        
        submerged_frames = 0
        for _ in range(120):
            if self.world.apply_buoyancy(handle=ball, surface_y=5.0, buoyancy=1.5, linear_drag=0.8, angular_drag=0.5, dt=1/60):
                submerged_frames += 1
            self.world.step(1/60)

        self.assertGreater(submerged_frames, 0)
        self.assertGreater(self.get_pos(ball)[1], 4.0, "Ball should be floating, not at bottom")

    def test_heightfield(self):
        grid_size = 64
        x = np.linspace(0, 10, grid_size, dtype=np.float32)
        xx, _ = np.meshgrid(x, x)
        heights = (xx * 0.5).astype(np.float32).tobytes()

        self.world.create_heightfield(pos=(0, 0, 0), rot=(0, 0, 0, 1), scale=(1, 1, 1), heights=heights, grid_size=grid_size)
        ball = self.world.create_body(pos=(5, 3.5, 5), shape=culverin.SHAPE_SPHERE, mass=10.0)
        
        self.world.step(0)
        start_x = self.get_pos(ball)[0]
        for _ in range(60): self.world.step(1/60)
        
        self.assertLess(self.get_pos(ball)[0], start_x, "Ball did not roll down slope")


class TestQueries(CulverinTestCase):
    def setUp(self):
        super().setUp()
        self.floor = self.world.create_body(pos=(0, -1, 0), size=(10, 1, 10), motion=culverin.MOTION_STATIC)
        self.world.step(0)

    def test_raycast(self):
        res = self.world.raycast(start=(0, 5, 0), direction=(0, -1, 0), max_dist=10.0)
        self.assertIsNotNone(res)
        handle, fraction, normal = res
        self.assertEqual(handle, self.floor)
        self.assertAlmostEqual(fraction, 0.5, places=3)
        np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-4)

    def test_batch_raycast(self):
        ray_count = 100
        starts = np.zeros((ray_count, 3), dtype=np.float32)
        starts[:, 1] = 10.0
        dirs = np.zeros((ray_count, 3), dtype=np.float32)
        dirs[:, 1] = -1.0
        
        res_raw = self.world.raycast_batch(starts.tobytes(), dirs.tobytes(), max_dist=20.0)
        dt = np.dtype([("h", np.uint64), ("f", np.float32), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("px", "f4"), ("py", "f4"), ("pz", "f4"), ("sub", "u4"), ("mat", "u4"), ("pad", "u4")])
        results = np.frombuffer(res_raw, dtype=dt)
        
        self.assertEqual(np.count_nonzero(results["h"]), ray_count)
        self.assertEqual(results["h"][0], self.floor)
        self.assertAlmostEqual(results["py"][0], 0.0, places=3)

    def test_shapecast(self):
        target = self.world.create_body(pos=(10, 5, 0), size=(1, 5, 5), motion=culverin.MOTION_STATIC)
        self.world.step(0)
        
        hit = self.world.shapecast(shape=culverin.SHAPE_SPHERE, pos=(0, 5, 0), rot=(0,0,0,1), dir=(20, 0, 0), size=(0.5,))
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], target)
        self.assertAlmostEqual(hit[3][0], -1.0, places=3) # Normal faces -X


class TestCollisionsAndEvents(CulverinTestCase):
    def test_collision_filtering(self):
        floor = self.world.create_body(pos=(0, 5, 0), size=(10, 0.1, 10), motion=culverin.MOTION_STATIC, category=1, mask=0xFFFF)
        
        # Added the size=(0.5, 0.5, 0.5) arguments back
        player = self.world.create_body(pos=(2, 10, 0), size=(0.5, 0.5, 0.5), category=2, mask=1|4) 
        ghost = self.world.create_body(pos=(2, 12, 0), size=(0.5, 0.5, 0.5), category=4, mask=1)    
        
        for _ in range(120): self.world.step(1/60)
        
        self.assertGreater(self.get_pos(player)[1], 5.0) # Player caught by floor
        self.assertLess(self.get_pos(ghost)[1], 5.6)     # Ghost passed through player, caught by floor

    def test_sensor_events(self):
        sensor = self.world.create_body(pos=(0, 5, 0), size=(2, 0.5, 2), is_sensor=True, motion=culverin.MOTION_STATIC)
        crate = self.world.create_body(pos=(0, 10, 0), motion=culverin.MOTION_DYNAMIC)
        
        hit = False
        for _ in range(60):
            self.world.step(1/60)
            for ev in self.world.get_contact_events_ex():
                if set(ev["bodies"]) == {sensor, crate}:
                    hit = True
        self.assertTrue(hit)


class TestCharactersAndVehicles(CulverinTestCase):
    def test_character_lifecycle_and_movement(self):
        char = self.world.create_character(pos=(0, 2, 0), height=1.8, radius=0.4)
        self.assertTrue(self.world.is_alive(char.handle))
        
        char.move((10, 0, 0), 1/60)
        self.world.step(1/60)
        self.assertGreater(char.get_position()[0], 0.0)
        
        # Test get_render_transform interpolation
        r_pos, r_rot = char.get_render_transform(0.5)
        self.assertGreater(r_pos[0], 0.0)

    def test_wheeled_vehicle(self):
        # 1. Add a floor with friction so the wheels can grip!
        self.world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC, friction=1.0)
        
        chassis = self.world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        wheels = [{"pos": (x, -0.5, z), "radius": 0.4} for x in [-0.8, 0.8] for z in [1.2, -1.2]]
        car = self.world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")
        self.world.step(0)
        
        # Settle
        for _ in range(60): self.world.step(1/60)
        
        car.set_input(forward=1.0)
        for _ in range(60): self.world.step(1/60)
        
        self.assertGreater(self.get_vel(chassis)[2], 1.0)
        self.assertEqual(car.wheel_count, 4)


class TestThreadSafety(CulverinTestCase):
    def test_blocking_mutation(self):
        # 1. Create a valid body to mutate
        h = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)
        
        stop = False
        def physics_worker():
            while not stop:
                try: self.world.step(1/60)
                except RuntimeError: pass
                
        t = threading.Thread(target=physics_worker)
        t.start()
        
        success = 0
        start = time.time()
        
        try:
            while time.time() - start < 0.5:
                # Should block safely until step is done
                self.world.set_position(handle=h, x=0, y=0, z=0)
                success += 1
        except Exception as e:
            self.fail(f"Mutation failed with unexpected error: {e}")
        finally:
            # ALWAYS stop the thread, preventing deadlock on failure
            stop = True
            t.join()
            
        self.assertGreater(success, 0, "Mutations should block and succeed")

if __name__ == '__main__':
    unittest.main(verbosity=2)
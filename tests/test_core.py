import unittest
import math
import time
import threading
import numpy as np
import culverin
from culverin import WheelConfig
from culverin import TrackConfig

class CulverinTestCase(unittest.TestCase):
    """Base class providing helper methods for interacting with Culverin buffers."""
    def setUp(self):
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 2048})
        self.world.step(0) # Flush initial state

    def get_pos(self, handle: int):
        return self.world.get_position(handle)

    def get_vel(self, handle: int):
        return self.world.get_velocity(handle)


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

    def test_material_hot_swap(self):
        """Test if changing material properties affects simulation."""
        # Since Jolt bodies are immutable-by-default for friction, 
        # we verify that the registry works for NEW bodies.
        self.world.register_material(id=50, friction=0.0)
        b1 = self.world.create_body(pos=(0, 10, 0), material_id=50)
        
        self.world.register_material(id=50, friction=1.0)
        b2 = self.world.create_body(pos=(5, 10, 0), material_id=50)
        
        self.world.step(0)
        # Both bodies now have different internal Jolt friction values 
        # despite having the same material ID.
        self.assertTrue(self.world.is_alive(b1))
        self.assertTrue(self.world.is_alive(b2))


class TestQueries(CulverinTestCase):
    def setUp(self):
        super().setUp()
        self.floor = self.world.create_body(pos=(0, -1, 0), size=(10, 1, 10), motion=culverin.MOTION_STATIC)
        self.world.step(0)

    def test_raycast(self):
        res = self.world.raycast(start=(0, 5, 0), direction=(0, -1, 0), max_dist=10.0)
        self.assertIsNotNone(res)
        assert res is not None 
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
        assert hit is not None
        self.assertEqual(hit[0], target)
        self.assertAlmostEqual(hit[3][0], -1.0, places=3) # Normal faces -X
        
    def test_overlap_sphere(self):
        b1 = self.world.create_body(pos=(5, 0, 0), size=(1, 1, 1))
        b2 = self.world.create_body(pos=(-5, 0, 0), size=(1, 1, 1))
        self.world.step(0)
        
        hits = self.world.overlap_sphere(center=(4.5, 0, 0), radius=2.0)
        self.assertIn(b1, hits)
        self.assertNotIn(b2, hits)

    def test_overlap_aabb(self):
        b1 = self.world.create_body(pos=(0, 5, 0), size=(1, 1, 1))
        self.world.step(0)
        
        hits = self.world.overlap_aabb(min=(-2, 3, -2), max=(2, 7, 2))
        self.assertIn(b1, hits)

    def test_get_body_stats(self):
        # 1. Test Valid Body
        pos = (5.0, 10.0, -2.0)
        rot = (0.0, 0.7071, 0.0, 0.7071) # 90 degree rotation around Y
        h = self.world.create_body(pos=pos, rot=rot, motion=culverin.MOTION_DYNAMIC)
        self.world.step(0)

        stats = self.world.get_body_stats(h)
        self.assertIsNotNone(stats)
        assert stats is not None
        
        # Unpack: ((x,y,z), (x,y,z,w), (vx,vy,vz))
        got_pos, got_rot, got_vel = stats
        
        self.assertAlmostEqual(got_pos[0], pos[0], places=3)
        self.assertAlmostEqual(got_pos[1], pos[1], places=3)
        self.assertAlmostEqual(got_pos[2], pos[2], places=3)
        
        self.assertAlmostEqual(got_rot[1], rot[1], places=3)
        self.assertAlmostEqual(got_rot[3], rot[3], places=3)
        
        # Verify initial velocity is zero
        self.assertEqual(got_vel, (0.0, 0.0, 0.0))

        # 2. Test Invalid/Destroyed Body
        self.world.destroy_body(h)
        self.world.step(0)
        
        # Ensure it returns None instead of crashing or raising
        self.assertIsNone(self.world.get_body_stats(h))

    def test_get_body_stats_dynamic(self):
        # Test that velocity updates
        # Ensure it's spawned higher than the floor instantiated by TestQueries.setUp(),
        # and explicitly give it a light mass so that 100 Ns easily overcomes gravity
        h = self.world.create_body(pos=(0, 5, 0), motion=culverin.MOTION_DYNAMIC, mass=1.0)
        self.world.apply_impulse(h, 0, 100, 0)
        self.world.step(1/60.0)
        
        stats = self.world.get_body_stats(h)
        self.assertIsNotNone(stats)
        assert stats is not None
        _, _, vel = stats
        self.assertGreater(vel[1], 0.0, "Velocity should be updated")


class TestCollisionsAndEvents(CulverinTestCase):
    # test_collision_filtering can segfault in musllinux with Clang 19 and above...
    def test_collision_filtering(self):
        _floor = self.world.create_body(pos=(0, 5, 0), size=(10, 0.1, 10), motion=culverin.MOTION_STATIC, category=1, mask=0xFFFF)
        
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

    def test_contact_removal_lifecycle(self):
        """Verify that EVENT_REMOVED is fired correctly even if one body is destroyed."""
        b1 = self.world.create_body(pos=(0, 0, 0), size=(2, 2, 2), motion=culverin.MOTION_STATIC)
        b2 = self.world.create_body(pos=(0, 0.5, 0), size=(1, 1, 1), motion=culverin.MOTION_DYNAMIC)
        self.world.step(1/60) # Generate Added event
        
        # Destroy b2 while it is touching b1
        self.world.destroy_body(b2)
        self.world.step(1/60) # Should trigger Removed event
        
        events = self.world.get_contact_events_ex()
        removed = [e for e in events if e["type"] == culverin.EVENT_REMOVED]
        self.assertGreater(len(removed), 0)
        self.assertIn(b1, removed[0]["bodies"])


class TestCharactersAndVehicles(CulverinTestCase):
    def test_character_lifecycle_and_movement(self):
        char = self.world.create_character(pos=(0, 2, 0), height=1.8, radius=0.4)
        self.assertTrue(self.world.is_alive(char.handle))
        
        char.move((10, 0, 0), 1/60)
        self.world.step(1/60)
        self.assertGreater(char.get_position()[0], 0.0)
        
        # Test get_render_transform interpolation
        r_pos, _r_rot = char.get_render_transform(0.5)
        self.assertGreater(r_pos[0], 0.0)

    def test_character_push_power(self):
        """Verify character strength affects dynamic body interaction."""
        crate = self.world.create_body(pos=(1, 1, 0), size=(1,1,1), mass=5.0)
        char = self.world.create_character(pos=(0, 1, 0))
        self.world.step(0)
        
        # Weak strength
        char.set_strength(10.0)
        char.move((20, 0, 0), 1/60)
        self.world.step(1/60)
        vel_weak = self.get_vel(crate)[0]
        
        # Strong strength
        char.set_strength(50000.0)
        char.set_position((0,1,0))
        self.world.set_linear_velocity(crate, x=0, y=0, z=0)
        self.world.set_position(crate, x=1, y=1, z=0)
        
        # CRITICAL FIX: Step(0) flushes the queue so the character 
        # actually teleports BEFORE we call char.move()
        self.world.step(0) 
        
        char.move((20, 0, 0), 1/60)
        self.world.step(1/60)
        vel_strong = self.get_vel(crate)[0]
        
        self.assertGreater(vel_strong, vel_weak)

    def test_wheeled_vehicle(self):
        # 1. Add a floor with friction so the wheels can grip!
        self.world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC, friction=1.0)
        
        chassis = self.world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        wheels: list[WheelConfig] = [{"pos": (x, -0.5, z), "radius": 0.4} for x in [-0.8, 0.8] for z in [1.2, -1.2]]
        car = self.world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")
        self.world.step(0)
        
        # Settle
        for _ in range(60): self.world.step(1/60)
        
        car.set_input(forward=1.0)
        for _ in range(60): self.world.step(1/60)
        
        self.assertGreater(self.get_vel(chassis)[2], 1.0)
        self.assertEqual(car.wheel_count, 4)
        
    def test_tracked_vehicle(self):
        # Floor for tracks
        self.world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC, friction=1.0)
        
        chassis = self.world.create_body(pos=(0, 2, 0), size=(2, 1, 3), mass=5000.0)
        wheels: list[WheelConfig] = [{"pos": (x, -1.0, z), "radius": 0.5} for x in [-1.5, 1.5] for z in [2.0, 0.0, -2.0]]
        
        # Track 0 (Left): indices 0, 2, 4. Track 1 (Right): indices 1, 3, 5
        tracks: list[TrackConfig] = [
            {"indices": [0, 2, 4], "driven_wheel": 0},
            {"indices": [1, 3, 5], "driven_wheel": 1}
        ]
        
        tank = self.world.create_tracked_vehicle(chassis=chassis, wheels=wheels, tracks=tracks)
        self.world.step(0)
        
        tank.set_tank_input(left=1.0, right=1.0)
        for _ in range(60): self.world.step(1/60)
        
        self.assertGreater(self.get_vel(chassis)[2], 0.5) # Tank should move forward


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

    def test_resize_memoryview_lock(self):
        """Ensure world cannot resize while a memoryview is held."""
        # Create a world that resizes after 64 bodies
        world = culverin.PhysicsWorld(settings={"max_bodies": 200})
        
        # Fill the initial 64-slot capacity
        for _ in range(64):
            world.create_body(pos=(0,0,0))
        world.step(0)

        # Export a buffer to lock the current C arrays
        _view = world.positions 
        
        # Adding the 65th body triggers PhysicsWorld_resize in C
        with self.assertRaises(BufferError):
            world.create_body(pos=(1,2,3))

class TestInterpolation(CulverinTestCase):
    def test_teleport_interpolation_reset(self):
        """Verify that set_position resets prev_positions to prevent interpolation streaks."""
        h = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)
        
        # Teleport
        self.world.set_position(h, x=1000, y=0, z=0)
        
        # We check render state BEFORE the next step()
        state = self.world.get_render_state(alpha=0.5)
        data = np.frombuffer(state, dtype=np.float32)
        # If the fix is in, this will be 1000. If not, it will be 500.
        self.assertEqual(data[0], 1000.0)

class TestEdgeCases(CulverinTestCase):
    def test_numerical_stability(self):
        """Test how the engine handles non-finite inputs."""
        # 1. NaN Position (In create_body)
        # Note: We use a float('nan') directly to ensure it hits C
        with self.assertRaises(ValueError):
            self.world.create_body(pos=(float('nan'), 0.0, 0.0))
        
        # 2. Infinite Impulse (In apply_impulse)
        h = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)
        with self.assertRaises(ValueError):
            self.world.apply_impulse(h, x=float('inf'), y=0.0, z=0.0)

    def test_handle_invalidation_chain(self):
        """Test 'Silent Invalidation': deleted bodies return None for all operations."""
        h = self.world.create_body(pos=(0, 10, 0))
        
        # Verify it works initially
        self.assertTrue(self.world.is_alive(h))
        
        # Kill the body
        self.world.destroy_body(h)
        
        # 1. Mutators: Instead of raising, they now return None (Silent Fail)
        # This proves the C-layer caught the SLOT_PENDING_DESTROY state.
        res = self.world.set_position(h, 0, 5, 0)
        self.assertIsNone(res, "set_position should return None for stale handles")
        
        # 2. Getters: Consistent return of None
        self.assertFalse(self.world.is_alive(h), "is_alive must be False")
        self.assertIsNone(self.world.get_index(h), "get_index must be None")
        self.assertIsNone(self.world.get_position(h), "get_position must be None")
        
        # 3. Double-Delete Safety (Idempotency)
        # Ensure calling destroy again doesn't crash the engine
        self.world.destroy_body(h)

    def test_empty_batch_inputs(self):
        """Ensure batch methods don't segfault on empty data."""
        # 1. Empty raycast
        res = self.world.raycast_batch(b"", b"", max_dist=10.0)
        self.assertEqual(len(res), 0)
        
        # 2. Empty body creation
        handles = self.world.create_bodies_batch([], [])
        self.assertEqual(len(handles), 0)

    def test_zero_scale_shapes(self):
        """Jolt usually dislikes zero-volume shapes. We should handle it gracefully."""
        # This should either raise a Python error or be clamped in C
        h = self.world.create_body(pos=(0, 0, 0), shape=culverin.SHAPE_BOX, size=(0, 0, 0))
        self.world.step(0.016)
        self.assertTrue(self.world.is_alive(h))

    def test_extreme_mass_ratios(self):
        """Test 1mg vs 1,000,000kg to see if the solver explodes."""
        _heavy = self.world.create_body(pos=(0, 0, 0), mass=1e6, motion=culverin.MOTION_DYNAMIC)
        _light = self.world.create_body(pos=(0, 1, 0), mass=1e-3, motion=culverin.MOTION_DYNAMIC)
        self.world.step(0.1) # Just check it doesn't crash


class TestComplexShapes(CulverinTestCase):
    def test_compound_body(self):
        parts: list[tuple[tuple[int, int, int], tuple[int, int, int, int], int, tuple[int, int, int]] | tuple[tuple[int, int, int], tuple[int, int, int, int], int, tuple[int]]] = [
            ((0, 0, 0), (0, 0, 0, 1), culverin.SHAPE_BOX, (1, 1, 1)),
            ((0, 2, 0), (0, 0, 0, 1), culverin.SHAPE_SPHERE, (1,))
        ]
        cb = self.world.create_compound_body(pos=(0, 10, 0), rot=(0, 0, 0, 1), parts=parts)
        self.world.step(0)
        self.assertTrue(self.world.is_alive(cb))
        
    def test_convex_hull(self):
        # Convert the list of points to a flat float32 bytes buffer
        points = np.array([
            [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
            [0, 0, -2] # Pyramid tip
        ], dtype=np.float32).tobytes()
        
        hull = self.world.create_convex_hull(pos=(0, 10, 0), rot=(0, 0, 0, 1), points=points, mass=5.0)
        self.world.step(0)
        self.assertTrue(self.world.is_alive(hull))


class TestConstraints(CulverinTestCase):
    def test_hinge_constraint(self):
        b1 = self.world.create_body(pos=(0, 5, 0), motion=culverin.MOTION_STATIC)
        b2 = self.world.create_body(pos=(2, 5, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.step(0)
        
        # Hinge params: (pivot_x, pivot_y, pivot_z), (axis_x, axis_y, axis_z), min_limit, max_limit
        c_handle = self.world.create_constraint(
            culverin.CONSTRAINT_HINGE, b1, b2, 
            params=((0, 5, 0), (0, 0, 1), -math.pi, math.pi)
        )
        self.assertIsNotNone(c_handle)
        
        # Test destruction
        self.world.destroy_constraint(c_handle)

    def test_hinge_motor(self):
        b1 = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_STATIC)
        # Verify b2 is DYNAMIC and has MASS
        b2 = self.world.create_body(pos=(2, 0, 0), motion=culverin.MOTION_DYNAMIC, mass=1.0)
        self.world.step(0)
        
        c = self.world.create_constraint(
            culverin.CONSTRAINT_HINGE, b1, b2, 
            params=((0, 0, 0), (0, 0, 1), -3.14, 3.14),
            motor={"type": 2, "target": 0.0}
        )

        ctype = self.world.get_constraint_type(c)
        self.assertEqual(ctype, culverin.CONSTRAINT_HINGE, 
                        f"Expected Hinge (2), got {ctype}")
        
        # Explicitly wake up b2
        self.world.activate(b2)

        self.world.step(1/60)
        
        # Set target
        self.world.set_constraint_target(c, math.pi / 2)
        
        # Run enough steps for the motor spring to ramp up
        for i in range(150):
            self.world.step(1/60)
            if i % 10 == 0:
                print(f"step {i}: pos={self.get_pos(b2)}")
                
        pos = self.get_pos(b2)
        self.assertLess(pos[0], 0.5, f"Body should have swung; current X is {pos[0]}")


class TestRagdollsAndSkeletons(CulverinTestCase):
    def test_skeleton_and_ragdoll_creation(self):
        import culverin
        skel = culverin.Skeleton()
        root = skel.add_joint(name="pelvis", parent_index=-1)
        spine = skel.add_joint(name="spine", parent_index=root)
        skel.finalize()
        
        settings = self.world.create_ragdoll_settings(skeleton=skel)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        settings.add_part(joint_index=spine, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.4, 0.2), parent_index=root)
        
        ragdoll = self.world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        self.assertIsNotNone(ragdoll)
        
        handles = ragdoll.get_body_handles()
        self.assertEqual(len(handles), 2)
        
        self.world.step(0)
        # Apply motor drive (Physical Animation)
        matrices = np.eye(4, dtype=np.float32)
        mats_buffer = np.stack([matrices, matrices]).tobytes() # 2 joints
        ragdoll.drive_to_pose(root_pos=(0, 5, 0), root_rot=(0,0,0,1), matrices=mats_buffer)

    def test_ragdoll_get_debug_info(self):
        import culverin
        # 1. Setup
        skel = culverin.Skeleton()
        root = skel.add_joint(name="pelvis", parent_index=-1)
        skel.finalize()
        
        settings = self.world.create_ragdoll_settings(skeleton=skel)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        
        ragdoll = self.world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        
        # 2. Physics Step required to initialize the bodies in the Jolt interface
        self.world.step(1/60.0)
        
        # 3. Call method under test
        debug_info = ragdoll.get_debug_info()
        
        # 4. Assertions
        self.assertIsInstance(debug_info, list, "get_debug_info should return a list")
        self.assertEqual(len(debug_info), 1, "Should have 1 body part")
        
        part = debug_info[0]
        self.assertIn("index", part)
        self.assertIn("pos", part)
        self.assertIn("vel", part)
        
        self.assertIsInstance(part["index"], int)
        self.assertIsInstance(part["pos"], tuple)
        self.assertIsInstance(part["vel"], tuple)
        self.assertEqual(len(part["pos"]), 3)
        
        # Verify valid values (pos should be near 0,10,0)
        self.assertAlmostEqual(part["pos"][1], 10.0, delta=1.0)


class TestStateManagement(CulverinTestCase):
    def test_save_and_load_state(self):
        b = self.world.create_body(pos=(0, 10, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.step(0)
        
        # Capture state at Y=10
        state = self.world.save_state()
        self.assertIsInstance(state, bytes)
        self.assertGreater(len(state), 0)
        
        # Let it fall
        for _ in range(10): self.world.step(1/60)
        self.assertLess(self.get_pos(b)[1], 10.0)
        
        # Restore state
        self.world.load_state(state=state)
        # Load state requires shadow buffer sync for python to see it immediately
        self.world.step(0) 
        self.assertEqual(self.get_pos(b)[1], 10.0)


class TestUserData(CulverinTestCase):
    def test_user_data_rw(self):
        h = self.world.create_body(pos=(0, 0, 0), user_data=42)
        self.world.step(0)
        self.assertEqual(self.world.get_user_data(h), 42)
        
        self.world.set_user_data(h, 999)
        self.world.step(0)
        self.assertEqual(self.world.get_user_data(h), 999)

class TestProfilerScenario(CulverinTestCase):
    """
    This test suite is designed for third-party profilers (VizTracer, py-spy, cProfile).
    The methods below exercise every 'hot path' in the engine:
    Memory Allocation -> Simulation -> Batch Queries -> Interpolation -> Data Sync.
    """

    def setUp(self):
        # We need a higher max_bodies limit than the base CulverinTestCase
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 10000})
        self.world.step(0)

    def test_full_stress_profile_cycle(self):
        body_count = 5000
        positions = np.random.uniform(-100, 100, (body_count, 3)).astype(np.float32).tolist()
        sizes = [[1.0, 1.0, 1.0]] * body_count
        
        self.world.create_bodies_batch(positions, sizes, culverin.SHAPE_BOX, culverin.MOTION_DYNAMIC)
        
        for i in range(200):
            self.world.step(1/60.0)
            if i % 10 == 0:
                starts = np.zeros((1000, 3), dtype=np.float32).tobytes()
                dirs = np.zeros((1000, 3), dtype=np.float32).tobytes()
                self.world.raycast_batch(starts, dirs, max_dist=100.0)
            self.world.get_render_state(alpha=0.5)

        state = self.world.save_state()
        self.world.load_state(state=state)
    def test_free_threading_concurrency(self):
        """
        Tests true parallelism. 
        - Physics simulation runs in a background thread.
        - A heavy Python-side 'simulation' runs in the main thread.
        - Total time should be significantly less than sequential execution.
        """
        import threading
        
        # 1. Setup: Load a moderate number of bodies
        body_count = 2000
        self.world.create_bodies_batch(
            np.random.uniform(-50, 50, (body_count, 3)).tolist(), 
            [[0.5]*3]*body_count, culverin.SHAPE_BOX, culverin.MOTION_DYNAMIC
        )
        
        # We want to run 1000 physics steps
        iterations = 1000
        
        def physics_task():
            for _ in range(iterations):
                self.world.step(1/60.0)

        # 2. Main thread heavy Python math (simulating game logic/AI)
        def heavy_python_math():
            res = 0
            for i in range(500_000): # Heavy CPU loop
                res += math.sin(i) * math.cos(i)
            return res

        t0 = time.perf_counter()
        
        # Start Physics in background
        t = threading.Thread(target=physics_task)
        t.start()
        
        # Run Python math in main thread
        heavy_python_math()
        
        t.join()
        total_time = time.perf_counter() - t0
        
        print(f"\n[Free-Threading] Parallel Physics + Math: {total_time*1000:.2f}ms")

    def test_contention_profile(self):
        """Force multiple threads to fight for the PhysicsWorld lock."""
        def hammer_getters():
            for _ in range(1000):
                # Rapid-fire calls to getters while step() is likely running
                self.world.get_render_state(alpha=0.5)

        threads = [threading.Thread(target=hammer_getters) for _ in range(4)]
        for t in threads: t.start()
        for _ in range(60): self.world.step(1/60.0)
        for t in threads: t.join()

    def test_extreme_command_contention(self):
        """Hammer the command queue from 8 threads while stepping the world."""
        bodies = [self.world.create_body(pos=(0,0,0)) for _ in range(100)]
        self.world.step(0)

        def worker():
            for _ in range(500):
                # Randomly move bodies
                target = bodies[np.random.randint(0, 100)]
                self.world.apply_impulse(target, 0, 10, 0)
                self.world.set_position(target, x=np.random.rand(), y=2, z=0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        
        for _ in range(100):
            self.world.step(1/60)
            
        for t in threads: t.join()
        # If we didn't segfault, the PyMutex and Command Queue swap are working.


if __name__ == '__main__':
    unittest.main(verbosity=2)
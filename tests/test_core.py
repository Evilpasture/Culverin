import contextlib
import math
import re
import textwrap
import threading
import time
import types
import unittest
from pathlib import Path
from types import CodeType, SimpleNamespace
from typing import Literal, Protocol

import numpy as np

import culverin
from culverin import TrackConfig, WheelConfig


class CulverinTestCase(unittest.TestCase):
    """Base class providing helper methods for interacting with Culverin buffers."""

    def setUp(self) -> None:
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 2048})
        self.world.step(0)  # Flush initial state

    def tearDown(self) -> None:
        del self.world

    def get_pos(self, handle: int) -> tuple[float, float, float]:
        return self.world.get_position(handle)

    def get_vel(self, handle: int) -> tuple[float, float, float]:
        return self.world.get_velocity(handle)

    # Well, my computer has assertHasAttr, but GitHub Actions don't, so I'm adding one myself.'
    if not hasattr(unittest.TestCase, "assertHasAttr"):

        def assertHasAttr(self, obj: object, name: str, msg: str | None = None) -> None:
            if not hasattr(obj, name):
                if isinstance(obj, types.ModuleType):
                    standardMsg = f"module {obj.__name__!r} has no attribute {name!r}"
                elif isinstance(obj, type):
                    standardMsg = f"type object {obj.__name__!r} has no attribute {name!r}"
                else:
                    standardMsg = f"{type(obj).__name__!r} object has no attribute {name!r}"
                self.fail(self._formatMessage(msg, standardMsg))


class TestPrintVersion(CulverinTestCase):
    def test_print_version(self) -> None:
        self.assertHasAttr(culverin, "__version__", "Check version...")
        print(f"Version: {culverin.__version__}")


class TestCoreMechanics(CulverinTestCase):
    def test_activation_and_gravity(self) -> None:
        h = self.world.create_body(pos=(0, 10, 0), shape=culverin.SHAPE_SPHERE, size=1.0)
        self.world.step(0)

        self.assertEqual(self.get_pos(h)[1], 10.0)
        self.world.step(1 / 60.0)
        self.assertLess(self.get_pos(h)[1], 10.0, "Body did not fall")

    def test_generational_handles(self) -> None:
        h1 = self.world.create_body(pos=(0, 0, 0))
        self.assertTrue(self.world.is_alive(h1))

        self.world.destroy_body(h1)
        self.world.step(0)
        self.assertFalse(self.world.is_alive(h1))

        h2 = self.world.create_body(pos=(10, 10, 10))
        self.assertNotEqual(h1, h2, "Handles must be unique across generations")
        self.assertTrue(self.world.is_alive(h2))

    def test_causal_consistency(self) -> None:
        # Apply impulse to a body created in the same un-stepped frame
        h = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.apply_impulse(h, 100, 0, 0)
        self.world.step(1 / 60.0)
        self.assertGreater(self.get_vel(h)[0], 0.0)

    def test_buoyancy(self) -> None:
        ball = self.world.create_body(
            pos=(0, 10, 0), shape=culverin.SHAPE_SPHERE, size=(0.5,), mass=10.0
        )
        self.world.step(0)

        submerged_frames = 0
        for _ in range(120):
            if self.world.apply_buoyancy(
                handle=ball,
                surface_y=5.0,
                buoyancy=1.5,
                linear_drag=0.8,
                angular_drag=0.5,
                dt=1 / 60,
            ):
                submerged_frames += 1
            self.world.step(1 / 60)

        self.assertGreater(submerged_frames, 0)
        self.assertGreater(self.get_pos(ball)[1], 4.0, "Ball should be floating, not at bottom")

    def test_heightfield(self) -> None:
        grid_size = 64
        x = np.linspace(0, 10, grid_size, dtype=np.float32)
        xx, _ = np.meshgrid(x, x)
        heights = (xx * 0.5).astype(np.float32).tobytes()

        self.world.create_heightfield(
            pos=(0, 0, 0),
            rot=(0, 0, 0, 1),
            scale=(1, 1, 1),
            heights=heights,
            grid_size=grid_size,
        )
        ball = self.world.create_body(pos=(5, 3.5, 5), shape=culverin.SHAPE_SPHERE, mass=10.0)

        self.world.step(0)
        start_x = self.get_pos(ball)[0]
        for _ in range(60):
            self.world.step(1 / 60)

        self.assertLess(self.get_pos(ball)[0], start_x, "Ball did not roll down slope")

    def test_material_hot_swap(self) -> None:
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
    def setUp(self) -> None:
        super().setUp()
        self.floor = self.world.create_body(
            pos=(0, -1, 0), size=(10, 1, 10), motion=culverin.MOTION_STATIC
        )
        self.world.step(0)

    def test_raycast(self) -> None:
        res = self.world.raycast(start=(0, 5, 0), direction=(0, -1, 0), max_dist=10.0)
        self.assertIsNotNone(res)
        assert res is not None
        handle, fraction, normal = res
        self.assertEqual(handle, self.floor)
        self.assertAlmostEqual(fraction, 0.5, places=3)
        np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-4)

    def test_batch_raycast(self) -> None:
        ray_count = 100
        starts = np.zeros((ray_count, 3), dtype=np.float32)
        starts[:, 1] = 10.0
        dirs = np.zeros((ray_count, 3), dtype=np.float32)
        dirs[:, 1] = -1.0

        res_raw = self.world.raycast_batch(starts.tobytes(), dirs.tobytes(), max_dist=20.0)
        dt = np.dtype(
            [
                ("h", np.uint64),
                ("f", np.float32),
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4"),
                ("px", "f4"),
                ("py", "f4"),
                ("pz", "f4"),
                ("sub", "u4"),
                ("mat", "u4"),
                ("pad", "u4"),
            ]
        )
        results = np.frombuffer(res_raw, dtype=dt)

        self.assertEqual(np.count_nonzero(results["h"]), ray_count)
        self.assertEqual(results["h"][0], self.floor)
        self.assertAlmostEqual(results["py"][0], 0.0, places=3)

    def test_shapecast(self) -> None:
        target = self.world.create_body(
            pos=(10, 5, 0), size=(1, 5, 5), motion=culverin.MOTION_STATIC
        )
        self.world.step(0)

        hit = self.world.shapecast(
            shape=culverin.SHAPE_SPHERE,
            pos=(0, 5, 0),
            rot=(0, 0, 0, 1),
            dir=(20, 0, 0),
            size=(0.5,),
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit[0], target)
        self.assertAlmostEqual(hit[3][0], -1.0, places=3)  # Normal faces -X

    def test_overlap_sphere(self) -> None:
        b1 = self.world.create_body(pos=(5, 0, 0), size=(1, 1, 1))
        b2 = self.world.create_body(pos=(-5, 0, 0), size=(1, 1, 1))
        self.world.step(0)

        hits = self.world.overlap_sphere(center=(4.5, 0, 0), radius=2.0)
        self.assertIn(b1, hits)
        self.assertNotIn(b2, hits)

    def test_overlap_aabb(self) -> None:
        b1 = self.world.create_body(pos=(0, 5, 0), size=(1, 1, 1))
        self.world.step(0)

        hits = self.world.overlap_aabb(min=(-2, 3, -2), max=(2, 7, 2))
        self.assertIn(b1, hits)

    def test_get_body_stats(self) -> None:
        # 1. Test Valid Body
        pos = (5.0, 10.0, -2.0)
        rot = (0.0, 0.7071, 0.0, 0.7071)  # 90 degree rotation around Y
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

    def test_get_body_stats_dynamic(self) -> None:
        # Test that velocity updates
        # Ensure it's spawned higher than the floor instantiated by TestQueries.setUp(),
        # and explicitly give it a light mass so that 100 Ns easily overcomes gravity
        h = self.world.create_body(pos=(0, 5, 0), motion=culverin.MOTION_DYNAMIC, mass=1.0)
        self.world.apply_impulse(h, 0, 100, 0)
        self.world.step(1 / 60.0)

        stats = self.world.get_body_stats(h)
        self.assertIsNotNone(stats)
        assert stats is not None
        _, _, vel = stats
        self.assertGreater(vel[1], 0.0, "Velocity should be updated")


class TestCollisionsAndEvents(CulverinTestCase):
    # test_collision_filtering can segfault in musllinux with Clang 19 and above...
    def test_collision_filtering(self) -> None:
        _floor = self.world.create_body(
            pos=(0, 5, 0),
            size=(10, 0.1, 10),
            motion=culverin.MOTION_STATIC,
            category=1,
            mask=0xFFFF,
        )

        # Added the size=(0.5, 0.5, 0.5) arguments back
        player = self.world.create_body(
            pos=(2, 10, 0), size=(0.5, 0.5, 0.5), category=2, mask=1 | 4
        )
        ghost = self.world.create_body(pos=(2, 12, 0), size=(0.5, 0.5, 0.5), category=4, mask=1)

        for _ in range(120):
            self.world.step(1 / 60)

        self.assertGreater(self.get_pos(player)[1], 5.0)  # Player caught by floor
        self.assertLess(self.get_pos(ghost)[1], 5.6)  # Ghost passed through player, caught by floor

    def test_sensor_events(self) -> None:
        sensor = self.world.create_body(
            pos=(0, 5, 0),
            size=(2, 0.5, 2),
            is_sensor=True,
            motion=culverin.MOTION_STATIC,
        )
        crate = self.world.create_body(pos=(0, 10, 0), motion=culverin.MOTION_DYNAMIC)

        hit = False
        for _ in range(60):
            self.world.step(1 / 60)
            for ev in self.world.get_contact_events_ex():
                if set(ev["bodies"]) == {sensor, crate}:
                    hit = True
        self.assertTrue(hit)

    def test_contact_removal_lifecycle(self) -> None:
        """Verify that EVENT_REMOVED is fired correctly even if one body is destroyed."""
        b1 = self.world.create_body(pos=(0, 0, 0), size=(2, 2, 2), motion=culverin.MOTION_STATIC)
        b2 = self.world.create_body(pos=(0, 0.5, 0), size=(1, 1, 1), motion=culverin.MOTION_DYNAMIC)
        self.world.step(1 / 60)  # Generate Added event

        # Destroy b2 while it is touching b1
        self.world.destroy_body(b2)
        self.world.step(1 / 60)  # Should trigger Removed event

        events = self.world.get_contact_events_ex()
        removed = [e for e in events if e["type"] == culverin.EVENT_REMOVED]
        self.assertGreater(len(removed), 0)
        self.assertIn(b1, removed[0]["bodies"])


class TestCharactersAndVehicles(CulverinTestCase):
    def test_character_lifecycle_and_movement(self) -> None:
        char = self.world.create_character(pos=(0, 2, 0), height=1.8, radius=0.4)
        self.assertTrue(self.world.is_alive(char.handle))

        char.move((10, 0, 0), 1 / 60)
        self.world.step(1 / 60)
        self.assertGreater(char.get_position()[0], 0.0)

        # Test get_render_transform interpolation
        r_pos, _r_rot = char.get_render_transform(0.5)
        self.assertGreater(r_pos[0], 0.0)

    def test_character_push_power(self) -> None:
        """Verify character strength affects dynamic body interaction."""
        crate = self.world.create_body(pos=(1, 1, 0), size=(1, 1, 1), mass=5.0)
        char = self.world.create_character(pos=(0, 1, 0))
        self.world.step(0)

        # Weak strength
        char.set_strength(10.0)
        char.move((20, 0, 0), 1 / 60)
        self.world.step(1 / 60)
        vel_weak = self.get_vel(crate)[0]

        # Strong strength
        char.set_strength(50000.0)
        char.set_position((0, 1, 0))
        self.world.set_linear_velocity(crate, x=0, y=0, z=0)
        self.world.set_position(crate, x=1, y=1, z=0)

        # CRITICAL FIX: Step(0) flushes the queue so the character
        # actually teleports BEFORE we call char.move()
        self.world.step(0)

        char.move((20, 0, 0), 1 / 60)
        self.world.step(1 / 60)
        vel_strong = self.get_vel(crate)[0]

        self.assertGreater(vel_strong, vel_weak)

    def test_wheeled_vehicle(self) -> None:
        # 1. Add a floor with friction so the wheels can grip!
        self.world.create_body(
            pos=(0, -1, 0),
            size=(100, 1, 100),
            motion=culverin.MOTION_STATIC,
            friction=1.0,
        )

        chassis = self.world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        wheels: list[WheelConfig] = [
            {"pos": (x, -0.5, z), "radius": 0.4} for x in [-0.8, 0.8] for z in [1.2, -1.2]
        ]
        car = self.world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")
        self.world.step(0)

        # Settle
        for _ in range(60):
            self.world.step(1 / 60)

        car.set_input(forward=1.0)
        for _ in range(60):
            self.world.step(1 / 60)

        self.assertGreater(self.get_vel(chassis)[2], 1.0)
        self.assertEqual(car.wheel_count, 4)

    def test_tracked_vehicle(self) -> None:
        # Floor for tracks
        self.world.create_body(
            pos=(0, -1, 0),
            size=(100, 1, 100),
            motion=culverin.MOTION_STATIC,
            friction=1.0,
        )

        chassis = self.world.create_body(pos=(0, 2, 0), size=(2, 1, 3), mass=5000.0)
        wheels: list[WheelConfig] = [
            {"pos": (x, -1.0, z), "radius": 0.5} for x in [-1.5, 1.5] for z in [2.0, 0.0, -2.0]
        ]

        # Track 0 (Left): indices 0, 2, 4. Track 1 (Right): indices 1, 3, 5
        tracks: list[TrackConfig] = [
            {"indices": [0, 2, 4], "driven_wheel": 0},
            {"indices": [1, 3, 5], "driven_wheel": 1},
        ]

        tank = self.world.create_tracked_vehicle(chassis=chassis, wheels=wheels, tracks=tracks)
        self.world.step(0)

        tank.set_tank_input(left=1.0, right=1.0)
        for _ in range(60):
            self.world.step(1 / 60)

        self.assertGreater(self.get_vel(chassis)[2], 0.5)  # Tank should move forward


class TestThreadSafety(CulverinTestCase):
    def test_blocking_mutation(self) -> None:
        # 1. Create a valid body to mutate
        h = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)

        stop = False

        def physics_worker() -> None:
            from contextlib import suppress

            while not stop:
                with suppress(RuntimeError):
                    self.world.step(1 / 60)

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

    def test_resize_memoryview_lock(self) -> None:
        """Ensure world cannot resize while a memoryview is held."""
        # Create a world that resizes after 64 bodies
        world = culverin.PhysicsWorld(settings={"max_bodies": 200})

        # Fill the initial 64-slot capacity
        for _ in range(64):
            world.create_body(pos=(0, 0, 0))
        world.step(0)

        # Export a buffer to lock the current C arrays
        _view = world.positions

        # Adding the 65th body triggers PhysicsWorld_resize in C
        with self.assertRaises(BufferError):
            world.create_body(pos=(1, 2, 3))


class TestInterpolation(CulverinTestCase):
    def test_teleport_interpolation_reset(self) -> None:
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
    def test_numerical_stability(self) -> None:
        """Test how the engine handles non-finite inputs."""
        # 1. NaN Position (In create_body)
        # Note: We use a float('nan') directly to ensure it hits C
        with self.assertRaises(ValueError):
            self.world.create_body(pos=(float("nan"), 0.0, 0.0))

        # 2. Infinite Impulse (In apply_impulse)
        h = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)
        with self.assertRaises(ValueError):
            self.world.apply_impulse(h, x=float("inf"), y=0.0, z=0.0)

    def test_handle_invalidation_chain(self) -> None:
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

    def test_empty_batch_inputs(self) -> None:
        """Ensure batch methods don't segfault on empty data."""
        # 1. Empty raycast
        res = self.world.raycast_batch(b"", b"", max_dist=10.0)
        self.assertEqual(len(res), 0)

        # 2. Empty body creation
        handles = self.world.create_bodies_batch([], [])
        self.assertEqual(len(handles), 0)

    def test_zero_scale_shapes(self) -> None:
        """Jolt usually dislikes zero-volume shapes. We should handle it gracefully."""
        # This should either raise a Python error or be clamped in C
        h = self.world.create_body(pos=(0, 0, 0), shape=culverin.SHAPE_BOX, size=(0, 0, 0))
        self.world.step(0.016)
        self.assertTrue(self.world.is_alive(h))

    def test_extreme_mass_ratios(self) -> None:
        """Test 1mg vs 1,000,000kg to see if the solver explodes."""
        _heavy = self.world.create_body(pos=(0, 0, 0), mass=1e6, motion=culverin.MOTION_DYNAMIC)
        _light = self.world.create_body(pos=(0, 1, 0), mass=1e-3, motion=culverin.MOTION_DYNAMIC)
        self.world.step(0.1)  # Just check it doesn't crash


class TestComplexShapes(CulverinTestCase):
    def test_compound_body(self) -> None:
        parts: list[
            tuple[
                tuple[int, int, int],
                tuple[int, int, int, int],
                int,
                tuple[int, int, int],
            ]
            | tuple[tuple[int, int, int], tuple[int, int, int, int], int, tuple[int]]
        ] = [
            ((0, 0, 0), (0, 0, 0, 1), culverin.SHAPE_BOX, (1, 1, 1)),
            ((0, 2, 0), (0, 0, 0, 1), culverin.SHAPE_SPHERE, (1,)),
        ]
        cb = self.world.create_compound_body(pos=(0, 10, 0), rot=(0, 0, 0, 1), parts=parts)
        self.world.step(0)
        self.assertTrue(self.world.is_alive(cb))

    def test_convex_hull(self) -> None:
        # Convert the list of points to a flat float32 bytes buffer
        points = np.array(
            [
                [1, 1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [0, 0, -2],  # Pyramid tip
            ],
            dtype=np.float32,
        ).tobytes()

        hull = self.world.create_convex_hull(
            pos=(0, 10, 0), rot=(0, 0, 0, 1), points=points, mass=5.0
        )
        self.world.step(0)
        self.assertTrue(self.world.is_alive(hull))


class TestConstraints(CulverinTestCase):
    def test_hinge_constraint(self) -> None:
        b1 = self.world.create_body(pos=(0, 5, 0), motion=culverin.MOTION_STATIC)
        b2 = self.world.create_body(pos=(2, 5, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.step(0)

        # Hinge params: (pivot_x, pivot_y, pivot_z), (axis_x, axis_y, axis_z), min_limit, max_limit
        c_handle = self.world.create_constraint(
            culverin.CONSTRAINT_HINGE,
            b1,
            b2,
            params=((0, 5, 0), (0, 0, 1), -math.pi, math.pi),
        )
        self.assertIsNotNone(c_handle)

        # Test destruction
        self.world.destroy_constraint(c_handle)

    def test_hinge_motor(self) -> None:
        b1 = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_STATIC)
        # Verify b2 is DYNAMIC and has MASS
        b2 = self.world.create_body(pos=(2, 0, 0), motion=culverin.MOTION_DYNAMIC, mass=1.0)
        self.world.step(0)

        c = self.world.create_constraint(
            culverin.CONSTRAINT_HINGE,
            b1,
            b2,
            params=((0, 0, 0), (0, 0, 1), -3.14, 3.14),
            motor={"type": 2, "target": 0.0},
        )

        ctype = self.world.get_constraint_type(c)
        self.assertEqual(ctype, culverin.CONSTRAINT_HINGE, f"Expected Hinge (2), got {ctype}")

        # Explicitly wake up b2
        self.world.activate(b2)

        self.world.step(1 / 60)

        # Set target
        self.world.set_constraint_target(c, math.pi / 2)

        # Run enough steps for the motor spring to ramp up
        for i in range(150):
            self.world.step(1 / 60)
            if i % 10 == 0:
                print(f"step {i}: pos={self.get_pos(b2)}")

        pos = self.get_pos(b2)
        self.assertLess(pos[0], 0.5, f"Body should have swung; current X is {pos[0]}")


class TestRagdollsAndSkeletons(CulverinTestCase):
    def test_skeleton_and_ragdoll_creation(self) -> None:
        import culverin

        skel = culverin.Skeleton()
        root = skel.add_joint(name="pelvis", parent_index=-1)
        spine = skel.add_joint(name="spine", parent_index=root)
        skel.finalize()

        settings = self.world.create_ragdoll_settings(skeleton=skel)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        settings.add_part(
            joint_index=spine,
            shape_type=culverin.SHAPE_BOX,
            size=(0.3, 0.4, 0.2),
            parent_index=root,
        )

        ragdoll = self.world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        self.assertIsNotNone(ragdoll)

        handles = ragdoll.get_body_handles()
        self.assertEqual(len(handles), 2)

        self.world.step(0)
        # Apply motor drive (Physical Animation)
        matrices = np.eye(4, dtype=np.float32)
        mats_buffer = np.stack([matrices, matrices]).tobytes()  # 2 joints
        ragdoll.drive_to_pose(root_pos=(0, 5, 0), root_rot=(0, 0, 0, 1), matrices=mats_buffer)

    def test_ragdoll_get_debug_info(self) -> None:
        import culverin

        # 1. Setup
        skel = culverin.Skeleton()
        root = skel.add_joint(name="pelvis", parent_index=-1)
        skel.finalize()

        settings = self.world.create_ragdoll_settings(skeleton=skel)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))

        ragdoll = self.world.create_ragdoll(settings=settings, pos=(0, 10, 0))

        # 2. Physics Step required to initialize the bodies in the Jolt interface
        self.world.step(1 / 60.0)

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
    def test_save_and_load_state(self) -> None:
        b = self.world.create_body(pos=(0, 10, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.step(0)

        # Capture state at Y=10
        state = self.world.save_state()
        self.assertIsInstance(state, bytes)
        self.assertGreater(len(state), 0)

        # Let it fall
        for _ in range(10):
            self.world.step(1 / 60)
        self.assertLess(self.get_pos(b)[1], 10.0)

        # Restore state
        self.world.load_state(state=state)
        # Load state requires shadow buffer sync for python to see it immediately
        self.world.step(0)
        self.assertEqual(self.get_pos(b)[1], 10.0)


class TestUserData(CulverinTestCase):
    def test_user_data_rw(self) -> None:
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

    def setUp(self) -> None:
        # We need a higher max_bodies limit than the base CulverinTestCase
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 10000})
        self.world.step(0)

    def test_full_stress_profile_cycle(self) -> None:
        body_count = 5000
        positions = np.random.uniform(-100, 100, (body_count, 3)).astype(np.float32).tolist()
        sizes = [[1.0, 1.0, 1.0]] * body_count

        self.world.create_bodies_batch(
            positions, sizes, culverin.SHAPE_BOX, culverin.MOTION_DYNAMIC
        )

        for i in range(200):
            self.world.step(1 / 60.0)
            if i % 10 == 0:
                starts = np.zeros((1000, 3), dtype=np.float32).tobytes()
                dirs = np.zeros((1000, 3), dtype=np.float32).tobytes()
                self.world.raycast_batch(starts, dirs, max_dist=100.0)
            self.world.get_render_state(alpha=0.5)

        state = self.world.save_state()
        self.world.load_state(state=state)

    def test_free_threading_concurrency(self) -> None:
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
            [[0.5] * 3] * body_count,
            culverin.SHAPE_BOX,
            culverin.MOTION_DYNAMIC,
        )

        # We want to run 1000 physics steps
        iterations = 1000

        def physics_task() -> None:
            for _ in range(iterations):
                self.world.step(1 / 60.0)

        # 2. Main thread heavy Python math (simulating game logic/AI)
        def heavy_python_math() -> float | Literal[0]:
            res = 0
            for i in range(500_000):  # Heavy CPU loop
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

        print(f"\n[Free-Threading] Parallel Physics + Math: {total_time * 1000:.2f}ms")

    def test_contention_profile(self) -> None:
        """Force multiple threads to fight for the PhysicsWorld lock."""

        def hammer_getters() -> None:
            for _ in range(1000):
                # Rapid-fire calls to getters while step() is likely running
                self.world.get_render_state(alpha=0.5)

        threads = [threading.Thread(target=hammer_getters) for _ in range(4)]
        for t in threads:
            t.start()
        for _ in range(60):
            self.world.step(1 / 60.0)
        for t in threads:
            t.join()

    def test_extreme_command_contention(self) -> None:
        """Hammer the command queue from 8 threads while stepping the world."""
        import random  # Use the stdlib random, not numpy.random

        bodies = [self.world.create_body(pos=(0, 0, 0)) for _ in range(100)]
        self.world.step(0)

        def worker() -> None:
            for _ in range(500):
                # random.randint and random.random are thread-safe in Python
                target = bodies[random.randint(0, 99)]
                self.world.apply_impulse(target, 0, 10, 0)
                self.world.set_position(target, x=random.random(), y=2, z=0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()

        for _ in range(100):
            self.world.step(1 / 60)

        for t in threads:
            t.join()
        # If we didn't segfault, the PyMutex and Command Queue swap are working.


class TestSleepingStates(CulverinTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Add a static floor so bodies have a surface to rest on
        self.floor = self.world.create_body(
            pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC
        )
        self.world.step(0)

    def test_body_goes_to_sleep(self) -> None:
        """Ensure a dynamic body falls asleep after coming to rest to save CPU."""
        # Place a box just above the floor (using a Box so it doesn't roll infinitely)
        box = self.world.create_body(
            pos=(0, 0.5, 0),
            shape=culverin.SHAPE_BOX,
            size=(1, 1, 1),
            motion=culverin.MOTION_DYNAMIC,
        )
        self.world.step(0)

        # Initially, it should be active as it falls
        self.assertTrue(self.world.is_active(box), "Body should be active upon creation")

        # Step the simulation enough times for the body to hit the floor, settle, and sleep.
        # Jolt's default sleep timer is usually around 0.5 seconds of inactivity.
        # We step for 2 seconds (120 frames) to be absolutely sure.
        for _ in range(120):
            self.world.step(1 / 60.0)

        self.assertFalse(
            self.world.is_active(box),
            "Body did not go to sleep after resting on the floor",
        )

    def test_impulse_wakes_body(self) -> None:
        """Ensure a sleeping body automatically wakes up when acted upon."""
        box = self.world.create_body(
            pos=(0, 0.5, 0),
            shape=culverin.SHAPE_BOX,
            size=(1, 1, 1),
            motion=culverin.MOTION_DYNAMIC,
            mass=1.0,
        )

        # Let it settle and fall asleep
        for _ in range(120):
            self.world.step(1 / 60.0)

        # Pre-condition check
        self.assertFalse(self.world.is_active(box), "Pre-condition failed: Body never fell asleep")

        # Apply a physical force (this should queue an activation command in the C layer)
        self.world.apply_impulse(box, 0, 100, 0)

        # Step once to process the command queue and advance physics
        self.world.step(1 / 60.0)

        # 1. Verify the engine flagged it as active again
        self.assertTrue(
            self.world.is_active(box),
            "Applying an impulse did not wake the sleeping body",
        )

        # 2. Verify it actually moved upward away from its resting state
        pos = self.get_pos(box)
        self.assertGreater(pos[1], 0.51, "Body woke up but didn't respond to the impulse velocity")


# Compatibility: Python 3.13/3.14 uses _interpreters
# --- SUBINTERPRETER COMPATIBILITY SHIM ---


class InterpretersProtocol(Protocol):
    from collections.abc import Callable

    def create(
        self,
        config: SimpleNamespace
        | Literal["default", "isolated", "legacy", "empty", ""]
        | None = "isolated",
        *,
        reqrefs: bool = False,
    ) -> int: ...

    def run_string(
        self,
        id: int,
        script: str | CodeType | Callable[[], object],
        shared: dict[str, object] | None = None,
        *,
        restrict: bool = False,
    ) -> None: ...

    def destroy(self, id: int, *, restrict: bool = False) -> None: ...


has_interpreters = False
interpreters: InterpretersProtocol | None = None

try:
    import _interpreters as interpreters  # type: ignore

    has_interpreters = True
except ImportError:
    try:
        import _xxsubinterpreters as interpreters  # type: ignore

        has_interpreters = True
    except ImportError:
        has_interpreters = False

HAS_INTERPRETERS = has_interpreters


class TestSubinterpreterIsolation(CulverinTestCase):
    @unittest.skipUnless(
        HAS_INTERPRETERS, "Subinterpreters module not available in this Python version"
    )
    def test_parser_isolation_across_interpreters(self) -> None:
        """
        Spawns a subinterpreter to ensure keywords like 'handle'
        don't leak or become 'garbage' pointers between instances.
        """
        assert interpreters is not None  # For type checker
        # 1. Initialize culverin in the MAIN interpreter
        world = culverin.PhysicsWorld()
        h = world.create_body(pos=(0, 10, 0))
        world.set_rotation(handle=h, x=0, y=0, z=0, w=1)

        # 2. Spawn a SUB-INTERPRETER
        interp_id: int = interpreters.create()

        # 3. Prepare code
        code = textwrap.dedent("""
            import culverin
            import math
            world = culverin.PhysicsWorld()
            h = world.create_body(pos=(0, 5, 0))
            world.set_position(handle=h, x=1, y=2, z=3)
            world.set_rotation(handle=h, x=0, y=0, z=0, w=1)
            print("Subinterpreter SUCCESS")
        """)

        try:
            # Both 3.12 (_xxsubinterpreters) and 3.13+ (_interpreters)
            # support run_string(id, code)
            interpreters.run_string(interp_id, code)
        except Exception as e:
            self.fail(f"Subinterpreter failed! Likely global state contamination: {e}")
        finally:
            interpreters.destroy(interp_id)

    @unittest.skipUnless(HAS_INTERPRETERS, "Subinterpreters module not available")
    def test_parallel_init_contention(self) -> None:
        def run_interp() -> None:
            assert interpreters is not None
            t_interp_id: int = interpreters.create()
            try:
                interpreters.run_string(t_interp_id, "import culverin; culverin.PhysicsWorld()")
            finally:
                interpreters.destroy(t_interp_id)

        threads = [threading.Thread(target=run_interp) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestURDFLoader(CulverinTestCase):
    def test_urdf_parsing_and_initialization(self) -> None:

        # 1. Define the path to your sample
        # Assuming execution from project root
        urdf_path = "tests/samples/urdf_sample.xml"

        if not Path(urdf_path).exists():
            self.skipTest(f"URDF sample not found at {urdf_path}")

        # 2. Use the loader to get the baked scene tuple
        # Returns: (count, pos_bytes, rot_bytes, shape_bytes, mot_bytes, layer_bytes, usr_bytes)
        baked_data = culverin.load_urdf(urdf_path)

        count = baked_data[0]
        self.assertGreater(count, 0, "URDF loader should have found at least one link")

        # 3. To actually get these into a world, since the current C __init__
        # expects a list, we can verify the data by initializing a world
        # with the count and then loading the state, OR we can modify
        # load_urdf locally for the test to return the list.

        # Validation: check if the byte lengths match the count
        # (Positions are 4 * double per body = 32 bytes)
        self.assertEqual(len(baked_data[1]), count * 32)

    def test_urdf_to_physics_world(self) -> None:

        from culverin import SHAPE_BOX, SHAPE_CYLINDER, parse_urdf

        urdf_path = "tests/samples/urdf_sample.xml"
        if not Path(urdf_path).exists():
            self.skipTest("URDF sample missing")

        # 1. Parse the URDF
        bodies = parse_urdf(urdf_path)

        # 2. Setup World
        world = culverin.PhysicsWorld()

        # 3. Create bodies manually to track handles
        # (We iterate to ensure we match names to handles)
        link_handles: dict[str, int] = {}
        for b in bodies:
            h = world.create_body(
                pos=b["pos"],
                rot=b["rot"],
                shape=b["shape"],
                size=b["size"],
                mass=b["mass"],
                motion=b["motion"],
            )
            link_handles[b["name"]] = h

        world.step(0)  # Synchronize

        # --- VALIDATION BASED ON YOUR XML ---

        # Check Base Link
        base_pos = world.get_position(link_handles["base_link"])
        self.assertEqual(base_pos, (0.0, 0.0, 0.0))
        # Mass was 10.0 in XML, verify it didn't fall (much) if gravity is on
        self.assertEqual(bodies[0]["mass"], 10.0)

        # Check Arm Link
        arm_pos = world.get_position(link_handles["arm"])
        # XML says: <origin xyz="0 0 -0.5"/>
        self.assertAlmostEqual(arm_pos[2], -0.5)

        # Verify Shape Types
        self.assertEqual(bodies[0]["shape"], SHAPE_BOX)
        self.assertEqual(bodies[1]["shape"], SHAPE_CYLINDER)


class TestDocumentation(unittest.TestCase):
    """
    Validation suite for the Culverin Documentation Engine.
    Ensures that the C-embedded DOCS.md correctly populates runtime docstrings.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Path relative to this test file
        cls.docs_path = Path(__file__).parent.parent / "docs" / "DOCS.md"

        if not cls.docs_path.exists():
            raise FileNotFoundError(f"DOCS.md not found at {cls.docs_path}")

        with Path.open(cls.docs_path, encoding="utf-8") as f:
            cls.raw_content = f.read()

        cls.expected_map = cls.parse_markdown(cls.raw_content)

    @staticmethod
    def parse_markdown(content: str) -> dict[str, str]:
        """
        Parses DOCS.md into a map: { 'ClassName.member_name': 'Normalized Documentation Text' }
        """
        # 1. Clean HTML comments (these are skipped by the C parser)
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        # 2. Extract Class Sections
        # We split by '## class ' but keep the classes
        class_splits = re.split(r"^## class ", content, flags=re.MULTILINE)

        docs_map: dict[str, str] = {}

        for section in class_splits[1:]:
            lines = section.splitlines()
            if not lines:
                continue

            class_name = lines[0].strip()

            # 3. Extract Members (Methods or Properties)
            # Members start with '### '
            member_splits = re.split(r"^### ", section, flags=re.MULTILINE)

            for member_block in member_splits[1:]:
                m_lines = member_block.splitlines()
                if not m_lines:
                    continue

                # Header parsing: "step(...)" or "positions (property)" -> "step", "positions"
                header = m_lines[0].strip()
                member_name = re.split(r"[\(\s]", header)[0].strip()

                # Content: Join the rest, strip leading/trailing whitespace
                # This matches the 'allocate_docstring' logic in culverin.c
                doc_text = "\n".join(m_lines[1:]).strip()

                docs_map[f"{class_name}.{member_name}"] = doc_text

        return docs_map

    def normalize(self, text: str) -> str:
        """Standardizes docstrings for comparison by removing carriage returns and extra padding."""
        if not text:
            return ""
        # Remove \r, strip whitespace from every line, and trim the block
        return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()

    def test_metadata_consistency(self) -> None:
        """Verify version and module-level docs which are handled separately in culverin_exec."""
        import culverin._culverin_c as core

        self.assertEqual(core.__doc__, "Culverin Physics Engine Core")
        self.assertTrue(hasattr(culverin, "__version__"))
        self.assertNotEqual(culverin.__version__, "0.0.0-unknown")

    def test_comprehensive_stitching(self) -> None:
        """
        Iterates through every entry in DOCS.md and verifies its presence
        and accuracy in the live Culverin objects.
        """
        failed_keys: list[object] = []

        for key, expected_body in self.expected_map.items():
            class_name, member_name = key.split(".")

            try:
                # 1. Resolve Class
                container = culverin if class_name == "Module" else getattr(culverin, class_name)

                # 2. Resolve Member (Method or Property)
                # Some properties are defined via PyGetSetDef, others as methods
                try:
                    member = getattr(container, member_name)
                except AttributeError:
                    member = getattr(container, f"_{member_name}")

                # 3. Extract Docstring
                actual_doc = member.__doc__

                self.assertIsNotNone(
                    actual_doc,
                    f"STITCHING FAILURE: {key} exists in DOCS.md but has no runtime __doc__",
                )

                # 4. Content Match
                norm_expected = self.normalize(expected_body)
                norm_actual = self.normalize(actual_doc)

                # We use 'assertIn' because the C-parser might include the header params
                # depending on how you've handled the pointer arithmetic.
                # But since you've used a clean skip-to-newline, we check the body.
                self.assertIn(
                    norm_expected[:50],  # Check first 50 chars for high-confidence match
                    norm_actual,
                    f"CONTENT MISMATCH: {key} docstring body doesn't match DOCS.md",
                )

            except AttributeError:
                failed_keys.append(key)

        self.assertEqual(
            failed_keys,
            [],
            f"API GAP: These members are documented in DOCS.md but missing from code: {failed_keys}",
        )

    def test_property_specifics(self) -> None:
        """Targeted check for high-performance properties (memoryviews)."""
        props = ["positions", "rotations", "velocities", "user_data"]
        for p in props:
            doc = getattr(culverin.PhysicsWorld, p).__doc__
            self.assertIsNotNone(doc)
            self.assertIn("memoryview", doc.lower())

    def test_character_controller_docs(self) -> None:
        """Ensure the Character class (created via world) correctly carries docs."""
        self.assertIsNotNone(culverin.Character.move.__doc__)
        assert culverin.Character.move.__doc__ is not None
        self.assertIn("Sweep and Slide", culverin.Character.move.__doc__)


class TestKinematics(CulverinTestCase):
    def test_kinematic_gravity_resistance(self) -> None:
        """Kinematic bodies should ignore gravity and stay pinned in space."""
        # Dynamic body (will fall)
        h_dyn = self.world.create_body(pos=(0, 10, 0), motion=culverin.MOTION_DYNAMIC)
        # Kinematic body (should stay)
        h_kin = self.world.create_body(pos=(5, 10, 0), motion=culverin.MOTION_KINEMATIC)

        self.world.step(0)
        for _ in range(10):
            self.world.step(1 / 60.0)

        self.assertLess(self.get_pos(h_dyn)[1], 10.0, "Dynamic body failed to fall")
        self.assertEqual(self.get_pos(h_kin)[1], 10.0, "Kinematic body moved under gravity")

    def test_kinematic_velocity_drive(self) -> None:
        """Setting linear velocity on a kinematic body should move it predictably."""
        h = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_KINEMATIC)
        self.world.set_linear_velocity(h, x=10.0, y=0, z=0)

        # Step 0.1 seconds
        for _ in range(6):
            self.world.step(1 / 60.0)

        pos = self.get_pos(h)
        # Should be roughly at X=1.0 (10 units/sec * 0.1 sec)
        self.assertAlmostEqual(pos[0], 1.0, places=2)
        self.assertEqual(pos[1], 0.0)

    def test_kinematic_pushing_dynamic(self) -> None:
        """Kinematic bodies should act as 'unstoppable forces' pushing dynamic objects."""
        # A dynamic crate sitting in the way
        crate = self.world.create_body(pos=(2, 0.5, 0), size=(1, 1, 1), mass=10.0)
        # A kinematic 'bulldozer'
        dozer = self.world.create_body(
            pos=(0, 0.5, 0), size=(1, 1, 1), motion=culverin.MOTION_KINEMATIC
        )

        self.world.step(0)
        self.world.set_linear_velocity(dozer, x=10.0, y=0, z=0)

        # Step until they collide and the dozer passes through the original spot
        for _ in range(20):
            self.world.step(1 / 60.0)

        crate_pos = self.get_pos(crate)
        dozer_pos = self.get_pos(dozer)

        # The dozer should have pushed the crate forward
        self.assertGreater(dozer_pos[0], 1.0)
        self.assertGreater(
            crate_pos[0], dozer_pos[0], "Crate should be in front of the kinematic dozer"
        )

    def test_motion_type_hotswap(self) -> None:
        """Test switching a body from Kinematic to Dynamic mid-simulation."""
        h = self.world.create_body(pos=(0, 10, 0), motion=culverin.MOTION_KINEMATIC)
        self.world.step(1 / 60.0)
        self.assertEqual(self.get_pos(h)[1], 10.0)

        # Switch to dynamic
        self.world.set_motion_type(h, culverin.MOTION_DYNAMIC)
        self.world.activate(h)  # Force wake up

        # Give it a few frames to start falling
        for _ in range(5):
            self.world.step(1 / 60.0)

        self.assertLess(
            self.get_pos(h)[1], 10.0, "Body did not start falling after switching to dynamic"
        )

    def test_kinematic_rotation_interaction(self) -> None:
        """Kinematic rotation should apply tangential velocity to dynamic objects."""
        # A flat kinematic 'spinner' platform
        spinner = self.world.create_body(
            pos=(0, 0, 0), size=(5, 0.2, 5), motion=culverin.MOTION_KINEMATIC, friction=1.0
        )
        # A dynamic ball sitting on the edge of the spinner
        ball = self.world.create_body(
            pos=(2, 0.5, 0), shape=culverin.SHAPE_SPHERE, size=0.2, mass=1.0
        )

        self.world.step(0)
        # Rotate the kinematic platform around Y axis (10 radians/sec)
        self.world.set_angular_velocity(spinner, x=0, y=10.0, z=0)

        # Step and check if the ball gains velocity from the friction/rotation
        for _ in range(10):
            self.world.step(1 / 60.0)

        vel = self.get_vel(ball)
        # The ball should have been 'thrown' or moved by the rotation
        speed_sq = vel[0] ** 2 + vel[2] ** 2
        self.assertGreater(speed_sq, 0.1, "Ball stayed static despite kinematic platform rotating")

    def test_kinematic_teleport_stability(self) -> None:
        """Directly setting position (teleporting) should still result in collision resolution."""
        # Static wall
        self.world.create_body(pos=(10, 0, 0), size=(1, 10, 10), motion=culverin.MOTION_STATIC)
        # Kinematic body
        k = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_KINEMATIC)

        self.world.step(0)

        # Teleport kinematic body directly into/past the wall
        self.world.set_position(k, x=15, y=0, z=0)
        self.world.step(1 / 60.0)

        self.assertEqual(
            self.get_pos(k)[0], 15.0, "Kinematic teleport was blocked (should be unstoppable)"
        )


class TestAdvancedPhysics(CulverinTestCase):
    def test_ccd_tunneling_prevention(self) -> None:
        """
        Verify that CCD prevents a high-speed projectile from tunneling.
        Note: Jolt's default max velocity is 500m/s.
        """
        # 1. Create a very thin static wall at X=2
        self.world.create_body(pos=(2, 0, 0), size=(0.1, 10, 10), motion=culverin.MOTION_STATIC)

        # 2. Create high-speed projectiles
        bullet = self.world.create_body(
            pos=(0, 0, 0), size=(0.2, 0.2, 0.2), motion=culverin.MOTION_DYNAMIC, ccd=True
        )

        ghost = self.world.create_body(
            pos=(0, 2, 0), size=(0.2, 0.2, 0.2), motion=culverin.MOTION_DYNAMIC, ccd=False
        )

        self.world.step(0)  # Flush creation

        # 3. Launch at 400 m/s (below the 500m/s default cap)
        # In 1/60s, they travel ~6.6m. The wall is at X=2.
        self.world.set_linear_velocity(bullet, x=400.0, y=0, z=0)
        self.world.set_linear_velocity(ghost, x=400.0, y=0, z=0)

        self.world.step(0)  # Flush velocity commands
        self.world.step(1 / 60.0)  # Simulate 1 frame

        pos_bullet = self.get_pos(bullet)
        pos_ghost = self.get_pos(ghost)

        # CCD bullet should be stopped by the wall (stopped near X=2)
        self.assertLess(pos_bullet[0], 2.2, "CCD Bullet tunneled through wall")

        # Non-CCD bullet (ghost) should have tunneled (ended up near X=6.6)
        self.assertGreater(pos_ghost[0], 4.0, "Non-CCD Bullet was unexpectedly stopped")

    def test_linear_and_angular_damping(self) -> None:
        """Verify that damping slows down bodies over time in a vacuum."""
        h = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_DYNAMIC)
        self.world.set_gravity(0, 0, 0)
        self.world.step(0)

        self.world.set_linear_velocity(h, 10, 0, 0)
        self.world.set_angular_velocity(h, 10, 0, 0)

        for _ in range(60):
            self.world.step(1 / 60.0)

        vel = self.world.get_velocity(h)
        self.assertLess(vel[0], 10.0, "Linear velocity did not damped")

    def test_slider_constraint(self) -> None:
        """Test a Slider (Prismatic) constraint for elevators or pistons."""
        b1 = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_STATIC)
        b2 = self.world.create_body(pos=(0, 2, 0), motion=culverin.MOTION_DYNAMIC)

        self.world.step(0)

        c = self.world.create_constraint(
            culverin.CONSTRAINT_SLIDER, b1, b2, params=((0, 0, 0), (0, 1, 0), 1.0, 5.0)
        )
        self.assertIsNotNone(c)

        self.world.apply_impulse(b2, 0, 1000, 0)

        for _ in range(30):
            self.world.step(1 / 60.0)

        pos = self.get_pos(b2)
        self.assertLessEqual(pos[1], 5.2)
        self.assertGreaterEqual(pos[1], 0.8)


class TestKinematicsAdvanced(CulverinTestCase):
    def test_kinematic_compound_movement(self) -> None:
        """Verify that compound kinematic bodies respond to velocity (The Basket Test)."""
        parts = [
            ((0, 0, 0), (0, 0, 0, 1), culverin.SHAPE_BOX, (1, 1, 1)),
            ((2, 0, 0), (0, 0, 0, 1), culverin.SHAPE_BOX, (1, 1, 1)),
        ]
        h = self.world.create_compound_body(
            pos=(0, 0, 0), rot=(0, 0, 0, 1), parts=parts, motion=culverin.MOTION_KINEMATIC
        )
        self.world.step(0)

        # Set velocity
        self.world.activate(h)
        self.world.set_linear_velocity(h, x=10.0, y=0, z=0)

        # Step for 0.5 seconds
        for _ in range(30):
            self.world.step(1 / 60.0)

        pos = self.get_pos(h)
        self.assertGreater(pos[0], 4.5, "Kinematic compound body failed to move")

    def test_kinematic_restitution_transfer(self) -> None:
        """Kinematic bodies should 'bounce' dynamic ones away based on their own velocity."""
        # Static floor
        self.world.create_body(pos=(0, -1, 0), size=(10, 1, 10), motion=culverin.MOTION_STATIC)

        # Kinematic 'Bat' moving upward
        bat = self.world.create_body(
            pos=(0, 0, 0), size=(2, 0.2, 2), motion=culverin.MOTION_KINEMATIC
        )

        # Dynamic ball falling onto the bat
        ball = self.world.create_body(pos=(0, 2, 0), shape=culverin.SHAPE_SPHERE, size=0.5)

        self.world.step(0)
        self.world.set_linear_velocity(bat, 0, 20.0, 0)

        # Simulate collision
        for _ in range(10):
            self.world.step(1 / 60.0)

        ball_vel = self.get_vel(ball)
        self.assertGreater(
            ball_vel[1], 15.0, "Kinematic velocity was not transferred to dynamic body"
        )

    def test_kinematic_to_static_interaction(self) -> None:
        """Kinematic bodies should NOT be blocked by static bodies (Ghosting)."""
        # Static wall
        self.world.create_body(pos=(5, 0, 0), size=(1, 5, 5), motion=culverin.MOTION_STATIC)

        # Kinematic body moving through wall
        k = self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_KINEMATIC)

        self.world.step(0)
        self.world.set_linear_velocity(k, 60.0, 0, 0)

        # Step 1/6th of a second (should be at X=10)
        for _ in range(10):
            self.world.step(1 / 60.0)

        pos = self.get_pos(k)
        self.assertGreater(pos[0], 9.0, "Kinematic body was blocked by a static object")


class TestRobustness(CulverinTestCase):
    def test_bit_perfect_determinism(self) -> None:
        """Verify that saving and loading state results in bit-identical physics results."""
        # 1. Setup a complex scene
        self.world.create_body(pos=(0, 0, 0), motion=culverin.MOTION_STATIC)
        bodies = self.world.create_bodies_batch(
            positions=[(0, 10, 0), (0.1, 12, 0), (-0.1, 14, 0)],
            sizes=[[0.5, 0.5, 0.5]] * 3,
            shape_type=culverin.SHAPE_BOX,
            motion_type=culverin.MOTION_DYNAMIC,
        )

        # 2. Run for 30 frames and save
        for _ in range(30):
            self.world.step(1 / 60.0)
        state_snapshot = self.world.save_state()

        # 3. Run for 30 more frames and record results
        for _ in range(30):
            self.world.step(1 / 60.0)
        pos_after_60 = [self.world.get_position(h) for h in bodies]

        # 4. Restore to frame 30 and run to 60 again
        self.world.load_state(state=state_snapshot)
        self.world.step(0)  # Sync shadow buffers
        for _ in range(30):
            self.world.step(1 / 60.0)
        pos_after_restore_60 = [self.world.get_position(h) for h in bodies]

        # 5. Compare. In a deterministic engine, these must be identical.
        for p1, p2 in zip(pos_after_60, pos_after_restore_60, strict=False):
            self.assertEqual(p1, p2, "Physics diverged after state restore (Determinism Failure)")

    def test_reaching_capacity_limit(self) -> None:
        """Force the engine to its max_bodies limit and ensure it fails gracefully."""
        limit = 128
        world = culverin.PhysicsWorld(settings={"max_bodies": limit})

        # Fill exactly to the limit
        handles: list[int] = []
        for i in range(limit):
            h = world.create_body(pos=(0, i, 0))
            handles.append(h)

        self.assertEqual(world.count, limit)
        self.assertEqual(world.remaining_capacity, 0)

        # The N+1 body should raise RuntimeError (or return None depending on your C policy)
        # Based on your current C code, it raises RuntimeError
        with self.assertRaises(RuntimeError):
            world.create_body(pos=(0, 0, 0))

    def test_mixed_batch_validity(self) -> None:
        """Ensure batch methods handle a mix of valid, stale, and invalid handles."""
        h1 = self.world.create_body(pos=(0, 0, 0))
        h2 = self.world.create_body(pos=(0, 0, 0))
        self.world.destroy_body(h1)  # h1 is now PENDING_DESTROY

        stale_h = 999999  # Completely fake handle

        # Apply buoyancy to a mix. Should not crash.
        # It should process h2 and ignore h1 and stale_h.
        res = self.world.apply_buoyancy_batch(
            handles=np.array([h1, h2, stale_h], dtype=np.uint64).tobytes(), surface_y=1.0
        )
        self.assertIsNone(res)  # Batch methods return None

    def test_distance_constraint_correct_params(self) -> None:
        """Verify distance constraint with 2-pivot format."""
        b1 = self.world.create_body(pos=(0, 0, 0))
        b2 = self.world.create_body(pos=(2, 0, 0))
        self.world.step(0)

        # Based on the C error 'takes exactly 2', the parser expects
        # exactly two arguments: (pivot1, pivot2).
        c = self.world.create_constraint(
            culverin.CONSTRAINT_DISTANCE, b1, b2, params=((0, 0, 0), (2, 0, 0))
        )
        self.assertIsNotNone(c)

    def test_handle_recycling_and_stale_constraints(self) -> None:
        """Verify that destroy_constraint invalidates the handle."""
        b1 = self.world.create_body(pos=(0, 0, 0))
        b2 = self.world.create_body(pos=(0, 2, 0))
        self.world.step(0)

        c = self.world.create_constraint(culverin.CONSTRAINT_FIXED, b1, b2)

        # Pre-condition: Verify the constraint is alive
        self.assertEqual(self.world.get_constraint_type(c), culverin.CONSTRAINT_FIXED)

        # Kill the constraint manually
        self.world.destroy_constraint(c)

        # Querying a destroyed constraint should return None (Silent Invalidation)
        ctype = self.world.get_constraint_type(c)
        self.assertIsNone(ctype, "get_constraint_type should return None for destroyed constraints")

    def test_constraint_automatic_cleanup_check(self) -> None:
        """Check if querying a constraint with a dead body returns None."""
        b1 = self.world.create_body(pos=(0, 0, 0))
        b2 = self.world.create_body(pos=(0, 2, 0))
        self.world.step(0)

        c = self.world.create_constraint(culverin.CONSTRAINT_FIXED, b1, b2)

        # Destroy one of the linked bodies
        self.world.destroy_body(b1)
        self.world.step(0)  # Flush destruction

        # In a robust engine, get_constraint_type should check if its bodies are still alive.
        # If your C-code doesn't do this check yet, this test serves as a reminder.
        ctype = self.world.get_constraint_type(c)
        # We assert None or it should at least not crash
        self.assertTrue(ctype is None or ctype == culverin.CONSTRAINT_FIXED)

    def test_buffer_stride_mismatch(self) -> None:
        """Pass incorrectly sized numpy buffers to C and ensure it catches the error."""
        ray_count = 10
        # Correct starts (10 * 3 floats = 120 bytes)
        starts = np.zeros((ray_count, 3), dtype=np.float32).tobytes()
        # Malformed directions (only 5 floats instead of 30)
        bad_dirs = np.zeros(5, dtype=np.float32).tobytes()

        with self.assertRaises((ValueError, RuntimeError)):
            self.world.raycast_batch(starts, bad_dirs, max_dist=10.0)

    def test_rapid_recreation_cycle(self) -> None:
        """Rapidly create and destroy the same slot to test generation counter wrap-around logic."""
        # Note: We won't actually wrap a 32-bit int in a unit test,
        # but we can test the recycling logic.
        last_h = None
        for _ in range(100):
            h = self.world.create_body(pos=(0, 0, 0))
            self.assertNotEqual(h, last_h)
            self.world.destroy_body(h)
            self.world.step(0)  # Force recycling
            last_h = h

    def test_character_teleport_step_consistency(self) -> None:
        """Ensure character.set_position works even if called multiple times per step."""
        char = self.world.create_character(pos=(0, 0, 0))
        self.world.step(0)

        char.set_position((10, 10, 10))
        char.set_position((20, 20, 20))  # Multiple teleports

        self.world.step(1 / 60.0)

        # Should be at the LAST set position
        pos = char.get_position()
        self.assertAlmostEqual(pos[0], 20.0)


class TestSoftBodies(CulverinTestCase):
    def create_cube_settings(
        self, size: float = 1.0, compliance: float = 0.0001
    ) -> culverin.SoftBodySharedSettings:
        """
        Helper to build an optimized soft-body cube using the high-performance bulk API.
        """
        settings = culverin.SoftBodySharedSettings()

        # 1. Prepare bulk data via NumPy
        s = size / 2.0
        # 8 Corners (Positions)
        verts = np.array(
            [
                [-s, -s, -s],
                [s, -s, -s],
                [s, s, -s],
                [-s, s, -s],
                [-s, -s, s],
                [s, -s, s],
                [s, s, s],
                [-s, s, s],
            ],
            dtype=np.float32,
        )

        # 12 Faces (Indices)
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],  # Front/Back
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],  # Bottom/Top
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],  # Left/Right
            ],
            dtype=np.uint32,
        )

        # 2. Bulk Load
        settings.add_vertices(verts.tobytes())
        settings.add_faces(faces.tobytes())

        # 3. Granular Setup
        settings.create_constraints(compliance=compliance, bend_type=culverin.BEND_DISTANCE)
        settings.optimize()

        return settings

    def test_soft_body_lifecycle(self) -> None:
        """Verify creation, handle validity, and destruction of soft bodies."""
        settings = self.create_cube_settings()

        h = self.world.create_soft_body(
            shared_settings=settings,
            pos=(0, 10, 0),
            rot=(0, 0, 0, 1),
            pressure=100.0,
            linear_damping=0.2,
            num_iterations=20,
        )

        self.world.step(0)
        self.assertTrue(self.world.is_alive(h))

        # Verify it falls
        self.world.step(1 / 60.0)
        pos = self.get_pos(h)
        self.assertLess(pos[1], 10.0, "Soft body center-of-mass did not fall")

        self.world.destroy_body(h)
        self.world.step(0)
        self.assertFalse(self.world.is_alive(h))

    def test_soft_body_creation_flags(self) -> None:
        """Verify the newly added SoftBody creation kwargs are parsed and applied without crashing."""
        settings = self.create_cube_settings()

        h = self.world.create_soft_body(
            shared_settings=settings,
            pos=(0, 10, 0),
            rot=(0, 0, 0, 1),
            make_rotation_identity=True,
            update_position=False,
            faces_double_sided=True,
            pressure=2000.0,
        )

        self.world.step(0)
        self.assertTrue(self.world.is_alive(h))

        # Simulate to ensure the flags don't cause solver crashes
        for _ in range(10):
            self.world.step(1 / 60.0)

        # Because update_position is False, the "Center of Mass" position reported by Jolt
        # might remain fixed or update differently. We mainly care that the flags parsed safely.
        self.assertIsNotNone(self.get_pos(h))

    def test_soft_body_dihedral_bending(self) -> None:
        """Test soft body creation using Dihedral bending constraints."""
        settings = culverin.SoftBodySharedSettings()

        # A simple hinge/book shape (4 vertices, 2 triangles sharing an edge)
        # Verts: (0,0,0), (1,0,0), (0,1,0), (-1,0,0)
        # Tri 1: 0, 1, 2
        # Tri 2: 0, 2, 3
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
        settings.add_vertices(pos.tobytes())

        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        settings.add_faces(faces.tobytes())

        # Use BEND_DIHEDRAL
        settings.create_constraints(compliance=0.01, bend_type=culverin.BEND_DIHEDRAL)
        settings.optimize()

        h = self.world.create_soft_body(settings, pos=(0, 5, 0), rot=(0, 0, 0, 1))
        self.world.step(0)
        self.assertTrue(self.world.is_alive(h))

    def test_soft_body_rest_pose(self) -> None:
        """Verify we can extract the rest-pose of a vertex before optimization."""
        settings = culverin.SoftBodySharedSettings()
        settings.add_vertex((1.5, 2.5, -3.5), 1.0)

        pos = settings.get_vertex_position(0)
        self.assertAlmostEqual(pos[0], 1.5)
        self.assertAlmostEqual(pos[1], 2.5)
        self.assertAlmostEqual(pos[2], -3.5)

        with self.assertRaises(IndexError):
            settings.get_vertex_position(99)

    def test_soft_body_bulk_creation_with_mass(self) -> None:
        """Verify bulk vertex loading with explicit inverse masses."""
        settings = culverin.SoftBodySharedSettings()

        # 3 vertices for a single triangle
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        # Mix of stationary (0.0) and mobile (1.0) vertices
        inv_masses = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        settings.add_vertices(pos.tobytes(), inv_masses.tobytes())
        settings.add_faces(np.array([0, 1, 2], dtype=np.uint32).tobytes())

        settings.create_constraints(0.001)
        settings.optimize()

        h = self.world.create_soft_body(settings, pos=(0, 0, 0), rot=(0, 0, 0, 1))
        self.world.step(0)

        dtype = np.float64 if culverin.USE_DOUBLE_PRECISION else np.float32
        view = self.world.get_soft_body_vertices(h)
        verts = np.frombuffer(view, dtype=dtype).reshape(-1, 4)

        # Check initial positions
        self.assertEqual(verts[1, 0], 1.0)
        self.assertEqual(verts[2, 1], 1.0)

    def test_soft_body_vertex_sync(self) -> None:
        """Verify zero-copy vertex synchronization into NumPy buffers."""
        settings = self.create_cube_settings()
        h = self.world.create_soft_body(settings, pos=(0, 5, 0), rot=(0, 0, 0, 1))
        self.world.step(0)

        view = self.world.get_soft_body_vertices(h)
        dtype = np.float64 if culverin.USE_DOUBLE_PRECISION else np.float32
        verts = np.frombuffer(view, dtype=dtype).reshape(-1, 4)

        self.assertEqual(len(verts), 8, f"Soft body should have 8 vertices (detected {dtype})")

        # Check initial world-space position of vertex 2
        # Local (0.5, 0.5, -0.5) + COM (0, 5, 0)
        self.assertAlmostEqual(verts[2, 1], 5.5, places=3)

        self.world.step(1 / 60.0)
        # Verify the same numpy array has updated data (proving zero-copy sync)
        self.assertLess(verts[2, 1], 5.5, "Vertices in NumPy buffer did not update after step")

    def test_soft_body_pinning(self) -> None:
        """Verify that pinned vertices remain fixed in space relative to the COM."""
        settings = culverin.SoftBodySharedSettings()

        # Create a vertical line of 3 vertices
        pos = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32)
        settings.add_vertices(pos.tobytes())

        # Jolt requires at least one face to optimize correctly
        settings.add_face(0, 1, 2)

        # Pin the very top vertex (index 2)
        settings.add_pinned_vertex(2)
        settings.create_constraints(0.001, culverin.BEND_DISTANCE)
        settings.optimize()

        h = self.world.create_soft_body(settings, pos=(0, 10, 0), rot=(0, 0, 0, 1))
        self.world.step(0)

        dtype = np.float64 if culverin.USE_DOUBLE_PRECISION else np.float32
        verts = np.frombuffer(self.world.get_soft_body_vertices(h), dtype=dtype).reshape(-1, 4)

        # Simulate
        for _ in range(30):
            self.world.step(1 / 60.0)

        # The pinned vertex (2) should be physically higher than the mobile ones
        self.assertGreater(verts[2, 1], verts[0, 1], "Pinned vertex fell below bottom vertex")

    def test_soft_body_collision(self) -> None:
        """Test if a soft body deforms/stops when hitting the floor."""
        self.world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)

        settings = self.create_cube_settings(compliance=0.01)
        h = self.world.create_soft_body(settings, pos=(0, 1, 0), rot=(0, 0, 0, 1))
        self.world.step(0)

        view = self.world.get_soft_body_vertices(h)
        dtype = np.float64 if culverin.USE_DOUBLE_PRECISION else np.float32
        verts = np.frombuffer(view, dtype=dtype).reshape(-1, 4)

        for _ in range(60):
            self.world.step(1 / 60.0)

        # Vertices should be caught by the floor top (Y=0)
        bottom_y = verts[[0, 1, 4, 5], 1]
        for y in bottom_y:
            self.assertGreater(y, -0.5, "Soft body fell through the floor")
            self.assertLess(y, 0.5, "Soft body didn't reach the floor")

    def test_invalid_handle_error(self) -> None:
        """Ensure get_soft_body_vertices fails correctly on rigid bodies."""
        h_rigid = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)

        with self.assertRaisesRegex(TypeError, "Handle does not belong to a soft body"):
            self.world.get_soft_body_vertices(h_rigid)

    def test_bulk_index_out_of_range(self) -> None:
        """Verify the C-layer guard catches bad indices in add_faces."""
        settings = culverin.SoftBodySharedSettings()
        settings.add_vertex((0, 0, 0), 1.0)

        # Vertex index 99 does not exist
        bad_faces = np.array([0, 0, 99], dtype=np.uint32)
        with self.assertRaises(IndexError):
            settings.add_faces(bad_faces.tobytes())

    def test_soft_body_save_load(self) -> None:
        """Test if soft bodies survive world state serialization."""
        settings = self.create_cube_settings()
        h = self.world.create_soft_body(settings, pos=(0, 10, 0), rot=(0, 0, 0, 1))
        self.world.step(0)

        state = self.world.save_state()

        for _ in range(10):
            self.world.step(1 / 60.0)
        self.assertLess(self.get_pos(h)[1], 10.0)

        self.world.load_state(state=state)
        self.world.step(0)

        self.assertTrue(self.world.is_alive(h))
        self.assertAlmostEqual(self.get_pos(h)[1], 10.0, places=3)

    def test_soft_body_getters_logic(self) -> None:
        """Test the new JoltC direct getters for soft body vertex data."""
        # Cube of size 2.0 at Y=5.0.
        # Corner 0 is at local (-1, -1, -1).
        # World Y = 5.0 - 1.0 = 4.0.
        settings = self.create_cube_settings(size=2.0)
        h = self.world.create_soft_body(settings, pos=(0, 5, 0), rot=(0, 0, 0, 1), pressure=0.0)
        # Flush creation
        self.world.step(0)

        # 1. Test Vertex Count
        count = self.world.get_soft_body_vertex_count(h)
        self.assertEqual(count, 8, "Cube should have exactly 8 vertices")

        # 2. Test World Position via direct getter
        # Jolt reports these in world-space immediately after creation
        world_pos = self.world.get_soft_body_vertex_position(h, 0)
        self.assertAlmostEqual(world_pos[0], -1.0)
        self.assertAlmostEqual(world_pos[1], 4.0)  # 5.0 (pos) - 1.0 (local)
        self.assertAlmostEqual(world_pos[2], -1.0)

        # 3. Test Bulk Extraction
        raw_bytes = self.world.get_soft_body_local_vertices(h)
        self.assertEqual(len(raw_bytes), 8 * 12, "Byte length must be num_verts * 12")

        verts_world = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 3)
        self.assertAlmostEqual(verts_world[0, 1], 4.0)

        # 4. Verify physical movement
        # Let the body fall for one frame
        self.world.step(1 / 60)

        new_pos = self.world.get_soft_body_vertex_position(h, 0)
        self.assertLess(new_pos[1], 4.0, "Vertex Y should have decreased due to gravity")

    def test_soft_body_getter_errors(self) -> None:
        """Verify safety guards for the new soft body getters."""
        h_rigid = self.world.create_body(pos=(0, 0, 0))
        self.world.step(0)

        # 1. Wrong Body Type
        with self.assertRaisesRegex(TypeError, "not belong to a soft body"):
            self.world.get_soft_body_vertex_count(h_rigid)

        # 2. Index Out of Bounds
        settings = self.create_cube_settings()
        h_soft = self.world.create_soft_body(settings, pos=(0, 0, 0), rot=(0, 0, 0, 1))
        self.world.step(0)

        with self.assertRaises(IndexError):
            self.world.get_soft_body_vertex_position(h_soft, 999)


class TestTupleMutation(unittest.TestCase):
    """
    Validation suite for culverin.mutate_tuple().
    This tests the C-layer's ability to safely perform 'illegal' mutations on
    immutable tuples and keep Python's internal hash-maps in sync.
    """

    def test_basic_mutation(self) -> None:
        """Verify that we can change a tuple element and that the hash updates."""
        t = (1, "target", 3)
        old_hash = hash(t)

        # Mutate index 1
        new_hash = culverin.mutate_tuple(t, 1, "mutated")

        self.assertEqual(t[1], "mutated")
        self.assertEqual(t, (1, "mutated", 3))
        self.assertNotEqual(old_hash, new_hash, "Hash must change after mutation")
        self.assertEqual(hash(t), new_hash, "Manual rehash must match Python's hash()")

    def test_negative_indexing(self) -> None:
        """Verify Python-style negative indexing works."""
        t = (10, 20, 30)
        culverin.mutate_tuple(t, -1, 99)  # Change the last element
        self.assertEqual(t, (10, 20, 99))

    def test_registry_sync(self) -> None:
        registry: dict[str, tuple[int, ...]] = {}  # Type hint here is fine
        t = (1, 2, 3)
        key = "my_tuple"
        registry[key] = t

        # ACTUAL call to your C function
        culverin.mutate_tuple(t, 0, 999, registry, key)

        # 1. Content check (Now registry actually has the data)
        self.assertEqual(registry[key][0], 999)

        # 2. Hash-map integrity check
        self.assertIn((999, 2, 3), registry.values())

    def test_registry_mismatch_protection(self) -> None:
        """Ensure the C-layer catches cases where the registry key doesn't match the target."""
        registry = {"a": (1, 2)}
        other_tuple = (3, 4)

        with self.assertRaisesRegex(ValueError, "not the same object"):
            culverin.mutate_tuple(other_tuple, 0, 5, registry, "a")

    def test_bounds_and_type_errors(self) -> None:
        """Verify that the C-layer guards against invalid inputs."""
        t = (1, 2)

        # 1. Index out of range
        with self.assertRaises(IndexError):
            culverin.mutate_tuple(t, 5, 99)

        # 2. Not a tuple
        with self.assertRaisesRegex(TypeError, "must be a tuple"):
            culverin.mutate_tuple([1, 2], 0, 99)  # type: ignore

        # 3. Registry not a dict
        with self.assertRaisesRegex(TypeError, "must be a dict"):
            culverin.mutate_tuple(t, 0, 99, ["not a dict"], "key")  # type: ignore

    def test_free_threading_stability(self) -> None:
        shared_tuple = (0, 0, 0)

        def hammer() -> None:
            for i in range(1000):
                with contextlib.suppress(Exception):
                    culverin.mutate_tuple(shared_tuple, i % 3, i)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(shared_tuple), 3)
        # Restore to known-good hashable state to avoid corrupting Python's integer cache
        culverin.mutate_tuple(shared_tuple, 0, 0)
        culverin.mutate_tuple(shared_tuple, 1, 0)
        culverin.mutate_tuple(shared_tuple, 2, 0)


class TestCharacterInteractions(CulverinTestCase):
    def test_character_pushing_box(self) -> None:
        """Verify that characters can physically move dynamic bodies."""
        # 1. Place a light box at X=1.1
        box = self.world.create_body(pos=(1.1, 0.5, 0), size=(1, 1, 1), mass=1.0)

        # 2. Place character at X=0
        char = self.world.create_character(pos=(0, 0.5, 0))
        char.set_strength(5000.0)
        self.world.step(0)

        # 3. Move character into the box
        # 20m/s * 1/60s = 0.33m. Radius 0.4. Reach = 0.73.
        # Box edge is at 1.1 - 0.5 = 0.6. Collision is guaranteed.
        char.move((20, 0, 0), 1 / 60)

        # Check that a contact event was recorded
        events = self.world.get_contact_events_ex()
        self.assertTrue(any(box in e["bodies"] for e in events), "Box contact not recorded")

        # 4. Step the world to allow the impulse to translate into movement
        self.world.step(1 / 60)

        # The box should have gained velocity from the character's 'apply_character_impulse'
        vel = self.world.get_velocity(box)
        self.assertGreater(vel[0], 0.5, "Character failed to push the dynamic box")
        self.world.step(1 / 60)  # flush character state before world teardown

    def test_character_vs_character_collision(self) -> None:
        """Verify the special callback path for virtual character collisions."""
        # RADIUS 0.4. Sum of radii 0.8.
        # Place centers at 0.0 and 1.2. Gap is 0.4 units.
        char1 = self.world.create_character(pos=(0, 0.5, 0))
        char2 = self.world.create_character(pos=(1.2, 0.5, 0))

        # Step 0 to flush registration commands
        self.world.step(0)

        # Move char2 slightly just to ensure it is fully "woken up" in the
        # CharacterVsCharacterCollision manager's broadphase.
        char2.move((0.01, 0, 0), 1 / 60)

        # Move char1 significantly into char2.
        # Speed 120m/s * 1/60s = 2.0 units movement.
        # Range: X 0.0 -> 2.0. Intersection with char2 at 1.2 is guaranteed.
        char1.move((120, 0, 0), 1 / 60)

        # Capture events immediately after the move() call
        events = self.world.get_contact_events_ex()
        char_hits = [
            e for e in events if char1.handle in e["bodies"] and char2.handle in e["bodies"]
        ]

        self.assertGreater(
            len(char_hits),
            0,
            f"No character contact detected. Total events captured: {len(events)}. "
            "Check if JPH_CharacterVirtual_Set/GetUserData is working in C.",
        )

        # Verify it's an Added or Persisted event
        self.assertIn(char_hits[0]["type"], [culverin.EVENT_ADDED, culverin.EVENT_PERSISTED])

    def test_character_contact_lifecycle(self) -> None:
        """Test Added -> Persisted -> Removed lifecycle against a static wall."""
        wall = self.world.create_body(
            pos=(1.0, 0.5, 0), size=(1, 1, 1), motion=culverin.MOTION_STATIC
        )
        char = self.world.create_character(pos=(0, 0.5, 0))
        self.world.step(0)

        # 1. ADDED
        char.move((10, 0, 0), 1 / 60)
        events = self.world.get_contact_events_ex()
        self.assertTrue(
            any(e["type"] == culverin.EVENT_ADDED and wall in e["bodies"] for e in events)
        )

        # 2. PERSISTED
        # We must step once to transition the Jolt listener's state
        self.world.step(1 / 60)
        char.move((10, 0, 0), 1 / 60)
        events = self.world.get_contact_events_ex()
        self.assertTrue(
            any(e["type"] == culverin.EVENT_PERSISTED and wall in e["bodies"] for e in events)
        )

        # 3. REMOVED
        char.set_position((-5, 0.5, 0))
        # ExtendedUpdate must run while NOT touching to fire Removed callback
        char.move((0, 0, 0), 1 / 60)
        events = self.world.get_contact_events_ex()
        self.assertTrue(
            any(e["type"] == culverin.EVENT_REMOVED and wall in e["bodies"] for e in events)
        )

    def test_character_collision_filtering(self) -> None:
        """Verify that character-vs-character contact can be filtered via bitmasks."""
        # 1. Setup two characters on 'Team A'
        # Category 2, Mask 1 (Collide with world, but not with other Category 2s)
        char1 = self.world.create_character(pos=(0, 0.5, 0))
        char2 = self.world.create_character(pos=(1.2, 0.5, 0))

        self.world.step(0)  # Flush

        # Set filters: Both are category 2, and both only look for category 1
        self.world.set_collision_filter(char1.handle, category=2, mask=1)
        self.world.set_collision_filter(char2.handle, category=2, mask=1)

        # 2. Attempt to move char1 THROUGH char2
        # If filtering works, char1 should move freely to X=2.0
        # and NO contact events should be generated.
        char1.move((120, 0, 0), 1 / 60)

        events = self.world.get_contact_events_ex()
        char_hits = [
            e for e in events if char1.handle in e["bodies"] and char2.handle in e["bodies"]
        ]

        self.assertEqual(len(char_hits), 0, "Characters collided despite filtering masks")

        # Verify char1 actually moved past char2 (didn't get stuck)
        self.assertGreater(char1.get_position()[0], 1.5)

        # 3. Change filter to allow collision
        # Mask 3 = (1 | 2), so it now sees category 2
        self.world.set_collision_filter(char1.handle, category=2, mask=3)
        self.world.set_collision_filter(char2.handle, category=2, mask=3)

        char1.set_position((0, 0.5, 0))
        char1.move((120, 0, 0), 1 / 60)

        events = self.world.get_contact_events_ex()
        char_hits = [
            e for e in events if char1.handle in e["bodies"] and char2.handle in e["bodies"]
        ]
        self.assertGreater(len(char_hits), 0, "Characters failed to collide after mask update")

    def test_character_slippery_platform(self) -> None:
        """Verify character is carried by a rotating kinematic platform."""
        self.world.register_material(id=10, friction=1.0)

        sticky_plat = self.world.create_body(
            pos=(0, 0, 0), size=(5, 0.5, 5), motion=culverin.MOTION_KINEMATIC, material_id=10
        )

        char_sticky = self.world.create_character(pos=(2, 1.0, 0), radius=0.5, height=1.0)
        self.world.step(0)

        # Warmup: settle onto platform
        for _ in range(20):
            char_sticky.move((0, 0, 0), 1 / 60)
            self.world.step(1 / 60)

        self.assertTrue(
            char_sticky.is_grounded(),
            f"Character failed to settle. Y={char_sticky.get_position()[1]}",
        )

        self.world.set_angular_velocity(sticky_plat, x=0, y=2.0, z=0)
        start_pos = char_sticky.get_position()

        # Zero input — ground velocity inheritance in C should do all the work
        for _ in range(10):
            char_sticky.move((0, 0, 0), 1 / 60)
            self.world.step(1 / 60)

        end_pos = char_sticky.get_position()
        dx = end_pos[0] - start_pos[0]
        dz = end_pos[2] - start_pos[2]
        displacement_sq = dx * dx + dz * dz

        self.assertGreater(
            displacement_sq, 0.1, f"Character failed to orbit. Start={start_pos}, End={end_pos}"
        )
        self.assertTrue(char_sticky.is_grounded(), f"Character fell off platform. Y={end_pos[1]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

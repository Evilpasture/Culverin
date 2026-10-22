"""Differential contract tests for the native C++23 rewrite.

These tests pin externally observable behavior of the legacy C23
implementation (buffer layouts, handle semantics, command deferral,
snapshot format, contact event records). They must pass unchanged on
both the old and the new native implementation; a failure here means
the rewrite changed a public contract, not an internal detail.
"""

from __future__ import annotations

import struct
import unittest

import culverin


class TestBufferLayoutContracts(unittest.TestCase):
    def test_shadow_buffer_strides(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 64})
        handles = [
            world.create_body(
                pos=(float(i), 10.0, 0.0),
                size=(0.5, 0.5, 0.5),
                shape=culverin.SHAPE_BOX,
                motion=culverin.MOTION_DYNAMIC,
            )
            for i in range(4)
        ]
        world.step(1 / 60)
        self.assertEqual(world.count, 4)
        # positions: float64 x4 stride; rotations/velocities: float32 x4 stride
        self.assertEqual(memoryview(world.positions).nbytes, 4 * 4 * 8)
        self.assertEqual(memoryview(world.rotations).nbytes, 4 * 4 * 4)
        self.assertEqual(memoryview(world.velocities).nbytes, 4 * 4 * 4)
        self.assertEqual(memoryview(world.angular_velocities).nbytes, 4 * 4 * 4)
        # Initial X positions survive in creation order in the dense buffer.
        xs = [struct.unpack_from("<d", world.positions, i * 32)[0] for i in range(4)]
        for i, handle in enumerate(handles):
            self.assertAlmostEqual(xs[world.get_index(handle)], float(i), places=5)

    def test_render_state_record_size(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        for i in range(3):
            world.create_body(
                pos=(0.0, 5.0 + i, 0.0),
                size=(0.5, 0.5, 0.5),
                shape=culverin.SHAPE_BOX,
                motion=culverin.MOTION_DYNAMIC,
            )
        world.step(1 / 60)
        state = world.get_render_state(alpha=0.5)
        self.assertEqual(len(state), 3 * 28)  # 7 float32 per body, no padding float


class TestHandleContracts(unittest.TestCase):
    def test_handle_packing_and_stale_detection(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        handle = world.create_body(
            pos=(0.0, 5.0, 0.0),
            size=(0.5, 0.5, 0.5),
            shape=culverin.SHAPE_BOX,
            motion=culverin.MOTION_DYNAMIC,
        )
        slot = handle & 0xFFFFFFFF
        generation = handle >> 32
        self.assertGreaterEqual(generation, 1)
        self.assertLess(slot, 16)
        world.step(1 / 60)
        self.assertTrue(world.is_alive(handle))

        world.destroy_body(handle)
        world.step(1 / 60)
        self.assertFalse(world.is_alive(handle))
        self.assertEqual(world.count, 0)

    def test_destroy_keeps_dense_buffer_packed(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        handles = [
            world.create_body(
                pos=(float(i * 2), 5.0, 0.0),
                size=(0.5, 0.5, 0.5),
                shape=culverin.SHAPE_BOX,
                motion=culverin.MOTION_DYNAMIC,
            )
            for i in range(5)
        ]
        world.step(1 / 60)
        world.destroy_body(handles[1])
        world.step(1 / 60)
        self.assertEqual(world.count, 4)
        self.assertEqual(memoryview(world.positions).nbytes, 4 * 4 * 8)
        for i in (0, 2, 3, 4):
            self.assertTrue(world.is_alive(handles[i]))
            pos = world.get_position(handles[i])
            self.assertIsNotNone(pos)
            self.assertAlmostEqual(pos[0], float(i * 2), places=4)


class TestCommandContracts(unittest.TestCase):
    def test_set_position_mirrored_before_step(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        handle = world.create_body(
            pos=(0.0, 5.0, 0.0),
            size=(0.5, 0.5, 0.5),
            shape=culverin.SHAPE_BOX,
            motion=culverin.MOTION_DYNAMIC,
        )
        world.step(1 / 60)
        world.set_position(handle, 7.0, 8.0, 9.0)
        pos = world.get_position(handle)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 7.0, places=5)
        self.assertAlmostEqual(pos[1], 8.0, places=5)
        self.assertAlmostEqual(pos[2], 9.0, places=5)

    def test_ops_on_pending_create_body(self) -> None:
        """Mutations queued before the first step must survive body creation."""
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        handle = world.create_body(
            pos=(0.0, 5.0, 0.0),
            size=(0.5, 0.5, 0.5),
            shape=culverin.SHAPE_BOX,
            motion=culverin.MOTION_DYNAMIC,
        )
        world.set_linear_velocity(handle, 3.0, 0.0, 0.0)
        world.step(1 / 60)
        vel = world.get_velocity(handle)
        self.assertIsNotNone(vel)
        self.assertGreater(vel[0], 1.0)

    def test_user_data_round_trip(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        handle = world.create_body(
            pos=(0.0, 5.0, 0.0),
            size=(0.5, 0.5, 0.5),
            shape=culverin.SHAPE_BOX,
            motion=culverin.MOTION_DYNAMIC,
            user_data=1234,
        )
        world.step(1 / 60)
        self.assertEqual(world.get_user_data(handle), 1234)
        world.set_user_data(handle, 98765)
        self.assertEqual(world.get_user_data(handle), 98765)


class TestSnapshotContracts(unittest.TestCase):
    def test_snapshot_round_trip_restores_positions(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        world.create_body(
            pos=(0.0, 0.0, 0.0), shape=culverin.SHAPE_PLANE, motion=culverin.MOTION_STATIC
        )
        handle = world.create_body(
            pos=(0.0, 5.0, 0.0),
            size=(0.5, 0.5, 0.5),
            shape=culverin.SHAPE_BOX,
            motion=culverin.MOTION_DYNAMIC,
        )
        for _ in range(30):
            world.step(1 / 60)
        saved = world.save_state()
        saved_positions = bytes(world.positions)
        for _ in range(30):
            world.step(1 / 60)
        world.load_state(state=saved)
        self.assertEqual(bytes(world.positions), saved_positions)
        self.assertTrue(world.is_alive(handle))


class TestResizeContracts(unittest.TestCase):
    def test_growth_beyond_initial_capacity(self) -> None:
        """Initial slot capacity is 64; creating more must grow storage."""
        world = culverin.PhysicsWorld(settings={"max_bodies": 512})
        handles = []
        for i in range(200):
            handles.append(
                world.create_body(
                    pos=(float(i % 10), 5.0 + float(i // 10), 0.0),
                    size=(0.25, 0.25, 0.25),
                    shape=culverin.SHAPE_BOX,
                    motion=culverin.MOTION_DYNAMIC,
                )
            )
            if i % 50 == 49:
                world.step(1 / 60)
        world.step(1 / 60)
        self.assertEqual(world.count, 200)
        self.assertTrue(all(world.is_alive(h) for h in handles))
        self.assertEqual(memoryview(world.positions).nbytes, 200 * 4 * 8)


if __name__ == "__main__":
    unittest.main()

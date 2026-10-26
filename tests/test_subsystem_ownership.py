"""Behavioural tests for the subsystem-ownership refactor.

Everything here runs under the project's normal Pytest workflow. There is no
standalone C++ test binary: the new ``BodyRegistry``, ``TransformBuffer``,
``CommandMailbox`` and ``DoubleBuffer`` primitives are exercised through the
public Python API, alongside the existing ``tests/test_core.py`` thread-safety
and generational-handle tests.

Observable guarantees covered here:

* zero-copy buffer strides are unchanged (float64 x4 positions, float32 x4
  rotations/velocities);
* storage is fixed-capacity: creating bodies beyond the old 64-slot growth
  boundary never invalidates a held memoryview and never raises ``BufferError``;
* exported transform frames stay pinned and readable across many steps;
* structural mutations (forces) are queued and only take effect on the step
  thread's flush;
* a reader thread can acquire zero-copy views while another thread steps
  without deadlocking or raising;
* ``world.get_render_state(alpha=...)`` still works.
"""

from __future__ import annotations

import struct
import threading
import unittest

import culverin


class TestZeroCopyStridesPreserved(unittest.TestCase):
    def test_strides_and_creation_order(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 64})
        for i in range(4):
            world.create_body(pos=(float(i) * 10.0, 10.0, 0.0))
        world.step(1 / 60)

        # Unchanged wire layout.
        self.assertEqual(memoryview(world.positions).nbytes, 4 * 4 * 8)
        self.assertEqual(memoryview(world.rotations).nbytes, 4 * 4 * 4)
        self.assertEqual(memoryview(world.velocities).nbytes, 4 * 4 * 4)
        self.assertEqual(memoryview(world.angular_velocities).nbytes, 4 * 4 * 4)

        # Creation-order X positions are visible in the published frame.
        xs = [struct.unpack_from("<d", world.positions, i * 32)[0] for i in range(4)]
        for i, x in enumerate(xs):
            self.assertAlmostEqual(x, float(i) * 10.0, places=5)


class TestFixedCapacityKeepsViewsValid(unittest.TestCase):
    def test_view_valid_across_creation_and_step(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 256})

        for _ in range(64):
            world.create_body(pos=(0.0, 0.0, 0.0))
        world.step(0)

        # memoryview() calls getbuffer once, pinning a raw pointer and a fixed
        # length. In the old design the 65th create reallocated that array.
        view = memoryview(world.positions)
        pinned_len = view.nbytes
        _ = bytes(view)

        # Growing past the old 64-slot boundary must not reallocate (no
        # BufferError) and must not move the pinned buffer.
        for i in range(64, 200):
            world.create_body(pos=(float(i), 0.0, 0.0))

        # A step publishes a fresh transform frame; the pinned view is still
        # readable (the step wrote the *other* frame) and keeps its length.
        world.step(1 / 60)
        self.assertEqual(view.nbytes, pinned_len)
        _ = bytes(view)

        self.assertEqual(world.count, 200)
        self.assertEqual(memoryview(world.positions).nbytes, 200 * 4 * 8)
        view.release()
        del view

    def test_every_channel_stays_pinned_across_many_steps(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 64, "gravity": (0, 0, 0)})
        world.create_body(pos=(1.0, 2.0, 3.0), motion=culverin.MOTION_DYNAMIC)
        world.step(0)

        names = ("positions", "rotations", "velocities", "angular_velocities")
        views = [memoryview(getattr(world, name)) for name in names]
        lengths = [view.nbytes for view in views]

        for _ in range(50):
            world.step(1 / 60)

        for view, length in zip(views, lengths):
            self.assertEqual(view.nbytes, length)
            self.assertEqual(len(bytes(view)), length)
            view.release()


class TestPureAsyncCommandMailbox(unittest.TestCase):
    def test_force_takes_effect_on_step_not_before(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16, "gravity": (0, 0, 0)})
        handle = world.create_body(pos=(1.0, 2.0, 3.0), motion=culverin.MOTION_DYNAMIC)
        world.step(0)

        def x_velocity() -> float:
            return struct.unpack_from("<f", world.velocities, 0)[0]

        self.assertAlmostEqual(x_velocity(), 0.0, places=6)
        world.apply_force(handle, 1000.0, 0.0, 0.0)
        self.assertAlmostEqual(x_velocity(), 0.0, places=6)

        world.step(1 / 60)
        self.assertGreater(x_velocity(), 0.0)


class TestBodyRegistryRecycling(unittest.TestCase):
    def test_generations_survive_recycling_without_resize(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 128})
        handles = [world.create_body(pos=(float(i), 0.0, 0.0)) for i in range(64)]

        for handle in handles[:10]:
            world.destroy_body(handle)
        world.step(0)
        for handle in handles[:10]:
            self.assertFalse(world.is_alive(handle))

        recycled = [world.create_body(pos=(100.0 + i, 0.0, 0.0)) for i in range(10)]
        for handle in recycled:
            self.assertTrue(world.is_alive(handle))
        self.assertEqual(len(set(handles[:10] + recycled)), 20)

        world.step(1 / 60)
        self.assertEqual(world.count, 64)


class TestConcurrentStepAndZeroCopyReads(unittest.TestCase):
    def test_step_and_memoryview_reader_are_safe(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 512, "gravity": (0, -10, 0)})
        for i in range(100):
            world.create_body(pos=(i * 3.0, 10.0, 0.0))
        world.step(0)

        expected_nbytes = 100 * 4 * 8
        barrier = threading.Barrier(3)
        errors: list[BaseException] = []

        def stepper() -> None:
            try:
                barrier.wait()
                for _ in range(200):
                    world.step(1 / 60)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        def reader() -> None:
            try:
                barrier.wait()
                for _ in range(1000):
                    with memoryview(world.positions) as view:
                        if view.nbytes != expected_nbytes:
                            errors.append(AssertionError(f"unexpected nbytes: {view.nbytes}"))
                        _ = bytes(view)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = (threading.Thread(target=stepper), threading.Thread(target=reader))
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=30)

        self.assertFalse(any(thread.is_alive() for thread in threads), "deadlock detected")
        self.assertEqual(errors, [])


class TestRenderStatePreserved(unittest.TestCase):
    def test_get_render_state(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 16})
        for i in range(3):
            world.create_body(pos=(0.0, 5.0 + i, 0.0))
        world.step(1 / 60)
        state = world.get_render_state(alpha=0.5)
        self.assertEqual(len(state), 3 * 28)  # 7 float32 per body, no padding


if __name__ == "__main__":
    unittest.main()

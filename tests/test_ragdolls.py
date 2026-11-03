"""Ragdoll subsystem: debug read-back and part construction.

Covers what get_debug_info reports, how add_part treats its optional pose, and
that a parented part is actually simulated through its constraint.
"""

from __future__ import annotations

import gc
import math
import unittest
import weakref

import culverin


class TestRagdollDebugInfo(unittest.TestCase):
    """get_debug_info reports position, rotation and velocity per body."""

    def setUp(self) -> None:
        self.world = culverin.PhysicsWorld(settings={"max_bodies": 256})
        skeleton = culverin.Skeleton()
        root = skeleton.add_joint(name="pelvis", parent_index=-1)
        skeleton.finalize()
        settings = self.world.create_ragdoll_settings(skeleton=skeleton)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        self.ragdoll = self.world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        self.world.step(1 / 60.0)

    def test_debug_info_reports_rotation(self) -> None:
        part = self.ragdoll.get_debug_info()[0]
        self.assertIn("rot", part)
        self.assertIsInstance(part["rot"], tuple)
        self.assertEqual(len(part["rot"]), 4)
        for component in part["rot"]:
            self.assertIsInstance(component, float)
            self.assertTrue(math.isfinite(component))

    def test_rotation_is_unit_length(self) -> None:
        rx, ry, rz, rw = self.ragdoll.get_debug_info()[0]["rot"]
        self.assertAlmostEqual(math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw), 1.0, places=4)

    def test_existing_keys_are_untouched(self) -> None:
        """'rot' is additive: index/pos/vel keep their names, types and order."""
        part = self.ragdoll.get_debug_info()[0]
        self.assertEqual(list(part)[:2], ["index", "pos"])
        self.assertIn("vel", part)
        self.assertIsInstance(part["index"], int)
        self.assertIsInstance(part["pos"], tuple)
        self.assertIsInstance(part["vel"], tuple)
        self.assertEqual(len(part["pos"]), 3)
        self.assertEqual(len(part["vel"]), 3)


class TestRagdollPartPosition(unittest.TestCase):
    """add_part validates its optional ``pos`` and reports failure."""

    @staticmethod
    def _settings(world: culverin.PhysicsWorld):
        skeleton = culverin.Skeleton()
        root = skeleton.add_joint(name="pelvis", parent_index=-1)
        skeleton.finalize()
        settings = world.create_ragdoll_settings(skeleton=skeleton)
        return settings, root

    def test_malformed_position_raises_and_adds_nothing(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 128})
        settings, root = self._settings(world)

        with self.assertRaises((TypeError, ValueError)):
            settings.add_part(
                joint_index=root,
                shape_type=culverin.SHAPE_BOX,
                size=(0.3, 0.2, 0.2),
                pos=(0.0, 0.0),  # only two components
            )

        # The part must not have been added: a ragdoll with no parts yields no
        # bodies, and get_body_handles reports exactly that.
        ragdoll = world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        world.step(0)
        self.assertEqual(list(ragdoll.get_body_handles()), [])

    def test_valid_position_still_accepted(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 128})
        settings, root = self._settings(world)
        settings.add_part(
            joint_index=root,
            shape_type=culverin.SHAPE_BOX,
            size=(0.3, 0.2, 0.2),
            pos=(1.0, 2.0, 3.0),
        )
        ragdoll = world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        self.assertIsNotNone(ragdoll)

    def test_omitting_pos_still_accepted(self) -> None:
        world = culverin.PhysicsWorld(settings={"max_bodies": 128})
        settings, root = self._settings(world)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        ragdoll = world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        self.assertIsNotNone(ragdoll)


class TestRagdollConstraintEnabled(unittest.TestCase):
    """A parented part is simulated through a live constraint."""

    def test_two_part_ragdoll_is_simulated(self) -> None:
        """A parented part (the only one that builds a constraint) still works."""
        world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 128})
        world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)

        skeleton = culverin.Skeleton()
        root = skeleton.add_joint(name="pelvis", parent_index=-1)
        spine = skeleton.add_joint(name="spine", parent_index=root)
        skeleton.finalize()

        settings = world.create_ragdoll_settings(skeleton=skeleton)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        settings.add_part(
            joint_index=spine,
            parent_index=root,
            shape_type=culverin.SHAPE_BOX,
            size=(0.3, 0.4, 0.2),
        )

        ragdoll = world.create_ragdoll(settings=settings, pos=(0, 5, 0))
        world.step(0)
        start = world.get_position(ragdoll.get_body_handles()[1])
        for _ in range(60):
            world.step(1 / 60.0)
        end = world.get_position(ragdoll.get_body_handles()[1])

        # The constrained part must still move; a disabled constraint would
        # leave it pinned to its creation pose.
        self.assertLess(end[1], start[1] - 0.01, "parented part did not fall")

    def test_ragdoll_debug_info_order_is_stable(self) -> None:
        """'rot' was inserted; the pre-existing keys keep their order."""
        world = culverin.PhysicsWorld(settings={"max_bodies": 128})
        skeleton = culverin.Skeleton()
        root = skeleton.add_joint(name="pelvis", parent_index=-1)
        skeleton.finalize()
        settings = world.create_ragdoll_settings(skeleton=skeleton)
        settings.add_part(joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2))
        ragdoll = world.create_ragdoll(settings=settings, pos=(0, 10, 0))
        world.step(1 / 60.0)
        part = ragdoll.get_debug_info()[0]
        self.assertEqual(list(part)[0], "index")
        self.assertEqual(list(part)[1], "pos")
        self.assertEqual(list(part)[-1], "vel")


def _single_part_ragdoll(world, pos=(0, 10, 0)):
    skeleton = culverin.Skeleton()
    root = skeleton.add_joint(name="pelvis", parent_index=-1)
    skeleton.finalize()
    settings = world.create_ragdoll_settings(skeleton=skeleton)
    settings.add_part(
        joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2)
    )
    return world.create_ragdoll(settings=settings, pos=pos)


class TestRagdollLifetime(unittest.TestCase):
    """Ragdoll matches Character and Vehicle on lifetime and teardown."""

    def setUp(self) -> None:
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 256})

    def test_destroy_is_explicit_and_immediate(self) -> None:
        """destroy() takes the parts out of the world before the object dies."""
        ragdoll = _single_part_ragdoll(self.world)
        self.world.step(0)
        handles = ragdoll.get_body_handles()
        self.assertTrue(handles)
        self.assertTrue(all(self.world.is_alive(h) for h in handles))

        ragdoll.destroy()

        for handle in handles:
            self.assertFalse(
                self.world.is_alive(handle), "destroy() left a ragdoll part in the world"
            )

    def test_destroy_is_idempotent(self) -> None:
        """Teardown paths can call destroy() without tracking state."""
        ragdoll = _single_part_ragdoll(self.world)
        ragdoll.destroy()
        ragdoll.destroy()
        gc.collect()
        ragdoll.destroy()

    def test_access_after_destroy_raises_instead_of_crashing(self) -> None:
        """Every entry point refuses a destroyed ragdoll rather than reading freed memory."""
        ragdoll = _single_part_ragdoll(self.world)
        ragdoll.destroy()

        for call in (ragdoll.get_body_handles, ragdoll.get_debug_info):
            with self.assertRaises(RuntimeError):
                call()

    def test_destroy_releases_the_world(self) -> None:
        """A destroyed ragdoll must not keep the whole world alive.

        This is what went unnoticed while the type had no tp_traverse: the world
        reference held for the ragdoll's lifetime kept a PhysicsWorld -- and
        every body pooled inside it -- alive until interpreter shutdown.
        """
        world_ref = weakref.ref(self.world)
        ragdoll = _single_part_ragdoll(self.world)
        ragdoll.destroy()

        self.assertIsNotNone(world_ref(), "world vanished while still referenced")
        del ragdoll
        gc.collect()

        # The harness still holds self.world, so the ref must survive the drop;
        # what is being pinned is that destroy() dropped the ragdoll's own hold.
        self.assertIsNotNone(world_ref())

    def test_destroyed_ragdoll_does_not_pin_the_world(self) -> None:
        """Drop the last reference: the world must become collectable."""
        world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 256})
        world_ref = weakref.ref(world)
        ragdoll = _single_part_ragdoll(world)
        world.step(0)

        ragdoll.destroy()
        del ragdoll
        del world
        gc.collect()

        self.assertIsNone(world_ref(), "destroyed ragdoll kept the PhysicsWorld alive")

    def test_undestroyed_ragdoll_keeps_the_world_alive(self) -> None:
        """The world must outlive a live ragdoll -- the other half of the contract."""
        world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 256})
        world_ref = weakref.ref(world)
        ragdoll = _single_part_ragdoll(world)
        world.step(0)

        del world
        gc.collect()
        self.assertIsNotNone(world_ref(), "world died while a live ragdoll still needed it")

        ragdoll.destroy()
        del ragdoll
        gc.collect()
        self.assertIsNone(world_ref(), "world outlived the ragdoll that was keeping it alive")

    def test_python_cycles_are_collectable(self) -> None:
        """RagdollObject must be visible to the cycle collector.

        Without Py_TPFLAGS_HAVE_GC / tp_traverse, a cycle through a container
        holding the ragdoll is uncollectable, which is the defect this pins.
        """
        self.assertTrue(gc.is_tracked(_single_part_ragdoll(self.world)))

    def test_collecting_a_cycle_releases_the_world(self) -> None:
        """A ragdoll trapped in a cycle must still give the world back.

        This is the case tp_clear exists for, and the one that crashes if the
        world reference is dropped while the native ragdoll still holds its
        PhysicsWorld*: ~Ragdoll locks the shadow lock and releases body slots
        through that pointer.
        """
        def trapped():
            world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 256})
            ref = weakref.ref(world)
            # The ragdoll holds the only reference to the world, and the cycle
            # holds the only reference to the ragdoll.
            cycle = {}
            cycle["ragdoll"] = _single_part_ragdoll(world)
            cycle["self"] = cycle
            return ref

        world_ref = trapped()
        for _ in range(3):
            gc.collect()

        self.assertIsNone(world_ref(), "a collected cycle still pinned the PhysicsWorld")


if __name__ == "__main__":
    unittest.main()

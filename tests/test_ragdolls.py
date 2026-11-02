"""Ragdoll subsystem: debug read-back and part construction.

Covers what get_debug_info reports, how add_part treats its optional pose, and
that a parented part is actually simulated through its constraint.
"""

from __future__ import annotations

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()

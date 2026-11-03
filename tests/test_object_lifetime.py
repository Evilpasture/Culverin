"""Garbage-collection protocol for every Python type that owns native state.

These are not feature tests. Each one pins a property of the *lifetime
contract* that is invisible at the API surface and expensive when it is wrong:
a leaked type reference, a cycle the collector cannot break, or a native object
released after the world reference it borrows. The implementation lives in
bindings/PyGcObject.hpp; these check that every type actually gets it.

Every type is exercised the same way so that a new GC type has an obvious place
to be added here.
"""

from __future__ import annotations

import gc
import sys
import unittest
import weakref

import culverin


# --- Factories: one per GC-managed type -------------------------------------
#
# Each returns a fresh instance bound to `world` (or, for BufferProxy, to a
# proxy of it). Adding a GC type to the extension means adding one entry here.


def _character(world: culverin.PhysicsWorld):
    return world.create_character(pos=(0, 2, 0))


def _ragdoll(world: culverin.PhysicsWorld):
    skeleton = culverin.Skeleton()
    root = skeleton.add_joint(name="pelvis", parent_index=-1)
    skeleton.finalize()
    settings = world.create_ragdoll_settings(skeleton=skeleton)
    settings.add_part(
        joint_index=root, shape_type=culverin.SHAPE_BOX, size=(0.3, 0.2, 0.2)
    )
    return world.create_ragdoll(settings=settings, pos=(0, 10, 0))


def _vehicle(world: culverin.PhysicsWorld):
    chassis = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
    world.step(0)
    return world.create_vehicle(
        chassis=chassis,
        wheels=[
            {"pos": (-1, -0.5, 1.5), "radius": 0.4},
            {"pos": (1, -0.5, -1.5), "radius": 0.4},
        ],
        drive="AWD",
    )


def _ship(world: culverin.PhysicsWorld):
    sled = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=800.0)
    return world.create_ship(
        sled=sled, kp=4000000.0, kd=1000000.0, throttle_force=500000.0, steer_speed=1.0
    )


def _proxy(world: culverin.PhysicsWorld):
    return world.positions


FACTORIES = {
    "Character": _character,
    "Ragdoll": _ragdoll,
    "Vehicle": _vehicle,
    "Ship": _ship,
    "BufferProxy": _proxy,
}


def _world() -> culverin.PhysicsWorld:
    world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": 512})
    world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)
    return world


def _cycle(make, count: int = 50) -> None:
    """Create `count` instances, drop them, and let the collector settle."""
    instances = [make() for _ in range(count)]
    del instances
    gc.collect()


def _type_refcount(type_obj: type) -> int:
    """Reference count of `type_obj`, with the collector flushed first.

    Only *differences* between readings are meaningful here. On the free-threaded
    build a heap type reports a large biased base value (2**60 + real count), and
    on every build CPython may substitute a sentinel count for immortal objects.

    The collection point matters just as much: free-threaded CPython defers
    reference count updates, merging them later rather than applying them
    inline, so a sample taken without one can read a count that is still in
    flight. Flushing first is what makes the comparison below exact.
    """
    gc.collect()
    return sys.getrefcount(type_obj)


class TestTypeReferenceIsBalanced(unittest.TestCase):
    """Creating and dropping instances must not leak references to the type.

    PyObject_GC_New takes a reference to the type on the caller's behalf, so a
    dealloc that does not decref it leaks one reference per instance -- for the
    process lifetime. It is invisible: no error, no growth in any object count,
    just a type that can never be freed.
    """

    def test_dropping_instances_does_not_leak_the_type(self) -> None:
        for name, factory in FACTORIES.items():
            with self.subTest(type=name):
                world = _world()
                make = lambda: factory(world)  # noqa: E731 - closes over world
                type_obj = type(make())

                # Warm up before sampling the baseline. The first cycle settles
                # one-time state and flushes the deferred refcount queue; sample
                # before that and the baseline reads a ref low, which makes
                # every later reading look like a leak of exactly one.
                _cycle(make)
                baseline = _type_refcount(type_obj)

                for cycle in range(3):
                    _cycle(make)
                    self.assertEqual(
                        _type_refcount(type_obj),
                        baseline,
                        f"{name} leaked type references after cycle {cycle}",
                    )


class TestCyclesAreCollectable(unittest.TestCase):
    """A cycle holding an instance must not keep the world alive.

    Without tp_traverse the collector cannot see the instance -> world edge, so
    the whole PhysicsWorld and every body pooled inside it survives until
    interpreter shutdown.
    """

    def test_cycle_does_not_pin_the_world(self) -> None:
        for name, factory in FACTORIES.items():
            with self.subTest(type=name):

                def trapped():
                    world = _world()
                    ref = weakref.ref(world)
                    # The instance holds the only reference to the world, and
                    # the dict holds the only reference to the instance.
                    cycle = {"instance": factory(world)}
                    cycle["self"] = cycle
                    return ref

                world_ref = trapped()
                for _ in range(3):
                    gc.collect()

                self.assertIsNone(
                    world_ref(), f"{name} in a cycle kept its PhysicsWorld alive"
                )

    def test_instances_are_tracked(self) -> None:
        """A GC type must be visible to the collector in the first place."""
        world = _world()
        for name, factory in FACTORIES.items():
            with self.subTest(type=name):
                self.assertTrue(
                    gc.is_tracked(factory(world)), f"{name} is not GC-tracked"
                )


class TestClearRunsBeforeDealloc(unittest.TestCase):
    """Native teardown must happen while the objects it borrows are alive.

    `tp_clear` may run before `tp_dealloc`, so any native object that reaches
    into the world through a raw pointer has to be released *before* the world
    reference is dropped. Getting that order wrong is a use-after-free during
    collection, which surfaces as a segfault or an allocator abort rather than
    an exception.
    """

    def test_reaching_a_clear_then_collecting_does_not_crash(self) -> None:
        # Reaching the clear directly is the whole point: collect() alone does
        # not guarantee the instance's clear runs before its dealloc here.
        for name, factory in FACTORIES.items():
            with self.subTest(type=name):
                world = _world()
                cycle = {"instance": factory(world)}
                cycle["self"] = cycle
                gc.collect()

                # The world is still usable afterwards: a native object
                # destroyed against a dangling pointer tends to corrupt the
                # world's slot bookkeeping even when it does not crash.
                survivor = world.create_character(pos=(0, 3, 0))
                world.step(0)
                self.assertIsNotNone(survivor.get_position())


class TestExplicitDestroy(unittest.TestCase):
    """Types with manual control must stay consistent with the GC path."""

    def test_ragdoll_destroy_is_idempotent_and_final(self) -> None:
        world = _world()
        ragdoll = _ragdoll(world)
        world.step(0)
        handles = ragdoll.get_body_handles()

        ragdoll.destroy()
        ragdoll.destroy()

        for handle in handles:
            self.assertFalse(world.is_alive(handle))
        with self.assertRaises(RuntimeError):
            ragdoll.get_body_handles()

    def test_ragdoll_destroy_then_collect(self) -> None:
        """The GC path must cope with a native object destroy() already freed."""
        world = _world()
        ragdoll = _ragdoll(world)
        ragdoll.destroy()

        cycle = {"ragdoll": ragdoll}
        cycle["self"] = cycle
        del cycle

        gc.collect()  # must not double-free


if __name__ == "__main__":
    unittest.main()

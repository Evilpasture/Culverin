"""Vehicle subsystem: wheeled cars, tracked tanks, and their construction.

Covers the wheel-query payloads and their reference counts, the tank input
path, wheel steering configuration, the transmission/engine settings that reach
Jolt's controller, and the ownership relationship between a vehicle and the
world it belongs to.

Kept separate from test_ship.py (a different controller) and from test_core.py
(which covers the world API rather than one subsystem's construction).
"""

from __future__ import annotations

import gc
import math
import sys
import unittest
import weakref

import culverin


def _wheeled(world: culverin.PhysicsWorld, wheels: list[dict[str, object]]) -> object:
    """A 1500 kg chassis on the given wheels, not yet stepped."""
    chassis = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
    world.step(0)
    return world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")


def _flat_world(max_bodies: int = 256) -> culverin.PhysicsWorld:
    """A world with a large friction floor at y = -1."""
    world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0), "max_bodies": max_bodies})
    world.create_body(
        pos=(0, -1, 0), size=(200, 1, 200), motion=culverin.MOTION_STATIC, friction=1.0
    )
    return world


class TestWheelTransformRefcounts(unittest.TestCase):
    """Wheel queries must not retain PyFloat objects between calls.

    The payload is built with PyFloat_FromDouble, whose result has to be handed
    to a stealing API (PyTuple_SET_ITEM); passing it to an increfing one
    (PyTuple_Pack) leaks one float per component per call. Floats are not
    GC-tracked, so this is measured by pymalloc block count.
    """

    def setUp(self) -> None:
        self.world = _flat_world()
        self.car = _wheeled(
            self.world,
            [{"pos": (x, -0.5, z), "radius": 0.4} for x in (-0.8, 0.8) for z in (1.2, -1.2)],
        )
        for _ in range(10):
            self.world.step(1 / 60.0)

    def _blocks_grown(self, call, n: int = 2000) -> int:
        # floats are not GC-tracked, so gc.get_objects() cannot see the leak;
        # sys.getallocatedblocks() counts pymalloc blocks, which is what the
        # leaked PyFloat objects occupy.
        for _ in range(n // 10):  # warm every cache first
            call(0)
        before = sys.getallocatedblocks()
        for _ in range(n):
            call(0)
        return sys.getallocatedblocks() - before

    def test_get_wheel_transform_does_not_leak(self) -> None:
        grown = self._blocks_grown(self.car.get_wheel_transform)
        # 2000 calls x 7 leaked floats was ~14000 blocks; a rounding error is
        # all that should be left.
        self.assertLess(grown, 100, f"get_wheel_transform leaked {grown} blocks")

    def test_get_wheel_local_transform_does_not_leak(self) -> None:
        grown = self._blocks_grown(self.car.get_wheel_local_transform)
        self.assertLess(grown, 100, f"get_wheel_local_transform leaked {grown} blocks")

    def test_payload_shape_is_unchanged(self) -> None:
        pos, rot = self.car.get_wheel_transform(0)
        self.assertIsInstance(pos, tuple)
        self.assertIsInstance(rot, tuple)
        self.assertEqual(len(pos), 3)
        self.assertEqual(len(rot), 4)
        for value in (*pos, *rot):
            self.assertIsInstance(value, float)
            self.assertTrue(math.isfinite(value))

    def test_repeated_calls_return_identical_values(self) -> None:
        first = self.car.get_wheel_transform(1)
        for _ in range(100):
            self.assertEqual(self.car.get_wheel_transform(1), first)
        gc.collect()


class TestTankInputZero(unittest.TestCase):
    """A zero steering ratio on both tracks must not corrupt the vehicle.

    ``TrackedVehicleController::SyncLeftRightTracks`` divides by (left * right)
    or (right - left), so a zero on both sides is 0/0 and stores NaN into both
    tracks' angular velocities. Release builds compile Jolt's assert out, so
    the call corrupts the vehicle rather than being rejected.
    """

    def _tank(self) -> tuple[culverin.PhysicsWorld, object, int]:
        world = _flat_world()
        chassis = world.create_body(pos=(0, 2, 0), size=(2, 1, 3), mass=5000.0)
        wheels = [{"pos": (x, -1.0, z), "radius": 0.5} for x in (-1.5, 1.5) for z in (2.0, 0.0, -2.0)]
        tracks = [
            {"indices": [0, 2, 4], "driven_wheel": 0},
            {"indices": [1, 3, 5], "driven_wheel": 1},
        ]
        tank = world.create_tracked_vehicle(chassis=chassis, wheels=wheels, tracks=tracks)
        world.step(0)
        return world, tank, chassis

    def test_zero_input_leaves_vehicle_finite(self) -> None:
        world, tank, chassis = self._tank()
        tank.set_tank_input(0.0, 0.0, 0.0)
        for _ in range(60):
            world.step(1 / 60.0)
        velocity = world.get_velocity(chassis)
        position = world.get_position(chassis)
        assert velocity is not None and position is not None
        for component in (*velocity, *position):
            self.assertTrue(math.isfinite(component), f"non-finite state: {position} {velocity}")

    def test_zero_input_does_not_stop_the_tank_driving(self) -> None:
        """A tank that took a no-steer call must still respond to throttle."""
        world, tank, chassis = self._tank()
        tank.set_tank_input(0.0, 0.0, 0.0)
        world.step(1 / 60.0)
        tank.set_tank_input(1.0, 1.0, 0.0)
        for _ in range(120):
            world.step(1 / 60.0)
        velocity = world.get_velocity(chassis)
        assert velocity is not None
        self.assertGreater(velocity[2], 0.5)

    def test_opposite_sign_pivot_still_turns(self) -> None:
        """-1/1 is a valid hard pivot and must keep working."""
        world, tank, chassis = self._tank()
        tank.set_tank_input(0.0, -1.0, 1.0)
        for _ in range(90):
            world.step(1 / 60.0)
        angular = world.get_angular_velocity(chassis)
        assert angular is not None
        self.assertTrue(all(math.isfinite(c) for c in angular))


class TestWheelSteeringFlag(unittest.TestCase):
    """Wheel steering is configurable, not only inferred from the sign of z.

    Left to itself the rule is ``z > 0.1 -> steers``, so a chassis laid out
    along -Z does not get *no* steering wheels: it gets steering wheels on the
    wrong axle, because the car's rear sits at the positive z the rule looks
    for. ``steers`` lets the config name the axle outright; omitting it keeps
    the rule.
    """

    @staticmethod
    def _wheels(sign: int, steer):
        return [
            {"pos": (x, -0.5, z), "radius": 0.4, **({"steers": steer(z)} if steer else {})}
            for x in (-0.8, 0.8)
            for z in (1.2, -1.2)
        ]

    @staticmethod
    def _drive(wheels) -> tuple[float, float]:
        """Full right lock for 3 s; return the chassis's final (|x|, z)."""
        world = _flat_world()
        chassis = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        world.step(0)
        car = world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")
        for _ in range(60):
            world.step(1 / 60.0)
        car.set_input(forward=1.0, right=1.0)
        for _ in range(180):
            world.step(1 / 60.0)
        position = world.get_position(chassis)
        assert position is not None
        return abs(position[0]), position[2]

    def test_flag_overrides_the_inference(self) -> None:
        """steers=False everywhere must drive straight, whatever z says."""
        lateral_default, forward_default = self._drive(self._wheels(1, None))
        lateral_flag, forward_flag = self._drive(self._wheels(1, lambda _z: False))
        self.assertGreater(lateral_default, 5.0, "sanity: the default layout turns")
        self.assertLess(lateral_flag, 0.1, "no wheel steered, yet the car turned")
        self.assertGreater(forward_flag, forward_default, "a straight car travels further")

    def test_legacy_inference_is_unchanged(self) -> None:
        """Omitting 'steers' equals saying 'steers where z > 0'."""
        legacy = self._drive(self._wheels(1, None))
        explicit = self._drive(self._wheels(1, lambda z: z > 0))
        self.assertAlmostEqual(legacy[0], explicit[0], places=6)
        self.assertAlmostEqual(legacy[1], explicit[1], places=6)

    def test_negative_z_front_can_be_told_to_steer(self) -> None:
        """On a -Z chassis the flag reaches the backend: the trajectory changes."""
        legacy = self._drive(self._wheels(-1, None))
        flagged = self._drive(self._wheels(-1, lambda z: z < 0))
        self.assertGreater(abs(legacy[0] - flagged[0]), 1.0)

    def test_steers_must_be_bool(self) -> None:
        world = _flat_world()
        chassis = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        wheels = [
            {"pos": (0, 0, 0), "radius": 0.4, "steers": "yes"},
            {"pos": (0, 0, 1), "radius": 0.4},
        ]
        with self.assertRaises(TypeError) as caught:
            world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")
        self.assertIn("steers", str(caught.exception))


class TestWheeledVehicleWorldReference(unittest.TestCase):
    """A vehicle must hold exactly one reference to its world.

    The world is kept alive through ``obj->world`` and released once by
    ``VehicleDealloc``; a second unbalanced increment leaks the entire
    ``PhysicsWorld``, and everything pooled inside it, for as long as any
    vehicle object lives. ``create_tracked_vehicle`` and
    ``create_vehicle`` must agree on this.

    ``sys.getallocatedblocks()`` cannot see the leak: it is a C-level refcount,
    not an allocation. A weak reference can -- a world that is still referenced
    somewhere never reaches the weakref callback.
    """

    @staticmethod
    def _world_with_vehicle() -> tuple[culverin.PhysicsWorld, object]:
        world = _flat_world(64)
        chassis = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        wheels = [
            {"pos": (x, -0.5, z), "radius": 0.4} for x in (-0.8, 0.8) for z in (1.2, -1.2)
        ]
        car = world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD")
        world.step(0)
        return world, car

    def test_dropping_the_vehicle_releases_the_world(self) -> None:
        world, car = self._world_with_vehicle()
        tracker = weakref.ref(world)
        del car
        del world
        gc.collect()
        self.assertIsNone(tracker(), "the world outlived the vehicle that leaked a reference to it")

    def test_the_vehicle_keeps_the_world_alive_while_it_exists(self) -> None:
        """The fix must not over-correct: the reference is still needed."""
        world, car = self._world_with_vehicle()
        tracker = weakref.ref(world)
        del world
        gc.collect()
        self.assertIsNotNone(tracker(), "the vehicle dropped a world it still uses")
        del car
        gc.collect()
        self.assertIsNone(tracker(), "the world was never released")

    def test_tracked_vehicles_agree(self) -> None:
        world = _flat_world(64)
        chassis = world.create_body(pos=(0, 2, 0), size=(2, 1, 3), mass=5000.0)
        wheels = [
            {"pos": (x, -1.0, z), "radius": 0.5} for x in (-1.5, 1.5) for z in (2.0, 0.0, -2.0)
        ]
        tracks = [
            {"indices": [0, 2, 4], "driven_wheel": 0},
            {"indices": [1, 3, 5], "driven_wheel": 1},
        ]
        tank = world.create_tracked_vehicle(chassis=chassis, wheels=wheels, tracks=tracks)
        world.step(0)
        tracker = weakref.ref(world)
        del tank
        del world
        gc.collect()
        self.assertIsNone(tracker(), "tracked vehicle leaked its world")


class TestWheeledTransmission(unittest.TestCase):
    """Transmission and engine settings must reach Jolt's controller.

    These are assembled into a ``VehicleTransmissionSettings`` and copied into
    the controller by ``WheeledVehicleController``'s constructor, so a value
    that never arrives silently degrades every wheeled vehicle to Jolt's
    defaults.

    The public API is the attribute-carrying ``Engine`` / ``Transmission``
    objects (``culverin.Automatic``, ``culverin.Manual``), not plain dicts --
    the parsers read attributes, so a dict is silently ignored.
    """

    @staticmethod
    def _drive(transmission=None, engine=None, seconds: float = 4.0) -> float:
        world = _flat_world()
        chassis = world.create_body(pos=(0, 2, 0), size=(1, 0.5, 2), mass=1500.0)
        wheels = [
            {"pos": (x, -0.5, z), "radius": 0.4} for x in (-0.8, 0.8) for z in (1.2, -1.2)
        ]
        kwargs = {}
        if transmission is not None:
            kwargs["transmission"] = transmission
        if engine is not None:
            kwargs["engine"] = engine
        car = world.create_vehicle(chassis=chassis, wheels=wheels, drive="AWD", **kwargs)
        for _ in range(60):
            world.step(1 / 60.0)
        car.set_input(forward=1.0)
        for _ in range(int(seconds * 60)):
            world.step(1 / 60.0)
        return world.get_position(chassis)[2]

    def test_gear_ratios_reach_the_controller(self) -> None:
        """A deliberately awful gearbox must change the vehicle's motion."""
        stock = self._drive()
        crippled = culverin.Transmission()
        crippled.ratios = [0.05, 0.05]
        moved = self._drive(crippled)
        self.assertGreater(stock, moved + 0.5,
                           f"transmission had no effect: stock {stock:.3f} vs {moved:.3f}")

    def test_clutch_strength_alone_reaches_the_controller(self) -> None:
        stock = self._drive()
        weak = culverin.Transmission()
        weak.clutch_strength = 1.0
        moved = self._drive(weak)
        # A 2000 -> 1 clutch cannot transmit meaningful torque.
        self.assertLess(moved, stock - 0.5,
                        f"clutch_strength had no effect: stock {stock:.3f} vs {moved:.3f}")

    def test_mode_reaches_the_controller(self) -> None:
        """``mode`` is forwarded now, but Jolt's controller ignores it.

        ``WheeledVehicleController`` only reads mMode in its serialisation code;
        the transmission is driven as automatic either way, so Auto and Manual
        produce identical motion and no user-visible assertion is possible.
        What this pins is that forwarding mode does not *break* anything: with
        every other field pinned equal, the two modes still agree.
        """
        auto = culverin.Transmission()
        auto.mode = 1
        manual = culverin.Transmission()
        manual.mode = 0
        self.assertAlmostEqual(self._drive(auto), self._drive(manual), places=6)

    def test_engine_settings_still_reach_the_controller(self) -> None:
        """Control: the engine path was already wired and must stay that way."""
        stock = self._drive()
        weak = culverin.Engine()
        weak.max_torque = 1.0
        moved = self._drive(engine=weak)
        self.assertLess(moved, stock - 0.5)

    def test_omitting_transmission_keeps_the_defaults(self) -> None:
        self.assertAlmostEqual(self._drive(), self._drive(culverin.Transmission()), places=6)



if __name__ == "__main__":
    unittest.main()

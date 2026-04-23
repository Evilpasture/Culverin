import numpy as np
import pytest

import culverin

# --- FIXTURES ---

@pytest.fixture
def world():
    """Provides a clean physics world for each test."""
    return culverin.PhysicsWorld(settings={"max_bodies": 100})

@pytest.fixture
def ship_setup(world: culverin.PhysicsWorld) -> tuple[culverin.PhysicsWorld, int, culverin.Ship]:
    """Creates a heavy sled and a native C ship controller."""
    sled = world.create_body(
        pos=(0, 10, 0),
        size=(5, 1, 10),
        mass=10000.0,
        motion=culverin.MOTION_DYNAMIC
    )
    world.step(0)

    # Increase KP for a faster "snap" back to upright
    # Increase KD to prevent the ship from overshooting and oscillating
    controller = world.create_ship(
        sled=sled,
        kp=4000000.0,      # Doubled from 2M to 4M
        kd=1000000.0,      # Increased damping
        throttle_force=500000.0,
        steer_speed=1.0
    )

    return world, sled, controller

# --- TESTS ---

def test_ship_creation(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Verify ship is created and carries the correct handle."""
    world, sled, controller = ship_setup
    assert controller is not None
    assert world.is_alive(sled)

def test_ship_invalid_handle(world: culverin.PhysicsWorld) -> None:
    """Ensure passing a bogus handle raises ValueError."""
    with pytest.raises(ValueError, match="Invalid sled handle"):
        world.create_ship(
            sled=999999, # Fake handle
            kp=1.0, kd=1.0, throttle_force=1.0, steer_speed=1.0
        )

def test_ship_stabilization(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Verify that the native C PD loop pulls the ship back to upright."""
    world, sled, _controller = ship_setup

    # Force a 45 degree tilt
    world.set_rotation(sled, 0, 0, 0.382, 0.923)
    world.step(0)

    stats = world.get_body_stats(sled)
    assert stats is not None, f"Sled {sled} should have valid body stats"

    # The linter now knows 'stats' is not None and can be unpacked safely
    _, rot_start, _ = stats

    # Give the ship 1 full second (60 frames) to stabilize
    for _ in range(60):
        world.step(1/60.0)

    stats_end = world.get_body_stats(sled)
    assert stats_end is not None

    _, rot_end, _ = stats_end

    # The Z component (roll) should have significantly decreased
    assert abs(rot_end[2]) < abs(rot_start[2])
    # With 4M gain and 60 frames, it should easily be under 0.1
    assert abs(rot_end[2]) < 0.1

def test_ship_throttle(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Verify C code applies forward force when input is set."""
    world, sled, controller = ship_setup

    # Initial state: static
    _, _, vel_start = world.get_body_stats(sled)
    assert vel_start[2] == 0.0

    # Set throttle
    controller.set_input(forward=1.0, right=0.0)

    # Step simulation
    for _ in range(10):
        world.step(1/60.0)

    _, _, vel_end = world.get_body_stats(sled)

    # Ship should be moving forward (Z axis)
    assert vel_end[2] > 1.0

def test_ship_steering(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Verify steering torque moves angular velocity Y towards target."""
    world, sled, controller = ship_setup

    # Set steer right
    controller.set_input(forward=0.0, right=1.0)

    # Give it more time to ramp up since it's now Torque-based, not velocity-override
    for _ in range(10):
        world.step(1/60.0)

    idx = world.get_index(sled)
    # Access the property .angular_velocities, not a method
    avel_buffer = np.frombuffer(world.angular_velocities, dtype=np.float32).reshape(-1, 4)

    # Verify we are moving towards the target (1.0)
    current_yaw_vel = avel_buffer[idx][1]
    assert current_yaw_vel > 0.1, "Torque steering should have started rotating the ship"

def test_ship_deallocation_safety(world: culverin.PhysicsWorld) -> None:
    """
    Critical Test: Ensure deleting the Python object removes the Jolt listener.
    If this fails, the world.step() will call a dangling pointer and segfault.
    """
    sled = world.create_body(pos=(0,0,0))
    world.step(0)

    controller = world.create_ship(sled, 1.0, 1.0, 1.0, 1.0)

    # Delete the controller (triggers Ship_dealloc)
    del controller

    # If the StepListener wasn't removed, this will crash the process
    try:
        for _ in range(10):
            world.step(1/60.0)
    except Exception as e:
        pytest.fail(f"World step crashed after ship deletion: {e}")

def test_ship_multi_interpreter_isolation(world: culverin.PhysicsWorld) -> None:
    """Tests if multiple ships can exist without clobbering each other's inputs."""
    s1_sled = world.create_body(pos=(-10, 10, 0), mass=1000)
    s2_sled = world.create_body(pos=(10, 10, 0), mass=1000)
    world.step(0)

    ship1 = world.create_ship(s1_sled, 0, 0, 10000, 0)
    ship2 = world.create_ship(s2_sled, 0, 0, 10000, 0)

    # Only ship 1 moves
    ship1.set_input(forward=1.0, right=0.0)
    ship2.set_input(forward=0.0, right=0.0)

    for _ in range(10):
        world.step(1/60.0)

    assert(v1 := world.get_velocity(s1_sled))
    assert(v2 := world.get_velocity(s2_sled))

    assert v1[2] > 0.5
    assert v2[2] == 0.0

def test_ship_banking_behavior(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Verify the ship rolls (banks) into its turns with sufficient gain."""
    world, sled, _ = ship_setup

    # FIX: Use 4M gain like the fixture, otherwise 10k kg mass won't move
    bank_controller = world.create_ship(
        sled=sled,
        kp=4000000.0, kd=100000.0,
        throttle_force=0,
        steer_speed=5.0,
        banking=0.5
    )

    bank_controller.set_input(forward=0.0, right=1.0)

    for _ in range(20): # 20 frames for physical response
        world.step(1/60.0)

    if (stats := world.get_body_stats(sled)):
        _, rot, _ = stats
        # Ensure banking torque has created a measurable tilt
        assert abs(rot[2]) > 0.01, f"Ship should have tilted. Got {abs(rot[2])}"

def test_ship_lateral_grip_performance(world: culverin.PhysicsWorld) -> None:
    """Verify that lateral grip significantly reduces sideways sliding."""
    s1 = world.create_body(pos=(-5, 2, 0), mass=1000, motion=culverin.MOTION_DYNAMIC)
    s2 = world.create_body(pos=(5, 2, 0), mass=1000, motion=culverin.MOTION_DYNAMIC)
    world.step(0)

    # 1. Crank the grip to 100,000.0.
    # With 1000kg mass, this provides strong correction without creating solver jitter.
    c_ice = world.create_ship(s1, 1000000, 10000, 5000, 0.5, lateral_grip=0.0)
    c_rail = world.create_ship(s2, 1000000, 10000, 5000, 0.5, lateral_grip=100000.0)

    # 2. Use very light steering (0.05) to isolate the lateral drift correction
    c_ice.set_input(forward=1.0, right=0.05)
    c_rail.set_input(forward=1.0, right=0.05)

    for _ in range(60):
        world.step(1/60.0)

    def get_lateral_slip(handle: int):
        if not (stats := world.get_body_stats(handle)):
            return 0.0
        _, rot, vel = stats
        qx, qy, qz, qw = rot
        # Calculate Local Right vector (X-axis of rotation)
        rx = 1 - 2 * (qy**2 + qz**2)
        ry = 2 * (qx * qy + qz * qw)
        rz = 2 * (qx * qz - qy * qw)
        # Dot product with velocity
        return vel[0] * rx + vel[1] * ry + vel[2] * rz

    slip_ice = abs(get_lateral_slip(s1))
    slip_rail = abs(get_lateral_slip(s2))

    # VALIDATION
    assert slip_ice > 0.05, f"Ice ship should be sliding. Got {slip_ice}"

    # Assert Rail ship has > 90% reduction in slip
    reduction = slip_rail / max(slip_ice, 0.001)
    assert reduction < 0.1, f"Grip failed to reduce slip sufficiently. Ratio: {reduction:.4f}. Rail: {slip_rail}, Ice: {slip_ice}"

def test_ship_terminal_velocity(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Verify that quadratic drag results in a terminal velocity."""
    world, sled, _ = ship_setup

    # Create a ship with very high drag
    drag_ship = world.create_ship(
        sled=sled, kp=1000, kd=100,
        throttle_force=100000.0, steer_speed=0,
        linear_drag=100.0 # Heavy resistance
    )

    drag_ship.set_input(forward=1.0, right=0.0)

    velocities = []
    # Run for 2 seconds
    for _ in range(120):
        world.step(1/60.0)
        if (v := world.get_velocity(sled)):
            velocities.append(v[2])

    # The acceleration should decrease over time
    accel_start = velocities[10] - velocities[0]
    accel_end = velocities[-1] - velocities[-11]

    assert accel_end < accel_start, "Drag should reduce acceleration over time"
    # Velocity should plateau (terminal velocity)
    assert velocities[-1] < 50.0, "Drag should prevent infinite acceleration"

def test_ship_destruction_mid_step(world: culverin.PhysicsWorld) -> None:
    """Verify that destroying a sled doesn't cause the Ship listener to segfault."""
    sled = world.create_body(pos=(0, 0, 0))
    ship = world.create_ship(sled, 100, 10, 1000, 1.0)

    world.step(1/60.0)

    # Destroy the body that the ship is controlling
    world.destroy_body(sled)

    # The next step should handle the sled_bid being invalid or the body being gone
    # without a C-level null pointer dereference.
    try:
        for _ in range(5):
            world.step(1/60.0)
            ship.set_input(forward=1.0, right=1.0)
    except Exception as e:
        pytest.fail(f"Ship controller failed to handle body destruction: {e}")

def test_ship_parameter_ranges(world: culverin.PhysicsWorld) -> None:
    """Test that extreme/unusual parameters don't explode the solver."""
    sled = world.create_body(pos=(0, 0, 0), mass=100)

    # Massive KP/KD or negative drag (should be clamped or handled)
    # Note: If C code doesn't clamp, this tests stability
    crazy_ship = world.create_ship(
        sled=sled,
        kp=99999999.0,
        kd=0.1,
        throttle_force=1e10,
        steer_speed=100.0,
        banking=10.0,
        lateral_grip=1e10,
        linear_drag=0.0
    )

    crazy_ship.set_input(forward=1.0, right=1.0)

    # If it doesn't crash or result in NaN, it's 'stable' enough
    for _ in range(10):
        world.step(1/60.0)

    if (stats := world.get_body_stats(sled)):
        pos, _, _ = stats
        # Check for NaN/Inf (using numpy or math)
        assert np.isfinite(pos).all(), "Ship parameters caused numerical explosion"

def test_ship_input_atomic_stability(ship_setup: tuple[culverin.PhysicsWorld, int, culverin.Ship]) -> None:
    """Stress test the atomic input setters with rapid changes."""
    world, _, controller = ship_setup

    # Rapidly toggle inputs
    for i in range(100):
        val = (i % 2) * 2.0 - 1.0 # Toggles between -1 and 1
        controller.set_input(forward=val, right=-val)
        # Step every few toggles
        if i % 10 == 0:
            world.step(1/60.0)

    # Ensure world is still ticking
    world.step(1/60.0)
    assert world.time > 0

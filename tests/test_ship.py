import pytest
import culverin
import math
import numpy as np

# --- FIXTURES ---

@pytest.fixture
def world():
    """Provides a clean physics world for each test."""
    return culverin.PhysicsWorld(settings={"max_bodies": 100})

@pytest.fixture
def ship_setup(world):
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

def test_ship_creation(ship_setup):
    """Verify ship is created and carries the correct handle."""
    world, sled, controller = ship_setup
    assert controller is not None
    assert world.is_alive(sled)

def test_ship_invalid_handle(world):
    """Ensure passing a bogus handle raises ValueError."""
    with pytest.raises(ValueError, match="Invalid sled handle"):
        world.create_ship(
            sled=999999, # Fake handle
            kp=1.0, kd=1.0, throttle_force=1.0, steer_speed=1.0
        )

def test_ship_stabilization(ship_setup):
    """Verify that the native C PD loop pulls the ship back to upright."""
    world, sled, controller = ship_setup
    
    # Force a 45 degree tilt
    world.set_rotation(sled, 0, 0, 0.382, 0.923)
    world.step(0)
    
    _, rot_start, _ = world.get_body_stats(sled)
    
    # Give the ship 1 full second (60 frames) to stabilize
    for _ in range(60):
        world.step(1/60.0)
        
    _, rot_end, _ = world.get_body_stats(sled)
    
    # The Z component (roll) should have significantly decreased
    assert abs(rot_end[2]) < abs(rot_start[2])
    # With 4M gain and 60 frames, it should easily be under 0.1
    assert abs(rot_end[2]) < 0.1

def test_ship_throttle(ship_setup):
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

def test_ship_steering(ship_setup):
    """Verify steering directly modifies angular velocity Y."""
    world, sled, controller = ship_setup
    
    # Set steer left
    controller.set_input(forward=0.0, right=1.0)
    
    world.step(1/60.0)
    
    # Check angular velocity from stats
    # stats: (pos, rot, vel) -> we need angular velocity separately or via custom C getter
    # For now, use the world.angular_velocities buffer directly
    idx = world.get_index(sled)
    avel_buffer = np.frombuffer(world.angular_velocities, dtype=np.float32).reshape(-1, 4)
    
    # Current angular velocity Y should match our steer_speed config (1.0)
    assert pytest.approx(avel_buffer[idx][1], abs=0.01) == 1.0

def test_ship_deallocation_safety(world):
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

def test_ship_multi_interpreter_isolation(world):
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
        
    v1 = world.get_velocity(s1_sled)
    v2 = world.get_velocity(s2_sled)
    
    assert v1[2] > 0.5
    assert v2[2] == 0.0
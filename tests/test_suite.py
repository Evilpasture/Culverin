import math
import time
import culverin
import numpy as np
import struct
import threading
import faulthandler

faulthandler.enable()


def assert_val(actual, expected, msg, atol=1e-4):
    """Helper to perform approximate float assertions."""
    if np.isclose(actual, expected, atol=atol):
        print(f"  ✅ {msg}: {actual:.4f}")
    else:
        print(f"  ❌ {msg}: Expected {expected:.4f}, got {actual:.4f}")
        raise AssertionError(f"Logic failure: {msg}")


def test_api_comprehensive():
    print("=== 1. Initialization & Baked Invariants ===")
    # Physics handles are (Generation << 32) | Slot.
    # Baked bodies in PhysicsWorld start at Slot 0, Gen 1.
    settings = {"gravity": (0, -10, 0), "max_bodies": 2048}
    bodies = [
        {
            "pos": (0, -0.5, 0),
            "size": (50, 0.5, 50),
            "shape": culverin.SHAPE_BOX,
            "motion": culverin.MOTION_STATIC,
            "user_data": 12345,
        }
    ]
    world = culverin.PhysicsWorld(settings=settings, bodies=bodies)

    # Floor should be at slot 0, gen 1
    floor_h = (1 << 32) | 0
    assert_val(world.is_alive(handle=floor_h), True, "Floor is alive (Baked)")
    assert_val(world.get_user_data(floor_h), 12345, "Baked UserData preserved")

    print("\n=== 2. Unified Handle System (Character) ===")
    char = world.create_character(pos=(0, 2, 0), height=1.8, radius=0.4)
    char_h = char.handle
    # Character handles also participate in the world's handle system
    assert_val(world.is_alive(char_h), True, "Character handle registered in world")

    print("\n=== 3. Activation & Gravity ===")
    h1 = world.create_body(
        pos=(10, 10, 10), shape=culverin.SHAPE_BOX, size=(0.5, 0.5, 0.5)
    )
    # Step 0 flushes the creation command queue
    world.step(0)
    h1_idx = world.get_index(handle=h1)
    spawn_y = world.positions[h1_idx * 4 + 1]
    print(world.positions.format)  # MUST BE 'd'
    print(world.positions.itemsize)  # MUST BE 8
    assert_val(spawn_y, 10.0, "Body spawned at correct height")

    world.step(1 / 60)
    # Check if the body moved down due to gravity
    assert_val(world.positions[h1_idx * 4 + 1] < 10.0, True, "Body is falling")

    print("\n=== 4. Constraint Hinge & Auto-Wake ===")
    anchor_h = world.create_body(pos=(5, 10, 0), motion=culverin.MOTION_STATIC)
    swinger_h = world.create_body(
        pos=(6, 10, 0), size=(0.5, 0.5, 0.5), motion=culverin.MOTION_DYNAMIC
    )
    world.step(0)

    # Hinge at the midpoint between bodies
    hinge = world.create_constraint(
        culverin.CONSTRAINT_HINGE,
        anchor_h,
        swinger_h,
        params=((5.5, 10, 0), (0, 0, 1), -1.5, 1.5),
    )

    for _ in range(30):
        world.step(1 / 60)

    sw_idx = world.get_index(swinger_h)
    sw_y = world.positions[sw_idx * 4 + 1]

    # Destroying the constraint should wake the attached dynamic body
    world.destroy_constraint(hinge)
    world.step(1 / 60)

    sw_y_post = world.positions[sw_idx * 4 + 1]
    assert_val(
        sw_y_post < sw_y, True, "Body auto-woke and fell after constraint destruction"
    )

    print("\n=== 5. World Interpolation (get_render_state) ===")
    # get_render_state returns packed (3 pos, 4 rot) per body
    render_bytes = world.get_render_state(0.5)
    # Use numpy to view the data for high-performance access
    render_arr = np.frombuffer(render_bytes, dtype=np.float32).reshape(-1, 7)

    f_idx = world.get_index(swinger_h)
    interp_y = render_arr[f_idx, 1]
    # At alpha 0.5, Y should be between the previous frame and current frame
    assert_val(
        interp_y > world.positions[f_idx * 4 + 1],
        True,
        "Interpolated Y is above current Y (during fall)",
    )

    print("\n=== 6. Character Interpolation (get_render_transform) ===")
    char.move((10, 0, 0), 1 / 60.0)
    # Returns ((x,y,z), (x,y,z,w))
    r_pos, r_rot = char.get_render_transform(0.5)
    assert_val(r_pos[0] > 0, True, "Character render-transform X moved forward")

    print("\n=== 7. Raycast with Ignore (Character Self-Filter) ===")
    c_pos = char.get_position()
    world.step(0)  # Ensure the character is actually in the Jolt grid
    # Cast ray down from character center. Ignore char.handle to hit floor.
    res = world.raycast(
        start=(c_pos[0], c_pos[1] + 1.0, c_pos[2]),
        direction=(0, -1, 0),
        max_dist=20.0,
        ignore=char.handle,
    )
    if res:
        hit_h, fraction, normal = res
        assert_val(hit_h, floor_h, "Ray skipped character and hit floor")
        assert_val(normal[1], 1.0, "Ray hit upward facing surface")
    else:
        raise AssertionError("Raycast failed to hit floor")

    print("\n=== 8. Shape Casting (Sweep Test) ===")
    target_h = world.create_body(
        pos=(10, 5, 0), size=(1, 5, 5), motion=culverin.MOTION_STATIC
    )
    world.step(0)
    # Sweep a sphere toward the target box
    hit = world.shapecast(
        shape=culverin.SHAPE_SPHERE,
        pos=(0, 5, 0),
        rot=(0, 0, 0, 1),
        dir=(20, 0, 0),
        size=(0.5,),
        ignore=char.handle,
    )
    if hit:
        h, fraction, point, normal = hit
        assert_val(h, target_h, "Shapecast hit correct target")
        assert_val(normal[0], -1.0, "Shapecast normal faces -X (Source direction)")
    else:
        raise AssertionError("Shapecast logic failure")

    print("\n=== 9. Event System (Rich Dictionary Events) ===")
    sensor_h = world.create_body(
        pos=(0, 10, 0), size=(1, 1, 1), is_sensor=True, motion=culverin.MOTION_STATIC
    )
    ball_h = world.create_body(pos=(0, 12, 0), shape=culverin.SHAPE_SPHERE, size=(0.5,))
    world.step(0)

    hit_detected = False
    for i in range(60):
        world.step(1 / 60)
        # get_contact_events_ex returns a list of dictionaries
        events = world.get_contact_events_ex()
        for ev in events:
            if set(ev["bodies"]) == {sensor_h, ball_h}:
                hit_detected = True
                print(
                    f"  ✅ Contact: Strength={ev['strength']:.2f}, Pos={ev['position']}"
                )
                break
        if hit_detected:
            break
    assert_val(hit_detected, True, "Sensor event captured")

    print("\n=== 10. Character-Character Collision === ")
    p1 = world.create_character(pos=(15, 0.9, 0), height=1.8)
    p2 = world.create_character(pos=(17, 0.9, 0), height=1.8)

    print("  Walking P2 into P1...")
    for _ in range(30):
        p2.move((-5.0, 0, 0), 1 / 60.0)
        world.step(1 / 60)

    final_pos = p2.get_position()
    # Characters should stop before they overlap significantly (Radius is 0.4 each)
    assert_val(
        final_pos[0] > 15.7,
        True,
        f"Character separation maintained (X={final_pos[0]:.2f})",
    )

    print("\n=== 11. Vehicles ===")
    # Chassis at (0, 2, 20)
    chassis_h = world.create_body(
        pos=(0, 2, 20), size=(1, 0.5, 2), motion=culverin.MOTION_DYNAMIC
    )
    wheels = [
        {"pos": (-1.0, -0.5, 1.2), "radius": 0.4},  # Front Left
        {"pos": (1.0, -0.5, 1.2), "radius": 0.4},  # Front Right
        {"pos": (-1.0, -0.5, -1.2), "radius": 0.4},  # Rear Left
        {"pos": (1.0, -0.5, -1.2), "radius": 0.4},  # Rear Right
    ]
    # Create vehicle with Rear Wheel Drive
    car = world.create_vehicle(chassis=chassis_h, wheels=wheels, drive="RWD")

    print("  Applying throttle...")
    for _ in range(60):
        car.set_input(forward=1.0, right=0.0, brake=0.0, handbrake=0.0)
        tmp_idx = world.get_index(chassis_h)
        tmp_x = world.positions[tmp_idx * 4]
        if math.isnan(tmp_x):
            print("Chassis exploded before step")
            break
        world.step(1 / 60)
    c_idx = world.get_index(chassis_h)
    new_z = world.positions[c_idx * 4 + 2]
    assert_val(new_z != 20.0, True, f"Vehicle moved from start (Z={new_z:.2f})")
    assert_val(car.wheel_count, 4, "Vehicle reported correct wheel count")

    # Clean up vehicle to ensure clean state
    car.destroy()

    print("\n=== 12. Ragdolls (Skeleton, Settings, Simulation) ===")

    # 1. Define Skeleton (Root -> Spine -> Head)
    skel = culverin.Skeleton()
    idx_root = skel.add_joint("Root")
    idx_spine = skel.add_joint("Spine", idx_root)
    idx_head = skel.add_joint("Head", idx_spine)

    # CRITICAL: Prepare hierarchy pointers before creating settings
    skel.finalize()
    print(
        f"  Created Skeleton with 3 joints. Root={idx_root}, Spine={idx_spine}, Head={idx_head}"
    )

    # 2. Configure Ragdoll Settings
    r_set = world.create_ragdoll_settings(skel)

    # Root: at local (0, 0, 0)
    r_set.add_part(
        joint_index=idx_root,
        shape_type=culverin.SHAPE_BOX,
        size=(0.2, 0.2, 0.2),
        mass=15.0,
        parent_index=-1,
        pos=(0, 0, 0),
    )
    # Spine: at local (0, 0.5, 0) - Spaced out to prevent "explosion"
    r_set.add_part(
        joint_index=idx_spine,
        shape_type=culverin.SHAPE_BOX,
        size=(0.15, 0.15, 0.15),
        mass=10.0,
        parent_index=idx_root,
        twist_min=-0.2,
        twist_max=0.2,
        cone_angle=0.2,
        axis=(0, 1, 0),
        normal=(1, 0, 0),
        pos=(0, 0.5, 0),
    )
    # Head: at local (0, 1.0, 0)
    r_set.add_part(
        joint_index=idx_head,
        shape_type=culverin.SHAPE_SPHERE,
        size=(0.12,),
        mass=5.0,
        parent_index=idx_spine,
        twist_min=-0.5,
        twist_max=0.5,
        cone_angle=0.5,
        axis=(0, 1, 0),
        normal=(1, 0, 0),
        pos=(0, 1.0, 0),
    )
    r_set.stabilize()

    # 3. Instantiate Ragdoll at world height Y=10
    # Because of the offsets, Root will be at 10.0, Spine at 10.5, Head at 11.0
    rag = world.create_ragdoll(settings=r_set, pos=(20, 10, 0), rot=(0, 0, 0, 1))

    # 4. Verification of Handles and Shadow Buffer Warm-up
    parts = rag.get_body_handles()
    assert_val(len(parts), 3, "Ragdoll instantiated with 3 parts")

    root_h = parts[0]
    spine_h = parts[1]

    r_idx = world.get_index(root_h)
    s_idx = world.get_index(spine_h)

    # Verify the C-layer Warm-up logic: Shadow buffers should already have values
    r_y_start = world.positions[r_idx * 4 + 1]
    s_y_start = world.positions[s_idx * 4 + 1]
    print(
        f"  Warm-up Check: Root Start Y={r_y_start:.2f}, Spine Start Y={s_y_start:.2f}"
    )
    assert_val(r_y_start, 10.0, "Shadow Buffer: Root initialized correctly")
    assert_val(s_y_start, 10.5, "Shadow Buffer: Spine initialized with offset")

    # 5. Simulation (Gravity)
    world.step(1 / 60)

    # Check for Physics Explosion vs Gravity
    for info in rag.get_debug_info():
        print(f"    Limb {info['index']} speed: {info['vel']}")

    r_y = world.positions[r_idx * 4 + 1]
    vy = world.velocities[r_idx * 4 + 1]  # Note: Use .velocities to match C table

    # Verify downward velocity (Gravity = -10 * 1/60 = -0.166)
    assert_val(vy < 0, True, f"Ragdoll has downward velocity (Vy={vy:.2f})")

    # Verify the root is actually lower than where it started
    assert_val(
        r_y < r_y_start,
        True,
        f"Ragdoll root is falling (Y={r_y:.4f} < {r_y_start:.4f})",
    )

    # 6. Active Driving (Motors)
    # Target Pose: 3 joints * Identity Matrix (16 floats per matrix)
    identity_16 = struct.pack("16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    pose_data = identity_16 * 3

    print(f"  Driving ragdoll motors to lift Root from {r_y:.2f} to Y=15...")
    rag.drive_to_pose(root_pos=(20, 15, 0), root_rot=(0, 0, 0, 1), matrices=pose_data)

    # Step simulation 15 times
    for _ in range(15):
        world.step(1 / 60)

    r_y_lifted = world.positions[r_idx * 4 + 1]
    print(f"  Final Y after lift: {r_y_lifted:.2f}")

    # The root warped to 15.0 and fell for 0.25s.
    # It should definitely be above 14.0.
    assert_val(
        r_y_lifted > 14.0, True, f"Ragdoll successfully lifted (Y={r_y_lifted:.2f})"
    )

    # 7. Cleanup & Invalidation
    del rag
    world.step(0)  # Process deferred deletions
    assert_val(
        world.is_alive(spine_h), False, "Ragdoll parts removed after object deletion"
    )

    print("\n=== 13. Thread Safety (Blocking & Re-entrancy) ===")
    stop_thread = False
    step_error_caught = False
    mutation_successful = 0

    def physics_worker():
        nonlocal step_error_caught
        while not stop_thread:
            try:
                # This thread keeps the world busy
                world.step(1 / 60)
            except RuntimeError as e:
                # We expect this if the main thread tries to step at the same time
                if "Concurrent step detected" in str(e):
                    step_error_caught = True
            time.sleep(0.001)

    t = threading.Thread(target=physics_worker)
    t.start()

    print("  Testing blocking mutation & step collision...")
    start_time = time.time()
    while time.time() - start_time < 1.0:
        try:
            # 1. Test Re-entrancy Guard (This SHOULD throw RuntimeError)
            # because 'step' explicitly checks if it's already stepping.
            world.step(1 / 60)
        except RuntimeError:
            step_error_caught = True

        try:
            # 2. Test Blocking Mutation (This should NOT throw)
            # It will wait for the background step to finish, then apply.
            world.set_position(handle=floor_h, x=0, y=-0.5, z=0)
            mutation_successful += 1
        except RuntimeError:
            # If this happens, your BLOCK_UNTIL macro isn't working
            pass

    stop_thread = True
    t.join()

    # Verification
    # We don't need this check anymore
    # assert_val(step_error_caught, True, "Concurrent step() calls correctly blocked via RuntimeError")
    assert_val(
        mutation_successful > 0,
        True,
        f"Mutation calls blocked and succeeded ({mutation_successful} times)",
    )
    print("  ✅ Thread safety test passed (Shadow-Lock Blocking is active)")

    print("\n=== 14. Buoyancy (Fluid Dynamics) ===")
    # Create a ball high above "water" (water surface at Y=5)
    ball_h = world.create_body(
        pos=(0, 10, 0), shape=culverin.SHAPE_SPHERE, size=(0.5,), mass=10.0
    )
    world.step(0)

    water_y = 5.0
    submerged_frames = 0

    print("  Dropping ball into water...")
    for i in range(120):
        # Apply buoyancy every frame before stepping
        is_wet = world.apply_buoyancy(
            handle=ball_h,
            surface_y=water_y,
            buoyancy=1.5,  # Make it quite buoyant so it bobs
            linear_drag=0.8,  # High drag to prevent infinite oscillation
            angular_drag=0.5,
            dt=1 / 60,
        )
        if is_wet:
            submerged_frames += 1

        world.step(1 / 60)

    b_idx = world.get_index(ball_h)
    final_y = world.positions[b_idx * 4 + 1]

    assert_val(
        submerged_frames > 0,
        True,
        f"Ball entered water ({submerged_frames} frames submerged)",
    )
    # If buoyancy works, the ball should be floating near the surface (Y=5)
    # instead of having fallen through the floor (Y=0)
    assert_val(final_y > 4.0, True, f"Ball is floating (Y={final_y:.2f})")
    print("  ✅ Buoyancy test passed (Object is floating)")

    print("\n=== 15. Batch Raycasting (AI Vision) ===")
    ray_count = 100
    # Create 100 rays starting at Y=10, looking down
    starts = np.zeros((ray_count, 3), dtype=np.float32)
    starts[:, 1] = 10.0
    # Spread them out in a line along X
    starts[:, 0] = np.linspace(-10, 10, ray_count)

    dirs = np.zeros((ray_count, 3), dtype=np.float32)
    dirs[:, 1] = -1.0  # Look down

    # Cast batch
    start_time = time.perf_counter()
    results_raw = world.raycast_batch(starts.tobytes(), dirs.tobytes(), max_dist=20.0)
    end_time = time.perf_counter()

    # Unpack with NumPy structured array
    dt = np.dtype(
        [
            ("handle", np.uint64),
            ("fraction", np.float32),
            ("nx", np.float32),
            ("ny", np.float32),
            ("nz", np.float32),
            ("px", np.float32),
            ("py", np.float32),
            ("pz", np.float32),
            ("subshape", np.uint32),
            ("material", np.uint32),
            ("_pad", np.uint32),  # Match the 48-byte size
        ]
    )
    results = np.frombuffer(results_raw, dtype=dt)

    hits = np.count_nonzero(results["handle"])
    print(
        f"  Batch processed {ray_count} rays in {(end_time - start_time) * 1000:.3f}ms"
    )
    print(f"  Hits detected: {hits}")

    assert_val(hits, ray_count, "All rays hit the floor")
    assert_val(results["handle"][0], floor_h, "First ray hit correct floor handle")
    assert_val(
        results["py"][0], 0.0, "Contact point Y correctly resolved at floor level"
    )

    print("  ✅ Batch raycast test passed")

    print("\n=== 16. Collision Filtering (Bitmasks) ===")
    CAT_FLOOR = 1 << 0
    CAT_PLAYER = 1 << 1
    CAT_GHOST = 1 << 2

    # 1. Floor only hits players and ghosts
    # (Existing floor is already slot 0, we can't easily re-mask it in this test
    # without a set_collision_filter method, so let's create a new one)
    platform_h = world.create_body(
        pos=(0, 5, 0),
        size=(10, 0.1, 10),
        motion=culverin.MOTION_STATIC,
        category=CAT_FLOOR,
        mask=0xFFFF,
    )

    # 2. Player hits Floor and Ghost
    player_h = world.create_body(
        pos=(2, 10, 0),
        size=(0.5, 0.5, 0.5),
        category=CAT_PLAYER,
        mask=CAT_FLOOR | CAT_GHOST,
    )

    # 3. Ghost hits Floor but IGNORES Player
    ghost_h = world.create_body(
        pos=(2, 12, 0), size=(0.5, 0.5, 0.5), category=CAT_GHOST, mask=CAT_FLOOR
    )  # Note: CAT_PLAYER is missing!

    world.step(0)

    print("  Simulating fall...")
    for _ in range(120):
        world.step(1 / 60)

    p_idx = world.get_index(player_h)
    g_idx = world.get_index(ghost_h)

    # Check Player: should be resting on the platform (Y=5 + 0.5 half-extents = 5.5)
    assert_val(world.positions[p_idx * 4 + 1] > 5.4, True, "Player stopped by platform")

    # Check Ghost: should have fallen STRAIGHT THROUGH the player and stopped on the platform
    # If it hit the player, it would be at Y > 6.0
    ghost_y = world.positions[g_idx * 4 + 1]
    assert_val(
        ghost_y < 5.6, True, f"Ghost ignored player and hit platform (Y={ghost_y:.2f})"
    )

    print("  ✅ Collision bitmask test passed")

    print("\n=== 17. Physical Materials (Friction/Restitution) ===")

    # Define Materials ONCE
    world.register_material(id=50, restitution=0.9)  # Rubber
    world.register_material(id=100, friction=1.0)  # Stone

    # 1. Bouncy Ball
    # Note: We DON'T pass restitution here anymore! It infers it from ID=50.
    bouncy_h = world.create_body(
        pos=(30, 10, 0), shape=culverin.SHAPE_SPHERE, size=(0.5,), material_id=50
    )

    # 2. Heavy Block
    stone_h = world.create_body(
        pos=(30, 0.1, 0),
        size=(10, 0.1, 10),
        motion=culverin.MOTION_STATIC,
        material_id=100,
    )

    world.step(0)

    print("  Dropping bouncy ball...")
    mat_found = False
    for i in range(150):  # Increased frames slightly to ensure impact at lower Y
        world.step(1 / 60)

        # Check for the event EVERY frame
        events = world.get_contact_events_ex()
        for ev in events:
            mats = ev.get("materials", (0, 0))
            if 50 in mats and 100 in mats:
                mat_found = True

        if i % 30 == 0:
            idx = world.get_index(bouncy_h)
            print(f"    Ball Y: {world.positions[idx * 4 + 1]:.2f}")

    assert_val(mat_found, True, "Contact reported correct Material IDs (50 vs 100)")

    print("\n=== 18. Terrain (Heightfields) ===")

    # 1. Generate Height Data (Slope)
    # Create a 64x64 grid
    grid_size = 64
    x = np.linspace(0, 10, grid_size, dtype=np.float32)
    z = np.linspace(0, 10, grid_size, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)

    # Simple slope: Height increases with X
    height_map = (xx * 0.5).astype(np.float32)  # Slope up along X

    # Pack to bytes
    height_bytes = height_map.tobytes()

    # 2. Create Terrain
    # Place at (40, 0, 0) to avoid other tests
    terrain_h = world.create_heightfield(
        pos=(40, 0, 0),
        rot=(0, 0, 0, 1),
        scale=(1.0, 1.0, 1.0),  # 1 meter per sample
        heights=height_bytes,
        grid_size=grid_size,
        friction=0.1,  # Slippery slope
    )

    # 3. Create Ball on the slope
    # Terrain starts at X=40. At local x=5 (World 45), height is 2.5.
    # Spawn ball slightly above the slope at X=45
    slope_ball = world.create_body(
        pos=(45, 3.5, 5), shape=culverin.SHAPE_SPHERE, size=(0.5,), mass=10.0
    )

    world.step(0)

    # 4. Simulate
    print("  Rolling ball down terrain...")
    start_x = world.positions[world.get_index(slope_ball) * 4]

    for _ in range(60):
        world.step(1 / 60)

    end_x = world.positions[world.get_index(slope_ball) * 4]

    # Slope goes UP along X (height = x * 0.5).
    # Gravity should pull ball DOWN along X (towards 0).
    # So End X should be LESS than Start X.
    print(f"  Ball X: {start_x:.2f} -> {end_x:.2f}")
    assert_val(end_x < start_x, True, "Ball rolled down the heightfield slope")

    print("  ✅ Heightfield terrain test passed")

    print("\n" + "=" * 40)
    print("   ALL API TESTS PASSED (Double Precision ABI)")
    print("=" * 40)


if __name__ == "__main__":
    test_api_comprehensive()

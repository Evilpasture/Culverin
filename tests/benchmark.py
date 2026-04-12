import argparse
import array
import os
import random
import threading
import time

import numpy as np
import psutil

import culverin


def get_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def run_leak_test(iterations: int = 50000) -> None:
    print("\n=== CULVERIN MEMORY LEAK TEST ===")
    world = culverin.PhysicsWorld(settings={"max_bodies": 10000})
    verts = array.array("f", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).tobytes()
    indices = array.array("I", [0, 1, 2]).tobytes()

    start_ram = get_ram_mb()
    print(f"Starting RAM: {start_ram:.2f} MB")

    start_time = time.time()
    for i in range(iterations):
        m = world.create_mesh_body(pos=(0, 0, 0), rot=(0, 0, 0, 1), vertices=verts, indices=indices)
        world.destroy_body(m)
        if i % 1000 == 0:
            world.step(0)  # Flush command queue

    world.step(0)  # Final flush

    end_ram = get_ram_mb()
    duration = time.time() - start_time
    leakage = end_ram - start_ram

    print(f"Completed {iterations} mesh cycles in {duration:.2f}s")
    print(f"RAM Shift: {leakage:+.2f} MB")
    if abs(leakage) > 5.0:
        print("❌ WARNING: Potential Memory Leak Detected")
    else:
        print("✅ SUCCESS: Memory is stable")


def run_threading_benchmark(duration: float = 5.0, num_bodies: int = 500) -> None:
    print("\n=== CULVERIN REALISTIC SIMULATION BENCHMARK ===")
    print(f"Simulating {num_bodies} active dynamic bodies across multiple cores for {duration}s...")

    # 1. SETUP: World with plenty of headroom to avoid RuntimeErrors
    world = culverin.PhysicsWorld(
        settings={"max_bodies": num_bodies + 2000, "max_pairs": num_bodies * 8}
    )

    # Create static floor (Giant Box)
    world.create_body(
        pos=(0, -5, 0), size=(500, 1, 500), shape=culverin.SHAPE_BOX, motion=culverin.MOTION_STATIC
    )

    # Create Dynamic Grid (Spawned in the air so they crash down)
    pos_list: list[tuple[float, float, float]] = []
    grid_size = int(np.cbrt(num_bodies)) + 1
    spacing = 1.5
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if len(pos_list) < num_bodies:
                    pos_list.append((x * spacing - 10, y * spacing + 10, z * spacing - 10))

    handles_raw = world.create_bodies_batch(
        positions=pos_list,
        sizes=[[0.5, 0.5, 0.5]] * len(pos_list),
        shape_type=culverin.SHAPE_BOX,
        motion_type=culverin.MOTION_DYNAMIC,
    )
    # Store handles in a mutable numpy array for thread-safe-ish updating
    handles = np.array(handles_raw, dtype=np.uint64)
    world.step(0)  # Initial push to BroadPhase

    # Thread States
    running = True
    stats = {"steps": 0, "rays": 0, "contacts": 0, "resets": 0, "mutations": 0}

    # --- THREAD 1: THE CORE STEPPER ---
    def worker_stepper() -> None:
        while running:
            try:
                # Run as fast as the CPU allows
                world.step(1.0 / 60.0)
                stats["steps"] += 1
            except RuntimeError:
                # Concurrent step/lock failure - just skip this loop
                pass

    # --- THREAD 2: SENSORS (RAYCASTS) ---
    def worker_sensors() -> None:
        batch_size = 500
        starts = array.array("f", [0.0] * (batch_size * 3))
        dirs = array.array("f", [0.0, -1.0, 0.0] * batch_size)
        while running:
            starts[1] = random.uniform(20, 50)  # Randomize height
            world.raycast_batch(starts=starts, directions=dirs, max_dist=100.0)
            stats["rays"] += batch_size
            time.sleep(0.01)  # ~100Hz

    # --- THREAD 3: GAMEPLAY LOGIC (ANTI-SLEEP & MEMORYVIEW) ---
    def worker_housekeeper() -> None:
        # Wrap the raw memoryview in a NumPy array
        pos_data = np.frombuffer(world.positions, dtype=np.float64).reshape(-1, 4)

        while running:
            # Find fallen bodies
            fallen_indices = np.where(pos_data[:num_bodies, 1] < 0.0)[0]

            for idx in fallen_indices:
                h = int(handles[idx])

                # ADDED: Check if the handle is still valid before calling C
                if world.is_alive(h):
                    try:
                        world.set_position(
                            h, random.uniform(-10, 10), 20.0, random.uniform(-10, 10)
                        )
                        world.set_linear_velocity(h, 0, 0, 0)
                        stats["resets"] += 1
                    except ValueError:
                        # Fallback: if it was destroyed between the is_alive check
                        # and the set_position call, just ignore it.
                        pass

            time.sleep(0.5)

    # --- THREAD 4: THE MUTATOR (CONTROLLED HAMMER) ---
    def worker_mutator() -> None:
        while running:
            try:
                # Destroy 5 bodies, create 5 bodies.
                # Stresses the BroadPhase AABB tree and synchronization locks.
                idx_to_replace = [random.randint(0, num_bodies - 1) for _ in range(5)]
                victims = [int(handles[i]) for i in idx_to_replace]

                world.destroy_bodies_batch(handles=victims)

                new_h = world.create_bodies_batch(
                    positions=[(0, 40, 0)] * 5,
                    sizes=[[1, 1, 1]] * 5,
                    shape_type=culverin.SHAPE_SPHERE,
                    motion_type=culverin.MOTION_DYNAMIC,
                )

                for i, h in enumerate(new_h):
                    handles[idx_to_replace[i]] = h
                    world.activate(int(h))  # Force awake

                stats["mutations"] += 5
            except RuntimeError:
                pass  # Lock contention or pool limit hit, try again later

            time.sleep(0.05)  # Run 20 times a second

    # Start Threads
    threads = [
        threading.Thread(target=worker_stepper, name="Stepper"),
        threading.Thread(target=worker_sensors, name="Sensors"),
        threading.Thread(target=worker_housekeeper, name="Housekeeper"),
        threading.Thread(target=worker_mutator, name="Mutator"),
    ]

    for t in threads:
        t.start()

    # Monitoring Loop
    start_t = time.time()
    try:
        while time.time() - start_t < duration:
            time.sleep(1.0)
            # Fetch raw events to clear the buffer (simulate event handling)
            contacts = world.get_contact_events_raw()
            if contacts:
                stats["contacts"] += 1

            print(
                f"[@ {time.time() - start_t:.1f}s] "
                f"Steps: {stats['steps']} | "
                f"Rays: {stats['rays']} | "
                f"Mutations: {stats['mutations']} | "
                f"Resets: {stats['resets']}"
            )

            if stats["steps"] == 0:
                print("❌ CRITICAL: Physics Thread is completely deadlocked")
                break
    finally:
        running = False
        for t in threads:
            t.join(timeout=1.0)

    fps = stats["steps"] / duration
    print("\n✅ TEST COMPLETE")
    print(f"Final Performance: {fps:.2f} FPS")
    print(f"Total Steps: {stats['steps']}")
    print(f"Total Raycasts: {stats['rays']}")


def run_churn_test(duration: float = 10.0) -> None:
    # There is a known memory issue. will investigate...
    print("\n=== CULVERIN FRAGMENTATION (CHURN) TEST ===")

    # Start with a world limited to 2000 bodies to force frequent re-use
    MAX_LIMIT = 2000
    world = culverin.PhysicsWorld(settings={"max_bodies": MAX_LIMIT})
    handles: list[int] = []

    world.create_body(
        pos=(0, -5, 0), size=(500, 1, 500), shape=culverin.SHAPE_BOX, motion=culverin.MOTION_STATIC
    )

    # Initial population: Fill to 50%
    print("-> Initializing world to 50% capacity...")
    handles = [
        world.create_body(pos=(random.random() * 10, 0, random.random() * 10))
        for _ in range(MAX_LIMIT // 2)
    ]
    world.step(0)

    start_t = time.time()
    ops = 0
    skipped = 0

    print(f"-> Starting Churn for {duration}s...")
    try:
        while time.time() - start_t < duration:
            # 1. RANDOM DESTRUCTION (50 bodies)
            # These slots become PENDING_DESTROY (still taking space)
            num_to_kill = min(len(handles), 50)
            for _ in range(num_to_kill):
                idx = random.randrange(len(handles))
                world.destroy_body(handles.pop(idx))

            # 2. THE PURGE
            # This flushes the command queue and actually frees the slots
            world.step(0.016)

            # 3. GRACEFUL SPAWNING
            # We look before we leap using your new getter
            available = world.remaining_capacity
            num_to_spawn = min(available, 50)

            for _ in range(num_to_spawn):
                h = world.create_body(pos=(random.random() * 10, 10, random.random() * 10))
                handles.append(h)
                ops += 1

            if num_to_spawn < 50:
                skipped += 50 - num_to_spawn

            if ops % 1000 == 0 and ops > 0:
                # Telemetry
                print(
                    f"  Cycle: {ops:6} ops | "
                    f"Active: {world.count:4}/{world.max_bodies} | "
                    f"RAM: {get_ram_mb():.2f}MB"
                )
            # Add a tiny sleep to allow the OS to reclaim memory and
            # let Jolt threads finish their work.
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping churn test...")
    finally:
        print("-> Churn finished. Waiting for cleanup...")
        handles.clear()
        for _ in range(10):
            world.step(0.1)  # Force multiple steps to flush buffers
            time.sleep(0.5)
            print(f"Post-Cleanup RAM: {get_ram_mb():.2f}MB")

    end_t = time.time()
    print("\n✅ CHURN COMPLETE")
    print(f" - Duration:      {end_t - start_t:.2f}s")
    print(f" - Total Created: {ops}")
    print(f" - Load Shedding: {skipped} spawns rejected gracefully (at limit)")
    print(f" - Final RAM:     {get_ram_mb():.2f}MB")


def run_soft_body_benchmark(duration: float = 10.0, num_bodies: int = 50, segments: int = 6) -> None:
    print("\n=== CULVERIN SOFT BODY STRESS TEST ===")
    print(f"Goal: Simulate {num_bodies} jelly cubes ({segments}^3 vertices each) for {duration}s")

    world = culverin.PhysicsWorld(settings={"max_bodies": 2000})
    world.create_body(pos=(0, -2, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)

    # 1. Measure Topology Setup (Shared Settings)
    t0 = time.perf_counter()
    settings = culverin.SoftBodySharedSettings()

    # Pre-generate coordinates to avoid nested loop overhead in Python
    lin = np.linspace(-1.0, 1.0, segments)
    grid = np.stack(np.meshgrid(lin, lin, lin), axis=-1).reshape(-1, 3)

    v_count = 0
    for pos in grid:
        # FIXED: Added required inv_mass argument
        settings.add_vertex(pos=tuple(pos), inv_mass=1.0)
        v_count += 1

    # Create simple structural integrity (connecting vertices in sequence)
    # Using distinct indices (i, i+1, i+2) to satisfy the C-guard
    for i in range(v_count - 2):
        settings.add_face(v1=i, v2=i + 1, v3=i + 2)

    # Standard granular setup
    settings.create_constraints(compliance=0.0001, bend_type=culverin.BEND_DISTANCE)
    settings.optimize()

    topology_time = (time.perf_counter() - t0) * 1000
    print(f"-> SharedSettings built in {topology_time:.2f}ms ({v_count} vertices)")

    handles: list[int] = []
    start_ram = get_ram_mb()
    dtype = np.float64 if culverin.USE_DOUBLE_PRECISION else np.float32

    stats = {"steps": 0, "verts_synced": 0}
    start_t = time.time()

    print("-> Starting simulation loop...")
    try:
        while time.time() - start_t < duration:
            # Maintain Population
            while len(handles) < num_bodies:
                h = world.create_soft_body(
                    shared_settings=settings,
                    pos=(random.uniform(-10, 10), random.uniform(10, 20), random.uniform(-10, 10)),
                    rot=(0, 0, 0, 1),
                    pressure=100.0,
                    linear_damping=0.2,
                    num_iterations=15,
                )
                handles.append(h)

            world.step(1 / 60.0)
            stats["steps"] += 1

            # Stress the Sync Layer
            for h in handles:
                view = world.get_soft_body_vertices(h)
                # This forces a read of the synchronized C memory into NumPy
                _ = np.frombuffer(view, dtype=dtype)
                stats["verts_synced"] += v_count

            # Churn to test memory reclamation of SoftBodyShadow buffers
            if stats["steps"] % 5 == 0:
                for _ in range(2):
                    if handles:
                        world.destroy_body(handles.pop(0))

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

    end_ram = get_ram_mb()
    total_time = time.time() - start_t

    print("\n✅ SOFT BODY RESULTS")
    print(f" - Performance:   {stats['steps'] / total_time:.2f} FPS")
    print(f" - Vertex Sync:  {stats['verts_synced'] / total_time / 1e6:.2f} Million Verts/sec")
    print(f" - RAM Usage:     {end_ram:.2f} MB (Delta: {end_ram - start_ram:+.2f} MB)")

    if abs(end_ram - start_ram) > 20.0:
        print("⚠️ WARNING: Significant RAM growth. Vertex shadow buffers might be leaking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Culverin Diagnostic Tools")
    parser.add_argument("--leak", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--churn", action="store_true")
    parser.add_argument("--soft", action="store_true", help="Run Soft Body Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.all or args.leak:
        run_leak_test()
    if args.all or args.stress:
        run_threading_benchmark()
    if args.all or args.churn:
        run_churn_test()
    if args.all or args.soft:
        run_soft_body_benchmark()

    if not any([args.all, args.leak, args.stress, args.churn, args.soft]):
        print("Please specify a test...")

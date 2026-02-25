import threading
import time
import random
import array
import numpy as np
import culverin
import sys

# --- CONFIGURATION ---
DURATION = 10.0
NUM_BODIES = 2000
NUM_QUERY_THREADS = 6  # Increased: Let's saturate the cores
BATCH_SIZE = 2000  # Increased: Reduce GIL overhead
MOVER_INTENSITY = 100  # Bodies moved per loop
HAMMER_BATCH_SIZE = 100  # Number of bodies to recycle per loop


def stress_test_v3():
    print(f"=== INITIALIZING MEGA-BATCH V3 ({DURATION}s) ===")

    # Enable aggressive culling to stress the "active" logic
    world = culverin.PhysicsWorld(settings={
        "max_bodies": NUM_BODIES + 10000,
        "max_pairs": NUM_BODIES * 2
    })

    # 1. SETUP
    print(f"-> Generating {NUM_BODIES} bodies...")
    rand_pos = np.random.uniform(-500, 500, (NUM_BODIES, 3)).astype(np.float32)
    # Stack them high so they might wake up if we disturb them
    rand_pos[:, 1] = np.linspace(0, 1000, NUM_BODIES)

    handles_raw = world.create_bodies_batch(
        positions=rand_pos.tolist(),
        sizes=[[0.5, 0.5, 0.5]] * NUM_BODIES,
        shape_type=culverin.SHAPE_BOX,
        motion_type=culverin.MOTION_DYNAMIC  # DYNAMIC! Let's see if Sleep works.
    )

    # Lock handles array for thread safety (NumPy arrays are generally thread-safe for reads)
    handles = np.array(handles_raw, dtype=np.uint64)
    print(f"-> Bodies created. Waking up Jolt...")

    # 2. SETTLE PHASE
    # Let them fall asleep so we test the "Selective Sync" transition
    for _ in range(60):
        world.step(1 / 60.0)
    print("-> World settled. Starting chaos...")

    # --- WORKERS ---
    running = True
    stats = {"steps": 0, "queries": 0, "mutations": 0, "moves": 0}

    def worker_stepper():
        while running:
            # Run as fast as CPU allows
            world.step(1 / 60.0)
            stats["steps"] += 1

    def worker_querier():
        # Pre-allocate large buffers to avoid Python GC overhead inside loop
        starts = array.array('f', [0.0] * (BATCH_SIZE * 3))
        dirs = array.array('f', [0.0, -1000.0, 0.0] * BATCH_SIZE)

        while running:
            # Jitter the start Y to prevent caching exact results
            starts[1] = random.uniform(500, 1000)
            # Parallel Raycast (GIL Released)
            world.raycast_batch(starts=starts, directions=dirs, max_dist=2000.0)
            stats["queries"] += 1

    def worker_hammer():
        """Batch-replaces bodies to stress the command queue and memory management."""
        while running:
            # 1. Select a batch of random victims
            # random.sample is faster than a loop of random.randint for collection
            v_indices = [random.randint(0, NUM_BODIES - 1) for _ in range(HAMMER_BATCH_SIZE)]
            victims = [int(handles[idx]) for idx in v_indices]

            # 2. BATCH DESTROY
            # Marks 100 bodies as dead in one shadow_lock cycle
            world.destroy_bodies_batch(handles=victims)

            # 3. BATCH CREATE
            # Prepare data for 100 new spheres
            new_pos = np.random.uniform(-50, 50, (HAMMER_BATCH_SIZE, 3)).astype(np.float32)
            new_pos[:, 1] = 1200.0  # Spawn height

            # This replaces 100 world.create_body calls with ONE call
            new_handles_raw = world.create_bodies_batch(
                positions=new_pos.tolist(),
                sizes=[[1.0, 1.0, 1.0]] * HAMMER_BATCH_SIZE,
                shape_type=culverin.SHAPE_SPHERE,
                motion_type=culverin.MOTION_DYNAMIC
            )

            # 4. UPDATE LOCAL CACHE
            for i, idx in enumerate(v_indices):
                handles[idx] = new_handles_raw[i]

            stats["mutations"] += HAMMER_BATCH_SIZE

            # Pace the hammer so it doesn't starve the physics stepper of lock time
            time.sleep(0.01)

    def worker_mover():
        while running:
            # Teleport bodies to force "Wake Up" logic in Jolt
            for _ in range(MOVER_INTENSITY):
                v_idx = random.randint(0, NUM_BODIES - 1)
                try:
                    h = int(handles[v_idx])
                    # Teleporting a Dynamic body forces it active
                    world.set_position(handle=h, x=0, y=1100, z=0)
                except:
                    pass
            stats["moves"] += MOVER_INTENSITY
            time.sleep(0.01)

    # --- LAUNCH ---
    threads = [threading.Thread(target=worker_stepper, name="Stepper")]
    for i in range(NUM_QUERY_THREADS):
        threads.append(threading.Thread(target=worker_querier, name=f"Ray-{i}"))
    threads.append(threading.Thread(target=worker_hammer, name="Hammer"))
    threads.append(threading.Thread(target=worker_mover, name="Mover"))

    for t in threads: t.start()

    # --- INSTANTANEOUS MONITORING ---
    start_time = time.time()
    last_time = start_time
    last_steps = 0
    last_queries = 0

    try:
        while time.time() - start_time < DURATION:
            time.sleep(1.0)
            now = time.time()
            dt = now - last_time

            curr_steps = stats["steps"]
            curr_queries = stats["queries"]

            fps = (curr_steps - last_steps) / dt
            rps = ((curr_queries - last_queries) * BATCH_SIZE) / dt

            print(f"[@ {now - start_time:.1f}s] "
                  f"FPS: {fps:.1f} | "
                  f"Rays/s: {rps / 1000:.1f}k | "
                  f"Mutations: {stats['mutations']} | "
                  f"Moves: {stats['moves']}")

            last_time = now
            last_steps = curr_steps
            last_queries = curr_queries

    finally:
        running = False
        for t in threads: t.join()

    print("=== TEST COMPLETE ===")


if __name__ == "__main__":
    stress_test_v3()
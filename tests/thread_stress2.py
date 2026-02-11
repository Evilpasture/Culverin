import threading
import time
import random
import array
import numpy as np
import culverin
import sys
import faulthandler

faulthandler.enable()

# --- CONFIGURATION ---
DURATION = 10.0
NUM_BODIES = 1_000_000  # One Million Bodies
NUM_QUERY_THREADS = 4
BATCH_SIZE = 200  # Raycasts per C call


def stress_test():
    print(f"=== INITIALIZING MEGA-STRESS TEST ({DURATION}s) ===")

    # Initialize world with high capacity to avoid reallocations
    world = culverin.PhysicsWorld(settings={
        "max_bodies": NUM_BODIES + 1000,
        "max_pairs": NUM_BODIES * 2
    })

    # 1. MEGA BATCH PRE-POPULATION
    print(f"-> Generating {NUM_BODIES} positions with NumPy...")
    # Generate all random data at once (vectorized)
    rand_pos = np.random.uniform(-50, 50, (NUM_BODIES, 3)).astype(np.float32)
    rand_pos[:, 1] += 60.0  # Shift Y up

    # Pre-allocate Handle array (uint64)
    handles = np.zeros(NUM_BODIES, dtype=np.uint64)

    print(f"-> Creating bodies (Native loop start)...")
    start_init = time.perf_counter()
    for i in range(NUM_BODIES):
        # Every 1000th body is Kinematic (so we can move it)
        # The rest are Static (to keep the solver fast)
        m_type = culverin.MOTION_KINEMATIC if i % 1000 == 0 else culverin.MOTION_STATIC

        handles[i] = world.create_body(
            pos=tuple(rand_pos[i]),
            size=(0.5, 0.5, 0.5),
            motion=m_type
        )
        if i % 100000 == 0:
            print(f"   Progress: {i / NUM_BODIES * 100:.0f}%")

    print(f"-> Initialization took {time.perf_counter() - start_init:.2f}s")
    print(f"-> Priming World (The First Step with 1M bodies is very heavy)...")
    start_prime = time.perf_counter()
    # This is likely where it is sitting for minutes
    world.step(0.016)
    print(f"-> First step completed in {time.perf_counter() - start_prime:.2f}s")

    # --- SHARED STATE ---
    running = True
    stats = {
        "steps": 0,
        "queries": 0,
        "mutations": 0,
        "moves": 0
    }

    # --- WORKERS ---

    def worker_stepper():
        while running:
            world.step(1 / 60.0)
            stats["steps"] += 1

    def worker_querier():
        """Uses large batches to maximize GIL-free time."""
        starts = array.array('f', [0, 100, 0] * BATCH_SIZE)
        dirs = array.array('f', [0, -200, 0] * BATCH_SIZE)
        while running:
            # Change start point slightly to invalidate Jolt's temporal cache
            starts[0] = random.uniform(-50, 50)
            world.raycast_batch(starts=starts, directions=dirs, max_dist=300.0)
            stats["queries"] += 1

    def worker_cache_hammer():
        """Randomly replaces bodies to stress cache and command queue."""
        # Pre-generate some random sizes
        random_sizes = np.random.uniform(0.1, 1.5, (1000, 3)).astype(np.float32)
        idx = 0
        while running:
            # Select random victim
            victim_idx = random.getrandbits(16) % NUM_BODIES

            # Destroy and Re-create
            world.destroy_body(handle=int(handles[victim_idx]))

            size = random_sizes[idx % 1000]
            new_h = world.create_body(pos=(0, 50, 0), size=tuple(size), shape=culverin.SHAPE_BOX)

            handles[victim_idx] = new_h
            stats["mutations"] += 1
            idx += 1

    def worker_mover():
        while running:
            # Batch 100 moves into one Python loop cycle
            # This reduces Python's 'for' loop overhead relative to the C call
            targets = [random.getrandbits(20) % NUM_BODIES for _ in range(100)]
            for t_idx in targets:
                try:
                    world.set_position(handle=int(handles[t_idx]), x=0, y=50, z=0)
                except: pass
            stats["moves"] += 100

    # --- LAUNCH ---
    threads = [
        threading.Thread(target=worker_stepper, name="Stepper"),
        threading.Thread(target=worker_querier, name="Querier-0"),
        threading.Thread(target=worker_querier, name="Querier-1"),
        threading.Thread(target=worker_cache_hammer, name="Hammer"),
        threading.Thread(target=worker_mover, name="Mover")
    ]

    for t in threads: t.start()

    # --- MONITORING ---
    start_time = time.time()
    last_steps = 0

    try:
        while time.time() - start_time < DURATION:
            time.sleep(1.0)
            current_steps = stats["steps"]
            rate = current_steps - last_steps
            last_steps = current_steps

            print(f"[@ {time.time() - start_time:.1f}s] "
                  f"FPS: {rate} | Batch-Queries: {stats['queries']} | "
                  f"Mutations: {stats['mutations']} | "
                  f"Shapes: {world.shape_count} | Moves: {stats['moves']}")

            if rate == 0 and (time.time() - start_time) > 2.0:
                print("\nCRITICAL FAILURE: SIMULATION FROZEN")
                sys.exit(1)

    finally:
        running = False
        for t in threads: t.join(timeout=2.0)

    print("\n=== FINAL RESULTS ===")
    print(f"Total Steps: {stats['steps']}")
    print(f"Total Rays: {stats['queries'] * BATCH_SIZE}")
    print("STRESS TEST PASSED")


if __name__ == "__main__":
    stress_test()
import threading
import time
import random
import array
import sys
import culverin
import faulthandler

print(f"DEBUG: Loading _culverin_c from: {culverin.__file__}")
faulthandler.enable()

# --- CONFIGURATION ---
DURATION = 5.0
NUM_BODIES = 2000
NUM_QUERY_THREADS = 4
NUM_MUTATOR_THREADS = 2
NUM_MOVER_THREADS = 2


def stress_test():
    print(f"=== INITIALIZING STRESS TEST ({DURATION}s) ===")
    print(">>> Goal: Stress JPH locks, Native CV, and Shape Cache Registry <<<")

    world = culverin.PhysicsWorld(settings={"max_bodies": 10000})
    world.create_body(pos=(0, -10, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)

    handles = []
    handle_lock = threading.Lock()

    print(f"-> Pre-populating {NUM_BODIES} bodies...")
    for i in range(NUM_BODIES):
        h = world.create_body(
            pos=(random.uniform(-10, 10), random.uniform(5, 50), random.uniform(-10, 10)),
            size=(0.5, 0.5, 0.5),  # Fixed size to prime the cache
            motion=culverin.MOTION_DYNAMIC
        )
        handles.append(h)

    world.step(0.016)

    running = True
    stats = {
        "steps": 0,
        "queries": 0,
        "mutations": 0,
        "moves": 0,
        "cache_hits": 0,
        "cache_misses": 0
    }

    # --- WORKERS ---

    def worker_stepper():
        while running:
            world.step(1 / 60.0)
            stats["steps"] += 1

    def worker_querier():
        count = 500
        starts = array.array('f', [0, 50, 0] * count)
        dirs = array.array('f', [0, -100, 0] * count)
        while running:
            starts[0] = random.uniform(-20, 20)
            world.raycast_batch(starts=starts, directions=dirs, max_dist=200.0)
            stats["queries"] += 1

    def worker_cache_hammer():
        """
        Stresses the find_or_create_shape function.
        Alternate between choosing from a small pool (Hits) and random (Misses).
        """
        pool_sizes = [(1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (2.0, 0.2, 2.0)]

        while running:
            # 80% chance of a cache hit, 20% chance of a new shape (miss)
            if random.random() > 0.2:
                size = random.choice(pool_sizes)
                stats["cache_hits"] += 1
            else:
                size = (random.uniform(0.1, 1.0), random.uniform(0.1, 1.0), random.uniform(0.1, 1.0))
                stats["cache_misses"] += 1

            h = world.create_body(pos=(0, 100, 0), size=size, shape=culverin.SHAPE_BOX)

            with handle_lock:
                handles.append(h)
                if len(handles) > NUM_BODIES:
                    victim = handles.pop(0)
                    world.destroy_body(handle=victim)
            stats["mutations"] += 1

    def worker_mover():
        while running:
            target = None
            with handle_lock:
                if handles: target = random.choice(handles)
            if target:
                try:
                    world.set_position(handle=target, x=random.uniform(-1, 1), y=20, z=random.uniform(-1, 1))
                    stats["moves"] += 1
                except ValueError:
                    pass

    # --- LAUNCH ---
    threads = [
        threading.Thread(target=worker_stepper, name="Stepper"),
        threading.Thread(target=worker_querier, name="Querier-0"),
        threading.Thread(target=worker_querier, name="Querier-1"),
        threading.Thread(target=worker_cache_hammer, name="CacheHammer-0"),
        threading.Thread(target=worker_cache_hammer, name="CacheHammer-1"),
        threading.Thread(target=worker_mover, name="Mover-0")
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

            # Check cache metrics if the C getter exists
            cache_info = ""
            if hasattr(world, "shape_count"):
                cache_info = f" | Shapes: {world.shape_count}"

            print(f"[@ {time.time() - start_time:.1f}s] "
                  f"FPS: {rate} | Queries: {stats['queries']} | "
                  f"Mutations: {stats['mutations']}{cache_info} | "
                  f"Moves: {stats['moves']}")

            if rate == 0:
                print("\nCRITICAL FAILURE: SIMULATION FROZEN (Deadlock detected)")
                sys.exit(1)

    finally:
        running = False
        for t in threads: t.join(timeout=1.0)

    print("\n=== TEST RESULTS ===")
    if stats['steps'] > 0:
        print(f"✅ Steps: {stats['steps']}")
        print(f"✅ Cache Hits: {stats['cache_hits']} / Misses: {stats['cache_misses']}")
        if hasattr(world, "shape_count"):
            print(f"✅ Final Shape Count: {world.shape_count}")
        print("✅ STRESS TEST PASSED")
    else:
        print("❌ Stepper failed")
        sys.exit(1)


if __name__ == "__main__":
    stress_test()
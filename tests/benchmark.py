import threading
import time
import random
import array
import numpy as np
import culverin
import argparse
import sys
import psutil
import os

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def run_leak_test(iterations=50000):
    print("\n=== CULVERIN MEMORY LEAK TEST ===")
    world = culverin.PhysicsWorld(settings={"max_bodies": 10000})
    verts = array.array('f', [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).tobytes()
    indices = array.array('I', [0, 1, 2]).tobytes()

    start_ram = get_ram_mb()
    print(f"Starting RAM: {start_ram:.2f} MB")
    
    start_time = time.time()
    for i in range(iterations):
        m = world.create_mesh_body(pos=(0, 0, 0), rot=(0, 0, 0, 1), vertices=verts, indices=indices)
        world.destroy_body(m)
        if i % 1000 == 0:
            world.step(0)  # Flush command queue
            
    world.step(0) # Final flush
    
    end_ram = get_ram_mb()
    duration = time.time() - start_time
    leakage = end_ram - start_ram
    
    print(f"Completed {iterations} mesh cycles in {duration:.2f}s")
    print(f"RAM Shift: {leakage:+.2f} MB")
    if abs(leakage) > 5.0:
        print("❌ WARNING: Potential Memory Leak Detected")
    else:
        print("✅ SUCCESS: Memory is stable")


def run_threading_benchmark(duration=5.0, num_bodies=5000):
    print(f"\n=== CULVERIN MEGA-BATCH THREAD STRESS TEST ===")
    print(f"Simulating {num_bodies} bodies across multiple cores for {duration}s...")
    
    world = culverin.PhysicsWorld(settings={"max_bodies": num_bodies + 5000, "max_pairs": num_bodies * 2})
    
    # Pre-populate
    rand_pos = np.random.uniform(-500, 500, (num_bodies, 3)).astype(np.float32)
    rand_pos[:, 1] += 500
    
    handles_raw = world.create_bodies_batch(
        positions=rand_pos.tolist(),
        sizes=[[0.5, 0.5, 0.5]] * num_bodies,
        shape_type=culverin.SHAPE_BOX,
        motion_type=culverin.MOTION_DYNAMIC
    )
    handles = np.array(handles_raw, dtype=np.uint64)
    world.step(0)
    
    # Thread States
    running = True
    stats = {"steps": 0, "queries": 0, "mutations": 0}

    def worker_stepper():
        while running:
            world.step(1 / 60.0)
            stats["steps"] += 1

    def worker_querier():
        batch_size = 1000
        starts = array.array('f', [0.0] * (batch_size * 3))
        dirs = array.array('f', [0.0, -100.0, 0.0] * batch_size)
        while running:
            starts[1] = random.uniform(200, 500)
            world.raycast_batch(starts=starts, directions=dirs, max_dist=1000.0)
            stats["queries"] += batch_size

    def worker_hammer():
        while running:
            # Recreate 10 bodies per loop
            v_idx = [random.randint(0, num_bodies - 1) for _ in range(10)]
            victims = [int(handles[i]) for i in v_idx]
            
            world.destroy_bodies_batch(handles=victims)
            new_h = world.create_bodies_batch(
                positions=np.random.uniform(-50, 50, (10, 3)).tolist(),
                sizes=[[1,1,1]]*10,
                shape_type=culverin.SHAPE_SPHERE
            )
            for i, idx in enumerate(v_idx): handles[idx] = new_h[i]
            stats["mutations"] += 10
            time.sleep(0.01)

    threads = [
        threading.Thread(target=worker_stepper, name="Stepper"),
        threading.Thread(target=worker_querier, name="Querier1"),
        threading.Thread(target=worker_querier, name="Querier2"),
        threading.Thread(target=worker_hammer, name="Hammer")
    ]
    
    for t in threads: t.start()
    
    start_t = time.time()
    try:
        while time.time() - start_t < duration:
            time.sleep(1.0)
            print(f"[@ {time.time()-start_t:.1f}s] Steps: {stats['steps']} | Rays: {stats['queries']} | Mutations: {stats['mutations']}")
            if stats['steps'] == 0:
                print("❌ CRITICAL: Physics Thread is deadlocked")
                break
    finally:
        running = False
        for t in threads: t.join(timeout=2.0)
        
    print(f"✅ STRESS TEST COMPLETE: {stats['steps']} steps, {stats['queries']} rays.")

def run_churn_test(duration=10.0):
    print("\n=== CULVERIN FRAGMENTATION (CHURN) TEST ===")
    
    # Start with a world limited to 2000 bodies to force frequent re-use
    MAX_LIMIT = 2000
    world = culverin.PhysicsWorld(settings={"max_bodies": MAX_LIMIT})
    handles = []
    
    # Initial population: Fill to 50%
    print(f"-> Initializing world to 50% capacity...")
    for i in range(MAX_LIMIT // 2):
        handles.append(world.create_body(pos=(random.random()*10, 0, random.random()*10)))
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
                h = world.create_body(pos=(random.random()*10, 10, random.random()*10))
                handles.append(h)
                ops += 1

            if num_to_spawn < 50:
                skipped += (50 - num_to_spawn)

            if ops % 1000 == 0 and ops > 0:
                # Telemetry
                print(f"  Cycle: {ops:6} ops | "
                      f"Active: {world.count:4}/{world.max_bodies} | "
                      f"RAM: {get_ram_mb():.2f}MB")
            # Add a tiny sleep to allow the OS to reclaim memory and 
            # let Jolt threads finish their work.
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping churn test...")

    end_t = time.time()
    print(f"\n✅ CHURN COMPLETE")
    print(f" - Duration:      {end_t - start_t:.2f}s")
    print(f" - Total Created: {ops}")
    print(f" - Load Shedding: {skipped} spawns rejected gracefully (at limit)")
    print(f" - Final RAM:     {get_ram_mb():.2f}MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Culverin Diagnostic Tools")
    parser.add_argument('--leak', action='store_true', help="Run the memory leak test")
    parser.add_argument('--stress', action='store_true', help="Run the multi-threaded stress test")
    parser.add_argument('--churn', action='store_true', help="Run the fragmentation churn test")
    
    args = parser.parse_args()
    
    if args.leak:
        run_leak_test()
    elif args.stress:
        run_threading_benchmark()
    elif args.churn:
        run_churn_test()
    else:
        print("Please specify a test to run: --leak or --stress or --churn")
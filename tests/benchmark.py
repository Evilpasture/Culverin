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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Culverin Diagnostic Tools")
    parser.add_argument('--leak', action='store_true', help="Run the memory leak test")
    parser.add_argument('--stress', action='store_true', help="Run the multi-threaded stress test")
    
    args = parser.parse_args()
    
    if args.leak:
        run_leak_test()
    elif args.stress:
        run_threading_benchmark()
    else:
        print("Please specify a test to run: --leak or --stress")
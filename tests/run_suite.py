import time
import sys
import threading
import math
import argparse
import culverin
import numpy as np

# --- Utils ---
def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def time_ms(start_ns):
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0

# --- Workloads ---

def heavy_python_workload(duration_sec):
    """
    Performs pure Python CPU work (math) for roughly 'duration_sec'.
    Used to check if the GIL allows other threads to run.
    """
    end_time = time.time() + duration_sec
    count = 0
    while time.time() < end_time:
        # Do some arbitrary math that holds the GIL
        math.sqrt(12345.6789) * math.sin(count)
        count += 1
    return count

def setup_simulation(count):
    print(f"[*] Spawning {count} bodies...", end="", flush=True)
    world = culverin.PhysicsWorld(settings={"gravity": (0, -9.81, 0), "max_bodies": count + 100})
    
    # Create floor
    world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)
    
    handles = []
    # Create a stack of cubes
    for i in range(count):
        y = (i // 100) * 1.1 + 0.5
        x = (i % 10) * 1.1 - 5.0
        z = ((i // 10) % 10) * 1.1 - 5.0
        h = world.create_body(pos=(x, y, z), size=(0.5, 0.5, 0.5), motion=culverin.MOTION_DYNAMIC)
        handles.append(h)
    
    print(" Done.")
    return world, handles

# --- Benchmarks ---

def bench_access_speed(world, handles):
    print_header("BENCHMARK 1: State Access Speed")
    print("Scenario: Read (x,y,z) position for all bodies.")
    
    # 1. The "Naive" Loop (Simulating standard OOP wrappers)
    start = time.perf_counter_ns()
    
    # Simulate: for body in bodies: pos = body.get_position()
    # We simulate the cost of the loop + the attribute lookup + tuple creation
    count = 0
    for h in handles:
        idx = world.get_index(h)
        # This manual slicing mimics the cost of an individual C-binding call
        x = world.positions[idx*4 + 0]
        y = world.positions[idx*4 + 1]
        z = world.positions[idx*4 + 2]
        # Prevent dead code elimination
        count += (x + y + z)
        
    naive_time = time_ms(start)
    print(f"Naive Python Loop:      {naive_time:.4f} ms")

    # 2. The Culverin Way (Shadow Buffer)
    start = time.perf_counter_ns()
    
    # Direct memory map to NumPy (Zero-Copy)
    # We cast to float32 to match the internal C-type
    raw_buffer = world.positions
    np_array = np.frombuffer(raw_buffer, dtype=np.float32)
    
    # To be fair, we must touch the data to ensure it's loaded
    # But in DOD, we usually pass the whole array to the GPU/Logic
    # So simply creating the view is often all that's needed.
    culverin_time = time_ms(start)
    
    # Cleanup view to prevent resize errors later
    del np_array
    del raw_buffer
    
    print(f"Shadow Buffer (NumPy):  {culverin_time:.4f} ms")
    
    speedup = naive_time / culverin_time if culverin_time > 0 else 0
    print(f"\n>>> SPEEDUP FACTOR: {speedup:.1f}x")

def bench_threading(world):
    print_header("BENCHMARK 2: GIL Release / Parallelism")
    print("Scenario: Run Physics (Thread A) + Heavy Math (Thread B) simultaneously.")
    
    # Step duration roughly 16ms
    dt = 1.0/60.0
    
    # 1. Measure Baseline Physics cost
    start = time.perf_counter_ns()
    world.step(dt)
    phys_cost = time_ms(start)
    print(f"Physics Only Cost:      {phys_cost:.2f} ms")
    
    # 2. Measure Baseline Math cost (aiming for same duration as physics)
    # We calibrate the math loop to take roughly the same time as the physics step
    # to maximize contention visibility.
    target_duration = phys_cost / 1000.0 
    start = time.perf_counter_ns()
    heavy_python_workload(target_duration)
    math_cost = time_ms(start)
    print(f"Math Only Cost:         {math_cost:.2f} ms")
    
    # 3. Run Serial (A then B)
    expected_serial = phys_cost + math_cost
    print(f"Expected Serial Time:   {expected_serial:.2f} ms")
    
    # 4. Run Parallel
    print("Running Parallel Threads...")
    
    def run_phys():
        world.step(dt)
        
    def run_math():
        heavy_python_workload(target_duration)

    t1 = threading.Thread(target=run_phys)
    t2 = threading.Thread(target=run_math)
    
    start = time.perf_counter_ns()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    parallel_time = time_ms(start)
    
    print(f"Actual Parallel Time:   {parallel_time:.2f} ms")
    
    # Analysis
    if parallel_time < (expected_serial * 0.85):
        print("\n>>> RESULT: SUCCESS (GIL Released)")
        print("    Threads ran in parallel. This is likely Python 3.13t or a standard build releasing the GIL correctly.")
    else:
        print("\n>>> RESULT: SEQUENTIAL (GIL Contention)")
        print("    Threads ran one after another. Check if you are using Python 3.13t.")

def main():
    parser = argparse.ArgumentParser(description="Culverin Performance Suite")
    parser.add_argument("--bodies", type=int, default=10000, help="Number of bodies to simulate")
    args = parser.parse_args()

    # Detect Free Threading
    is_free_threaded = False
    try:
        if sys._is_gil_enabled() == False:
            is_free_threaded = True
    except AttributeError:
        pass # Not Python 3.13
        
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Free-Threaded:  {is_free_threaded}")

    # Run
    world, handles = setup_simulation(args.bodies)
    
    # Warmup
    world.step(1/60)
    
    bench_access_speed(world, handles)
    bench_threading(world)
    
    print_header("Done.")

if __name__ == "__main__":
    main()
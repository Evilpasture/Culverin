import culverin
import time
import numpy as np
import csv
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
ITERATIONS = 1_000_000
BODY_COUNT = 1_000
CSV_FILENAME = "culverin_stepping_benchmark.csv"
STEP_SIZE = 1/60.0

def run_benchmark():
    print(f"Initializing Culverin World with {BODY_COUNT} bodies...")
    
    # 1. Setup World
    world = culverin.PhysicsWorld(settings={
        "max_bodies": BODY_COUNT + 10,
        "gravity": (0, -9.81, 0)
    })

    # 2. Add a static floor
    world.create_body(pos=(0, -1, 0), size=(100, 1, 100), motion=culverin.MOTION_STATIC)

    # 3. Batch create dynamic cubes in a grid
    positions = []
    for x in range(10):
        for y in range(10):
            for z in range(10):
                positions.append((x * 2.0, 10.0 + (y * 2.0), z * 2.0))
    
    world.create_bodies_batch(
        positions=positions,
        sizes=[(0.5, 0.5, 0.5)] * BODY_COUNT,
        shape_type=culverin.SHAPE_BOX,
        motion_type=culverin.MOTION_DYNAMIC
    )

    # Flush initial creation
    world.step(0)

    # 4. Pre-allocate NumPy arrays for data collection (Speed is key!)
    # Column 0: Step Duration (Nanoseconds)
    # Column 1: Accumulated World Time (Seconds)
    data_points = np.zeros((ITERATIONS, 2), dtype=np.float64)

    print(f"Starting {ITERATIONS:,} steps. This may take a minute...")
    
    start_bench = time.perf_counter()

    # ==========================================================
    # CORE BENCHMARK LOOP
    # ==========================================================
    for i in range(ITERATIONS):
        # Measure only the step() call
        t_start = time.perf_counter_ns()
        world.step(STEP_SIZE)
        t_end = time.perf_counter_ns()

        # Store data
        data_points[i, 0] = t_end - t_start
        data_points[i, 1] = world.time

        if i % 100_000 == 0 and i > 0:
            print(f" Progress: {i:,} / {ITERATIONS:,} steps completed...")

    end_bench = time.perf_counter()
    # ==========================================================

    total_duration = end_bench - start_bench
    avg_step_ms = (np.mean(data_points[:, 0]) / 1_000_000)
    
    print("\n" + "="*40)
    print(f"BENCHMARK COMPLETE")
    print(f"Total Wall Time: {total_duration:.2f} seconds")
    print(f"Average Step:    {avg_step_ms:.4f} ms")
    print(f"Throughput:      {ITERATIONS / total_duration:,.0f} steps/sec")
    print("="*40)

    # 5. Export to CSV
    print(f"Writing data to {CSV_FILENAME}...")
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step_index", "duration_ns", "world_time_s"])
        
        # Write in chunks to handle memory efficiently if needed, 
        # but for 1M rows, writing the whole block is fine.
        for idx in range(ITERATIONS):
            writer.writerow([idx, int(data_points[idx, 0]), data_points[idx, 1]])

    print(f"Success! {Path(CSV_FILENAME).absolute()}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark aborted by user.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
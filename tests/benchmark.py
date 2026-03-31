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


def run_threading_benchmark(duration=5.0, num_bodies=500):
    print(f"\n=== CULVERIN REALISTIC SIMULATION BENCHMARK ===")
    print(f"Simulating {num_bodies} active dynamic bodies across multiple cores for {duration}s...")
    
    # 1. SETUP: World with plenty of headroom to avoid RuntimeErrors
    world = culverin.PhysicsWorld(settings={
        "max_bodies": num_bodies + 2000, 
        "max_pairs": num_bodies * 8
    })
    
    # Create static floor (Giant Box)
    world.create_body(
        pos=(0, -5, 0), size=(500, 1, 500), 
        shape=culverin.SHAPE_BOX, motion=culverin.MOTION_STATIC
    )
    
    # Create Dynamic Grid (Spawned in the air so they crash down)
    pos_list = []
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
        motion_type=culverin.MOTION_DYNAMIC
    )
    # Store handles in a mutable numpy array for thread-safe-ish updating
    handles = np.array(handles_raw, dtype=np.uint64)
    world.step(0) # Initial push to BroadPhase
    
    # Thread States
    running = True
    stats = {"steps": 0, "rays": 0, "contacts": 0, "resets": 0, "mutations": 0}

    # --- THREAD 1: THE CORE STEPPER ---
    def worker_stepper():
        while running:
            try:
                # Run as fast as the CPU allows
                world.step(1.0 / 60.0)
                stats["steps"] += 1
            except RuntimeError:
                # Concurrent step/lock failure - just skip this loop
                pass 

    # --- THREAD 2: SENSORS (RAYCASTS) ---
    def worker_sensors():
        batch_size = 500
        starts = array.array('f', [0.0] * (batch_size * 3))
        dirs = array.array('f', [0.0, -1.0, 0.0] * batch_size)
        while running:
            starts[1] = random.uniform(20, 50) # Randomize height
            world.raycast_batch(starts=starts, directions=dirs, max_dist=100.0)
            stats["rays"] += batch_size
            time.sleep(0.01) # ~100Hz

    # --- THREAD 3: GAMEPLAY LOGIC (ANTI-SLEEP & MEMORYVIEW) ---
    def worker_housekeeper():
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
                        world.set_position(h, random.uniform(-10, 10), 20.0, random.uniform(-10, 10))
                        world.set_linear_velocity(h, 0, 0, 0)
                        stats["resets"] += 1
                    except ValueError:
                        # Fallback: if it was destroyed between the is_alive check 
                        # and the set_position call, just ignore it.
                        pass
                
            time.sleep(0.5)

    # --- THREAD 4: THE MUTATOR (CONTROLLED HAMMER) ---
    def worker_mutator():
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
                    motion_type=culverin.MOTION_DYNAMIC
                )
                
                for i, h in enumerate(new_h):
                    handles[idx_to_replace[i]] = h
                    world.activate(int(h)) # Force awake
                    
                stats["mutations"] += 5
            except RuntimeError:
                pass # Lock contention or pool limit hit, try again later
                
            time.sleep(0.05) # Run 20 times a second

    # Start Threads
    threads = [
        threading.Thread(target=worker_stepper, name="Stepper"),
        threading.Thread(target=worker_sensors, name="Sensors"),
        threading.Thread(target=worker_housekeeper, name="Housekeeper"),
        threading.Thread(target=worker_mutator, name="Mutator")
    ]
    
    for t in threads: t.start()
    
    # Monitoring Loop
    start_t = time.time()
    try:
        while time.time() - start_t < duration:
            time.sleep(1.0)
            # Fetch raw events to clear the buffer (simulate event handling)
            contacts = world.get_contact_events_raw()
            if contacts:
                stats["contacts"] += 1
                
            print(f"[@ {time.time()-start_t:.1f}s] "
                  f"Steps: {stats['steps']} | "
                  f"Rays: {stats['rays']} | "
                  f"Mutations: {stats['mutations']} | "
                  f"Resets: {stats['resets']}")
            
            if stats['steps'] == 0:
                print("❌ CRITICAL: Physics Thread is completely deadlocked")
                break
    finally:
        running = False
        for t in threads: t.join(timeout=1.0)
        
    fps = stats['steps'] / duration
    print(f"\n✅ TEST COMPLETE")
    print(f"Final Performance: {fps:.2f} FPS")
    print(f"Total Steps: {stats['steps']}")
    print(f"Total Raycasts: {stats['rays']}")

def run_churn_test(duration=10.0):
    # There is a known memory issue. will investigate...
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
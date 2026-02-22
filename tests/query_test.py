import threading
import time
import random
import numpy as np
import culverin
import array

# --- CONFIGURATION ---
DURATION = 15.0
NUM_BODIES = 20000
# Scale these based on your CPU cores
RAY_THREADS = 4
OVERLAP_THREADS = 4
SHAPECAST_THREADS = 2
BATCH_SIZE = 1000

def lock_free_query_stress():
    print(f"=== CULVERIN LOCK-FREE QUERY STRESS TEST ===")
    
    world = culverin.PhysicsWorld(settings={
        "max_bodies": NUM_BODIES + 5000,
        "max_pairs": NUM_BODIES * 2
    })

    # 1. SETUP: Create a dense "forest" of shapes to make BVH traversal expensive
    print(f"-> Spawning {NUM_BODIES} objects...")
    positions = np.random.uniform(-100, 100, (NUM_BODIES, 3)).astype(np.float32)
    # Give them some height
    positions[:, 1] = np.random.uniform(0, 50, NUM_BODIES)
    
    # Mix of boxes and spheres
    handles = world.create_bodies_batch(
        positions=positions.tolist(),
        sizes=[[1.0, 1.0, 1.0]] * NUM_BODIES,
        shape_type=culverin.SHAPE_BOX,
        motion_type=culverin.MOTION_STATIC # Static is faster for stressing the BVH
    )
    
    print("-> World ready. Launching concurrent query threads...")

    # Shared state
    running = True
    stats = {
        "rays": 0,
        "overlaps": 0,
        "shapecasts": 0,
        "steps": 0,
        "mutations": 0
    }

    def ray_worker():
        """Stresses the batch raycaster and the new rsqrt math."""
        starts = array.array('f', [0.0] * (BATCH_SIZE * 3))
        # Pointing randomly but with a set distance
        dirs = array.array('f', [0.0] * (BATCH_SIZE * 3))
        for i in range(BATCH_SIZE * 3):
            dirs[i] = random.uniform(-1, 1)

        while running:
            # Shift the start pos slightly to invalidate caches
            starts[0] = random.uniform(-100, 100)
            starts[2] = random.uniform(-100, 100)
            
            # This releases the GIL and runs without a global mutex!
            world.raycast_batch(starts=starts, directions=dirs, max_dist=500.0)
            stats["rays"] += BATCH_SIZE

    def overlap_worker():
        """Stresses overlap_sphere and overlap_aabb."""
        while running:
            center = (random.uniform(-100, 100), 10.0, random.uniform(-100, 100))
            
            # Alternate between Sphere and AABB
            if random.random() > 0.5:
                world.overlap_sphere(center=center, radius=10.0)
            else:
                world.overlap_aabb(
                    min=(center[0]-5, 0, center[2]-5),
                    max=(center[0]+5, 20, center[2]+5)
                )
            stats["overlaps"] += 1

    def shapecast_worker():
        """Stresses the complex sweep logic."""
        while running:
            pos = (random.uniform(-50, 50), 20.0, random.uniform(-50, 50))
            # Sweep downwards
            world.shapecast(
                shape=culverin.SHAPE_SPHERE,
                pos=pos,
                rot=(0,0,0,1),
                dir=(0, -1, 0),
                size=1.0,
                ignore=0
            )
            stats["shapecasts"] += 1

    def mutation_worker():
        """
        The Chaos Thread. 
        Constantly creates and destroys bodies to force the Shadow Lock 
        to fight with the Query threads for priority.
        """
        while running:
            # Create a temporary body
            h = world.create_body(
                pos=(0, 100, 0), 
                shape=culverin.SHAPE_BOX, 
                motion=culverin.MOTION_DYNAMIC
            )
            # Instantly apply a force (Command Queue stress)
            world.apply_impulse(h, 0, -10, 0)
            # Destroy it
            world.destroy_body(h)
            
            stats["mutations"] += 1
            # Small sleep to let queries actually run
            time.sleep(0.01)

    def stepper_worker():
        """The simulation heartbeat."""
        while running:
            world.step(1/60.0)
            stats["steps"] += 1
            time.sleep(0.01)

    # Launching
    threads = []
    for _ in range(RAY_THREADS): threads.append(threading.Thread(target=ray_worker))
    for _ in range(OVERLAP_THREADS): threads.append(threading.Thread(target=overlap_worker))
    for _ in range(SHAPECAST_THREADS): threads.append(threading.Thread(target=shapecast_worker))
    threads.append(threading.Thread(target=mutation_worker))
    threads.append(threading.Thread(target=stepper_worker))

    for t in threads: t.start()

    # Monitoring
    start_t = time.time()
    last_t = start_t
    
    try:
        while time.time() - start_t < DURATION:
            time.sleep(1.0)
            dt = time.time() - last_t
            last_t = time.time()
            
            # Snapshot stats
            r_rate = stats["rays"] / (time.time() - start_t)
            o_rate = stats["overlaps"] / (time.time() - start_t)
            s_rate = stats["shapecasts"] / (time.time() - start_t)
            
            print(f"Elapsed: {time.time() - start_t:.1f}s | "
                  f"Rays/s: {r_rate/1000:.1f}k | "
                  f"Overlaps/s: {o_rate:.1f} | "
                  f"Shapecasts/s: {s_rate:.1f} | "
                  f"Steps: {stats['steps']}")
                  
    finally:
        running = False
        for t in threads: t.join()

    print("=== TEST COMPLETE ===")
    print(f"Final Totals:")
    print(f" - Total Rays Cast: {stats['rays']}")
    print(f" - Total Overlaps: {stats['overlaps']}")
    print(f" - Total Step() calls: {stats['steps']}")
    print(f" - Mutations successfully synced: {stats['mutations']}")

if __name__ == "__main__":
    lock_free_query_stress()
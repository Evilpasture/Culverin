import pybullet as p
import time
import numpy as np

# Configuration
BODY_COUNT = 10000

def main():
    print(f"Initializing PyBullet (DIRECT mode)...")
    # DIRECT mode is headless (fairest comparison to Culverin)
    p.connect(p.DIRECT)
    p.setGravity(0, -9.81, 0)

    # create a simple collision shape
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.25])
    
    print(f"Spawning {BODY_COUNT} bodies...")
    body_ids = []
    
    # Batch creation isn't standard in PyBullet, so we loop
    for i in range(BODY_COUNT):
        # Stack them so they don't sleep immediately
        uid = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col_id,
            basePosition=[(i % 10)*0.6, (i // 10)*0.6, 0]
        )
        body_ids.append(uid)
        
    print("Done spawning.")
    
    # Warmup step
    p.stepSimulation()

    # --- BENCHMARK 1: PHYSICS STEP ---
    start = time.perf_counter()
    p.stepSimulation()
    end = time.perf_counter()
    step_time_ms = (end - start) * 1000.0
    print(f"Physics Step Time:      {step_time_ms:.4f} ms")

    # --- BENCHMARK 2: STATE ACCESS ---
    print(f"Reading positions for {BODY_COUNT} bodies...")
    
    start = time.perf_counter()
    
    # This is the standard way to get data in PyBullet
    # There is no native "Zero-Copy" view for all base positions at once.
    # You MUST allocate a tuple for every single body.
    chk = 0.0
    for uid in body_ids:
        # Returns ((x,y,z), (x,y,z,w))
        pos, rot = p.getBasePositionAndOrientation(uid)
        chk += pos[0] # Prevent dead code elimination
        
    end = time.perf_counter()
    read_time_ms = (end - start) * 1000.0
    
    print(f"Read Time (Loop):       {read_time_ms:.4f} ms")
    
    # --- COMPARISON ---
    # Based on your Culverin result (0.02ms)
    culverin_time = 0.02 
    ratio = read_time_ms / culverin_time
    
    print("\n" + "="*40)
    print("COMPARISON")
    print("="*40)
    print(f"PyBullet Read: {read_time_ms:.4f} ms")
    print(f"Culverin Read: {culverin_time:.4f} ms")
    print(f"Speedup Factor: {ratio:.1f}x")

    p.disconnect()

if __name__ == "__main__":
    main()
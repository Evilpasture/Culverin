import threading
import time
import random
import culverin
import numpy as np
from typing import List, Dict, Optional

class StressTest:
    def __init__(self, duration: float, target_bodies: int, num_threads: int):
        self.duration = duration
        self.target_bodies = target_bodies
        self.num_threads = num_threads
        
        self.running = True
        self.handle_lock = threading.Lock()
        self.handles: List[int] = []
        
        # Initialize world immediately so Pylance knows it's not None
        self.world = culverin.PhysicsWorld(settings={"max_bodies": target_bodies + 1000})
        
        self.stats: Dict[str, int] = {
            "steps": 0,
            "created": 0,
            "destroyed": 0,
            "moved": 0,
            "read_ops": 0,
            "stale_access_caught": 0,
            "buffer_locks_hit": 0
        }

    def physics_loop(self):
        print("[Thread-Physics] Started")
        while self.running:
            self.world.step(1.0/60.0)
            self.stats["steps"] += 1
            time.sleep(0.001)

    def spawner_loop(self):
        print("[Thread-Spawner] Started")
        while self.running:
            if len(self.handles) < self.target_bodies:
                try:
                    h = self.world.create_body(
                        pos=(random.uniform(-50, 50), random.uniform(10, 100), random.uniform(-50, 50)),
                        size=(0.5, 0.5, 0.5),
                        motion=culverin.MOTION_DYNAMIC
                    )
                    with self.handle_lock:
                        self.handles.append(h)
                    self.stats["created"] += 1
                except BufferError:
                    self.stats["buffer_locks_hit"] += 1
                    time.sleep(0.001)
            time.sleep(0.001)

    def killer_loop(self):
        print("[Thread-Killer] Started")
        while self.running:
            to_kill = None
            with self.handle_lock:
                if self.handles:
                    idx = random.randint(0, len(self.handles) - 1)
                    to_kill = self.handles.pop(idx)
            
            if to_kill:
                try:
                    self.world.destroy_body(to_kill)
                    self.stats["destroyed"] += 1
                except BufferError:
                    self.stats["buffer_locks_hit"] += 1
            time.sleep(0.002)

    def mover_loop(self):
        print("[Thread-Mover] Started")
        while self.running:
            target = None
            with self.handle_lock:
                if self.handles:
                    target = random.choice(self.handles)
            
            if target:
                try:
                    self.world.set_linear_velocity(target, 0, 10, 0)
                    self.stats["moved"] += 1
                except ValueError:
                    self.stats["stale_access_caught"] += 1
                except Exception:
                    pass
            time.sleep(0.001)

    def reader_loop(self):
        print("[Thread-Reader] Started")
        while self.running:
            positions = None
            arr = None
            try:
                positions = self.world.positions
                if len(positions) > 0:
                    arr = np.frombuffer(positions, dtype=np.float32)
                    _ = np.sum(arr)
                self.stats["read_ops"] += 1
            except BufferError:
                self.stats["buffer_locks_hit"] += 1
            except Exception as e:
                print(f"CRITICAL READER ERROR: {e}")
                break
            finally:
                if arr is not None:
                    del arr
                if positions is not None:
                    del positions
            time.sleep(0.001)

    def run(self):
        print(f"Stress Test: {self.duration} seconds, {self.num_threads} worker threads.")
        
        threads = [
            threading.Thread(target=self.physics_loop),
            threading.Thread(target=self.spawner_loop),
            threading.Thread(target=self.killer_loop),
            threading.Thread(target=self.mover_loop),
            threading.Thread(target=self.reader_loop)
        ]

        start_time = time.time()
        for t in threads:
            t.start()

        try:
            while time.time() - start_time < self.duration:
                time.sleep(1)
                print(f"\r[Time: {time.time()-start_time:.1f}s] "
                      f"Bodies: {len(self.handles)} | "
                      f"Ops(C/D/M): {self.stats['created']}/{self.stats['destroyed']}/{self.stats['moved']} | "
                      f"Reads: {self.stats['read_ops']} | "
                      f"Contention: {self.stats['buffer_locks_hit']}", end="")
        except KeyboardInterrupt:
            print("\nStopping early...")

        self.running = False
        for t in threads:
            t.join()

        print("\n\n" + "="*40)
        print("STRESS TEST RESULTS")
        print("="*40)
        print(f"Total Physics Steps:   {self.stats['steps']}")
        print(f"Bodies Created:        {self.stats['created']}")
        print(f"Bodies Destroyed:      {self.stats['destroyed']}")
        print(f"Buffer Reads:          {self.stats['read_ops']}")
        print("-" * 20)
        print(f"Stale Handles Caught:  {self.stats['stale_access_caught']}")
        print(f"Buffer Contentions:    {self.stats['buffer_locks_hit']}")
        print("="*40)

        if self.stats['steps'] > 100 and self.stats['read_ops'] > 100:
            print("SUCCESS: Engine handled high concurrency without crashing.")
        else:
            print("FAILURE: Threads stalled.")

if __name__ == "__main__":
    test = StressTest(duration=30.0, target_bodies=2000, num_threads=4)
    test.run()
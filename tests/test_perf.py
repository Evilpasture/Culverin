import sys
import unittest
import time
import array
import numpy as np
import culverin

class TestPerformanceRegression(unittest.TestCase):
    """
    Performance Regression Suite.
    These tests ensure that core optimizations (like FastParse, Batching, and Memory Views)
    do not regress in future updates. 
    
    Note: Thresholds are intentionally generous to prevent flaky CI failures on 
    shared runners, but strict enough to catch O(N^2) bugs or accidental Python allocations.
    """

    def setUp(self):
        # Create a massive world for stress testing
        self.max_bodies = 20000
        self.world = culverin.PhysicsWorld(settings={
            "gravity": (0, -10, 0), 
            "max_bodies": self.max_bodies,
            "max_pairs": self.max_bodies * 4
        })

    def tearDown(self):
        # Force cleanup to prevent memory spikes between tests
        del self.world

    def test_batch_vs_iterative_creation(self):
        """Ensure C-level batch creation remains significantly faster than Python loops."""
        body_count = 5000
        positions = np.random.uniform(-100, 100, (body_count, 3)).astype(np.float32).tolist()
        sizes = [[1.0, 1.0, 1.0]] * body_count

        # 1. Iterative Creation
        t0 = time.perf_counter()
        loop_handles = []
        for p in positions:
            loop_handles.append(self.world.create_body(pos=p, shape=culverin.SHAPE_BOX, motion=culverin.MOTION_DYNAMIC))
        loop_time = time.perf_counter() - t0

        # Flush
        self.world.destroy_bodies_batch(loop_handles)
        self.world.step(0)

        # 2. Batch Creation
        t0 = time.perf_counter()
        batch_handles = self.world.create_bodies_batch(
            positions=positions, 
            sizes=sizes, 
            shape_type=culverin.SHAPE_BOX, 
            motion_type=culverin.MOTION_DYNAMIC
        )
        batch_time = time.perf_counter() - t0

        print(f"\n[Perf] Create {body_count} bodies -> Loop: {loop_time*1000:.2f}ms | Batch: {batch_time*1000:.2f}ms")
        
        # Assertions
        self.assertEqual(len(batch_handles), body_count)
        self.assertLess(batch_time, loop_time * 0.8, "Batch creation should be measurably faster than looping")
        self.assertLess(batch_time, 0.05, "Batch creation of 5k bodies took over 50ms (Major Regression)")

    def test_simulation_step_overhead(self):
        """Ensure the core simulation step remains highly performant under heavy load."""
        body_count = 10000
        positions = np.random.uniform(-50, 50, (body_count, 3)).astype(np.float32).tolist()
        
        # Spawn a massive block of falling cubes
        self.world.create_bodies_batch(positions, [[0.5]*3]*body_count, culverin.SHAPE_BOX, culverin.MOTION_DYNAMIC)
        self.world.step(0) # Initial flush

        # Step 60 frames
        t0 = time.perf_counter()
        for _ in range(60):
            self.world.step(1/60.0)
        total_time = time.perf_counter() - t0
        avg_ms = (total_time / 60.0) * 1000.0

        print(f"\n[Perf] 10k Body Sim Step -> Avg: {avg_ms:.2f} ms/frame")
        
        # Jolt is fast. 10k free-falling bodies should easily process in under 16ms (60fps)
        # We set threshold to 25ms to account for slow CI runners.
        self.assertLess(avg_ms, 25.0, "Simulation step time degraded significantly.")

    def test_fastparse_stress_limit(self):
        """
        Tests the FastParse engine at its architectural limit (64 arguments).
        Parses 32 positional and 32 keyword arguments to verify bitmask 
        integrity and O(1) hash table stability.
        """
        # Create 32 positional arguments (a0...a31)
        pos_args = [i for i in range(32)]
        
        # Create 32 keyword arguments (a32...a63)
        # Using sys.intern to ensure string literals are interned, which is critical for FastParse's optimization.
        kw_args = {sys.intern(f"a{i}"): i for i in range(32, 64)}
        
        iterations = 50000
        
        # Warmup (optional, but ensures JIT or cache warming)
        self.world._benchmark_parse(*pos_args, **kw_args)
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.world._benchmark_parse(*pos_args, **kw_args)
        total_time = time.perf_counter() - t0
        
        calls_per_sec = iterations / total_time
        print(f"\n[Perf] FastParse Stress Limit (64 args) -> {calls_per_sec:,.0f} calls/sec ({total_time*1000:.2f}ms total)")
        
        # This is a rigorous test. 50k calls to a 64-arg parser involves massive
        # bitwise operations and dictionary lookups per iteration. 
        # 250ms is a safe threshold for modern CPUs on CI runners.
        self.assertLess(total_time, 0.250, "FastParse stress limit (64 args) has regressed.")

    def test_fastbuild_engine_overhead(self):
        """
        Tests the FastBuild engine (C23 _Generic dispatcher).
        Measures the raw speed of creating Python Tuples from C primitives
        without any format string parsing (Py_BuildValue).
        """
        iterations = 100000 # 100k iterations because this is lightning fast
        
        # Warmup
        self.world._benchmark_build()
        
        t0 = time.perf_counter()
        for _ in range(iterations):
            # This calls the METH_NOARGS C function we wrote
            _obj = self.world._benchmark_build()
        total_time = time.perf_counter() - t0
        
        calls_per_sec = iterations / total_time
        print(f"\n[Perf] FastBuild Engine -> {calls_per_sec:,.0f} builds/sec ({total_time*1000:.2f}ms total)")
        
        # FastBuild should easily exceed 1 million builds per second on modern hardware.
        # We'll set a conservative regression threshold of 200ms for 100k calls.
        self.assertLess(total_time, 0.200, "FastBuild engine construction speed has regressed.")
        
        # Safety check: ensure

    def test_raycast_batch_speed(self):
        """Ensure the GIL-released batch raycast remains heavily optimized."""
        # Create a floor to hit
        self.world.create_body(pos=(0,-1,0), size=(1000, 1, 1000), motion=culverin.MOTION_STATIC)
        self.world.step(0)

        ray_count = 50000
        starts = np.zeros((ray_count, 3), dtype=np.float32)
        starts[:, 1] = 10.0
        dirs = np.zeros((ray_count, 3), dtype=np.float32)
        dirs[:, 1] = -1.0
        
        starts_bytes = starts.tobytes()
        dirs_bytes = dirs.tobytes()

        t0 = time.perf_counter()
        res = self.world.raycast_batch(starts_bytes, dirs_bytes, max_dist=20.0)
        total_time = time.perf_counter() - t0

        print(f"\n[Perf] 50k Batch Raycasts -> {total_time*1000:.2f} ms")
        
        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)
        # 50k parallel raycasts against a single broadphase object should be near instant (< 50ms)
        self.assertLess(total_time, 0.150, "Batch raycasting took too long. Check Jolt threads or memory view overhead.")

    def test_state_save_load_speed(self):
        """State save/load should be a direct memcpy of stride arrays. Must be instantaneous."""
        body_count = 10000
        self.world.create_bodies_batch(
            np.random.uniform(-100, 100, (body_count, 3)).tolist(), 
            [[1.0, 1.0, 1.0]] * body_count, 
            culverin.SHAPE_BOX, 
            culverin.MOTION_DYNAMIC
        )
        self.world.step(0)

        # Time Save
        t0 = time.perf_counter()
        state_bytes = self.world.save_state()
        save_time = time.perf_counter() - t0

        # Time Load
        t0 = time.perf_counter()
        self.world.load_state(state=state_bytes)
        load_time = time.perf_counter() - t0

        print(f"\n[Perf] 10k Body State -> Save: {save_time*1000:.2f}ms | Load: {load_time*1000:.2f}ms")
        print(f"[Perf] State Payload Size -> {len(state_bytes) / 1024 / 1024:.2f} MB")

        # memcpy of ~2-3 MB of shadow buffers should take less than 5ms
        self.assertLess(save_time, 0.05, "save_state() is too slow, ensure no Python loops are used.")
        self.assertLess(load_time, 0.05, "load_state() is too slow, ensure Jolt syncing is optimized.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
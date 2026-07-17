import os
import sys
import threading
import time
import unittest

import numpy as np

import culverin


# ==============================================================================
# PERFORMANCE THRESHOLDS CONFIGURATION
# ==============================================================================
# Adjust these values to tune the strictness of the CI performance pipeline.
# Time values are in seconds unless specified as _MS (milliseconds).
class THRESHOLDS:
    # Detect if we are running in GitHub Actions/CI
    IS_CI = os.getenv("CI", "false").lower() == "true"

    # Scale factors based on your actual CI regression logs:
    # Physics is close (1.5x), but Python-to-C plumbing is hit hard (6x)
    # We increase the base scale from 1.0 to 1.5 or 2.0 locally to accommodate
    # developers running background applications (IDEs, web browsers, etc.)
    PHYS_SCALE = 3.0 if IS_CI else 1.5
    API_SCALE = 6.0 if IS_CI else 2.0  # <--- Relaxed local scale from 1.0 to 2.0

    # Creation & Lifecycle
    BATCH_CREATE_RATIO = 0.7 if IS_CI else 0.5
    BATCH_CREATE_MAX_S_5K = 0.04 * PHYS_SCALE

    # Core Simulation
    SIM_STEP_MAX_MS_10K = 2.0 * PHYS_SCALE
    BULK_MUTATION_MAX_MS = 2.0 * PHYS_SCALE
    CONTENTION_STEP_MAX_MS = 1.8 * PHYS_SCALE

    # C-API Bindings
    # These are now scaled by the updated API_SCALE to give the CPU some breathing room
    FASTPARSE_MAX_S_50K = 0.025 * API_SCALE
    FASTBUILD_MAX_S_100K = 0.01 * API_SCALE

    # Queries
    RAYCAST_MAX_S_50K = 0.020 * PHYS_SCALE

    # State Management
    STATE_SAVE_MAX_S = 0.025 * PHYS_SCALE
    STATE_LOAD_MAX_S = 0.025 * PHYS_SCALE


# ==============================================================================


class TestPerformanceRegression(unittest.TestCase):
    """
    Performance Regression Suite.
    These tests ensure that core optimizations (like FastParse, Batching, and Memory Views)
    do not regress in future updates.
    """

    def setUp(self) -> None:
        # Create a massive world for stress testing
        self.max_bodies = 50000
        self.world = culverin.PhysicsWorld(
            settings={
                "gravity": (0, -10, 0),
                "max_bodies": self.max_bodies,
                "max_pairs": self.max_bodies * 4,
            }
        )

    def tearDown(self) -> None:
        # Force cleanup to prevent memory spikes between tests
        del self.world

    def test_batch_vs_iterative_creation(self) -> None:
        """Ensure C-level batch creation remains significantly faster than Python loops."""
        body_count = 20000
        rng = np.random.default_rng()
        positions = rng.uniform(-100, 100, (body_count, 3)).astype(np.float32).tolist()
        sizes = [[1.0, 1.0, 1.0]] * body_count

        # 1. Iterative Creation (Fair comparison: passes the same 'size' structures)
        t0 = time.perf_counter()
        loop_handles = [
            self.world.create_body(
                pos=p,
                size=sizes[i],
                shape=culverin.SHAPE_BOX,
                motion=culverin.MOTION_DYNAMIC,
            )
            for i, p in enumerate(positions)
        ]
        loop_time = time.perf_counter() - t0

        # Flush
        self.world.destroy_bodies_batch(loop_handles)
        self.world.step(0)

        # 2. Batch Creation (Tiny warmup)
        self.world.create_bodies_batch(
            positions[:10], sizes[:10], culverin.SHAPE_BOX, culverin.MOTION_DYNAMIC
        )

        t0 = time.perf_counter()
        batch_handles = self.world.create_bodies_batch(
            positions=positions,
            sizes=sizes,
            shape_type=culverin.SHAPE_BOX,
            motion_type=culverin.MOTION_DYNAMIC,
        )
        batch_time = time.perf_counter() - t0

        print(
            f"\n[Perf] Create {body_count} bodies -> Loop: {loop_time * 1000:.2f}ms | Batch: {batch_time * 1000:.2f}ms"
        )

        # Assertions
        self.assertEqual(len(batch_handles), body_count)
        self.assertLess(
            batch_time,
            loop_time * THRESHOLDS.BATCH_CREATE_RATIO,
            "Batch creation should be measurably faster than looping",
        )
        self.assertLess(
            batch_time,
            THRESHOLDS.BATCH_CREATE_MAX_S_5K,
            "Batch creation of 5k bodies took too long (Major Regression)",
        )

    def test_simulation_step_overhead(self) -> None:
        """Ensure the core simulation step remains highly performant under heavy load."""
        body_count = 10000
        rng = np.random.default_rng()
        positions = rng.uniform(-50, 50, (body_count, 3)).astype(np.float32).tolist()

        # Spawn a massive block of falling cubes
        self.world.create_bodies_batch(
            positions, [[0.5] * 3] * body_count, culverin.SHAPE_BOX, culverin.MOTION_DYNAMIC
        )
        self.world.step(0)  # Initial flush

        # Step 600 frames
        t0 = time.perf_counter()
        for _ in range(600):
            self.world.step(1 / 60.0)
        total_time = time.perf_counter() - t0
        avg_ms = (total_time / 600.0) * 1000.0

        print(f"\n[Perf] 10k Body Sim Step -> Avg: {avg_ms:.2f} ms/frame")

        self.assertLess(
            avg_ms, THRESHOLDS.SIM_STEP_MAX_MS_10K, "Simulation step time degraded significantly."
        )

    def test_fastparse_stress_limit(self) -> None:
        """
        Tests the FastParse engine at its architectural limit (64 arguments).
        Parses 32 positional and 32 keyword arguments to verify bitmask
        integrity and O(1) hash table stability.
        """
        # Create 32 positional arguments (a0...a31)
        pos_args = list(range(32))

        # Create 32 keyword arguments (a32...a63)
        # Using sys.intern to ensure string literals are interned.
        kw_args = {sys.intern(f"a{i}"): i for i in range(32, 64)}

        iterations = 50000

        # Warmup (optional, but ensures JIT or cache warming)
        self.world._benchmark_parse(*pos_args, **kw_args)

        t0 = time.perf_counter()
        for _ in range(iterations):
            self.world._benchmark_parse(*pos_args, **kw_args)
        total_time = time.perf_counter() - t0

        calls_per_sec = iterations / total_time
        print(
            f"\n[Perf] FastParse Stress Limit (64 args) -> {calls_per_sec:,.0f} calls/sec ({total_time * 1000:.2f}ms total)"
        )

        self.assertLess(
            total_time,
            THRESHOLDS.FASTPARSE_MAX_S_50K,
            "FastParse stress limit (64 args) has regressed.",
        )

    def test_fastbuild_engine_overhead(self) -> None:
        """
        Tests the FastBuild engine (C23 _Generic dispatcher).
        Measures the raw speed of creating Python Tuples from C primitives
        without any format string parsing (Py_BuildValue).
        """
        iterations = 100000  # 100k iterations because this is lightning fast

        # Warmup
        self.world._benchmark_build()

        t0 = time.perf_counter()
        for _ in range(iterations):
            # This calls the METH_NOARGS C function we wrote
            _obj = self.world._benchmark_build()
        total_time = time.perf_counter() - t0

        calls_per_sec = iterations / total_time
        print(
            f"\n[Perf] FastBuild Engine -> {calls_per_sec:,.0f} builds/sec ({total_time * 1000:.2f}ms total)"
        )

        self.assertLess(
            total_time,
            THRESHOLDS.FASTBUILD_MAX_S_100K,
            "FastBuild engine construction speed has regressed.",
        )

        # Safety check: ensure it's returning the expected object type
        res = self.world._benchmark_build()
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 9)

    def test_raycast_batch_speed(self) -> None:
        """Ensure the GIL-released batch raycast remains heavily optimized."""
        # Create a floor to hit
        self.world.create_body(pos=(0, -1, 0), size=(1000, 1, 1000), motion=culverin.MOTION_STATIC)
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

        print(f"\n[Perf] 50k Batch Raycasts -> {total_time * 1000:.2f} ms")

        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)
        self.assertLess(
            total_time,
            THRESHOLDS.RAYCAST_MAX_S_50K,
            "Batch raycasting took too long. Check Jolt threads or memory view overhead.",
        )

    def test_state_save_load_speed(self) -> None:
        """State save/load should be a direct memcpy of stride arrays. Must be instantaneous."""
        body_count = 10000
        self.world.create_bodies_batch(
            np.random.uniform(-100, 100, (body_count, 3)).tolist(),
            [[1.0, 1.0, 1.0]] * body_count,
            culverin.SHAPE_BOX,
            culverin.MOTION_DYNAMIC,
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

        print(
            f"\n[Perf] 10k Body State -> Save: {save_time * 1000:.2f}ms | Load: {load_time * 1000:.2f}ms"
        )
        print(f"[Perf] State Payload Size -> {len(state_bytes) / 1024 / 1024:.2f} MB")

        self.assertLess(save_time, THRESHOLDS.STATE_SAVE_MAX_S, "save_state() is too slow.")
        self.assertLess(load_time, THRESHOLDS.STATE_LOAD_MAX_S, "load_state() is too slow.")

    def test_bulk_mutation_throughput(self) -> None:
        """
        Stress test for the Command Queue.
        Applies a force to 5,000 bodies every frame.
        This tests the O(1) command queue write path and command capacity reallocs.
        """
        body_count = 5000
        # 1. Setup: Bulk Create
        handles = self.world.create_bodies_batch(
            np.random.uniform(-50, 50, (body_count, 3)).tolist(),
            [[0.5] * 3] * body_count,
            culverin.SHAPE_BOX,
            culverin.MOTION_DYNAMIC,
        )
        self.world.step(0)

        # 2. Stress Loop: Apply 5,000 forces per frame for 100 frames
        t0 = time.perf_counter()
        for _ in range(100):
            for h in handles:
                self.world.apply_force(h, 0, 10, 0)

            # Step triggers the flush of these 5,000 commands
            self.world.step(1 / 60.0)

        total_time = time.perf_counter() - t0
        avg_ms = (total_time / 100.0) * 1000.0

        print(f"\n[Perf] Bulk Mutation (5k forces/frame) -> Avg: {avg_ms:.2f} ms/frame")

        self.assertLess(
            avg_ms,
            THRESHOLDS.BULK_MUTATION_MAX_MS,
            "Bulk mutation overhead (Command Queue) is too high.",
        )

    def test_fastparse_morphism_overhead(self) -> None:
        """
        Benchmark the Monomorphic stubs (Speculative) vs the Generic parser (Fallback).
        - Positional: Hits the 'fp_speculate_p4_naked' stub (O(1) direct call).
        - Keywords: Bails to 'fp_parse_vector' (Generic loop).
        """
        h = self.world.create_body(pos=(0, 0, 0))
        iterations = 1_000_000

        # 1. Benchmark Positional (Targeting fp_speculate_p4_naked)
        t0 = time.perf_counter()
        calls_pos = 0
        for _ in range(iterations):
            self.world.set_linear_velocity(h, 1.0, 2.0, 3.0)
            calls_pos += 1
        t_pos = time.perf_counter() - t0

        # 2. Benchmark Keywords (Targeting fp_parse_vector fallback)
        t0 = time.perf_counter()
        calls_kw = 0
        for _ in range(iterations):
            # We use keywords to force the bail-out logic in C
            self.world.set_linear_velocity(h, x=1.0, y=2.0, z=3.0)
            calls_kw += 1
        t_kw = time.perf_counter() - t0
        # 3. Benchmark for loop (Because parsing is that quick)
        t0 = time.perf_counter()
        calls_raw = 0
        for _ in range(iterations):
            calls_raw += 1
        t_raw = time.perf_counter() - t0
        # 4.
        t0 = time.perf_counter()
        calls_func = 0

        def test_func(h: int, x: float, y: float, z: float) -> None:
            return None

        for _ in range(iterations):
            test_func(h, x=1.0, y=2.0, z=3.0)
            calls_func += 1
        t_func = time.perf_counter() - t0
        # 5. Benchmark NumPy (Small Vector Creation/Update)
        # This simulates the overhead of using common NumPy operations in a loop
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t0 = time.perf_counter()
        calls_np = 0
        for _ in range(iterations):
            # Benchmarking a simple update or creation
            # NumPy has significant 'C-API' transition overhead for these
            np.add(vec, 1.0, out=vec)
            calls_np += 1
        t_np = time.perf_counter() - t0

        # Calculate nanoseconds per call for the "Naked" truth
        ns_pos = (t_pos / iterations) * 1e9
        ns_kw = (t_kw / iterations) * 1e9
        ns_raw = (t_raw / iterations) * 1e9
        ns_func = (t_func / iterations) * 1e9
        ns_np = (t_np / iterations) * 1e9

        print("\n[Perf] FastParse Morphism Analysis:")
        print(
            f"       PATH: Positional (Speculative) | Calls: {calls_pos:,} | Avg: {ns_pos:.1f} ns/call"
        )
        print(
            f"       PATH: Keywords   (Generic)     | Calls: {calls_kw:,} | Avg: {ns_kw:.1f} ns/call"
        )
        print(
            f"       PATH: Raw         (Loop)       | Calls: {calls_raw:,} | Avg: {ns_raw:.1f} ns/call"
        )
        print(
            f"       PATH: Function    (Generic)    | Calls: {calls_func:,} | Avg: {ns_func:.1f} ns/call"
        )
        print(
            f"       PATH: NumPy Add   (External C) | Calls: {calls_np:,} | Avg: {ns_np:.1f} ns/call"
        )
        print(f"       RESULT: {t_kw / t_pos:.2f}x Speedup in Speculative path")
        print(f"       RESULT: {ns_np / ns_pos:.2f}x Speedup vs NumPy overhead")

        # Verification of logic
        self.assertEqual(calls_pos, iterations)
        self.assertEqual(calls_kw, iterations)

    def test_contention_efficiency(self) -> None:
        """
        Ensure the Lock-Free Handover (BufferProxy) allows parallel execution
        without causing the Physics Thread to stall.
        """
        body_count = 5000
        self.world.create_bodies_batch(
            np.random.uniform(-50, 50, (body_count, 3)).tolist(),
            [[0.5] * 3] * body_count,
            culverin.SHAPE_BOX,
            culverin.MOTION_DYNAMIC,
        )
        self.world.step(0)

        def worker_math() -> None:
            # Continuously read from the proxy (simulates heavy Numpy/AI work)
            # This tests if releasebuffer/getbuffer/CV overhead is low.
            for _ in range(500):
                with memoryview(self.world.positions) as mv:
                    _ = np.frombuffer(mv, dtype=np.float64).sum()

        t0 = time.perf_counter()
        math_thread = threading.Thread(target=worker_math)
        math_thread.start()

        # Run 100 physics steps in parallel with the math thread
        for _ in range(100):
            self.world.step(1 / 60.0)

        math_thread.join()
        total_time = time.perf_counter() - t0
        avg_ms = (total_time / 100.0) * 1000.0

        print(f"\n[Perf] Contention (Step + Proxy Read) -> {avg_ms:.2f} ms/frame")

        self.assertLess(
            avg_ms,
            THRESHOLDS.CONTENTION_STEP_MAX_MS,
            "Stepper is stalling due to Proxy contention.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

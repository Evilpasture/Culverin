# Culverin Performance Benchmarks

Culverin is designed to solve the "Python Physics Bottleneck." These benchmarks compare Culverin's Data-Oriented approach against traditional Object-Oriented (OO) wrappers.

*Tests conducted on: Python 3.14.2t (Free-Threaded)*

## 1. State Access Speed
**Scenario:** Read the (x, y, z) positions of 10,000 active rigid bodies.

| Method | Time (ms) | Speedup |
| :--- | :--- | :--- |
| **Traditional OO** (`for b in bodies: b.get_pos()`) | 4.83 ms | 1x |
| **Culverin Shadow Buffer** (`np.frombuffer(world.positions)`) | **0.02 ms** | **235x** |

**Why?** 
In traditional wrappers, every call to `get_pos()` involves Python function call overhead, argument parsing, and tuple allocation. Culverin exposes a direct pointer to the C-array, allowing **Zero-Copy** access via NumPy or memoryviews.

## 2. The "Heavy Logic" Parallelism (GIL Release)
**Scenario:** A simulation with 10,000 bodies. We run ~42ms of heavy Python "AI logic" (pure math) per frame alongside the ~42ms physics step.

| Execution Mode | Total Frame Time | Result |
| :--- | :--- | :--- |
| **Serial (Standard Wrapper)** | 83.18 ms | Physics blocks Python. Logic waits for Physics. |
| **Culverin (Parallel)** | **42.85 ms** | **Physics runs on Core A, Python runs on Core B.** |

**The Result:** 
Because `world.step()` releases the GIL and Culverin is thread-safe, the physics simulation ran effectively **in parallel**. The physics step added only **1.2ms** of overhead to the total frame time, despite doing **41ms** of work.

## 3. Scaling (Broadphase Performance)
Jolt is famous for its broadphase. Here is how Culverin scales with body count (Box-on-Box stack):

| Body Count | Simulation Step (ms) | Shadow Buffer Sync (ms) |
| :--- | :--- | :--- |
| 1,000 | 0.8 ms | 0.01 ms |
| 5,000 | 3.2 ms | 0.02 ms |
| 10,000 | 41.6 ms | 0.04 ms |

## 4. Concurrency & Stability (Chaos Test)
**Scenario:** A 30-second "Fuzz Test" running **5 concurrent threads** attacking the engine simultaneously to provoke race conditions:
1.  **Physics Thread:** Steps simulation at 60Hz.
2.  **Spawner Thread:** Rapidly creates bodies (forcing C++ memory reallocation).
3.  **Killer Thread:** Randomly deletes bodies (creating holes in sparse arrays).
4.  **Mover Thread:** Tries to move bodies that may have just been deleted.
5.  **Reader Thread:** Constantly exports `numpy` views of the shadow buffers.

| Metric | Result |
| :--- | :--- |
| **Total Duration** | 30.0s |
| **Active Threads** | 5 |
| **Physics Steps** | ~14,800 |
| **Buffer Reads** | ~19,400 |
| **Segmentation faults / Crashes** | **0** |
| **Safety Intercepts** | **8,409** |

**The Result:**
The engine successfully intercepted **8,409** unsafe memory access attempts (where the Spawner tried to resize memory while the Reader was reading it). Instead of crashing with a Segmentation Fault in C/C++, Culverin raised a safe Python `BufferError` Exception, allowing the logic to back off and retry.

## 5. Comparison vs. PyBullet
A comparison with Culverin against PyBullet (CPython build) on the same machine running Python 3.14t. The test involved spawning 10,000 rigid bodies and reading their positions into Python for rendering.

| Metric | PyBullet | Culverin | Difference |
| :--- | :--- | :--- | :--- |
| **GIL Status** | **Forced ON** (Warning emitted) | **OFF** (True Parallelism) | Architectural Win |
| **Data Access Time** | 11.92 ms | **0.02 ms** | **596x Faster** |
| **Frame Budget Used** | 72% (of 16ms) | 0.1% | Enables 10k+ objects |

*Note: PyBullet forces the re-enabling of the GIL upon import, neutralizing the threading benefits of Python 3.13t/3.14t. Culverin runs fully free-threaded.*

## How to run benchmarks
Scripts are provided in the `/tests/` directory:

```bash
# Run speed and threading speedup tests
python tests/run_suite.py --bodies 10000

# Run the stability test
python tests/thread_stress.py

# For good measure, run the PyBullet script for reference.
python tests/pybullet_bench.py
```
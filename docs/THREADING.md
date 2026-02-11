# Multithreading

This document outlines the technical architecture of Culverin's concurrency model, specifically optimized for **Free-Threaded Python (3.13t/3.14t)**.

## The Synchronization Model

Culverin uses a multi-layered locking strategy to ensure that the high-speed Jolt C++ core and the Python memory model never collide.

### 1. The Lock Hierarchy
To prevent deadlocks, Culverin strictly enforces an acquisition order. Violating this order in C results in a deadlock; following it ensures safety:
1.  **Shadow Lock (`PyMutex`/`ShadowMutex`):** Protects Python-facing metadata, handles, and slot maps.
2.  **GIL (Python Global Interpreter Lock):** Only present in standard Python; ignored in free-threaded builds.
3.  **Native Mutex (`SRWLOCK`/`pthread_mutex_t`):** Hard-serializes access to the Jolt C++ `TempAllocator` and the `PhysicsSystem`.

### 2. Stepper Priority (Anti-Starvation)
Simulation steps have absolute priority over asynchronous queries (like raycasts). 
*   When `world.step()` is called, it sets a `step_requested` atomic flag.
*   New Raycast/Overlap calls will see this flag and **block on a Condition Variable** immediately, even if the step hasn't started yet.
*   This ensures that a high volume of background queries cannot "starve" the physics simulation.

## Thread Safety Categories

### Concurrent Reading (Safe)
The "Shadow Buffers" (`world.positions`, `world.rotations`) are contiguous C-arrays. Reading these from any thread is **always safe and lock-free**. 
*   If you read while `world.step()` is running, you are reading the state from the *previous* frame.
*   Data is updated atomically at the end of the step.

### State Mutation (Blocking)
Functions that modify the world state (e.g., `create_body`, `destroy_body`, `set_position`, `apply_impulse`) are **Serialized**.
*   If the physics thread is currently inside a `world.step()`, these calls will pause the calling thread.
*   Once the step completes, the mutations are flushed from an internal Command Queue into the Jolt core.

### Asynchronous Queries (Serialized)
Raycasts, Shapecasts, and Overlap queries release the GIL and can be called from any thread. 

*   **Thread-Safe Allocation:** ~~Culverin uses a custom JoltC build with `TempAllocatorMalloc`. This replaces Jolt's default stack-based memory with a thread-safe heap allocator. This eliminates the "Freeing in wrong order" crashes common in other Jolt wrappers.~~ 

Continues using `TempAllocatorImplWithMallocFallback`.
*   **Broadphase Integrity:** Although memory allocation is now thread-safe, queries are still **hard-serialized** against the simulation step (`world.step()`) using a Native Mutex. This ensures that a raycast never attempts to read the acceleration structure while the simulation is actively re-balancing the tree, which avoids undefined behavior even with thread-safe memory.
*   **High Throughput:** Because queries are extremely fast (often microsecond scale), this serialization provides massive throughput for AI and visibility logic without the risk of read-after-write hazards.


## High-Performance Rendering

Culverin provides a "Snapshot" system to decouple physics frequency from rendering frequency.

### The Interpolation Pattern
Physics usually runs at 60Hz, while displays may run at 144Hz or higher.
1.  `world.step()` automatically snapshots the "Previous" state of every object.
2.  `world.get_render_state(alpha)` performs a vectorized **Lerp** (Position) and **NLerp** (Rotation) between the Previous and Current state.

**Example Multi-Threaded Loop:**

```python
# Thread 1: Physics & Logic
def physics_thread():
    while True:
        # Releases GIL. Mutations from other threads block here.
        world.step(1/60)
        # Logic is safe to run here
        do_gameplay_logic()

# Thread 2: Rendering (Main Thread)
def render_thread():
    while True:
        # Calculate how far we are between physics frames (0.0 - 1.0)
        alpha = time_since_last_step / (1/60)
        
        # Returns a single, packed 'bytes' object of ALL transforms.
        # This is GIL-free and thread-safe.
        buffer = world.get_render_state(alpha)
        
        # Fast upload to GPU (e.g., OpenGL/Vulkan/DirectX)
        gpu_upload_vbo(buffer)
        draw_scene()
```

## Technical Constraints
*   **Recursive Stepping:** A single thread cannot call `world.step()` inside a collision callback. This will trigger a `RuntimeError` to prevent self-deadlock.
*   **Memoryview Safety:** Accessing `world.positions` returns a memoryview. This view is invalidated if the world capacity is resized (e.g., creating more bodies than the initial `max_bodies`). Always re-wrap the buffer if you perform mass object creation.

## Stability

* **The Hard Serializer**: g_jph_trampoline_lock (Native OS Mutex) ensures Jolt's internal C++ state is never accessed by two threads at once.
* **The Priority Escalator**: step_requested ensures that queries immediately yield to the simulation, preventing the "Raycast Starvation" deadlock.
* **The Condition Variable**: Native Condition Variables replaced busy-waiting, reducing CPU overhead and making the "wait" logic instant and reliable.
* **The GIL-Native Handshake**: By releasing the GIL before requesting the Native Lock, the Priority Inversion Deadlock(Triangle of Death) between the Python interpreter and the OS scheduler is eliminated.
* **The Memory Anchor**: Replacing the Binder's stack allocator with a Malloc-based allocator removed the most common source of Jolt crashes, trading single-threaded performance for multithreading
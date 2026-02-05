# Culverin Architecture

Culverin is not a standard Python wrapper. It does not wrap C++ objects in Python objects for every entity. Instead, it uses a **Data-Oriented Design** (DOD) approach to minimize cache misses and Python interpreter overhead.

## The Shadow Buffer System
When you create a body, Culverin adds it to the internal Jolt simulation *and* to a set of flat C-arrays called **Shadow Buffers**.

*   **Jolt (C++):** Calculates physics, collisions, and constraints.
*   **Shadow Buffers (C/Python):** Mirrors the state (Position, Rotation, Velocity) of every active body.

When you access `world.positions`, you are not calling a function. You are getting a raw `memoryview` into the C array.

### Zero-Copy Rendering
Because the data is contiguous, you can pass it directly to modern graphics APIs (OpenGL/Vulkan) or NumPy without looping in Python.

```python
# The "Wrong" Way (Traditional OOP)
# Slow: 10,000 function calls, 10,000 tuple allocations
for body in bodies:
    pos = body.get_position()
    renderer.draw(pos)

# The "Culverin" Way (Data Oriented)
# Fast: 0 loops, 0 allocations. Direct memory map.
# positions is [x, y, z, pad, x, y, z, pad...]
all_positions = np.frombuffer(world.positions, dtype=np.float32)
renderer.upload_instance_buffer(all_positions)
```

### Memory Layouts
All shadow buffers use `float32` and are 16-byte aligned to ensure SIMD compatibility.

*   **`world.positions`**: 16 bytes per body. `[float x, float y, float z, float 0.0]`. 
    *   *The 4th float is padding for SIMD alignment.*
*   **`world.rotations`**: 16 bytes per body. `[float x, float y, float z, float w]`.
*   **`world.velocities`**: 16 bytes per body. `[float vx, float vy, float vz, float 0.0]`.
*   **`world.get_render_state(alpha)`**: 28 bytes per body. `[float x, y, z, float x, y, z, w]`.
    *   *Used for direct GPU Instance Buffer uploads.*

### Render State Layout (28 Bytes)
When calling `get_render_state(alpha)`, the returned buffer is packed like this:

| Offset | Type | Component |
| :--- | :--- | :--- |
| 0 | float32 | Position X |
| 4 | float32 | Position Y |
| 8 | float32 | Position Z |
| 12 | float32 | Rotation X |
| 16 | float32 | Rotation Y |
| 20 | float32 | Rotation Z |
| 24 | float32 | Rotation W |

**Note:** Unlike the raw shadow buffers, this interpolation buffer **removes the 4th padding float** from the position to save bandwidth during GPU uploads.

## Generational Handles
Culverin does not return object references. It returns `uint64` handles.
*   **Bits 0-31:** Index in the dense array.
*   **Bits 32-63:** Generation ID.

If you destroy a body and creating a new one reuses that slot, the Generation ID increments. Old handles stored in your logic will safely fail `is_alive(handle)` checks instead of causing C++ Use-After-Free crashes.

## The Command Queue
State changes (setPosition, setVelocity) are **Deferred**.
1. You call `world.set_position(handle, ...)`
2. Culverin pushes a `CMD_SET_POS` struct to a C-queue.
3. You call `world.step()`
4. Culverin acquires the lock, flushes the queue, applies changes to Jolt, and then steps the simulation.

This ensures thread safety and deterministic application of logic updates.
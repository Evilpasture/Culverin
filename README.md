# Culverin Physics

[![PyPI - Version](https://img.shields.io/pypi/v/culverin)](https://pypi.org/project/culverin/)
[![Python Version](https://img.shields.io/pypi/pyversions/culverin)](https://pypi.org/project/culverin/)

Culverin is a middleware bridge between the **Jolt Physics** engine and the Python Virtual Machine. It is designed for 3D games and simulations that require high performance and multi-threaded execution, for the Python game engine developers.

### Key Concepts

*   **Free-Threading Support:** Designed for Python 3.13t and 3.14t. The engine releases the Global Interpreter Lock (GIL) during physics updates and raycast batches.
*   **Shadow Buffers:** All body positions, rotations, and velocities are stored in contiguous C-arrays. You can access this data via `memoryview`, `NumPy` or any dependency that supports the buffer protocol without the overhead of creating Python objects for every body.
*   **Thread-Safe API:** The engine uses a priority-based locking system. Simulation steps, state mutations, and queries can run on different threads without causing deadlocks or memory corruption.
*   **Generational Handles:** Bodies are referenced by 64-bit handles rather than pointers. This ensures that using a handle for a deleted object will not crash the program.
*   **Double-Precision Positions:** Uses double-precision floats (`float64`) for world positions to prevent physics jitter in massive open worlds, while using `float32` for rotations and velocities to save memory and computing resources.

### Features

*   **Standard Primitives:** Box, Sphere, Capsule, Cylinder, and Plane shapes.
*   **Complex Shapes:** Support for Convex Hulls, Heightfields (Terrain), and static Meshes.
*   **Compound Bodies:** Create single bodies composed of multiple child shapes.
*   **Character Controller:** A virtual character controller with built-in support for climbing stairs, sliding down slopes, and pushing objects.
*   **Vehicles:** Support for wheeled vehicles and tracked vehicles (tanks) with physical treads and skid-steering.
*   **Ragdolls & Skeletons:** Multi-body articulated physics with active motorized poses.
*   **Allowed Degrees of Freedom:** Restrict rigid bodies to specific world-space translation and rotation axes, including 2D-style motion.
*   **Constraints:** Fixed, Point, Hinge, Slider, Distance, Swing-Twist, and Cone constraints.
*   **Queries:** Efficient single and batch Raycasting, Shapecasting (sweeps), and Overlap queries.
*   **Collision Events:** Native event buffer for contact added, persisted, and removed events.
*   **Soft Bodies:** Support for soft bodies from Jolt Physics, with fine-grained configurations.

### Installation

```bash
pip install culverin
```

### 📚 Documentation

Detailed API references for every class and method are available:
*   **Online Docs:** [View the Interactive API Reference](https://evilpasture.github.io/Culverin/)
*   **In-Engine:** Use the standard Python `help()` system: `help(culverin.PhysicsWorld.create_body)` or `help(culverin)`

### Quick Start

```python
import numpy as np

import culverin

# Initialize the world with a capacity limit
world = culverin.PhysicsWorld(settings={"max_bodies": 1000})

# Create a static ground plane
# a plane normal of default (0.577, 0.577, 0.577) represents an infinite 45-degree tilted ramp.
world.create_body(pos=(0, 0, 0), shape=culverin.SHAPE_PLANE, motion=culverin.MOTION_STATIC)

# Create a dynamic box
handle = world.create_body(
    pos=(0, 10, 0), size=(1, 1, 1), shape=culverin.SHAPE_BOX, motion=culverin.MOTION_DYNAMIC
)

# Simulation loop
for _ in range(1000):
    world.step(1 / 60)

    # Zero-Copy Data Access
    # Wrap the internal engine memory in NumPy without copying a single byte.
    # This gives you O(1) access to all 1,000+ bodies at once.
    positions = np.frombuffer(world.positions, dtype=np.float64).reshape(-1, 4)

    # Print the Y position of our box
    idx = world.get_index(handle)
    print(f"Box Height: {positions[idx, 1]:.2f}m")
```

### Technical Specifications

| Spec | Standard |
| :--- | :--- |
| **Units** | Metric (1.0 = 1 meter) |
| **Coordinate System** | Right-Handed (Y-Up) |
| **Angle Units** | Radians |
| **Quaternion Format** | `(x, y, z, w)` |
| **Position Buffer** | `Float64` (Stride 4: x, y, z, pad) |
| **Rotation/Velocity Buffers** | `Float32` (Stride 4: x, y, z, w/pad) |
| **Minimum Python** | 3.12 (3.13t+ recommended for multi-threading) |

### Performance & Rendering Note
For maximum performance when updating a renderer (like OpenGL or Vulkan), do not loop through individual handles in Python. Instead, use the interpolating export method:

```python
# Returns a single, tightly packed bytes object of Float32s 
# formatted as [px, py, pz, rx, ry, rz, rw] for every active body.
# The 'alpha' parameter (0.0 to 1.0) interpolates between the previous 
# and current physics steps for buttery-smooth rendering.
render_data = world.get_render_state(alpha=0.5)
```
To read data for gameplay logic, use `numpy.frombuffer(world.positions, dtype=np.float64)` and `numpy.frombuffer(world.rotations, dtype=np.float32)` to wrap the internal engine memory without making copies.

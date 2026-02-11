# Culverin Physics

Culverin is a Python wrapper for the **Jolt Physics** engine. It is designed for 3D games and simulations that require high performance and multi-threaded execution.

### Key Concepts

*   **Free-Threading Support:** Designed for Python 3.13t and 3.14t. The engine releases the Global Interpreter Lock (GIL) during physics updates and raycast batches.
*   **Shadow Buffers:** All body positions, rotations, and velocities are stored in contiguous C-arrays. You can access this data via `memoryview` or `numpy` without the overhead of creating Python objects for every body.
*   **Thread-Safe API:** The engine uses a priority-based locking system. Simulation steps, state mutations, and queries can run on different threads without causing deadlocks or memory corruption.
*   **Generational Handles:** Bodies are referenced by 64-bit handles rather than pointers. This ensures that using a handle for a deleted object will not crash the program.
*   **Double-Precision Internal:** Uses double-precision floats for world positions to prevent physics jitter in large environments, while mirroring data to float32 buffers for rendering efficiency.

### Features

*   **Standard Primitives:** Box, Sphere, Capsule, Cylinder, and Plane shapes.
*   **Complex Shapes:** Support for Convex Hulls, Heightfields (Terrain), and static Meshes.
*   **Compound Bodies:** Create single bodies composed of multiple child shapes.
*   **Character Controller:** A virtual character controller with built-in support for climbing stairs, sliding down slopes, and pushing objects.
*   **Vehicles:** Support for wheeled vehicles and tracked vehicles (tanks) with physical treads and skid-steering.
*   **Constraints:** Fixed, Point, Hinge, Slider, Distance, and Cone constraints.
*   **Queries:** Efficient single and batch Raycasting, Shapecasting (sweeps), and Overlap queries.
*   **Collision Events:** Native event buffer for contact added, persisted, and removed events.

### Installation

Building from source requires CMake and a C++ compiler (Visual Studio on Windows, GCC or Clang on Linux/macOS).

```bash
# Clone the repository including submodules
git clone --recursive https://github.com/Evilpasture/culverin.git
cd culverin

# Install the package
pip install .
```

### Quick Start

```python
import culverin
import numpy as np

# Initialize the world with 500 bodies capacity
world = culverin.PhysicsWorld(settings={"gravity": (0, -9.81, 0), "max_bodies": 1000})

# Create a ground plane
world.create_body(pos=(0, 0, 0), shape=culverin.SHAPE_PLANE, motion=culverin.MOTION_STATIC)

# Create a dynamic box
handle = world.create_body(pos=(0, 10, 0), size=(1, 1, 1), shape=culverin.SHAPE_BOX, motion=culverin.MOTION_DYNAMIC)

# Simulation loop
for _ in range(1000):
    world.step(1/60)
    
    # Access position directly from the shadow buffer
    idx = world.get_index(handle)
    pos = world.positions[idx * 4 : idx * 4 + 3]
    print(f"Box Height: {pos[1]}")
```

### Technical Specifications

| Spec | Standard |
| :--- | :--- |
| **Units** | Metric (1.0 = 1 meter) |
| **Coordinate System** | Right-Handed (Y-Up) |
| **Angle Units** | Radians |
| **Quaternion Format** | `(x, y, z, w)` |
| **Internal Precision** | Float64 (Double) |
| **Buffer Precision** | Float32 |
| **Minimum Python** | 3.11 (3.13+ recommended for multi-threading) |

### Performance Note
For maximum performance when reading state, use the `world.positions` and `world.rotations` attributes. These return `memoryview` objects that point directly to the engine's internal memory. Use `numpy.frombuffer(world.positions, dtype=np.float32)` to wrap them in a NumPy array without copying the data.
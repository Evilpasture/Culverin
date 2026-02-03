# Culverin Physics Engine

**Culverin** is a high-performance Python binding for **Jolt Physics**, engineered specifically for high-fidelity 3D games and simulations. It is designed to bridge the gap between the speed of C/C++ and the ease of Python.

### Key Features
*   **Built for Python 3.13t/3.14t:** Optimized for the Free-Threaded interpreter. Releases the GIL during simulation steps.
*   **Shadow Buffer Architecture:** State is mirrored in high-speed C arrays. Read positions and rotations via `memoryview` or `numpy` without a single Python overhead per object.
*   **Double-Precision Core:** Uses `RVec3` (double-precision) internally to eliminate "jitter" and "shaking" in large-scale open worlds.
*   **Generational Handles:** 64-bit handles prevent crashes. Even if an object is deleted, Python logic won't access "stale" pointers.
*   **Professional Features:**
    *   **Character Controllers:** Native Jolt virtual character with stair-climbing and slope-sliding.
    *   **Native Tanks:** Real physical treads and skid-steering (not just raycast boxes).
    *   **Ragdolls & Motors:** Full skeleton support with motor-driven poses.
    *   **Trigger Lifecycle:** Native `ENTER`, `STAY`, and `EXIT` events for gameplay zones.
    *   **CCD:** Continuous Collision Detection for high-speed projectiles.

### Performance
Culverin is designed for **Zero-Copy** workflows.
*   **Batch Raycasting:** Fire hundreds of rays in a single call with the GIL released.
*   **Render State Slicing:** Get interpolated [Position, Rotation] for all 10,000 objects in one `bytes` block for instant GPU upload.

### Installation (Build from Source)
```bash
git clone https://github.com/Evilpasture/culverin
cd culverin
pip install .
```
or in your Python project terminal with your virtual environment activated
```bash
pip install git+https://github.com/Evilpasture/culverin.git
```
*(Requires CMake and a C++ compiler for Jolt integration)*

### 🏎 Quick Start
```python
import culverin

# Create World
world = culverin.PhysicsWorld(settings={"gravity": (0, -9.81, 0)})

# Create a Dynamic Body
box = world.create_body(pos=(0, 10, 0), size=(0.5, 0.5, 0.5), shape=culverin.SHAPE_BOX)

# Main Loop
while True:
    world.step(1/60)
    pos = world.positions[world.get_index(box) * 4 : world.get_index(box) * 4 + 3]
    print(f"Box is at: {pos}")
```

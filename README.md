# Culverin 🏹
**High-performance Jolt Physics wrapper for Python 3.14+ (Free-Threaded)**

Culverin is a "Data-Oriented" wrapper for the Jolt Physics engine. It is designed specifically for the experimental free-threaded (No-GIL) builds of Python, utilizing a "Shadow Buffer" architecture to achieve astronomical simulation speeds.

### 🚀 Performance
- **Simulation Speed:** ~25,000+ FPS (1,000 active bodies)
- **Step Time:** ~0.04ms per frame
- **Overhead:** Zero-copy data access via NumPy-wrapped memoryviews.
- **Concurrency:** Fully No-GIL ready; run physics and game logic on separate cores simultaneously.

### 🏗️ Architecture: The Culverin Pipeline
1. **Bake:** Python validates and packs body data into flat binary buffers.
2. **Step:** C-extension releases the GIL and runs Jolt's multithreaded solver.
3. **Sync:** Jolt state is mirrored into aligned C-arrays (Shadow Buffers).
4. **View:** Python accesses results via `numpy.asarray()` views of the Shadow Buffers.

### 🛠️ Installation

```bash
# Requires scikit-build-core and a C++17 compiler
git clone --recursive https://github.com/Evilpasture/culverin
cd culverin
pip install .
```

### Usage
```python
import culverin
import numpy as np

# Define bodies
bodies = [{"pos": (0, 10, 0), "mass": 1.0}]

# Initialize
world = culverin.PhysicsWorld(bodies=bodies)

# Zero-copy view
positions = np.asarray(world.positions)

# Step
world.step(1/60)
print(positions[0]) # Updated automatically!
```
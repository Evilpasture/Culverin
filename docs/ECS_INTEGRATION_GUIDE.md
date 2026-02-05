# Integrating with ECS (PyBullet Eviction Guide)

So, you're tired of PyBullet. You're tired of the clunky API, the overhead of creating Python tuples for every single position update, and the struggle to run physics on a separate thread without the GIL fighting you. And you want a game engine in Python with fast physics.

Culverin was built to solve this. It is designed to sit inside a **Physics System** within an ECS architecture.

## The Mental Shift

| Concept | PyBullet / Standard Wrapper | Culverin |
| :--- | :--- | :--- |
| **Object Ref** | You hold an integer ID (0, 1, 2...). Reused IDs cause bugs if you aren't careful. | You hold a **Generational Handle**. Stale handles are safe. |
| **Data Access** | `p.getBasePositionAndOrientation(id)`<br>*(Allocates a tuple, holds GIL, slow)* | `world.positions[i*4 : i*4+3]`<br>*(Raw memory view, zero allocation, releases GIL)* |
| **Updates** | You call functions one by one. | You queue commands; `world.step()` flushes them in bulk. |
| **Syncing** | You loop through entities and copy data manually. | You use **Bulk Buffer Transfer** (memcpy). |

## 1. The Component

Your ECS needs a component to store the Culverin handle. Do not store raw physics data (mass, friction) here after initialization; let Culverin own the physics state.

```python
from dataclasses import dataclass
import culverin

@dataclass
class RigidBody:
    # Config (Used only during creation)
    shape: int = culverin.SHAPE_BOX
    mass: float = 1.0
    motion: int = culverin.MOTION_DYNAMIC
    collider_size: tuple = (1, 1, 1)
    
    # Runtime State
    handle: int = 0  # The Culverin Generational Handle
    initialized: bool = False
```

## 2. The Physics System

This is where the magic happens. The system manages the lifecycle and the **Data Pump** between your ECS `Transform` components and Culverin's `Shadow Buffers`.

### Initialization
When the system starts, create the world.

```python
class PhysicsSystem:
    def __init__(self):
        # 1. Init World
        self.world = culverin.PhysicsWorld(settings={"gravity": (0, -9.81, 0)})
        
        # 2. Optimization: Pre-allocate reusable numpy arrays for bulk sync
        # avoiding allocation every frame is the key to performance.
        self.id_buffer = None 
        self.pos_buffer = None
        self.rot_buffer = None
```

### The Update Loop (Logic)
In your game loop, `update()` should handle three phases: **Push**, **Step**, and **Pull**.

```python
    def update(self, registry, dt):
        # --- PHASE 1: INIT NEW ENTITIES ---
        # Find entities with RigidBody but handle == 0
        for entity, (body, transform) in registry.get_components(RigidBody, Transform):
            if not body.initialized:
                body.handle = self.world.create_body(
                    pos=transform.position,
                    rot=transform.rotation,
                    size=body.collider_size,
                    shape=body.shape,
                    motion=body.motion,
                    # CRITICAL: Store Entity ID in User Data!
                    # This allows you to map Collisions back to Entities.
                    user_data=entity 
                )
                body.initialized = True

        # --- PHASE 2: PUSH LOGIC (Teleporting) ---
        # If your game logic moved an object (e.g. respawn), tell physics.
        for entity, (body, transform) in registry.get_components(RigidBody, Transform):
            if transform.is_dirty: # Assuming your Transform has a dirty flag
                self.world.set_transform(body.handle, transform.position, transform.rotation)
                # If dynamic, wake it up so it reacts to the move
                self.world.activate(body.handle) 

        # --- PHASE 3: STEP ---
        # Releases GIL. Python can do other things here if threaded.
        self.world.step(dt)

        # --- PHASE 4: PULL (Bulk Sync) ---
        # The "Secret Sauce" of Culverin. Don't iterate one by one.
        
        # 1. Get indices of all ACTIVE bodies (sleeping ones don't need syncing)
        # This returns a bytes object of uint32 indices
        active_indices_bytes = self.world.get_active_indices()
        
        if not active_indices_bytes:
            return

        import numpy as np
        
        # View into indices
        indices = np.frombuffer(active_indices_bytes, dtype=np.uint32)
        
        # View into Shadow Buffers
        # Note: positions is (N, 4) floats [x,y,z,w]
        all_pos = np.frombuffer(self.world.positions, dtype=np.float32).reshape(-1, 4)
        all_rot = np.frombuffer(self.world.rotations, dtype=np.float32).reshape(-1, 4)
        all_userdata = np.frombuffer(self.world.user_data, dtype=np.uint64)

        # 2. Extract Data for Active Bodies only
        # This uses NumPy fancy indexing (very fast C-level copy)
        active_pos = all_pos[indices]
        active_rot = all_rot[indices]
        active_entities = all_userdata[indices]

        # 3. Write back to ECS
        # Ideally, your ECS supports bulk writes. If not, you loop here.
        # Even looping here is faster than calling 10,000 getters.
        for i, entity_id in enumerate(active_entities):
            # Update your ECS transform directly
            # Assuming registry has a fast lookup...
            transform = registry.transforms[entity_id]
            transform.position = active_pos[i, :3] # x,y,z
            transform.rotation = active_rot[i]     # x,y,z,w
```

## 3. Handling Collisions

Unlike PyBullet's `getContactPoints` which allocates massive lists of tuples, Culverin gives you options.

### The Fast Way (Raw Buffer)
Use `get_contact_events_raw()` for zero-allocation reading.

```python
def process_collisions(self):
    # Returns a memoryview of C structs
    raw_contacts = self.world.get_contact_events_raw()
    
    # Wrap in NumPy for easy access
    # Struct format: uint64 body1, uint64 body2, float32 pos[3], normal[3], impulse...
    # See docs for exact stride (64 bytes)
    contacts = np.frombuffer(raw_contacts, dtype=MY_CONTACT_DTYPE)
    
    for c in contacts:
        entity_a = self.world.get_user_data(c['body1'])
        entity_b = self.world.get_user_data(c['body2'])
        
        # Dispatch event to your game logic
        Events.emit("collision", entity_a, entity_b, c['impulse'])
```

### The "Trigger Event" timing 

`get_contact_events()` returns the events accumulated *during* the most recent `world.step()`. 

Therefore, collision logic should usually run immediately **after** the step.

## 4. Interpolation (Render Threading)

If you are running physics at 60Hz and rendering at 144Hz, PyBullet looks jittery. Culverin solves this natively.

In your **Render Loop**:

```python
# alpha = (current_time - last_physics_time) / fixed_dt
# 0.0 = Exactly at previous step, 1.0 = Exactly at current step
alpha = calculate_alpha() 

# Returns a packed buffer of [Pos(3f), Rot(4f)] interpolated 
# between Previous and Current physics states.
render_buffer = self.world.get_render_state(alpha)

# Upload 'render_buffer' DIRECTLY to your GPU Instance Buffer.
# No Python loops. No unpacking.
gpu.update_buffer(render_buffer)
```

## Migration Cheat Sheet

| Task | PyBullet Code | Culverin Code |
| :--- | :--- | :--- |
| **Load URDF** | `p.loadURDF("duck.urdf")` | `world.create_body(shape=SHAPE_MESH, ...)` *(Load mesh via trimesh/obj)* |
| **Move Body** | `p.resetBasePositionAndOrientation(...)` | `world.set_transform(handle, pos, rot)` |
| **Get Pos** | `pos, rot = p.getBasePositionAndOrientation(id)` | `idx = world.get_index(h); p = world.positions[idx*4:idx*4+3]` |
| **Raycast** | `p.rayTest(start, end)` | `world.raycast(start, end)` |
| **Batch Ray** | `p.rayTestBatch(starts, ends)` | `world.raycast_batch(starts_bytes, ends_bytes)` |
| **Keep Upright** | (Manual constraint setup) | `world.create_constraint(CONSTRAINT_HINGE, ...)` |
| **Character** | (Manual capsule sliding/ghost object) | `world.create_character(...)` (Native Jolt Controller) |
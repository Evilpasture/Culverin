# Culverin Physics Engine - Method Documentation

## class Module

### _dump_schema_json(...)
Internal: Dumps the current FastParse schema registry to `culverin_schema.json`. 
Used primarily for generating type stubs or debugging parser configurations.

### mutate_tuple(...)
Bypasses Python's immutability to modify a tuple index in-place.
**Arguments:**
- **target, index, value, [registry, key]**
- If 3 arguments: Swaps the pointer at `index` and recomputes the hash.
- If 5 arguments: Atomically pops from `registry`, mutates, and re-inserts to ensure dict integrity.
**Warning:** High-performance tool. Improper use can corrupt dictionary keys.

## class PhysicsWorld

### step(...)

Advances the physics simulation clock and synchronizes the internal state. 

This method acts as the "Commit" phase for the world. Because Culverin uses a deferred command architecture for thread safety, calls to create bodies, destroy bodies, or apply impulses are queued and only physically executed during this call.

**Arguments:**
- **`dt` (float):** The time delta to simulate (default is `1/60.0`). 
    - If `dt > 0`: The engine flushes the command buffer, performs collision detection, solves constraints, and integrates motion.
    - If `dt == 0`: The engine flushes the command buffer and optimizes the Broadphase (useful for structural maintenance), but does not move objects or advance the internal `time` property.

**Thread Safety & Concurrency:**
- **Blocking Behavior:** If other Python threads are currently performing queries (like `raycast` or `overlap_sphere`), `step` will block until those queries complete to ensure structural consistency.
- **Anti-Starvation:** Includes internal logic to yield to waiting mutator threads, preventing the physics loop from locking out the main application logic in high-load scenarios.
- **Re-entrancy Guard:** Raises a `RuntimeError` if called concurrently from multiple threads on the same world instance.

**Side Effects:**
1. **Command Execution:** All pending `create_body`, `destroy_body`, and `apply_impulse` calls are executed in the order they were received.
2. **Buffer Swap:** Synchronizes Jolt’s internal simulation results back into the contiguous C-arrays exposed by the `positions`, `rotations`, and `velocities` properties.
3. **Interpolation Preparation:** Moves the current state to the "previous" state buffer, allowing `get_render_state` to calculate smooth sub-frame motion.
4. **Event Generation:** Resets and repopulates the contact event buffers accessible via `get_contact_events`.


### create_body(...)

Allocates and initializes a new rigid body within the simulation. 

**Returns:**
- **`handle` (int):** A unique 64-bit identifier. This handle is **generational**; if a body is destroyed and its memory slot is reused, the old handle will become invalid, preventing "dangling pointer" bugs in your Python logic.

**Arguments:**
- **`shape` (int):** The geometry type (e.g., `culverin.SHAPE_BOX`, `SHAPE_SPHERE`).
- **`pos` (tuple):** Initial `(x, y, z)` world position.
- **`rot` (tuple):** Initial `(x, y, z, w)` quaternion. Defaults to identity `(0, 0, 0, 1)`.
- **`size` (tuple/float):** Dimensions for the shape (e.g., half-extents for a Box, radius for a Sphere).
- **`motion` (int):** Mobility type:
    - `MOTION_STATIC`: Unmovable, zero CPU cost (e.g., floors, walls).
    - `MOTION_DYNAMIC`: Full physics simulation (e.g., crates, debris).
    - `MOTION_KINEMATIC`: Position controlled via script; pushes dynamic objects but is not moved by forces.
- **`mass` (float):** Mass in kg. If positive, Culverin automatically calculates the moment of inertia. Defaults to 1.0.
- **`category` / `mask` (int):** Bitmasks for collision filtering. Bodies only collide if `(A.category & B.mask)` and `(B.category & A.mask)` are both non-zero.
- **`ccd` (bool):** Enables **Continuous Collision Detection**. Set to `True` for small, fast-moving objects (bullets) to prevent them from "tunneling" through walls.
- **`is_sensor` (bool):** If `True`, the body generates collision events but does not physically push other objects (trigger zones).

**Lifecycle & Performance:**
- **Deferred Insertion:** Calling this method queues a creation command. The body is registered in the `positions` and `rotations` buffers immediately, but it does not physically interact with the world until the next `world.step()`.
- **Memory Locality:** Bodies are packed into contiguous C-arrays. Creating a body automatically expands these arrays if the `capacity` is reached, which may involve a brief internal reallocation.
- **Numerical Safety:** Culverin performs "Sanity Checks" on input coordinates. Passing `NaN` or `Infinity` for position or rotation will raise a `ValueError` immediately rather than crashing the C++ solver later.


### create_bodies_batch(...)

Massively parallelized creation of multiple rigid bodies.

**Returns:**
- **`handles` (list[int]):** A Python list containing the unique 64-bit generational handles for every body created, in the order they were provided.

**Arguments:**
- **`positions` (list[tuple]):** A sequence of `(x, y, z)` tuples.
- **`sizes` (list[tuple/float]):** A sequence of size parameters. This list must have the same length as the `positions` list.
- **`shape` (int):** The geometry type (e.g., `SHAPE_BOX`) applied to **all** bodies in this batch.
- **`motion` (int):** The mobility type (e.g., `MOTION_DYNAMIC`) applied to **all** bodies in this batch.

**Performance & Optimization:**
- **GIL Optimization:** Culverin releases the Python Global Interpreter Lock (GIL) during the shape-building phase. This allows Jolt to construct collision settings in parallel with other Python logic.
- **Cache Locality:** Unlike calling `create_body` in a loop, this method populates the internal shadow buffers (`positions`, `rotations`) in a single contiguous memory sweep, maximizing CPU cache efficiency.
- **Reduced Lock Contention:** The engine acquires the internal `shadow_lock` only once for the entire batch, rather than once per body. This significantly reduces overhead when spawning 1,000+ objects at once.
- **Batch Command Queuing:** Creation commands are flushed into the command buffer as a single transaction, minimizing the structural work required during the next `world.step()`.

**Constraints:**
- All bodies in a single batch must share the same `shape` and `motion` type.
- Raises `ValueError` if the `positions` and `sizes` lists are of unequal length.
- Numerical validation is performed on the entire batch; if any coordinate is `NaN` or `Inf`, the operation will fail before any bodies are added to the world.


### destroy_body(...)

Queues a rigid body for removal from the physics simulation.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body to destroy.

**Lifecycle & Safety:**
- **Deferred Destruction:** The body is not immediately deleted. Instead, it is marked as `PENDING_DESTROY`. The actual memory cleanup and removal from the Jolt solver occur at the beginning of the next `world.step()`.
- **Immediate Invalidation:** As soon as this method is called, the body is considered "dead." Subsequent calls to `apply_impulse`, `set_position`, or `is_alive` using this handle will return immediately (or raise an error if `STRICT_HANDLE_ENABLED` is set in your build), even before the next `step()` occurs.
- **Handle Safety:** If the handle is invalid or stale (i.e., the body was already destroyed or replaced), the call is safely ignored.


### destroy_bodies_batch(...)

High-efficiency removal of multiple rigid bodies in a single operation.

**Arguments:**
- **`handles` (Sequence[int]):** A list, tuple, or NumPy array of 64-bit handles to be destroyed.

**Performance & Optimization:**
- **Transaction Overhead:** Drastically reduces the overhead associated with the Python-to-C bridge by processing the entire sequence in a single C-native loop.
- **Lock Optimization:** Acquired the internal `shadow_lock` only once for the entire collection. This prevents "lock-stutter" which can occur when destroying hundreds of objects (like expired particles) individually.
- **Atomic State Transition:** Every valid body in the batch is atomically transitioned to the `PENDING_DESTROY` state, ensuring that other threads (like a rendering thread calling `get_render_state`) immediately stop processing these bodies.
- **Robustness:** If the sequence contains a mix of valid and invalid handles, the engine will safely destroy all valid bodies and skip the invalid ones without halting the batch.


### create_mesh_body(...)

Instantiates a complex 3D triangle mesh as a static physics body.

**Returns:**
- **`handle` (int):** The unique 64-bit generational handle for the mesh body.

**Arguments:**
- **`pos` (tuple):** The `(x, y, z)` world-space anchor for the mesh.
- **`rot` (tuple):** The `(x, y, z, w)` quaternion orientation.
- **`vertices` (Buffer):** A flat, contiguous array of `float32` values representing vertex positions `[x0, y0, z0, x1, y1, z1, ...]`.
- **`indices` (Buffer):** A flat, contiguous array of `uint32` values representing triangle vertex indices `[t0_a, t0_b, t0_c, t1_a, ...]`.
- **`user_data` (int):** An optional 64-bit integer associated with the body.
- **`category` / `mask` (int):** Bitmasks for collision filtering.

**Physical Properties:**
- **Static Only:** Mesh bodies are forced to `MOTION_STATIC`. They do not possess mass or velocity and cannot be moved by forces. They are intended to act as the "ground" or "architecture" of your world.
- **Collision Precision:** Unlike primitive shapes (boxes/spheres), meshes provide per-triangle collision accuracy.

**Performance & Memory:**
- **Parallel BVH Construction:** Building the internal Bounding Volume Hierarchy (BVH) for a large mesh is a computationally expensive task. Culverin executes this build phase **outside the Python GIL**, allowing your application to remain responsive while Jolt processes the geometry on background threads.
- **Zero-Copy Intent:** This method accepts any object supporting the Python Buffer Protocol (e.g., `bytes`, `bytearray`, or `numpy.ndarray`).
- **Immediate Data Release:** Once the Jolt internal representation is built, Culverin immediately releases the Python buffers. You do not need to keep the original vertex/index arrays alive in Python memory to maintain the physics shape.
- **Safety Checks:** The engine validates that indices do not point to out-of-bounds vertices and checks for degenerate triangles. If the mesh is invalid, a `ValueError` or `RuntimeError` is raised before the body is added.


### create_constraint(...)

Creates a mechanical joint (e.g., Hinge, Slider, Distance) that restricts the relative motion between two rigid bodies.

**Returns:**
- **`handle` (int):** A unique 64-bit generational constraint handle. Like body handles, these are invalid after destruction to prevent stale logic.

**Arguments:**
- **`type` (int):** A `culverin.CONSTRAINT_*` constant defining the joint logic.
- **`body1` / `body2` (int):** The generational handles of the two bodies to be joined.
- **`params` (tuple/list):** Joint-specific geometry settings. (See **Constraint Types** below).
- **`motor` (dict, optional):** Configuration for motorized joints (e.g., `{'max_torque': 500, 'max_force': 1000}`).

**Constraint Types & Params:**
- **`CONSTRAINT_FIXED`**: Locks two bodies together completely. *Params: None.*
- **`CONSTRAINT_POINT`**: A "ball-and-socket" joint. *Params: `(pivot_x, pivot_y, pivot_z)`.*
- **`CONSTRAINT_HINGE`**: Restricted rotation around a single axis. *Params: `((pivot), (axis), [min_limit, max_limit])`.*
- **`CONSTRAINT_SLIDER`**: Restricts motion to sliding along a single axis. *Params: `((pivot), (axis), [min_dist, max_dist])`.*
- **`CONSTRAINT_DISTANCE`**: Maintains a specific distance range. *Params: `(min_dist, max_dist)`.*
- **`CONSTRAINT_CONE`**: Restricted "swing" rotation within a half-angle. *Params: `((pivot), (axis), half_angle)`.*

**Internal Stability & Concurrency:**
- **Deadlock Prevention:** Culverin automatically sorts body IDs before locking to ensure that multithreaded joint creation never results in a circular lock dependency.
- **Automatic Wake-up:** Creating a constraint between two bodies automatically activates them, ensuring the joint reacts immediately to the new mechanical configuration.
- **Structural Integrity:** If a body is currently `PENDING_CREATE`, Culverin flushes the command buffer before creating the constraint to ensure the underlying Jolt bodies are available for linking.


### destroy_constraint(...)

Safely removes a mechanical joint and invalidates its handle.

**Arguments:**
- **`handle` (int):** The 64-bit constraint handle to destroy.

**Lifecycle & Side Effects:**
- **Immediate Wake-up:** When a constraint is destroyed, Culverin automatically wakes up the connected bodies. This prevents "phantom physics" where objects remain floating in the air after the joint holding them has been removed.
- **Safety:** Attempting to destroy a constraint that has already been removed is a no-op.
- **Structural Timing:** Like body destruction, constraint removal is synchronized with the physics solver to prevent crashes during active query iterations.


### get_constraint_type(...)

Returns the internal type identifier of a mechanical joint. This is primarily used for diagnostic purposes or for logic systems managing heterogeneous collections of constraints.

**Returns:**
- **`type_id` (int):** An integer matching a `culverin.CONSTRAINT_*` constant. Returns `None` if the handle is invalid or stale.

**Arguments:**
- **`handle` (int):** The 64-bit handle of the constraint to query.

**Performance:**
- Executes an **O(1)** generational lookup. It is safe to call every frame within game logic loops.


### create_vehicle(...)

Creates a fully simulated, drivable wheeled vehicle (Car, Truck, etc.). Unlike primitive bodies, this returns a specialized **`Vehicle` object** used for controlling acceleration, steering, and transmission.

**Returns:**
- **`vehicle` (Vehicle):** A control object for the vehicle instance.

**Arguments:**
- **`chassis` (int):** The generational handle of the rigid body that acts as the car's frame.
- **`wheels` (list[dict]):** A list of dictionaries, one per wheel.
    - **`pos` (tuple):** Local position relative to the chassis.
    - **`radius` / `width` (float):** Physical dimensions of the tire.
    - **`suspension` (float):** Maximum suspension travel distance.
    - **`friction` (float):** Tire grip multiplier (default: 1.0).
- **`drive` (str):** Drivetrain layout. Options: `"AWD"` (All-Wheel), `"RWD"` (Rear), `"FWD"` (Front).
- **`engine` (Engine, optional):** An `Engine` config object defining torque and RPM curves.
- **`transmission` (Transmission, optional):** A `Manual` or `Automatic` config object defining gear ratios.

**Simulation Details:**
- **Raycast Suspension:** Culverin automatically attaches a `VehicleCollisionTesterRay` to the vehicle. This simulates suspension by casting rays from the chassis toward the ground, ensuring high-performance wheel-to-terrain interaction.
- **Drivetrain Logic:** The vehicle includes a simulated differential, engine inertia, and a multi-gear transmission.
- **Arcade Steering:** By default, Culverin identifies wheels with positive `Z` local coordinates as "steering wheels" and those with negative `Z` as "drive wheels" for simplified setup, though this can be overridden.
- **Thread Safety:** If the chassis body was created in the same frame, Culverin will automatically flush the world's command buffer to ensure the vehicle constraint can link to the underlying physics body.


### create_tracked_vehicle(...)

Creates a heavy-duty tracked vehicle constraint (e.g., Tank, Bulldozer, Excavator). Unlike wheeled vehicles, tracked vehicles use a specialized controller that manages track tension and differential steering.

**Returns:**
- **`vehicle` (Vehicle):** A control object specialized for track-based inputs (Left/Right track power).

**Arguments:**
- **`chassis` (int):** The generational handle of the rigid body acting as the vehicle hull.
- **`wheels` (list[dict]):** Configuration for road wheels, idlers, and sprockets.
    - Road wheels should be placed along the bottom of the hull to support the vehicle's weight.
- **`tracks` (list[dict]):** Configuration defining how road wheels are grouped into tracks.
    - **`indices` (list[int]):** The indices of the wheels (from the `wheels` list) that belong to this track.
    - **`driven_wheel` (int):** The index of the wheel that receives engine torque (the sprocket).
- **`max_torque` / `max_rpm` (float):** Engine power settings specific to heavy machinery.

**Drivetrain & Logic:**
- **Differential Steering:** Tracked vehicles use "Tank Drive." Steering is achieved by varying the power delivered to the left and right tracks independently.
- **Suspension Integration:** Every wheel in the track uses Culverin's raycast suspension system, allowing for smooth traversal over rugged terrain without the CPU cost of simulated physical track links.
- **Automatic Sync:** Like all constraints, if the chassis is newly created, the command buffer is automatically flushed to ensure a valid link.


### create_ragdoll_settings(...)

Initializes a configuration container used to define a ragdoll's physical structure. This is a non-simulated "blueprint" object; it does not add anything to the world until `create_ragdoll()` is called.

**Returns:**
- **`settings` (RagdollSettings):** A configuration object used to add parts (limbs) and joints.

**Arguments:**
- **`skeleton` (Skeleton):** A pre-finalized `Skeleton` object defining the bone hierarchy and names.

**Workflow Context:**
1. Create a **`Skeleton`**.
2. Define the hierarchy using `skeleton.add_joint()`.
3. Call `world.create_ragdoll_settings(skeleton)` to get this object.
4. Use `settings.add_part()` to map collision shapes (boxes/capsules) to the skeleton's joints.
5. Finally, call `world.create_ragdoll(settings)` to spawn the physical ragdoll.

**Memory Management:**
- This object is owned by the Python garbage collector. It can be reused to spawn multiple identical ragdoll instances (e.g., a squad of soldiers) efficiently.


### create_ragdoll(...)

Instantiates a physical multi-body ragdoll into the world based on a pre-defined blueprint.

**Returns:**
- **`ragdoll` (Ragdoll):** A control object used to drive the ragdoll's limbs toward animated poses or retrieve individual limb handles.

**Arguments:**
- **`settings` (RagdollSettings):** The blueprint object containing the limb shapes and joint constraints.
- **`pos` (tuple):** The `(x, y, z)` world position where the root of the ragdoll will be spawned.
- **`rot` (tuple):** The `(x, y, z, w)` quaternion orientation for the entire ragdoll.
- **`user_data` (int):** An optional 64-bit integer associated with the ragdoll.
- **`category` / `mask` (int):** Bitmasks for collision filtering. These are applied to every limb in the ragdoll.

**Internal Logic & Management:**
- **Automatic Pose Initialization:** Culverin takes the "neutral" bind pose defined in the skeleton, applies the requested root transform, and initializes all limbs in the correct relative positions. This prevents the ragdoll from "exploding" due to overlapping parts on spawn.
- **Limb Registration:** Although managed as a ragdoll, every limb (e.g., forearm, thigh, torso) is registered as a unique entry in the global `positions` and `rotations` buffers. You can retrieve individual limb handles using `ragdoll.get_body_handles()`.
- **Constraint Solver:** The ragdoll is added to the Jolt solver with high-priority activation, ensuring it reacts naturally to gravity and collisions immediately upon spawning.
- **Buffer Safety:** Spawning a ragdoll consumes multiple slots in the world's body capacity (one per limb). Culverin automatically resizes the internal shadow buffers if the ragdoll's parts exceed the current remaining capacity.

### create_heightfield(...)

Creates a static terrain body from a 2-dimensional grid of elevation values.

**Returns:**
- **`handle` (int):** The unique 64-bit generational handle for the terrain body.

**Arguments:**
- **`pos` (tuple):** The `(x, y, z)` world-space position of the terrain origin.
- **`rot` (tuple):** The `(x, y, z, w)` quaternion orientation.
- **`scale` (tuple):** The `(x, y, z)` scaling factor.
    - `scale.x` and `scale.z` define the horizontal distance between grid points.
    - `scale.y` acts as a vertical multiplier for the height values.
- **`heights` (Buffer):** A flat, contiguous array of `float32` values representing the elevation grid. 
- **`grid_size` (int):** The number of vertices along one side of the square grid. The `heights` buffer must contain exactly `grid_size * grid_size` floats.
- **`material_id` (int):** Optional ID to apply specific friction/restitution properties to the entire terrain.

**Terrain Mechanics:**
- **Static Only:** Heightfields are optimized for immovable geography. They are forced to `MOTION_STATIC` and do not have mass or velocity.
- **Memory Efficiency:** Instead of storing thousands of triangles, Culverin uses a compact representation that only stores the height at each vertex, allowing for massive worlds (e.g., 2048x2048 grids) with very low overhead.
- **Hole Support:** (Internal) Culverin's heightfields support "holes" (non-colliding areas) by setting height values to a specific sentinel value, allowing for caves or tunnels that pass through the terrain.

**Performance Notes:**
- **Asynchronous Build:** The internal Bounding Volume Hierarchy (BVH) for the heightfield is built **outside the Python GIL**. This ensures that loading a massive 4-million-point map does not freeze your application's main thread.
- **NumPy Integration:** This method is designed to work directly with `numpy.ndarray`. You can generate a landscape using NumPy math and pass the array directly to this method without any intermediate Python-level copies.


### create_convex_hull(...)

Creates a rigid body by "shrink-wrapping" a cloud of points into a convex shell. This is the most efficient way to create physics for irregular, organic-shaped objects (like rocks, debris, or hand-held tools) that need to be **DYNAMIC**.

**Returns:**
- **`handle` (int):** The unique 64-bit generational handle.

**Arguments:**
- **`points` (Buffer):** A flat array of `float32` values representing `(x, y, z)` coordinates. Supports `bytes`, `bytearray`, or `numpy.ndarray`.
- **`motion` (int):** Defaults to `MOTION_DYNAMIC`. Unlike standard meshes, convex hulls can be fully simulated with mass and velocity.
- **`mass` (float):** The total mass. Culverin will automatically compute the correct center of mass and moment of inertia based on the volume of the hull.
- **`pos` / `rot` (tuple):** Initial world-space transform.

**Performance & Precision:**
- **Asynchronous Hull Generation:** The math required to compute the convex shell is executed **outside the Python GIL**.
- **Numerical Robustness:** Culverin automatically handles redundant points and simplifies highly complex point clouds to ensure the resulting hull is numerically stable for the physics solver.
- **Limitation:** As a convex shape, the hull will not have "holes" or indentations (it acts like a tightly wrapped balloon). For concave objects, use `create_compound_body` or `create_mesh_body`.


### create_compound_body(...)

Creates a single rigid body composed of multiple distinct sub-shapes. This is ideal for man-made objects like chairs, vehicles, or multi-part machinery where a single box or sphere is insufficient.

**Returns:**
- **`handle` (int):** The unique 64-bit generational handle.

**Arguments:**
- **`parts` (list[tuple]):** A list of sub-shapes. Each entry must be a tuple:
    - `((local_x, local_y, local_z), (local_rx, local_ry, local_rz, local_rw), shape_type, size)`
- **`pos` / `rot` (tuple):** The world-space transform of the body's primary anchor.
- **`motion` (int):** Supports `STATIC`, `KINEMATIC`, or `DYNAMIC`.
- **`mass` (float):** If positive, Culverin calculates a unified moment of inertia based on the combined distribution of all sub-shapes.

**Internal Features:**
- **Single Handle Management:** Even if a body has 50 sub-shapes, it is managed as a single Entity in the physics world. Moving the handle moves all sub-shapes atomically.
- **Shape Cache Integration:** Culverin's C-level **Shape Cache** identifies identical sub-shapes across different compound bodies. If you create 100 identical chairs, the underlying Jolt geometry is shared in memory, drastically reducing the cache footprint.
- **Performance:** Compound bodies are significantly faster than creating multiple bodies and joining them with `FixedConstraints`. Use this method whenever parts of an object never move relative to each other.

### create_soft_body(...)

Instantiates a physically simulated soft body (deformable mesh) into the world. 

Unlike rigid bodies, soft bodies do not have a fixed shape; they are composed of vertices and constraints (edges/springs) that allow the mesh to squish, stretch, and jiggle.

**Returns:**
- **`handle` (int):** A unique 64-bit generational handle. Use this to apply forces or retrieve vertex data via `get_soft_body_vertices`.

**Arguments:**
- **`shared_settings` (SoftBodySharedSettings):** The topological blueprint defining the mesh and constraints.
- **`pos` (tuple):** Initial world-space position of the body's center of mass.
- **`rot` (tuple):** Initial world-space rotation (Quaternion: `x, y, z, w`).
- **`pressure` (float):** Internal gas pressure (default: `0.0`). High values (e.g., `500.0+`) make the body behave like an inflated balloon. Requires a manifold (closed) mesh to work correctly.
- **`vertex_radius` (float):** The physical thickness of the vertices for collision (default: `0.05`).
- **`linear_damping` (float):** Resistance to motion (default: `0.1`). Vital for stopping "numerical explosions" in high-energy deformable objects.
- **`num_iterations` (int):** Solver quality (default: `10`). Increase to `20-30` for stiffer, more structurally sound objects or complex cloth.
- **`max_linear_velocity` (float):** Hard safety cap (default: `500.0`). Prevents vertices from teleporting off-screen if the simulation math destabilizes.
- **`gravity_factor` (float):** Multiplier for global gravity. Use `0.0` for weightless cloth or `0.5` for "moon-jelly."
- **`friction` (float):** Surface friction (default: `0.2`).
- **`restitution` (float):** Bounciness/Energy return (default: `0.0`).
- **`make_rotation_identity` (bool):** If `True`, the initial rotation provided in `rot` is applied directly to the vertex positions during initialization, and the body's actual transform rotation is reset to identity. (Default: `False`).
- **`update_position` (bool):** If `True`, the Center of Mass (COM) position is updated every frame based on the average movement of the vertices. If `False`, the COM remains fixed at the origin of the body while vertices deform around it. (Default: `True`).
- **`faces_double_sided` (bool):** If `True`, collisions will be detected against both the front and back faces of the triangles. Essential for single-layered cloth meshes or open shells where the "inside" is reachable. (Default: `False`).
- **`user_data` (int):** Optional 64-bit integer for custom identification.
- **`category` / `mask` (int):** Bitmasks for collision filtering.

**Operational Mechanics:**
- **Two-Tier Tracking:** Culverin tracks the Center of Mass in the global `positions` buffer, while individual vertices are synced to a specialized shadow buffer.
- **Structural Integrity:** Soft bodies are numerically sensitive. For high-speed collisions, it is recommended to use sub-stepping (e.g., calling `world.step()` 4 times per frame with `dt/4`).

### create_ship(...)

Creates a specialized native C-controller for buoyant vessels. 

This method instantiates a high-frequency **Ship Controller** that runs directly within the Jolt Physics solver's sub-step loop. This eliminates the one-frame latency found in Python-based stabilizers, providing rock-solid stability for heavy ships even in turbulent water.

**Returns:**
- **`ship` (Ship):** A native control object for the ship instance.

**Arguments:**
- **`sled` (int):** The generational handle of the rigid body that acts as the ship's physical core (typically a compound body containing the hull volume and ballast).
- **`kp` (float):** The proportional gain for the upright stabilizer. Higher values make the ship snap back to upright faster.
- **`kd` (float):** The derivative gain (damping) for the stabilizer. Prevents the ship from oscillating or "vibrating" when reaching the upright position.
- **`throttle_force` (float):** The linear force in Newtons applied when the throttle is active.
- **`steer_speed` (float):** The target angular velocity in radians per second for turning.

**Thread Safety:**
- This method flushes the command buffer and acquires the global trampoline lock to register a `StepListener` in Jolt. It should be called during initialization rather than in high-frequency loops.


### get_soft_body_vertices(...)

Returns a zero-copy `memoryview` pointing to the real-time, world-space positions of every vertex in a simulated soft body.

**Returns:**
- **`view` (memoryview):** A contiguous buffer of vertices.
- **Format:** `float32` (or `float64` if using double precision build).
- **Stride:** 4 (X, Y, Z, W). The `W` component is purely for SIMD alignment.
- **NumPy Usage:** `verts = np.frombuffer(world.get_soft_body_vertices(h), dtype=np.float32).reshape(-1, 4)`

**Arguments:**
- **`handle` (int):** The 64-bit handle of the soft body.

**Performance:**
- Culverin's C++ sync loop performs the local-to-world transformation for every vertex at the end of every `world.step()`, ensuring the buffer is display-ready with zero Python overhead.


### apply_impulse(...)

Applies an instantaneous change in linear momentum to a body. This results in an immediate change in the body's linear velocity without affecting its rotation.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the target body.
- **`x`, `y`, `z` (float):** The impulse vector (in Newton-seconds). The direction of the vector is the direction of the "push," and the magnitude defines the strength.

**Execution & Concurrency:**
- **Center of Mass:** The impulse is applied exactly at the body's center of mass, meaning it will not cause the object to spin.
- **Deferred vs. Immediate:**
    - If the body is already active in the simulation (`SLOT_ALIVE`), the impulse is applied **immediately** on a background thread, bypassing the Python GIL.
    - If the body was created in the current frame and hasn't "stepped" yet (`PENDING_CREATE`), Culverin queues the impulse in an internal **Command Buffer** and applies it the moment the body is physically instantiated during the next `world.step()`.
- **Thread Safety:** This method is safe to call from multiple Python threads simultaneously. It uses a non-blocking architecture that only synchronizes if the physics solver is currently swapping state buffers.


### apply_angular_impulse(...)

Applies an instantaneous change in rotational momentum, causing the body to spin around its center of mass.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the target body.
- **`x`, `y`, `z` (float):** The angular impulse vector. The direction of the vector defines the axis of rotation (using the right-hand rule), and the magnitude defines the rotational speed increase.

**Operational Notes:**
- **Inertia Tensor:** The resulting change in rotational velocity is automatically scaled by the body's moment of inertia. Heavier or larger objects will require a larger angular impulse to achieve the same spin rate as smaller objects.
- **Dynamic Only:** Like linear impulses, this has no effect on `STATIC` bodies. For `KINEMATIC` bodies, use `set_angular_velocity`.
- **Causal Consistency:** Because of Culverin's command queuing, you can `create_body` and `apply_angular_impulse` in the same Python function block; the engine guarantees the spin will be applied correctly when the body "wakes up" in the solver.


### apply_impulse_at(...)

Applies an instantaneous impulse at a specific point in world space. Because the impulse is usually offset from the body's center of mass, this operation typically results in **both** a change in linear velocity and a change in angular velocity (torque).

**Arguments:**
- **`handle` (int):** The 64-bit handle of the target body.
- **`ix`, `iy`, `iz` (float):** The impulse vector (Newton-seconds).
- **`px`, `py`, `pz` (float):** The specific world-space coordinate where the impact occurs.

**Physics & Logic:**
- **Torque Generation:** If `(px, py, pz)` is not the center of mass, the body will begin to spin. This is the standard method for simulating projectile impacts on specific limbs or "kicking" one corner of a crate.
- **Causal Consistency:** Just like standard impulses, this can be called on a body that was created in the same frame (`PENDING_CREATE`). Culverin will store the world-space impact point and apply it correctly as soon as the body is physically added to the world.
- **Automatic Activation:** Applying an impulse at a point automatically wakes the body if it was sleeping.


### apply_force(...)

Applies a continuous linear force to a body’s center of mass. Forces are intended for gradual acceleration over time rather than instantaneous changes.

**Arguments:**
- **`handle` (int):** The 64-bit handle of the target body.
- **`x`, `y`, `z` (float):** The force vector in Newtons ($N = kg \cdot m/s^2$).

**Operational Best Practices:**
- **Frequency:** Unlike impulses, which should usually be called once per event, `apply_force` should be called **every frame** (or every simulation step) for as long as the force is active.
- **Accumulation:** Jolt's internal force buffers are thread-safe accumulators. You can call `apply_force` from multiple threads for the same body (e.g., one thread for wind, another for a thruster), and Culverin will sum them correctly before the next solver step.
- **Mass Dependency:** A force of 10N will accelerate a 1kg object much faster than a 100kg object ($a = F/m$). For mass-independent acceleration, use impulses or modify velocity directly.
- **Solver Reset:** Note that the force accumulator is cleared by the engine at the end of every `world.step()`.


### apply_torque(...)

Applies a continuous rotational force (torque) to a body, causing it to accelerate its spin around its center of mass.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the target body.
- **`x`, `y`, `z` (float):** The torque vector in Newton-meters ($N \cdot m$). The vector direction defines the axis of rotation, and the magnitude defines the rotational "twist" strength.

**Execution Details:**
- **Accumulation:** Like linear forces, torque is additive. Multiple calls within a single frame (from wind, motors, or stabilizers) are summed by the engine before the solver runs.
- **Frame-Based:** To maintain a constant spin acceleration, this method must be called every frame. The torque accumulator is automatically reset to zero at the end of every `world.step()`.
- **Inertia Scaling:** The resulting angular acceleration is dependent on the body's moment of inertia. A long, thin rod requires more torque to spin end-over-end than to spin like a drill.
- **Immediate vs. Deferred:** Supports Culverin’s dual-path execution: active bodies are updated immediately off-GIL, while pending bodies have their torque stored in the command buffer.


### set_gravity(...)

Dynamically updates the global gravity vector for the entire physics world.

**Arguments:**
- **`x`, `y`, `z` (float):** The acceleration vector in meters per second squared ($m/s^2$). Standard Earth gravity is typically `(0, -9.81, 0)`.

**Global Side Effects:**
- **Mass-Wide Activation:** Changing gravity is a significant structural event. Culverin automatically **wakes up every body in the simulation** when this is called. This ensures that objects resting on the floor immediately react if gravity is reversed, weakened, or shifted sideways.
- **Consistency Guard:** This operation blocks until the current physics step (if any) is complete to ensure the new gravity is applied uniformly across all sub-steps.
- **Zero Gravity:** Passing `(0, 0, 0)` effectively creates a weightless environment. Note that objects will maintain their current momentum unless acted upon by drag or manual forces.
- **Numerical Safety:** As with all coordinate inputs, Culverin validates that the gravity components are finite. Passing non-finite values will raise a `ValueError` to prevent the solver from destabilizing.

### get_gravity(...)

Gets the current gravity of the physics world.

**Returns:** A tuple of 3 floats.


### apply_buoyancy(...)

Calculates and applies the physical impulses required to simulate a body floating in a fluid.

**Returns:**
- **`submerged` (bool):** Returns `True` if any part of the body is currently below the fluid surface.

**Arguments:**
- **`handle` (int):** The generational handle of the body.
- **`surface_y` (float):** The world-space height ($Y$ coordinate) of the fluid surface.
- **`buoyancy` (float):** The upward force multiplier. A value of `1.0` typically counters gravity for an object with the same density as the fluid.
- **`linear_drag` / `angular_drag` (float):** Hydrodynamic resistance. High values prevent objects from "jittering" or spinning infinitely in the water.
- **`fluid_velocity` (tuple):** A `(x, y, z)` vector representing the current of the fluid. The body will naturally drift in this direction.
- **`dt` (float):** The time step (should match your `world.step()` delta).

**Technical Behavior:**
- **Archimedes' Principle:** Culverin calculates the displaced volume of the shape below the `surface_y` plane to determine the upward impulse.
- **Concurrency Safety:** This method is registered as an **Active Query**. If the physics thread attempts to destroy this body while buoyancy is being calculated, it will wait for the calculation to finish, preventing memory corruption.
- **Off-GIL Execution:** The heavy geometric math required to find the submerged volume is performed **outside the Python GIL**, allowing for smooth performance in multi-threaded applications.


### apply_buoyancy_batch(...)

Executes the buoyancy simulation for a large collection of bodies in a single high-speed pass. This is the recommended method for simulating ocean debris, floating cargo, or large-scale nautical environments.

**Arguments:**
- **`handles` (Buffer):** A list, tuple, or NumPy array of uint64 body handles.
- **Other Arguments:** Same as `apply_buoyancy`, applied uniformly to all bodies in the batch.

**Performance Benefits:**
- **Reduced Python Overhead:** By passing a buffer of handles (such as a NumPy array), you eliminate the overhead of calling the C-API thousands of times per frame.
- **Lock Optimization:** The engine resolves all body IDs in one brief critical section and then performs the actual buoyancy impulses in parallel without holding any locks.
- **Vectorization Friendly:** Because the handles are read from a contiguous buffer, the CPU can efficiently pre-fetch body data, making this significantly faster than a Python-level loop.



### set_position(...)

Instantly teleports a body to a specific coordinate in world space.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`x`, `y`, `z` (float):** The target world-space coordinates.

**Operational Features:**
- **Zero-Streak Reset:** Culverin performs a "hard reset" on the internal shadow buffers. It updates both the *current* and *previous* position arrays simultaneously. This ensures that `get_render_state` (interpolation) returns the exact new position immediately, preventing the "visual streak" or motion-blur artifacts that typically occur when an object teleports across the screen.
- **Support for Characters:** This method works for both standard rigid bodies and virtual `Character` controllers, making it the standard way to handle "Respawning" or "Warping" logic.
- **Solver Interaction:** Teleporting a **DYNAMIC** body that overlaps with other geometry will cause a massive "physics pop" in the next frame as the solver tries to resolve the intersection. For physical movement, prefer `set_linear_velocity` or `apply_force`.
- **Causal Consistency:** You can teleport a body in the same frame it was created (`PENDING_CREATE`).


### set_rotation(...)

Instantly sets the orientation of a body using a quaternion.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`x`, `y`, `z`, `w` (float):** The target orientation as a unit quaternion.

**Operational Features:**
- **Interpolation Guard:** Similar to `set_position`, this method updates both the current and previous rotation buffers. This prevents the renderer from trying to "spin" the object from its old orientation to the new one over the course of a frame, resulting in a perfectly clean orientation snap.
- **Normalization Validation:** Culverin validates that the provided components are finite. While Jolt is robust, passing a non-unit quaternion may result in slight scaling artifacts; ensure your input quaternion is normalized for best results.
- **Kinematic Control:** This is the primary way to control the orientation of `MOTION_KINEMATIC` bodies (like rotating platforms or elevator doors).


### set_linear_velocity(...)

Directly overrides the current translational velocity of a body or character controller.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`x`, `y`, `z` (float):** The new velocity vector in meters per second ($m/s$).

**Operational Features:**
- **Causal Consistency Mirror:** This method employs an immediate shadow-buffer update. If you call `set_linear_velocity` and then immediately query the velocity (via `get_body_stats` or the `.velocities` buffer) in the same frame, Culverin will return the **new** value. This prevents the "one-frame lag" common in other physics wrappers.
- **Kinematic Pathfinding:** This is the ideal method for controlling `MOTION_KINEMATIC` bodies, such as elevators or moving platforms, where you want them to move at a specific speed regardless of obstacles.
- **Character Support:** Can be used to apply sudden velocity changes to a virtual `Character` (e.g., jump velocity or external launch pads), though standard movement should typically use `character.move()`.
- **Zero-Frame Creation:** Fully compatible with bodies created in the same frame. The velocity is stored in the command buffer and applied the instant the body is born in the solver.


### set_angular_velocity(...)

Directly overrides the current rotational velocity of a body, causing it to spin at a specific rate.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`x`, `y`, `z` (float):** The new angular velocity vector in radians per second ($rad/s$). The vector's direction is the axis of rotation.

**Operational Features:**
- **Mass Independence:** Unlike `apply_torque`, which is resisted by the body's moment of inertia, `set_angular_velocity` forces the body to spin at the exact requested rate regardless of its weight or shape.
- **Same-Frame Feedback:** Like its linear counterpart, the new angular velocity is mirrored to the shadow buffers immediately, ensuring your game logic sees the updated state without waiting for the next `world.step()`.
- **Dynamic Stability:** Setting extremely high angular velocities on dynamic bodies can cause physics instability. It is recommended to keep rotation rates within realistic bounds for the solver's time step.
- **Constraint Interaction:** Note that if a body is restricted by a constraint (like a `FixedConstraint`), the solver may overwrite this velocity in the next step to satisfy the joint's requirements.


### set_transform(...)

Atomically updates both the world position and the orientation of a body or character controller. This is the recommended method for "teleporting" an object, as it ensures both properties are updated in a single operation.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`pos` (tuple):** The target `(x, y, z)` coordinates.
- **`rot` (tuple):** The target `(x, y, z, w)` quaternion orientation.

**Operational Features:**
- **Visual Consistency (Zero-Streak):** Like the individual setters, `set_transform` updates both the current and the previous state buffers. This forces the interpolation engine to snap immediately to the new transform, preventing "motion blur" or visual stretching artifacts during the teleport frame.
- **Atomic Execution:** The change is applied as a single transaction in the internal command buffer, ensuring the physics solver never sees a "half-updated" state where position is changed but rotation remains old.
- **Character Respawning:** Fully supports virtual characters, making it the ideal method for moving a player to a checkpoint or start point.
- **Numerical Guard:** Automatically validates that all 7 components (3 pos + 4 rot) are finite numbers before allowing the operation.


### set_collision_filter(...)

Dynamically updates the collision category and mask for a body that is already in the simulation. This allows you to change what an object can collide with on the fly (e.g., making a ghost character walk through walls or enabling a projectile to hit a specific team).

**Arguments:**
- **`handle` (int):** The 64-bit handle of the body.
- **`category` (int):** A 32-bit bitmask representing "what this object is."
- **`mask` (int):** A 32-bit bitmask representing "what this object can hit."

**Logic & Safety:**
- **Bidirectional Rejection:** Collision only occurs if `(A.category & B.mask)` and `(B.category & A.mask)` are **both** non-zero.
- **Structural Safety:** Because changing collision filters can invalidate existing broadphase pairs, this operation is highly synchronized. It will block until any active simulation steps **and** any active queries (like a background raycast) are complete to prevent memory corruption.
- **Immediate Effect:** Unlike movement commands which are often deferred, collision filter updates are written directly to the shadow buffers, ensuring that the very next query (even in the same frame) respects the new filtering rules.


### register_material(...)

Defines or updates a physics material preset in the world's global registry.

**Arguments:**
- **`id` (int):** A unique identifier for the material (typically `0-255`). This ID is used when creating bodies or meshes to assign these properties.
- **`friction` (float):** The coefficient of friction.
    - `0.0`: Perfectly slippery (like ice).
    - `0.5`: Standard surface.
    - `1.0+`: Extremely high grip (like rubber).
- **`restitution` (float):** The "bounciness" of the surface.
    - `0.0`: No bounce (plastic deformation).
    - `1.0`: Perfectly elastic bounce.
    - *Note:* Values slightly above 1.0 are physically possible in Jolt but can cause simulation instability.

**Operational Mechanics:**
- **Global Registry:** Once a material is registered, any body assigned that `material_id` will automatically inherit these properties. If you update a material's properties via this method, every body using that ID will immediately reflect the change in the next physics step.
- **Contact Event Integration:** When a collision occurs, the `material_id` of both involved bodies is included in the data returned by `get_contact_events`. This allows your game logic to trigger specific sound effects (e.g., "metal on wood") or particle systems based on the surface type.
- **Structural Safety:** Registering a material is a structural change. To ensure consistency, this method blocks until the current physics step and all active queries are complete. 
- **Dynamic Growth:** The material registry is stored in a contiguous C-array that automatically expands as you register more IDs.

**Default Behavior:**
- If a body is created with a `material_id` that has not been registered, Culverin defaults to a friction of `0.2` and a restitution of `0.0`. 
- `ID 0` is generally reserved for the default material.


### set_constraint_target(...)

Sets the goal for a motorized joint. The effect of this call depends on the type of constraint and its current motor configuration.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the motorized constraint.
- **`target` (float):** The target value for the motor:
    - **Hinge Constraints:** The target angle in **radians**.
    - **Slider Constraints:** The target position (offset from pivot) or target velocity, depending on the motor state.

**Supported Constraints:**
- **`CONSTRAINT_HINGE`**: Drives the joint to a specific rotation. If the hinge motor is active, it will apply torque to reach and maintain the `target` angle.
- **`CONSTRAINT_SLIDER`**: Drives the joint along its axis. 
    - If the motor is in *Position* mode, it drives to the `target` distance.
    - If the motor is in *Velocity* mode, it maintains the `target` speed.

**Operational Features:**
- **Automatic Activation:** To ensure the joint responds immediately, this method **automatically wakes up both bodies** connected to the constraint. This prevents a common physics bug where a motor fails to move because the connected entities have gone to "sleep" to save CPU.
- **Dynamic Updates:** You can call this method every frame (e.g., for smooth servo-driven movement) or once (e.g., to open a door).
- **Thread Safety:** This operation is synchronized with the physics solver. If the world is currently stepping, the call will block briefly until it is safe to update the joint parameters.
- **Generational Safety:** If the constraint has been destroyed and replaced, Culverin will identify the stale handle and ignore the command, preventing crashes.


### get_motion_type(...)

Queries the current mobility state of a body or character controller.

**Returns:**
- **`motion_type` (int):** An integer matching one of the following constants:
    - `culverin.MOTION_STATIC` (0): The body is immovable and has zero CPU simulation cost.
    - `culverin.MOTION_KINEMATIC` (1): The body is controlled by script/velocity and is not affected by forces or gravity.
    - `culverin.MOTION_DYNAMIC` (2): The body is fully simulated by the physics solver.
- Returns `None` if the handle is invalid or stale.

**Operational Notes:**
- **Character Support:** Fully supports virtual characters. While characters are typically kinematic, this method allows you to verify the underlying state of the controller's physics representation.
- **Consistency Guard:** This method blocks if the physics engine is currently swapping buffers to ensure the returned state is accurate to the most recent simulation step.



### set_motion_type(...)

Dynamically changes how a body behaves in the physics world. This allows you to toggle an object's mobility on the fly without destroying and recreating it.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`motion_type` (int):** The new mobility constant (`STATIC`, `KINEMATIC`, or `DYNAMIC`).

**Use Cases:**
- **Performance Optimization:** Set a `DYNAMIC` body to `STATIC` once it has come to a complete rest and is no longer needed for gameplay, effectively removing its CPU cost.
- **Gameplay Mechanics:** Transition a `KINEMATIC` platform to `DYNAMIC` when a joint breaks, allowing it to tumble and fall naturally.
- **Building Systems:** Convert a dynamic held item into a static world object when it is "placed" or "built" into a structure.

**Technical Behavior:**
- **Deferred Execution:** Changes to motion type are structural and are queued in the internal **Command Buffer**. The change will take effect at the start of the next `world.step()`.
- **Causal Consistency:** Like movement commands, you can set the motion type of a body in the same frame it was created (`PENDING_CREATE`).
- **Activation Side-Effect:** Switching a body to `DYNAMIC` automatically wakes it up, ensuring it begins simulating immediately.
- **Constraint Handling:** Be aware that changing a body to `STATIC` while it is part of a complex constraint chain may result in the joints being disabled or becoming rigid.


### activate(...)

Forces a sleeping body or character to "wake up" and resume physical simulation immediately.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body to wake.

**Operational Features:**
- **Island Re-integration:** When a body is activated, the physics solver re-adds it to the active simulation islands. It will immediately react to gravity, forces, and collisions with other objects.
- **Triggering Reactions:** Use this if you are manually modifying an object's environment (e.g., deleting the floor beneath a sleeping crate) to ensure it notices the change and starts falling.
- **Causal Consistency:** This method is compatible with bodies created in the current frame (`PENDING_CREATE`).
- **Thread Safety:** Safe to call from multiple threads. The activation is queued in the command buffer and executed at the start of the next `world.step()`.



### deactivate(...)

Forces an active body or character to enter a "sleeping" state, effectively pausing its simulation.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body to put to sleep.

**Use Cases & Performance:**
- **CPU Optimization:** If you have thousands of objects that you know do not need to move (e.g., props far away from the player), manually deactivating them can drastically reduce the CPU load of the physics solver.
- **Logic Freezing:** Use this to temporarily "freeze" an object in place for gameplay reasons, such as a cinematic pause or a status effect.

**Operational Constraints:**
- **Automatic Wake-up:** Note that a deactivated body will automatically wake up again if it is struck by a dynamic object, if a force is applied to it, or if its `motion_type` is changed.
- **Island Management:** Deactivating a body that is currently supporting other objects may cause those objects to fall or jitter as the solver recalculates the physical island.
- **Deferred Execution:** Like `activate`, this is a queued command that takes physical effect during the next simulation step.



### set_ccd(...)

Enables or disables **Continuous Collision Detection** (CCD) for a body or character controller. This is a high-precision collision mode used to combat "tunneling"—a phenomenon where small, fast-moving objects pass through thin walls because they move further in a single frame than the thickness of the obstacle.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body.
- **`enabled` (bool):** 
    - `True`: Enables **Linear Casting**. The engine will "sweep" the body’s volume between its previous and current position to ensure no collisions were skipped.
    - `False`: Resumes standard **Discrete** collision detection (default).

**When to use CCD:**
- **Projectiles:** High-speed bullets or arrows hitting thin geometry.
- **High-Speed Gameplay:** Sports games with small fast-moving balls.
- **Small Scale Simulations:** When working with very small objects relative to the world's gravity and velocity.

**Performance & Trade-offs:**
- **CPU Cost:** CCD is computationally more expensive than standard collision. It is recommended to enable it only for "active" high-speed objects rather than static scenery or slow-moving props.
- **Dynamic Only:** CCD is primarily effective for `MOTION_DYNAMIC` and `MOTION_KINEMATIC` bodies. 
- **Deferred Execution:** Like motion type changes, this is a structural update queued in the **Command Buffer**. The precision shift will take effect during the next `world.step()`.
- **Character Support:** Culverin allows CCD to be toggled for virtual characters, providing extra stability when characters are moved at extreme speeds.



### raycast(...)

Casts a single ray through the world to detect the first physical intersection.

**Returns:**
- **`(handle, fraction, normal)` (tuple):**
    - `handle` (int): The 64-bit handle of the body struck by the ray.
    - `fraction` (float): The distance to the hit point, expressed as a fraction of `max_dist` (0.0 to 1.0).
    - `normal` (tuple): The `(x, y, z)` surface normal at the point of impact.
- Returns **`None`** if no intersection was detected within `max_dist`.

**Arguments:**
- **`start` (tuple):** The `(x, y, z)` world-space origin of the ray.
- **`direction` (tuple):** The `(x, y, z)` direction vector.
- **`max_dist` (float):** The maximum distance to project the ray. Defaults to `1000.0`.
- **`ignore` (int, optional):** The handle of a body to exclude from the search (e.g., the character firing the ray).

**Performance & Safety:**
- **Off-GIL Execution:** The geometric search is performed outside the Python GIL, allowing other Python logic to run in parallel.
- **Thread Safety:** Multiple threads can perform raycasts simultaneously.
- **Integrational Integrity:** Culverin atomically verifies that hits belong to bodies that are still valid and active before returning the handle to Python.


### raycast_batch(...)

Executes a large volume of raycasts in a single high-speed operation. This is the recommended method for implementing LiDAR sensors, AI visibility sweeps, or complex multi-segment sensors.

**Returns:**
- **`results` (bytes):** A packed binary buffer containing the hit data for every ray. This is designed for zero-copy processing in NumPy. Each hit is a 48-byte structure containing:
    - `handle` (uint64)
    - `fraction` (float32)
    - `normal` (3x float32)
    - `position` (3x float32)
    - `sub_shape_id` (uint32)
    - `material_id` (uint32)

**Arguments:**
- **`starts` (Buffer):** A flat `float32` array of `(x, y, z)` origins.
- **`directions` (Buffer):** A flat `float32` array of `(x, y, z)` direction vectors.
- **`max_dist` (float):** The maximum distance applied to all rays in the batch.

**Optimization Features:**
- **Bulk Processing:** Bypasses the overhead of thousands of individual Python-to-C calls.
- **NumPy Integration:** You can pass NumPy arrays directly as inputs and cast the output `bytes` directly into a structured NumPy array for analysis.
- **Lock-Free Loop:** Once the batch begins, it processes rays without holding internal engine locks, maximizing CPU throughput on multi-core systems.


### shapecast(...)

Performs a volume-based sweep of a 3D shape through the world to detect the first intersection. Unlike a raycast, which is infinitely thin, a shapecast tests a "thick" path, making it the ideal tool for simulating large projectiles, checking character clearance (e.g., "can I fit through this gap?"), or predicting collisions for player movement.

**Returns:**
- **`(handle, fraction, contact_point, normal)` (tuple):**
    - `handle` (int): The 64-bit handle of the body struck.
    - `fraction` (float): The distance traveled along the sweep vector before impact (0.0 to 1.0).
    - `contact_point` (tuple): The `(x, y, z)` world-space coordinate where the shapes first touched.
    - `normal` (tuple): The surface normal of the impacted object at the contact point.
- Returns **`None`** if the shape traveled the full distance without hitting anything.

**Arguments:**
- **`shape` (int):** The geometry type to sweep (e.g., `culverin.SHAPE_SPHERE`, `SHAPE_BOX`).
- **`pos` (tuple):** The `(x, y, z)` world-space starting position of the shape.
- **`rot` (tuple):** The `(x, y, z, w)` orientation of the shape during the sweep.
- **`dir` (tuple):** The sweep vector. The shape is projected from `pos` to `pos + dir`.
- **`size` (tuple/float):** The dimensions of the shape being swept (e.g., radius for a sphere, half-extents for a box).
- **`ignore` (int, optional):** A body handle to exclude from the sweep (typically the body initiating the query).

**Operational Features:**
- **Volumetric Accuracy:** This method detects collisions that a thin raycast would miss, such as a large sphere grazing the corner of a crate.
- **Off-GIL Execution:** The complex geometric "GJK/EPA" algorithms used for shape-sweeping are executed **outside the Python GIL**, ensuring high performance even with complex shapes.
- **Active Query Guard:** This query is synchronized with the physics solver. If the simulation thread is currently moving or destroying objects, the query will safely wait for a consistent world state before executing.
- **Handle Validation:** Culverin atomically verifies that the struck object is still valid and has not been destroyed in the same frame, returning a reliable handle to your Python logic.


### overlap_sphere(...)

Identifies all bodies whose collision volume intersects with a defined sphere in world space.

**Returns:**
- **`handles` (list[int]):** A list of unique 64-bit generational handles for all overlapping bodies. Returns an empty list if no intersections are found.

**Arguments:**
- **`center` (tuple):** The `(x, y, z)` world-space center of the search volume.
- **`radius` (float):** The radius of the sphere.

**Technical Features:**
- **Narrow-Phase Precision:** This query uses Jolt's narrow-phase collision detection, meaning it accounts for the exact shape of every object (e.g., it correctly detects if a point of a complex mesh is inside the sphere).
- **Off-GIL Collection:** The intersection testing and hit collection occur **outside the Python GIL**. Culverin uses a high-speed stack-allocated collector to minimize memory pressure.
- **Atomic Validation:** Before returning handles to Python, Culverin performs an atomic check on every hit to ensure the bodies are still alive and valid, filtering out any entities that were marked for destruction in the current frame.


### overlap_aabb(...)

Identifies all bodies whose bounding volumes intersect with an **Axis-Aligned Bounding Box** (AABB).

**Returns:**
- **`handles` (list[int]):** A list of unique 64-bit generational handles for overlapping bodies.

**Arguments:**
- **`min` (tuple):** The `(x, y, z)` minimum corner of the box.
- **`max` (tuple):** The `(x, y, z)` maximum corner of the box.

**Performance & Use Cases:**
- **Extreme Speed:** This query operates exclusively on the **Broad-phase (Dynamic Tree)**. It does not check precise geometry, only the "bounding boxes" of objects. This makes it significantly faster than `overlap_sphere` for querying large regions of space.
- **Optimization Tool:** Ideal for frustum culling, broad proximity checks, or identifying groups of entities to be processed by more expensive logic.
- **Concurrency Safety:** Like all spatial queries, this blocks until the world state is consistent and is protected by the `active_queries` guard.


### get_index(...)

Returns the current dense array index associated with a body handle. This index corresponds directly to the rows in the raw `positions`, `rotations`, and `velocities` memoryviews.

**Returns:**
- **`index` (int):** The integer index ($0$ to `count-1`) representing the body's location in the contiguous C-arrays.
- Returns **`None`** if the handle is invalid or if the body has been destroyed.

**Use Cases & Performance:**
- **Direct Memory Access:** If you are using NumPy or a custom shader to process the world's `positions` buffer in bulk, you use `get_index` to find exactly which slot in that buffer belongs to a specific game entity.
- **ECS Integration:** This is the primary key for mapping physics data into an external Entity Component System.
- **Temporal Stability:** Note that a body's index may change after a structural optimization or if bodies are removed and the arrays are defragmented. It is recommended to query the index within the same frame you intend to use it.


### is_alive(...)

Checks if a 64-bit handle points to a valid, existing body or character controller.

**Returns:**
- **`alive` (bool):** `True` if the body exists and is currently part of the simulation; `False` if the handle is stale (the body was destroyed) or invalid.

**Operational Details:**
- **Generational Validation:** This is more than a simple "null check." Because Culverin uses generational handles, `is_alive` will return `False` if a body was destroyed and its internal memory slot was reallocated to a *new* body. This prevents logic errors where a script continues to move an object that no longer exists.
- **Lifecycle Awareness:** Returns `True` for bodies in the `PENDING_CREATE` state, allowing you to verify the success of a `create_body` call before the next physics step.
- **Thread Safety:** Performs a high-speed atomic load of the body's state. It is safe to call from any thread at any time without blocking the physics solver.


### is_active(...)

Checks if a specific body is currently "awake" and participating in the active physics simulation.

**Returns:**
- **`active` (bool):** `True` if the body is simulating (moving or being touched); `False` if the body has "settled" into a sleeping state or is `STATIC`.
- Raises a `ValueError` (or returns `False` depending on build settings) if the handle is stale.

**Operational Details:**
- **Sleeping Logic:** Jolt Physics automatically puts bodies to sleep when their velocity falls below a certain threshold for a specific period.
- **Character Support:** Virtual characters are almost always "active" when being moved, but this method can verify if they have been physically deactivated.
- **Pending Bodies:** Bodies that were just created but have not yet undergone their first `world.step()` will return `False`, as they have not yet been integrated into the solver's active islands.


### get_active_indices(...)

Returns a packed binary buffer containing the array indices of all bodies currently active in the simulation. This is an extremely high-performance tool for selectively updating graphics or AI only for objects that are actually moving.

**Returns:**
- **`indices` (bytes):** A flat buffer of `uint32` integers. 
- You can process this directly in NumPy: `np.frombuffer(world.get_active_indices(), dtype=np.uint32)`.

**Optimization Features:**
- **Deadlock-Safe Snapshot:** This method uses a sophisticated "snapshot and query" strategy. It briefly locks the world to copy the internal body IDs and then performs the activity checks **outside the lock**. This prevents the physics solver from stalling while you query the state of thousands of objects.
- **Selective Rendering:** Instead of iterating over all `count` bodies to update your GPU buffers, you can use these indices to perform "sparse updates," drastically reducing the work done by your rendering engine.
- **Zero-Allocation (Internal):** Culverin utilizes high-speed internal scratch memory to gather these indices before packing them for Python, ensuring minimal garbage collection pressure.


### shape_count (property)

The number of unique collision geometries currently stored in the engine's internal deduplication cache.

**Details:**
- **Memory Optimization:** Culverin automatically identifies identical shapes (e.g., if you create 1,000 crates of the same size, they all share a single memory-resident geometry).
- **Efficiency:** Monitoring this allows you to see the effectiveness of shape reuse in your scene. A lower `shape_count` relative to your total body `count` indicates better memory efficiency and faster collision detection.


### is_step_pending (property)

A high-speed boolean flag indicating if the physics solver is currently in the middle of a simulation step or a buffer swap.

**Details:**
- **Non-Blocking Logic:** Use this in performance-critical logic to decide whether to perform a structural change (like creating or destroying bodies). 
- **Concurrency Hint:** If `True`, calling a mutating method (like `set_collision_filter`) may block your thread briefly until the solver finishes.
- **Thread Safety:** This property uses an atomic load and is safe to query from any thread without incurring lock overhead.


### max_bodies (property)

The absolute hard limit of bodies and characters allowed in this world instance.

**Details:**
- **Fixed at Initialization:** This value is determined by the `max_bodies` setting provided during `__init__`. 
- **Pre-allocation:** Culverin uses this to pre-size critical Jolt Physics structures, ensuring that once the simulation starts, memory allocation is minimized.
- **System Constraints:** Exceeding this limit during `create_body` will raise a `RuntimeError` to prevent simulation instability.


### remaining_capacity (property)

The number of body slots currently available before the world reaches its `max_bodies` limit.

**Details:**
- **Dynamic Headroom:** Calculated as `max_bodies - count`. 
- **Usage:** This is useful for load-balancing or "pooling" systems that need to know if they have enough room to spawn a large batch of objects (like a ragdoll or a building) without triggering a capacity error.
- **Atomic Precision:** Returns the current state of the engine's internal free-list using thread-safe counters.


### positions (property)

A read-only `memoryview` of the world-space positions for all bodies.

- **Data Type:** `float64` (Double Precision).
- **Layout:** Contiguous array with a **stride of 4**. Each body occupies 32 bytes: `[x, y, z, padding]`.
- **Note:** The 4th component is purely for SIMD alignment and contains undefined data. 
- **NumPy Usage:** `pos = np.frombuffer(world.positions, dtype=np.float64).reshape(-1, 4)[:, :3]`


### rotations (property)

A read-only `memoryview` of the world-space orientations for all bodies.

- **Data Type:** `float32` (Single Precision).
- **Layout:** Contiguous array with a **stride of 4**. Each body occupies 16 bytes: `[x, y, z, w]`.
- **Note:** These are unit quaternions.
- **NumPy Usage:** `rots = np.frombuffer(world.rotations, dtype=np.float32).reshape(-1, 4)`


### velocities (property)

A read-only `memoryview` of the linear velocities for all bodies.

- **Data Type:** `float32` (Single Precision).
- **Layout:** Contiguous array with a **stride of 4**. Each body occupies 16 bytes: `[vx, vy, vz, padding]`.
- **Note:** Units are in meters per second ($m/s$).
- **NumPy Usage:** `vels = np.frombuffer(world.velocities, dtype=np.float32).reshape(-1, 4)[:, :3]`


### angular_velocities (property)

A read-only `memoryview` of the rotational velocities for all bodies.

- **Data Type:** `float32` (Single Precision).
- **Layout:** Contiguous array with a **stride of 4**. Each body occupies 16 bytes: `[wx, wy, wz, padding]`.
- **Note:** Units are in radians per second ($rad/s$).
- **NumPy Usage:** `ang_vels = np.frombuffer(world.angular_velocities, dtype=np.float32).reshape(-1, 4)[:, :3]`


### count (property)

The current number of active bodies and characters in the simulation.

- **Data-Oriented Role:** This value defines the valid range of the data buffers. If `count` is 1000, only the first 1000 rows of the `positions`, `rotations`, and `user_data` memoryviews contain valid simulation data.
- **Performance:** Queries an internal atomic counter. It is safe to access from any thread to monitor population density or to calculate buffer slices for NumPy processing.
- **Note:** This count includes bodies that are `PENDING_CREATE` but excludes those that have been fully destroyed and purged.


### time (property)

The total accumulated simulation time (in seconds) since the world was initialized.

- **Physics Time vs. Wall Time:** This property only advances when `world.step(dt)` is called with a positive `dt`. It represents the "internal clock" of the physics solver, regardless of how much real-world time has passed.
- **Precision:** Stored as a `float64` (double) to maintain high precision even in simulations that run for weeks or months of simulated time.
- **Usage:** Ideal for syncing animations, determining the age of physical effects, or identifying time-stamped collision events.


### user_data (property)

A high-speed `memoryview` of the 64-bit integers attached to every body in the world.

- **Data Type:** `uint64`.
- **Layout:** Contiguous array with a **stride of 1**. 
- **The "Logic Bridge":** This buffer is the primary way to map physics bodies back to your Python objects. By storing a unique ID or a pointer-as-integer in `user_data`, you can use NumPy to filter physics results and immediately know which game entity they belong to without performing thousands of individual `get_user_data()` calls.
- **NumPy Usage:** 
  ```python
  # Get all user IDs as a simple array
  ids = np.frombuffer(world.user_data, dtype=np.uint64)
  ```
- **Sync Behavior:** Changes made via `set_user_data()` are mirrored here immediately, ensuring that this buffer is always a "Source of Truth" for your application logic.


### get_render_state(...)

Calculates smooth, interpolated transforms for every body in the simulation to eliminate "physics jitter." This is the primary tool for synchronizing high-frequency physics (e.g., 60Hz) with high-frequency displays (e.g., 144Hz+).

**Returns:**
- **`state` (bytes):** A packed binary buffer of `float32` values. 
- **Layout:** Contiguous rows of 7 floats per body: `[x, y, z, rx, ry, rz, rw]`.
- **Total Size:** `count * 28` bytes.

**Arguments:**
- **`alpha` (float):** The interpolation factor, typically between `0.0` and `1.0`. 
    - `0.0`: Returns the physics state from the *previous* frame.
    - `1.0`: Returns the physics state from the *current* frame.
    - `0.5`: Returns a perfectly smooth halfway point.

**Technical Precision & Math:**
- **Double-Precision LERP:** Positions are interpolated using `float64` (double) precision before being packed into the `float32` output. This prevents "jitter" and "shimmering" artifacts when objects are located far from the world origin.
- **Shortest-Path NLERP:** Rotations are calculated using Normalized Linear Interpolation with a dot-product check. This ensures that objects always take the shortest rotational path and that quaternions remain unit-length.
- **Visual Teleportation Guard:** This method respects the "Zero-Streak" logic of `set_position`. If an object was teleported in the current frame, it will not interpolate from its old position, preventing visual "stretching."

**Performance Optimization:**
- **GPU-Ready:** The output buffer is formatted specifically for high-speed transfer to a Graphics API. You can map this `bytes` object directly to a **Vertex Buffer Object (VBO)** or a **Structured Buffer** in Vulkan, DirectX, or OpenGL.
- **C-Native Loop:** The entire interpolation for thousands of bodies is performed in a single optimized C loop, bypassing Python overhead entirely.
- **Thread Safety:** The operation takes a consistent snapshot of the physics state. It is safe to call from a dedicated rendering thread while the physics solver prepares the next step on a background thread.


### get_debug_data(...)

Generates high-performance geometric data representing the current state of all physics collision shapes and mechanical constraints.

**Returns:**
- **`(lines, triangles)` (tuple):**
    - **`lines` (bytes):** A binary buffer of vertices defining the edges of shapes and constraints. Render as a **Line List**.
    - **`triangles` (bytes):** A binary buffer of vertices defining solid surfaces. Render as a **Triangle List**.

**Binary Vertex Format (16 bytes per vertex):**

---

| Offset | Type | Description |
| :--- | :--- | :--- |
| `0` | `float32` | Position X |
| `4` | `float32` | Position Y |
| `8` | `float32` | Position Z |
| `12` | `uint32` | Packed Color (RGBA) |

**Arguments:**
- **`shapes` (bool):** If `True`, draws the actual collision geometry (default: `True`).
- **`constraints` (bool):** If `True`, draws joints, pivots, and movement limits (default: `True`).
- **`bbox` (bool):** If `True`, draws Axis-Aligned Bounding Boxes for every body (default: `False`).
- **`centers` (bool):** If `True`, draws the center-of-mass transform for every body (default: `False`).
- **`wireframe` (bool):** If `True`, renders shapes as lines rather than solid triangles (default: `True`).

**Technical Features:**
- **Jolt Integration:** This method hooks directly into Jolt's `DebugRenderer` API. It captures everything from sphere wireframes to the complex limit-cones of ragdoll joints.
- **GPU-Ready:** The output is a raw vertex stream. You can upload these `bytes` objects directly to your GPU's vertex buffers without any per-vertex processing in Python.
- **Efficiency:** Culverin reuses internal scratch buffers to gather this data, minimizing heap allocations. The final `bytes` objects are created in a single high-speed copy.
- **Consistency:** Automatically blocks until the simulation is idle, ensuring the geometry you see is perfectly synchronized with the most recent `world.step()`.

**Usage Tip:**
To render this in a modern engine, create two dynamic vertex buffers. Map the `lines` bytes to the first and the `triangles` bytes to the second. Use a simple shader that passes the vertex color to the fragment output.


### get_body_stats(...)

Retrieves the current, exact physical state of a body or character controller in a single, atomic operation.

**Returns:**
- **`stats` (tuple):** A nested tuple structure: `((px, py, pz), (rx, ry, rz, rw), (vx, vy, vz))`
    - **Position:** `(float, float, float)` in world-space (Double Precision).
    - **Rotation:** `(float, float, float, float)` unit quaternion.
    - **Velocity:** `(float, float, float)` linear velocity in $m/s$.
- Returns **`None`** if the handle is invalid or the body has been destroyed.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body to query.

**Operational Features:**
- **Snapshot Consistency:** This method uses an internal synchronization lock (`shadow_lock`) to ensure that the position, rotation, and velocity returned are all from the same simulation sub-step. This prevents "tearing" where position might be from the current frame but velocity is from a previous one.
- **Character Support:** Fully supports virtual characters. This is the standard way to retrieve a character's current position and velocity for game logic (e.g., triggering footstep sounds or UI markers).
- **High-Speed Building:** Culverin uses a custom C-level tuple-building engine to instantiate the result. This bypasses standard Python object creation overhead, making this call significantly faster than a series of individual property lookups.
- **Double Precision:** Position coordinates are extracted directly from the double-precision shadow buffers, ensuring sub-millimeter accuracy even for entities far from the world origin.

**When to use this vs. Properties:**
- Use **`get_body_stats(handle)`** for per-entity logic, such as a script checking if a specific player is inside a goal or calculating a single object's kinetic energy.
- Use **`world.positions`** and **`world.velocities`** (via NumPy) when you need to process hundreds or thousands of bodies at once (e.g., for rendering or global AI calculations).



### get_user_data(...)

Retrieves the custom 64-bit integer associated with a body or character controller. This is the primary mechanism for linking a physics entity back to your high-level Python game objects or logic systems.

**Returns:**
- **`data` (int):** The unsigned 64-bit integer previously assigned to this body.
- Returns **`None`** if the handle is invalid or stale (depending on your engine build settings).

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the body to query.

**Operational Features:**
- **Logic Mapping:** By storing a unique object ID, an array index, or even a memory address (cast to an integer) in the user data, you can immediately identify which game entity was involved in a collision or hit by a raycast.
- **Causal Consistency:** This method can be called on a body immediately after `create_body`, even before the first `world.step()`. Culverin ensures that data assigned during creation is readable in the same frame.
- **Support for Characters:** Fully compatible with virtual character handles, allowing you to attach unique IDs to players and NPCs.
- **Concurrency Safety:** Queries the internal shadow buffer using an atomic-safe path. It will block if the physics engine is currently performing a buffer swap to ensure the data returned is synchronized.

**Efficiency Note:**
- Use **`get_user_data(handle)`** when you need the ID of a single specific object (e.g., in response to a raycast hit).
- If you need to map *all* active bodies to their IDs (e.g., for a bulk update loop), use the **`world.user_data`** property to get a single `memoryview` and process it via NumPy for significantly higher performance.


### set_user_data(...)

Assigns a custom, unsigned 64-bit integer to a body or character controller. This value is stored within the engine's internal shadow buffers and persists until the body is destroyed or the data is overwritten.

**Arguments:**
- **`handle` (int):** The unique 64-bit handle of the target body.
- **`user_data` (int):** An unsigned 64-bit integer (e.g., a unique Entity ID, a database primary key, or a memory address).

**Operational Features:**
- **Causal Consistency Mirror:** When you call this method, Culverin immediately updates the local shadow buffer (`world.user_data`). This ensures that if you call `get_user_data()` or inspect the raw memoryview later in the same Python frame, you will see the updated value instantly without waiting for a physics step.
- **Callback Integration:** In addition to the local buffer, this method queues a command to update the Jolt Physics body’s internal metadata. This is critical for collision callbacks; when a collision occurs, the `user_data` you set here is what will be reported in the `get_contact_events` result.
- **Support for Characters:** Fully supports virtual characters, allowing you to associate NPC or Player IDs directly with their physical controllers.
- **Creation-Frame Safe:** Compatible with bodies in the `PENDING_CREATE` state. You can set the user data of a body the moment it is spawned.

**Technical Constraints:**
- The value must be a non-negative 64-bit integer.
- This operation is synchronized. If the physics solver is currently performing a buffer swap, the call will block briefly to ensure memory integrity.


### get_contact_events(...)

Retrieves a simplified list of collision interactions that occurred during the most recent `world.step()`. This is the high-performance choice for basic gameplay triggers like "Did the bullet hit a player?" or "Did the crate touch the floor?"

**Returns:**
- **`events` (list[tuple]):** A list where each entry is a 4-tuple:
    - **`handle1` (int):** 64-bit handle of the first body.
    - **`handle2` (int):** 64-bit handle of the second body.
    - **`impulse` (float):** The magnitude of the impact force.
    - **`sliding_speed_sq` (float):** The squared tangential velocity (useful for detecting "scratching" or "grinding" sounds).

**Operational Mechanics:**
- **Canonical Ordering:** Culverin automatically sorts the handles (`handle1 < handle2`). This ensures that a collision between Object A and Object B is always reported the same way, regardless of which body Jolt processed first.
- **Consumption Model:** Calling this method **clears the internal event buffer**. You should call it once per frame after `world.step()` to process the results.
- **Performance:** This method uses Culverin's **FastBuild** engine to instantiate tuples directly in C, bypassing the significant overhead of standard Python object creation.


### get_contact_events_ex(...)

Retrieves detailed collision data, including the exact coordinates and surface properties of the impact. Use this method when you need to spawn particle effects at the point of impact or play material-specific sounds (e.g., "footstep on gravel" vs. "footstep on metal").

**Returns:**
- **`events` (list[dict]):** A list of dictionaries. Each dictionary contains:
    - **`bodies` (tuple):** `(handle1, handle2)` generational handles.
    - **`position` (tuple):** `(x, y, z)` world-space contact point.
    - **`normal` (tuple):** `(nx, ny, nz)` surface normal of the collision.
    - **`impulse` (float):** Impact magnitude.
    - **`slide_sq` (float):** Squared sliding speed.
    - **`materials` (tuple):** `(mat_id1, mat_id2)` as defined in `register_material`.
    - **`type` (int):** The event phase:
        - `EVENT_ADDED` (0): New collision started this frame.
        - `EVENT_PERSISTED` (1): Collision is continuing from a previous frame.
        - `EVENT_REMOVED` (2): Bodies have stopped touching.


### get_contact_events_raw(...)

Returns a zero-copy `memoryview` providing direct access to the engine's internal contact event buffer. This is the recommended path for processing large-scale collision data using **NumPy** or **Pandas**.

**Returns:**
- **`buffer` (memoryview):** A contiguous binary view of the contact events. Each event is a **64-byte structured record**.

**Record Format (`<QQfffffffffIIII`):**

---

| Byte Offset | Type | Member | Description |
| :--- | :--- | :--- | :--- |
| `0` | `uint64` | `body1` | Handle of the first body (canonical order). |
| `8` | `uint64` | `body2` | Handle of the second body. |
| `16` | `float32` | `px, py, pz` | World-space contact position. |
| `28` | `float32` | `nx, ny, nz` | World-space contact normal. |
| `40` | `float32` | `impulse` | Impact force magnitude. |
| `44` | `float32` | `slide_sq` | Squared tangential (sliding) velocity. |
| `48` | `uint32` | `mat1, mat2` | Material IDs for both bodies. |
| `56` | `uint32` | `type` | Event phase (`ADDED`, `PERSISTED`, `REMOVED`). |
| `60` | `uint32` | `_pad` | Padding for 8-byte alignment. |

**Operational Features:**
- **Zero-Copy Snapshot:** This method creates a high-speed binary snapshot of the internal buffer. Accessing the data does not block the physics engine from starting the next simulation step.
- **Consumption Model:** Just like the high-level event methods, calling this **resets the internal event count to zero** for the next frame.
- **NumPy Integration:** This is designed to be cast directly into a structured NumPy array:
  ```python
  import numpy as np

  # Define the structure matching the C-struct
  contact_dtype = np.dtype([
      ('bodies', np.uint64, (2,)),
      ('position', np.float32, (3,)),
      ('normal', np.float32, (3,)),
      ('impulse', np.float32),
      ('slide_sq', np.float32),
      ('materials', np.uint32, (2,)),
      ('type', np.uint32),
      ('_pad', np.uint32)
  ])

  # Extract and process in bulk
  raw_view = world.get_contact_events_raw()
  events = np.frombuffer(raw_view, dtype=contact_dtype)

  # Example: Find all collisions with high impulse
  explosive_hits = events[events['impulse'] > 100.0]
  ```

**When to use this:**
- Use this when you expect more than 100 collision events per frame.
- Use this for Reinforcement Learning (RL) observations where you need to feed collision data directly into a tensor.
- Use this for custom audio engines that need to process thousands of "scratches" or "slides" simultaneously.


### save_state(...)

Captures a complete, consistent binary snapshot of the entire physics world.

**Returns:**
- **`state` (bytes):** An opaque binary blob containing the exact state of all bodies, their velocities, internal generational handles, and mapping tables.

**Operational Features:**
- **Deterministic Snapshots:** This captures not only the positions and rotations but also the linear/angular velocities, the current simulation time, and the internal Entity Component System (ECS) mappings.
- **Consistency Guard:** The engine blocks until the world is in a quiescent state (not currently stepping or performing queries) to ensure the snapshot is atomically consistent.
- **Handle Persistence:** Because it saves the internal generational counters, all `uint64` handles held by your Python logic will remain valid after a subsequent `load_state` call.
- **Networking Ready:** The resulting `bytes` object is compact and suitable for transmission over a network to implement client-side prediction or server-side state synchronization.


### load_state(...)

Restores the physics world to a previously saved state. 

**Arguments:**
- **`state` (bytes):** A binary blob previously generated by `save_state()`.

**Operational Features:**
- **Full World Restoration:** This method immediately overwrites the current simulation state. It restores all bodies to their saved positions and velocities and repopulates the engine's internal free-lists and mapping tables.
- **Jolt Synchronization:** After restoring the raw memory buffers, Culverin automatically performs a "C++ Sync," updating the underlying Jolt Physics bodies and re-integrating them into the solver's spatial acceleration structures (Broad-phase).
- **Wake-up Logic:** Every restored body is automatically activated to ensure the simulation resumes smoothly from the saved point.
- **Capacity Validation:** Culverin validates that the snapshot matches the world’s current `max_bodies` configuration. Attempting to load a snapshot from a world with a different capacity will raise a `ValueError`.

**Warning:**
This is a destructive operation. Any bodies created after the `save_state()` call was made will be purged, and any ongoing physics interactions will be overwritten.


### create_character(...)

Creates a high-level **Kinematic Character Controller** (Virtual Character). Unlike standard dynamic bodies, a character controller is designed specifically for player or NPC movement, providing smooth interaction with stairs, slopes, and walls without the "jitter" associated with purely physical movement.

**Returns:**
- **`character` (Character):** A specialized control object used for movement and state queries.

**Arguments:**
- **`pos` (tuple):** Initial `(x, y, z)` world position.
- **`height` (float):** The total height of the character (default: `1.8`).
- **`radius` (float):** The radius of the character's capsule (default: `0.4`).
- **`step_height` (float):** The maximum vertical obstacle height the character can automatically "step up" without jumping (default: `0.4`).
- **`max_slope` (float):** The maximum incline angle (in degrees) the character can walk up before sliding down (default: `45.0`).

**Key Features:**
- **Capsule-Based Physics:** Characters are represented internally as capsules. This geometry is ideal for navigating complex environments as it prevents the character from getting caught on small floor edges or corners.
- **Automated Navigation:** The controller automatically handles "sliding" along walls and "stepping" over small curbs or stairs, providing the fluid movement expected in FPS or RPG titles.
- **Dynamic Interaction:** Although kinematic, characters can physically push **DYNAMIC** objects. The force they exert is controlled by `character.set_strength()`.
- **Integrated Shadow Buffers:** Characters are first-class citizens in Culverin. They occupy slots in the global `positions` and `rotations` arrays, allowing them to be rendered and queried exactly like standard rigid bodies.
- **Collision Events:** Characters generate full collision data. You can detect when a player touches an object using `get_contact_events`.

**Operational Workflow:**
Instead of applying forces or setting velocity via the world, you move the character by calling **`character.move(velocity, dt)`**. This executes a sweep-and-slide algorithm to resolve collisions and find the final position.

### get_soft_body_vertex_count(...)

Retrieves the number of vertices currently comprising a simulated soft body.

**Returns:**
- **`count` (int):** The unsigned 32-bit vertex count.
- Returns `None` (or raises) if the handle is invalid or does not belong to a soft body.

**Arguments:**
- **`handle` (int):** The 64-bit handle of the soft body.


### get_soft_body_vertex_position(...)

Retrieves the current **World Space** position of a specific vertex. 

**Returns:**
- **`pos` (tuple):** The `(x, y, z)` coordinates of the vertex relative to the body's Center of Mass.

**Arguments:**
- **`handle` (int):** The 64-bit handle of the soft body.
- **`index` (int):** The vertex index to query.

**Note:** This provides the "deformed" local position from Jolt's solver. To get the world-space position, you must either use the shadow buffer via `get_soft_body_vertices` or manually multiply this result by the body's world transform.


### get_soft_body_local_vertices(...)

Despite the name, Jolt returns these in World Space. This method extracts the deformed positions of all vertices in a single bulk operation.

**Returns:**
- **`buffer` (bytes):** A packed binary buffer of `float32` values.
- **Layout:** Contiguous `[x, y, z]` triplets (12 bytes per vertex).
- **NumPy Usage:** `verts = np.frombuffer(world.get_soft_body_local_vertices(h), dtype=np.float32).reshape(-1, 3)`

**Arguments:**
- **`handle` (int):** The 64-bit handle of the soft body.

**Performance:**
This method uses direct Jolt memory access and is faster than calling `get_soft_body_vertex_position` in a loop. Unlike `get_soft_body_vertices` (which returns a persistent `memoryview`), this method returns a **new copy** of the data as a `bytes` object. Use this when you specifically need local-space offsets for skeletal skinning or mesh-morphing logic.


### _benchmark_parse(...)

Internal macro overhead benchmark. (For engine developers).


### _benchmark_build(...)

Internal tuple generation benchmark. (For engine developers).


## class Character

### handle (property)

The unique 64-bit generational handle associated with this character.

- **Role:** This handle identifies the character within the `PhysicsWorld`. You can pass this handle to world-level methods such as `set_user_data`, `get_body_stats`, or as an `ignore` parameter in a `raycast`.
- **Atomic Access:** Retrieving the handle is a lock-free operation.


### move(...)

Moves the character through the world using a velocity-based "Sweep and Slide" algorithm. This is the primary method for controlling player or NPC movement.

**Arguments:**
- **`velocity` (tuple):** The desired movement vector in meters per second $(x, y, z)$. 
    - Note: This should include gravity (e.g., `-9.81` on the Y-axis) if you want the character to fall, as kinematic characters do not automatically react to global gravity.
- **`dt` (float):** The time delta for the movement (should match your `world.step()` delta).

**Operational Mechanics:**
- **Kinematic Integration:** Unlike dynamic bodies, which are moved by the solver, the character controller manually "sweeps" its capsule shape through the world. If it hits an obstacle, it automatically calculates the "slide" vector to move along walls or up slopes.
- **Auto-Stepping & Slopes:** The movement logic automatically handles walking up stairs (defined by `step_height`) and ensures the character remains "stuck" to the ground when walking down slopes, preventing jittery "bunny-hopping" behavior.
- **Dynamic Interaction:** While moving, the character will automatically detect and push **DYNAMIC** rigid bodies in its path. The strength of this push is configured via `set_strength()`.
- **Visual Smoothness:** `move()` automatically updates the character’s **Previous Position** shadow buffer. This ensures that `get_render_transform` provides perfectly smooth interpolation even if the character’s movement frequency differs from the display refresh rate.

**Concurrency & Performance:**
- **Off-GIL Execution:** The heavy collision-sweep calculations are performed **outside the Python GIL**, allowing for multiple characters to be moved in parallel across different threads.
- **Consistency Guard:** This method synchronizes with the `PhysicsWorld`. It will block briefly if the world is currently in the middle of a simulation step to ensure movement occurs against a stable world state.


### get_position(...)

Retrieves the character’s current world-space coordinates.

**Returns:**
- **`pos` (tuple):** A 3-tuple of floats `(x, y, z)` representing the center of the character's capsule.
- **Precision:** Returned in **Double Precision** (`float64`), ensuring accuracy even when the character is kilometers away from the world origin.

**Operational Features:**
- **Consistency Lock:** This method briefly acquires the world’s internal `shadow_lock` to ensure the coordinates are retrieved atomically. This prevents "tearing" if the position is queried while a movement thread is simultaneously updating the character.
- **SIMD Optimized:** The underlying C-implementation utilizes aligned memory access to retrieve the position with minimal CPU overhead.



### set_position(...)

Instantly teleports the character to a new world coordinate. Use this for "warping" a character (e.g., entering a building, respawning, or moving between levels).

**Arguments:**
- **`pos` (tuple):** The target `(x, y, z)` world-space coordinates.

**Operational Features:**
- **Zero-Streak Reset:** This is a high-end rendering feature. When you call `set_position`, Culverin instantly updates both the **current** and **previous** position shadow buffers. This forces the interpolation engine (`get_render_transform`) to snap to the new location immediately.
- **Visual Integrity:** By resetting the interpolation history, you prevent the character from "smearing" or creating a giant visual streak across the screen during the teleport frame.
- **Collision Safety:** Note that `set_position` does not perform collision checks. If you teleport a character into a wall or another body, it will be stuck until moved out or until the physics solver forces a separation.
- **Synchronized Execution:** This method blocks if the physics solver is currently mid-step to ensure the underlying Jolt Virtual Character and the world's memory buffers remain perfectly in sync.


### set_rotation(...)

Instantly sets the orientation of the character controller using a quaternion.

**Arguments:**
- **`rot` (tuple):** The target unit quaternion `(x, y, z, w)`.

**Operational Features:**
- **Interpolation Guard:** Like `set_position`, this method performs a "Zero-Streak" reset on the rotation shadow buffers. Both the current and previous orientations are updated simultaneously to prevent the character from "spinning rapidly" for a single frame when teleporting or snapping to a new direction.
- **Unit Validation:** Culverin performs high-speed numerical checks to ensure the provided components are finite. While the physics solver is robust, it is recommended to pass a normalized unit quaternion for consistent results.
- **Capsule Alignment:** Note that standard character capsules are typically aligned vertically. Setting a rotation that tilts the character away from the world's "Up" vector may affect the controller's ability to walk up stairs or slopes.


### is_grounded(...)

Queries the character's current relationship with the ground. This is the primary trigger for logic such as "can jump," "play falling animation," or "apply air friction."

**Returns:**
- **`grounded` (bool):** 
    - `True`: The character is currently supported by a walkable surface (ground, floor, or shallow slope).
    - `False`: The character is currently in the air, sliding down a steep slope, or touching a vertical wall.

**Operational Features:**
- **Solver Precision:** This status is calculated during the character's most recent movement sweep. It accounts for the `max_slope` setting provided during creation; if the character is touching a slope steeper than that limit, `is_grounded` will return `False`.
- **Constraint Awareness:** The grounded state is physically verified against both rigid bodies and static environment meshes.
- **State Synchronization:** This check is synchronized with the world’s internal state, ensuring it remains accurate even if queried from a non-movement thread.


### set_strength(...)

Configures the maximum physical force the character can exert when colliding with dynamic objects in the environment.

**Arguments:**
- **`strength` (float):** The "push" multiplier. Higher values allow the character to shove heavy crates or vehicles aside with ease; lower values make the character stop or slow down when hitting obstacles.

**Operational Mechanics:**
- **Physical Weight:** This value determines the magnitude of the impulse applied to dynamic bodies during the character's movement sweep.
- **Thread Safety:** The strength value is stored in an **Atomic Float**. You can update a character's strength (e.g., for a "super-strength" power-up) from any thread, and the physics solver will pick up the new value in its next internal callback immediately.
- **Realism:** To maintain a realistic feel, set this value proportional to the "expected" mass of the character. If `strength` is too high, the character may launch objects with explosive force upon walking into them.


### get_render_transform(...)

Calculates the character's smooth, sub-frame position and rotation for the current display frame. This is the recommended method for retrieving data to position your character mesh or camera.

**Returns:**
- **`(pos, rot)` (tuple):**
    - **`pos` (tuple):** The interpolated `(x, y, z)` world position (Double Precision).
    - **`rot` (tuple):** The interpolated `(x, y, z, w)` quaternion (Shortest-path NLERP).

**Arguments:**
- **`alpha` (float):** The interpolation factor (typically `0.0` to `1.0`). This should represent the progress between the last and current physics steps.

**Technical Precision:**
- **Double-Precision LERP:** Like the world-level render state, position interpolation is performed in 64-bit floats to eliminate jitter and shimmering in large worlds.
- **Shortest-Path Rotations:** Rotations are calculated using Normalized Linear Interpolation with double-cover handling. This ensures the character always rotates via the most direct path and orientation always stays unit-length.
- **Visual Teleport Protection:** This method automatically respects `set_position()` and `set_rotation()`. If the character was teleported in the current frame, interpolation is bypassed to ensure an instantaneous visual snap.
- **Efficiency:** Retrieves the "Previous" state from Culverin's internal C-arrays and the "Current" state directly from the Jolt virtual character object in a single synchronized C-API call.


## class Vehicle

### wheel_count (property)

The total number of physical wheels or road-wheels attached to the vehicle.

- **Role:** This value is used as the upper bound when iterating through wheels to retrieve their visual transforms for rendering.
- **Immutable:** The number of wheels is fixed during the `create_vehicle` or `create_tracked_vehicle` call and cannot be modified for the lifetime of the vehicle instance.
- **Performance:** Accessing this property is an **O(1)** operation that reads directly from the engine's internal metadata.




### set_input(...)

Applies high-level driver inputs to a wheeled vehicle. This method manages the complex coordination between the throttle, braking, and steering, using an internal state machine to handle gear transitions and arcade-style driving logic.

**Arguments:**
- **`forward` (float):** Driving intent ranging from `-1.0` to `1.0`.
    - **Positive values:** Acceleration and automatic shifting into forward gears.
    - **Negative values:** Intent to reverse; automatically shifts the transmission into the reverse gear.
    - **Zero:** Shifts the transmission to Neutral and allows the vehicle to coast.
- **`right` (float):** Steering input from `-1.0` (Full Left) to `1.0` (Full Right).
- **`brake` (float):** Standard service brake force from `0.0` to `1.0`.
- **`handbrake` (float):** Rear-wheel lockup force from `0.0` to `1.0`. Useful for drifting or emergency stops.

**Operational Features:**
- **Arcade State Machine:** Culverin implements a sophisticated "Smart Shift" logic. If you are moving forward and press the reverse key (negative `forward`), the engine initially applies the brakes to bring the car to a stop before automatically engaging the reverse gear.
- **Automatic Activation:** Calling this method automatically wakes the vehicle's chassis if it has gone to sleep. This ensures the vehicle responds immediately to user input without requiring a manual `activate()` call.
- **Rolling Resistance:** When `forward` is zero, the engine automatically applies a subtle "rolling resistance" to the brakes, preventing the vehicle from slowly rolling forever on flat surfaces.
- **Concurrency Safety:** This method is synchronized with the world’s internal buffers. It blocks briefly if the physics solver is currently swapping state to ensure inputs are applied to a consistent drivetrain state.

**Technical Tip:**
For the most realistic feel, call this method every frame based on your game's controller or keyboard state. The vehicle's simulated engine and transmission will handle the sub-frame torque integration.



### set_tank_input(...)

Applies specialized driver inputs to a tracked vehicle (e.g., Tanks, Bulldozers). This method utilizes a differential steering model, where steering is achieved by varying the torque delivered to the left and right tracks independently.

**Arguments:**
- **`left` (float):** Power applied to the left track, ranging from `-1.0` (Full Reverse) to `1.0` (Full Forward).
- **`right` (float):** Power applied to the right track, ranging from `-1.0` to `1.0`.
- **`brake` (float):** Overall braking force from `0.0` to `1.0`, applied to both tracks simultaneously.

**Operational Features:**
- **Differential Steering (Tank Drive):** By providing different values for `left` and `right`, you can perform "Pivot Turns" (turning in place by running tracks in opposite directions) or "S-turns" while moving.
- **Kickstart Transmission Logic:** Tracked vehicles in Culverin include an automated gearbox. To prevent the vehicle from "crawling" when idle, the transmission defaults to Neutral. The moment the total throttle exceeds a minimal threshold, the engine automatically engages 1st gear to begin movement.
- **Automatic Wake-up:** Like wheeled vehicles, calling this method automatically wakes up the vehicle’s hull. This ensures the tank remains physically active and responsive to control inputs.
- **Neutral Safety:** If no input is detected on either track, the vehicle automatically shifts back to Neutral, allowing it to remain stationary on slopes without burning simulated fuel or jittering.

**Simulation Tip:**
For heavy machinery, abrupt changes from full forward to full reverse can cause the vehicle to "buck" due to the high simulated torque. For smooth movement, interpolate your input values over a few frames to simulate a human driver's throttle response.


### get_wheel_transform(...)

Retrieves the current world-space position and orientation of a specific wheel. This is the simplest way to render wheels if your graphics engine uses a flat world-space hierarchy.

**Returns:**
- **`(pos, rot)` (tuple):**
    - **`pos` (tuple):** The `(x, y, z)` world position of the wheel center (Double Precision).
    - **`rot` (tuple):** The `(x, y, z, w)` unit quaternion representing the wheel's combined steering and rolling rotation.
- Returns **`None`** if the wheel index is out of bounds.

**Arguments:**
- **`index` (int):** The index of the wheel (from $0$ to `wheel_count - 1`).

**Operational Features:**
- **Suspension Integration:** The returned position reflects the real-time compression of the suspension springs. As the vehicle drives over bumps, the wheel's $Y$ coordinate will fluctuate relative to the chassis.
- **Spin & Steer:** The rotation quaternion accounts for both the steering angle (yaw) and the physical rotation of the tire (roll) as it moves across the ground.
- **Snapshot Consistency:** This method blocks briefly if the physics world is currently swapping state buffers, ensuring the transform is perfectly in sync with the most recent chassis position.


### get_wheel_local_transform(...)

Retrieves the position and orientation of a wheel relative to the vehicle's chassis. This is the recommended method for game engines that use a parent-child transform hierarchy (where the wheel is a child of the car body).

**Returns:**
- **`(pos, rot)` (tuple):**
    - **`pos` (tuple):** The `(x, y, z)` local offset from the chassis center of mass.
    - **`rot` (tuple):** The `(x, y, z, w)` local rotation relative to the chassis orientation.

**Arguments:**
- **`index` (int):** The index of the wheel.

**Operational Features:**
- **Suspension Animation:** Because the coordinates are local, the $X$ and $Z$ positions are typically constant, while the $Y$ position represents the current "suspension offset." This is ideal for animating shock absorbers and mechanical linkages.
- **Coordinate System:** Culverin automatically detects the wheel's side (Left vs. Right) based on the local coordinates provided during creation, ensuring that rotation and steering axes are correctly mirrored.
- **High-Speed Extraction:** Like the world-space version, this is optimized in C to perform the matrix-to-quaternion conversion with minimal overhead.


### destroy(...)

Immediately unregisters the vehicle from the physics simulation and releases all associated C-level resources.

**Operational Features:**
- **Manual Resource Management:** While Culverin will clean up the vehicle when the Python object is garbage collected, `destroy()` allows you to free the memory and unregister the Jolt constraints immediately. Use this when a vehicle is "deleted" or "reset" in your game logic to ensure the physics solver stops processing it instantly.
- **Structural Cleanup:** This method removes the internal `StepListener` (which processes engine/transmission logic) and the `VehicleConstraint` from the world. 
- **Safety:** Once called, the vehicle object becomes invalid. Any subsequent calls to `set_input` or `get_wheel_transform` will be ignored or return `None`.
- **Synchronized:** This is a blocking operation. It waits for the physics solver to be idle to ensure the vehicle is not being accessed by Jolt while its memory is being freed.


### get_debug_state(...)

Prints a comprehensive real-time telemetry dump of the vehicle's internal state to the system console (`stderr`). This is an invaluable tool for tuning suspension stiffness, gear ratios, and tire grip.

**Telemetry Provided:**
- **Drivetrain:** Current Engine RPM, Torque Output, Gear Index, and Clutch Friction.
- **Wheel State:** For every wheel, it reports:
    - **Contact:** Whether the tire is currently touching the ground.
    - **Suspension:** The current compression length in meters.
    - **Speed:** Angular velocity (rad/s) and tire surface speed (m/s).
    - **Slip (Lambda):** Longitudinal and Lateral slip values, indicating how much the tire is skidding or sliding.

**Operational Notes:**
- **Debug Build Only:** This method is typically only functional if the Culverin C-extension was compiled with the `CULVERIN_DEBUG` flag enabled. In production builds, this call is a no-op.
- **Performance:** Because it prints to the console and acquires a snapshot lock, this should only be used during development and not in final production code.
- **Solver Safety:** Does not execute if the physics world is currently in the middle of a simulation step.


## class Skeleton

### add_joint(...)

Adds a new joint (bone) to the skeleton definition.

**Returns:**
- **`joint_index` (int):** The unique integer index assigned to this joint. You will use this index to specify the `parent_index` for child joints and to map collision shapes in `RagdollSettings`.

**Arguments:**
- **`name` (str):** A unique string identifier for the joint (e.g., `"Hips"`, `"LowerArm_L"`).
- **`parent_index` (int):** The index of the parent joint. 
    - Use **`-1`** to designate a root joint (the starting point of the skeleton).
    - For all other joints, provide the index returned by a previous `add_joint` call.

**Structural Rules:**
- **Hierarchical Ordering:** Culverin (and the underlying Jolt engine) requires that a parent joint be added **before** its children. Failure to follow this order will result in an error during the `finalize()` call.
- **Root Joints:** A skeleton can have multiple root joints if necessary, though most character skeletons use a single root (typically the Hips or Pelvis).
- **Naming:** While joint names are used for lookup, the engine identifies joints primarily by their index for maximum performance during physics-to-animation blending.

**Workflow Example:**
```python
skel = culverin.Skeleton()
hips = skel.add_joint("Hips", -1)          # Root
spine = skel.add_joint("Spine", hips)      # Child of hips
head = skel.add_joint("Head", spine)      # Child of spine
```


### get_joint_index(...)

Retrieves the integer index assigned to a joint based on its name. This is the recommended way to resolve bone indices when working with external animation data or skeletal meshes.

**Returns:**
- **`index` (int):** The unique index of the joint.
- Returns **`-1`** if no joint with that name exists in the skeleton.

**Arguments:**
- **`name` (str):** The string name of the joint provided during `add_joint`.

**Operational Features:**
- **Lookup Performance:** This method performs a string-to-index lookup within the skeleton's internal dictionary. While efficient, it is recommended to cache the returned index if you need to access it every frame.


### finalize(...)

Validates and "bakes" the skeleton hierarchy into its final internal representation. This method must be called exactly once after all joints have been added and before the skeleton can be used to create ragdolls.

**Operational Side Effects:**
- **Parental Index Calculation:** Culverin pre-calculates the complete parent-child mapping, optimizing the skeleton for high-speed coordinate transformations during physics simulation.
- **Structural Validation:** The engine performs a rigorous check to ensure:
    1. Every joint has a valid parent index (or is a root).
    2. No circular dependencies exist (e.g., a joint cannot be its own ancestor).
    3. Joints are ordered correctly (parents are defined before children).
- **Runtime Preparation:** Once finalized, the skeleton becomes read-only. You cannot add further joints, but you can safely pass the skeleton to `world.create_ragdoll_settings()`.

**Constraint:**
- If the skeleton structure is invalid, this method raises a `RuntimeError` with a detailed description of the violation (e.g., "Skeleton joints are out of order").


## class Ragdoll

### drive_to_pose(...)

Uses internal motorized joints to drive every limb of the ragdoll toward a target animated pose. This is the foundation of **Physical Animation**, allowing a character to follow a motion-captured animation while still reacting physically to collisions, weight, and external forces.

**Arguments:**
- **`root_pos` (tuple):** The desired `(x, y, z)` world position of the skeleton's root (typically the hips).
- **`root_rot` (tuple):** The desired `(x, y, z, w)` world orientation of the root.
- **`matrices` (Buffer):** A flat binary buffer containing local-space transformation matrices for every joint in the skeleton.
    - **Format:** Each joint requires a 4x4 matrix of `float32` values (64 bytes per joint).
    - **Total Size:** `joint_count * 64` bytes.
    - **NumPy Usage:** `ragdoll.drive_to_pose(pos, rot, my_animated_matrices.tobytes())`

**Operational Mechanics:**
- **Hybrid Simulation:** When this method is called, Culverin performs a two-step operation:
    1. **Teleport (Optional Snap):** It updates the underlying physics bodies to match the target pose.
    2. **Motorized Tracking:** It activates the ragdoll's internal motors to "stiffen" the joints, attempting to maintain that pose against gravity and obstacles.
- **Active Ragdolls:** Unlike a "limp" ragdoll, a ragdoll being "driven" can stand upright, walk, or perform combat moves while remaining physically solid. If the character is hit by a massive object, the motors will be overpowered, causing the character to reel back or fall realistically.
- **Efficiency:** The matrix processing and joint-target calculations are performed in a highly optimized C loop that directly interfaces with Jolt's `DriveToPoseUsingMotors` API.

**Technical Tip:**
To achieve a "Hit Reaction" or "Death" effect, simply stop calling `drive_to_pose()`. The motors will turn off, and the ragdoll will immediately transition to a fully passive physical state (collapsing under its own weight).


### get_body_handles(...)

Returns a list of the 64-bit generational handles for every physical body (limb) that constitutes the ragdoll.

**Returns:**
- **`handles` (list[int]):** A list of handles in the order they were defined in the skeleton. 
- **Note:** If a specific limb has been destroyed or is invalid, the corresponding entry in the list will be **`None`**.

**Use Cases:**
- **Per-Limb Interaction:** Use these handles with standard `PhysicsWorld` methods. For example, you can call `world.apply_impulse(hand_handle, ...)` to make a character drop a weapon or `world.get_body_stats(head_handle)` to detect high-velocity head impacts.
- **Limb Specific Logic:** Attach unique identifiers to specific limbs using `world.set_user_data()` so your collision callbacks can distinguish between a hit to the torso and a hit to the foot.


### get_debug_info(...)

Retrieves a detailed snapshot of the physical state for every limb in the ragdoll. This is an essential tool for verifying that your physical rig matches your visual skeleton and for debugging motorized pose-following.

**Returns:**
- **`info` (list[dict]):** A list of dictionaries, one per limb, containing:
    - **`index` (int):** The joint index within the ragdoll.
    - **`pos` (tuple):** The current `(x, y, z)` world-space position of the limb.
    - **`vel` (tuple):** The current `(vx, vy, vz)` linear velocity of the limb.

**Operational Features:**
- **Snapshot Consistency:** All data is retrieved in a single synchronized pass. This ensures that the positions and velocities for all 20+ limbs are from the exact same simulation sub-step, preventing "visual jitter" in debug overlays.
- **FastBuild Optimized:** Although this method returns Python dictionaries, the creation of these objects is handled by Culverin's C-native **FastBuild** engine, making it significantly more efficient than manual dictionary construction in a Python loop.
- **Diagnostic Usage:** While highly optimized, this method still creates Python objects for every limb. For high-performance data extraction (e.g., feeding a neural network), prefer using the global `world.positions` and `world.velocities` memoryviews combined with the limb handles.


## class RagdollSettings

### add_part(...)

Assigns a physical body and a mechanical joint to a specific skeleton bone. This effectively "skins" the abstract skeleton with physical properties.

**Arguments:**
- **`joint_index` (int):** The index of the skeleton joint being physicalized.
- **`shape_type` (int):** The collision geometry for this limb (e.g., `SHAPE_CAPSULE` for limbs, `SHAPE_BOX` for the torso).
- **`size` (tuple/float):** Dimensions of the shape.
- **`mass` (float):** The mass of this specific limb in kg (default: `10.0`).
- **`parent_index` (int):** The index of the joint this part should physically attach to. 
    - Use **`-1`** for the root part (no constraint).
    - Providing a parent index automatically creates a **Swing-Twist Constraint** between this part and its parent.
- **`twist_min` / `twist_max` (float):** The rotation limits (in radians) around the limb's primary axis.
- **`cone_angle` (float):** The half-angle limit for the limb's "swing" motion.
- **`axis` / `normal` (tuple):** Vector pairs defining the local orientation of the joint's constraint.

**Operational Features:**
- **Automatic Scaling:** Culverin calculates the correct inertia tensor for each limb based on its shape and mass.
- **Integrated Shape Cache:** Like standard bodies, limb shapes are deduplicated. If both arms use the same capsule size, they will share memory for the underlying collision geometry.
- **Physical Hierarchy:** Defining a `parent_index` here creates a high-performance motorized joint, which is required for `Ragdoll.drive_to_pose()`.


### stabilize(...)

Performs an automated structural analysis of the ragdoll hierarchy to ensure physical stability.

**Returns:**
- **`success` (bool):** `True` if the ragdoll was successfully stabilized.

**Why use this:**
- **Jitter Prevention:** In complex ragdolls, limbs often overlap at the joints. Without stabilization, the physics engine would constantly try to "push" the connected limbs apart, causing the character to shake or vibrate.
- **Automated Collision Filtering:** This method automatically identifies connected limbs and disables collisions between them. It also adjusts joint positions to perfectly match the skeletal bind pose.
- **Workflow Tip:** Always call `stabilize()` after you have finished adding all parts but **before** you call `world.create_ragdoll()`.


## class SoftBodySharedSettings

Defines the topology, physical mass distribution, and internal constraints of a soft body. This is a non-simulated "blueprint" object.

### add_vertex(...)

Adds a single vertex (node) to the soft body definition.

**Arguments:**
- **`pos` (tuple):** The `(x, y, z)` local-space coordinate of the vertex.
- **`inv_mass` (float):** The inverse mass ($1/m$). 
    - `1.0`: 1kg vertex. 
    - `0.0`: Technically pins the vertex (though `add_pinned_vertex` is the preferred API).


### add_face(...)

Defines a triangular surface on the mesh by linking three vertex indices. Edges of these faces automatically become distance springs when `create_constraints` is called.

**Arguments:**
- **`v1`, `v2`, `v3` (int):** Vertex indices in the order they were added.


### add_pinned_vertex(...)

[Positional Only] Marks a specific vertex as **Pinned**. Pinned vertices have zero inverse mass and are fixed in world-space (relative to the body's center of mass).

**Arguments:**
- **`index` (int):** The vertex index to anchor.

**IMPORTANT:** Must be called **BEFORE** `create_constraints()` to ensure the structural springs correctly account for the anchored point.


### create_constraints(...)

Generates the internal physical structure (the "bones") of the soft body.

**Arguments:**
- **`compliance` (float):** The inverse of stiffness. 
    - `0.0`: Perfectly rigid.
    - `0.0001`: Stiff Jelly/Rubber.
    - `0.01`: Floppy Paper/Cloth.
- **`bend_type` (int):** How the mesh resists folding.
    - `culverin.BEND_NONE` (0): No bending resistance.
    - `culverin.BEND_DISTANCE` (1): Standard "internal struts" (Best for cubes/blobs).
    - `culverin.BEND_DIHEDRAL` (2): Resists changes in triangle angles (Best for cloth/capes).

**Technical Feature:**
This method automatically applies the provided compliance to both the **Edge Constraints** and the **Shear Constraints**, providing significantly higher integrity than simple distance springs alone.


### optimize()

[No Arguments] Finalizes the blueprint and builds the spatial acceleration structures (BVH) required for high-performance collision detection.

**Workflow Requirement:**
This must be the **final call** on a settings object. Once optimized, you should not add more vertices or faces.


### get_vertex_position(...)

[Positional Only] Retrieves the **Rest Pose** position of a vertex from the shared settings.

**Returns:**
- **`pos` (tuple):** The `(x, y, z)` local coordinates of the vertex as originally defined.

**Arguments:**
- **`index` (int):** The vertex index to query.

### add_vertices(...)

Massively parallelized addition of vertices to the soft body blueprint. This is the recommended method for creating complex meshes (e.g., loading from an OBJ file or generating via NumPy).

**Arguments:**
- **`positions` (Buffer):** A flat, contiguous array of `float32` values representing `(x, y, z)` coordinates.
- **`inv_masses` (Buffer, optional):** A flat array of `float32` values defining the inverse mass for each vertex. 
    - If provided, the length must exactly match the number of vertices in the `positions` buffer.
    - If `None` (default), all vertices in this batch are assigned a mass of 1.0kg.

**Performance & Memory:**
- **Single Allocation:** Unlike calling `add_vertex` in a loop, this method performs exactly one heap allocation by pre-reserving memory for the entire batch.
- **SIMD Optimized:** The underlying C++ loop is designed to be auto-vectorized by the compiler for high-speed coordinate conversion.
- **NumPy Integration:** Designed to ingest NumPy arrays directly:
  ```python
  positions = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)
  settings.add_vertices(positions.tobytes())
  ```

**Constraints:**
- Raises `ValueError` if the buffer size is not a multiple of 12 bytes (3x float32).
- Raises `RuntimeError` if called after `optimize()`.


### add_faces(...)

High-speed batch definition of the soft body's surface triangles.

**Arguments:**
- **`indices` (Buffer):** A flat, contiguous array of `uint32` values representing vertex indices. Every 3 values define one triangular face.

**Operational Details:**
- **Safety Validation:** Culverin performs a C-native bounds check on every index provided. If an index points to a non-existent vertex, the method raises an `IndexError` before any data is sent to Jolt.
- **Winding Order:** Standard counter-clockwise winding is expected for correct surface normal generation.
- **Edge Generation:** The edges defined by these faces will be automatically converted into physical distance springs when `create_constraints()` is called.

**Usage Example:**
```python
# Create a single triangle connecting vertices 0, 1, and 2
indices = np.array([0, 1, 2], dtype=np.uint32)
settings.add_faces(indices.tobytes())
```

**Constraints:**
- Raises `ValueError` if the buffer size is not a multiple of 12 bytes (3x uint32).
- Raises `RuntimeError` if called after `optimize()`.

## class Registry

A native, high-performance **Sparse Set ECS (Entity Component System)** registry. 

This class is designed to manage game state data in contiguous memory buffers, making it the perfect companion for `PhysicsWorld`. It allows you to process thousands of game entities using NumPy with zero Python-loop overhead.

### create()
[No Arguments] Creates a new 64-bit entity.
**Returns:**
- **`entity` (int):** A unique 64-bit generational handle.

### destroy(entity)
Removes an entity and all its associated components from the registry.
**Arguments:**
- **`entity` (int):** The 64-bit handle. 
**Note:** This invalidates the handle. The internal memory slot will be recycled with a new generation ID in future `create()` calls.

### register_component(size_bytes)
Defines a new component type.
**Returns:**
- **`comp_id` (int):** An integer ID used to add/remove this component.
**Arguments:**
- **`size_bytes` (int):** The fixed size of the component data in bytes (e.g., 12 for a 3-float vector).

### add(entity, comp_id, data=None)
Attaches a component to an entity.
**Arguments:**
- **`entity` (int):** The entity handle.
- **`comp_id` (int):** The ID from `register_component`.
- **`data` (Buffer, optional):** The initial data. Must match the registered size. If `None`, memory is zero-initialized.

### remove(entity, comp_id)
Removes a specific component from an entity.
**Note:** This triggers an internal "Swap-and-Pop" to keep the storage buffer contiguous.

### has(entity, comp_id)
**Returns:** `True` if the entity possesses the component.

### get_view(comp_id)
Returns a **writable zero-copy `memoryview`** of all active data for a component type.
- **Layout:** Contiguous C-array of the size specified during registration.
- **Usage:** Ideal for bulk updates using `np.frombuffer`.

### get_entities(comp_id)
Returns a **read-only zero-copy `memoryview`** of entity handles.
- **Layout:** `uint64` handles.
- **Relationship:** The handle at index `i` corresponds to the data at index `i` in the `get_view` buffer.

### is_alive(...)
Returns a boolean whether a handle is alive.

**Returns:** `True` if the handle is alive, otherwise `False`.


### clear()
[No Arguments] Instantly destroys all entities and wipes all component data.

**Note:** This is highly optimized for scene transitions or level resets. It safely increments generation counters internally, which instantly invalidates all existing entity handles without the overhead of destroying them one by one.

### get(entity, comp_id)
Retrieves the raw component data for a single entity.

**Arguments:**
- **`entity` (int):** The 64-bit entity handle.
- **`comp_id` (int):** The ID of the component.
**Returns:**
- **`data` (bytes | None):** A raw `bytes` object containing the data, or `None` if the entity does not possess the component or is invalid.

### get_active_count()
**Returns:**
- **`count` (int):** The total number of currently alive entities in the registry.

### get_component_count(comp_id)
**Arguments:**
- **`comp_id` (int):** The ID of the component.
**Returns:**
- **`count` (int):** The total number of entities that currently possess this specific component.

### sync_from_world(world, handle_comp_id, pos_comp_id, rot_comp_id)

Performs a high-performance, C-native bulk synchronization of body transforms (position and rotation) from the `PhysicsWorld` into the ECS `Registry`. 

This method solves the primary performance bottleneck in Python-based ECS architectures: mapping physics simulation results back to game entities. By executing the entire handle-lookup and data-transfer loop in C, it eliminates the $O(N)$ cost of Python list comprehensions and NumPy indexing.

**Arguments:**
- **`world` (PhysicsWorld):** The source world to read simulation results from.
- **`handle_comp_id` (int):** The ECS component storing the `uint64` physics handles.
- **`pos_comp_id` (int):** The ECS component where the `float32` positions will be written. Pass `-1` to skip syncing positions.
- **`rot_comp_id` (int):** The ECS component where the `float32` quaternions will be written. Pass `-1` to skip syncing rotations.

**Technical Requirements:**
- **Handle Buffer:** The component registered as `handle_comp_id` must have an `element_size` of exactly **8 bytes** (`uint64`).
- **Position Buffer:** If mapped, `pos_comp_id` must have an `element_size` of exactly **12 bytes** (3x `float32`).
- **Rotation Buffer:** If mapped, `rot_comp_id` must have an `element_size` of exactly **16 bytes** (4x `float32`).
- **Handle Validation:** The method utilizes `unpack_handle` internally. If an entity possesses a stale or invalid physics handle, its transform will be safely skipped without interrupting the batch.

**Performance & Precision:**
- **C-Native Loop:** The synchronization is performed in a single contiguous memory sweep, staying entirely within the CPU's L1/L2 cache. Both Position and Rotation are synced in the same pass.
- **Precision Downcasting:** If the engine is built with `DOUBLE_PRECISION`, this method automatically performs the cast from `float64` (Physics Position) to `float32` (ECS Position) during the copy, saving you from doing it manually in NumPy.
- **Lock Consistency:** This method acquires the world's `shadow_lock` for the duration of the transfer, ensuring that the ECS receives a perfectly consistent snapshot of the physics world.

**Usage Example:**
```python
# In your main game loop:
world.step(1/60)

# Sync thousands of entities in sub-millisecond time
registry.sync_from_world(
    world, 
    COMP_PHYSICS_HANDLE, 
    COMP_WORLD_POSITION,
    COMP_WORLD_ROTATION
)
```

**Constraints:**
- Raises `TypeError` if the component sizes do not match the required bytes.
- Raises `ValueError` if the provided component IDs are invalid.

## class MathService

A high-performance **SIMD-accelerated math utility** designed for Culverin. 

This class provides a direct bridge to C++ math routines (typically Jolt or GLM-based). It utilizes a specialized **Speculative FastParse** system to achieve call overheads as low as **77ns**, making it significantly faster than equivalent NumPy scalar operations or standard Python functions for high-frequency matrix calculations.

### get_perspective(fovy, aspect, near, far)
Computes a standard 4x4 perspective projection matrix.

**Arguments:**
- **`fovy` (float):** Field of view in the y-direction, in radians.
- **`aspect` (float):** Aspect ratio (width/height).
- **`near` (float):** Distance to the near clipping plane.
- **`far` (float):** Distance to the far clipping plane.
**Returns:**
- **`matrix` (tuple):** A 16-element tuple representing the 4x4 matrix in column-major order.

### get_ortho(left, right, bottom, top, near, far)
Computes a 4x4 orthographic projection matrix.

**Arguments:**
- **`left`, `right` (float):** Coordinates for the left and right vertical clipping planes.
- **`bottom`, `top` (float):** Coordinates for the bottom and top horizontal clipping planes.
- **`near`, `far` (float):** Distances to the near and far depth clipping planes.
**Returns:**
- **`matrix` (tuple):** A 16-element tuple (4x4 matrix).

### get_look_at(eye, target, up)
Computes a 4x4 View Matrix (LookAt).

**Arguments:**
- **`eye` (tuple):** 3-element tuple (x, y, z) of the camera position.
- **`target` (tuple):** 3-element tuple (x, y, z) of the point to look at.
- **`up` (tuple):** 3-element tuple (x, y, z) defining the world "up" vector.
**Returns:**
- **`matrix` (tuple):** A 16-element tuple (4x4 matrix).

### get_trs(translation, rotation, scale)
Computes a 4x4 **Translation-Rotation-Scale** transformation matrix.

**Arguments:**
- **`translation` (tuple):** 3-element tuple (x, y, z).
- **`rotation` (tuple):** 4-element tuple (x, y, z, w) representing a quaternion.
- **`scale` (tuple):** 3-element tuple (x, y, z).
**Returns:**
- **`matrix` (tuple):** A 16-element tuple (4x4 matrix).

### get_trs_batch(translations, rotations, scales)
Performs a high-performance **batch generation** of TRS matrices. This method is the primary tool for updating ECS transform components or preparing instance data for a GPU.

**Arguments:**
- **`translations` (Buffer):** Tightly packed `float32` data (3 per element).
- **`rotations` (Buffer):** Tightly packed `float32` data (4 per element).
- **`scales` (Buffer):** Tightly packed `float32` data (3 per element).

**Returns:**
- **`data` (bytes):** A raw bytes object containing the concatenated $16 \times \text{float32}$ matrices.

**Technical Notes:**
- **Zero-Copy Intent:** The returned `bytes` object can be cast to a `memoryview` or `numpy.ndarray` with zero copying, allowing for direct upload to a Vulkan/OpenGL buffer.
- **SIMD Parallelism:** Internally utilizes CPU SIMD instructions to calculate multiple matrices simultaneously where possible.
- **Memory Safety:** Automatically releases all input buffers immediately after calculation.

**Usage Example:**
```python
# Batch calculate 10,000 matrices from ECS memoryviews
matrices_raw = math_service.get_trs_batch(
    registry.get_view(COMP_POS),
    registry.get_view(COMP_ROT),
    registry.get_view(COMP_SCALE)
)

# Upload directly to GPU or wrap in NumPy
matrix_array = np.frombuffer(matrices_raw, dtype=np.float32).reshape(-1, 4, 4)
```

**Constraints:**
- All input buffers must have matching element counts.
- Input buffers must support the Python Buffer Protocol (e.g., `bytes`, `memoryview`, `numpy.ndarray`).

### inverse(matrix)
Computes the inverse of a 4x4 matrix.

**Arguments:**
- **`matrix` (tuple):** A 16-element tuple (4x4 matrix).
**Returns:**
- **`matrix` (tuple):** The inverted 16-element tuple.

**Technical Notes:**
- Highly optimized for View Matrix generation (inverting a Camera's World matrix).
- Utilizes SIMD-accelerated Cramer's rule or Gaussian elimination based on CPU architecture.

### matmul(a, b)
Multiplies two 4x4 matrices.

**Arguments:**
- **`a` (tuple):** The left-hand 16-element matrix.
- **`b` (tuple):** The right-hand 16-element matrix.
**Returns:**
- **`matrix` (tuple):** The resulting 16-element matrix.

### transform_vec3(matrix, vector)
Applies a 4x4 transformation matrix to a 3D vector.

**Arguments:**
- **`matrix` (tuple):** A 16-element tuple.
- **`vector` (tuple):** A 3-element tuple (x, y, z).
**Returns:**
- **`vector` (tuple):** The transformed 3-element tuple.

**Technical Notes:**
- Performs full affine transformation including translation.
- Internally handles the $w$ component as 1.0 for position transformation.

### matmul_batch(matrix, batch)
Multiplies a single 4x4 matrix by a buffer of 4x4 matrices. This is the optimal way to calculate **MVP (Model-View-Projection)** matrices for a group of entities.

**Arguments:**
- **`matrix` (tuple):** A single 16-element tuple (usually a View-Projection matrix).
- **`batch` (Buffer):** Tightly packed `float32` data representing $N$ matrices.

**Returns:**
- **`data` (bytes):** A raw bytes object containing the concatenated results.

### cull_aabb(vp_matrix, min, max)
Performs a frustum culling check for a single Axis-Aligned Bounding Box (AABB).

**Arguments:**
- **`vp_matrix` (tuple):** A 16-element View-Projection matrix.
- **`min` (tuple):** 3-element tuple (x, y, z) for AABB minimum corner.
- **`max` (tuple):** 3-element tuple (x, y, z) for AABB maximum corner.
**Returns:**
- **`visible` (bool):** `True` if the box is inside or intersecting the frustum; `False` otherwise.

### cull_aabb_batch(vp_matrix, aabbs)
Performs a high-velocity **frustum culling** check on a batch of AABBs.

**Arguments:**
- **`vp_matrix` (tuple):** A 16-element View-Projection matrix.
- **`aabbs` (Buffer):** Tightly packed `float32` data in the format `[minX, minY, minZ, maxX, maxY, maxZ, ...]`.

**Returns:**
- **`mask` (bytearray):** A visibility mask where each byte is `1` (visible) or `0` (culled).

**Technical Notes:**
- **Gribb-Hartmann Extraction:** Planes are extracted from the `vp_matrix` using SIMD rows.
- **SIMD Culling:** Utilizes Jolt's `GetSupport` logic to test AABBs against 6 planes in parallel without branching.
- **ECS Integration:** Designed to be used as a filter mask before submitting draw calls.

**Usage Example:**
```python
# Cull 5,000 entities against the current camera frustum
visibility_mask = math_service.cull_aabb_batch(view_proj, aabb_buffer)

# Filter entity IDs for the renderer
visible_entities = [id for i, id in enumerate(entities) if visibility_mask[i]]
```

### vec3_normalize(v)
Scales a 3D vector to have a length of 1.0.

**Arguments:**
- **`v` (tuple):** A 3-element vector (x, y, z).
**Returns:**
- **`vector` (tuple):** The normalized 3-element vector. Returns `(0, 0, 0)` for zero-length inputs.

### vec3_normalize_batch(vecs)
Scales a buffer of 3D vectors to have a length of 1.0.

**Arguments:**
- **`vecs` (Buffer):** Tightly packed `float32` vector data.
**Returns:**
- **`data` (bytes):** A new bytes object with the normalized vector data.

### vec3_lerp_batch(vecs_a, vecs_b, alpha)
Performs linear interpolation between two buffers of 3D vectors.

**Arguments:**
- **`vecs_a` (Buffer):** The starting vectors.
- **`vecs_b` (Buffer):** The ending vectors.
- **`alpha` (float):** The interpolation factor, clamped to [0, 1].
**Returns:**
- **`data` (bytes):** A new bytes object with the interpolated vector data.

### vec3_dot(v1, v2)
Calculates the dot product of two 3D vectors.

**Arguments:**
- **`v1`, `v2` (tuple):** 3-element vectors.
**Returns:**
- **`result` (float):** The scalar dot product.

### vec3_cross(v1, v2)
Calculates the cross product of two 3D vectors.

**Arguments:**
- **`v1`, `v2` (tuple):** 3-element vectors.
**Returns:**
- **`vector` (tuple):** The resulting 3-element vector, perpendicular to both inputs.

### vec3_distance(v1, v2)
Calculates the Euclidean distance between two 3D points.

**Arguments:**
- **`v1`, `v2` (tuple):** 3-element vectors.
**Returns:**
- **`distance` (float):** The scalar distance.

### vec3_distance_batch(vecs_a, vecs_b)
Calculates the Euclidean distance between pairs of vectors from two buffers.

**Arguments:**
- **`vecs_a`, `vecs_b` (Buffer):** Buffers of 3D vector data.
**Returns:**
- **`data` (bytes):** A bytes object containing a flat list of `float32` distances.

### vec3_reflect(v, normal)
Reflects an incident vector off a surface normal.

**Arguments:**
- **`v` (tuple):** The incoming direction vector.
- **`normal` (tuple):** The unit normal of the surface.
**Returns:**
- **`vector` (tuple):** The reflected direction vector.

### quat_from_euler(x, y, z)
Creates a quaternion from Euler angles (in radians).

**Arguments:**
- **`x`, `y`, `z` (float):** Rotation angles around the X, Y, and Z axes.
**Returns:**
- **`quaternion` (tuple):** The resulting 4-element quaternion (x, y, z, w).

### quat_to_euler(x, y, z, w)
Converts a quaternion back to Euler angles (in radians).

**Arguments:**
- **`x`, `y`, `z`, `w` (float):** The components of the quaternion.
**Returns:**
- **`angles` (tuple):** A 3-element tuple of Euler angles (x, y, z).

### quat_slerp(q1, q2, t)
Performs Spherical Linear Interpolation between two quaternions.

**Arguments:**
- **`q1`, `q2` (tuple):** 4-element quaternions.
- **`t` (float):** The interpolation factor [0, 1].
**Returns:**
- **`quaternion` (tuple):** The interpolated quaternion.

### quat_mul(a, b)
Multiplies two quaternions, combining their rotations. The rotation `b` is applied first, followed by `a`.

**Arguments:**
- **`a`, `b` (tuple):** 4-element quaternions.
**Returns:**
- **`quaternion` (tuple):** The resulting combined rotation.

### quat_inverse(q)
Computes the inverse of a quaternion.

**Arguments:**
- **`q` (tuple):** A 4-element quaternion.
**Returns:**
- **`quaternion` (tuple):** The inverse quaternion.

### quat_rotate_vec3(q, v)
Rotates a 3D vector by a quaternion.

**Arguments:**
- **`q` (tuple):** The rotation quaternion.
- **`v` (tuple):** The 3-element vector to rotate.
**Returns:**
- **`vector` (tuple):** The rotated 3-element vector.

### quat_rotate_vec3_inverse(q, v)
Rotates a 3D vector by the inverse (conjugate) of a quaternion. This is the standard way to convert a world-space offset to local space.

**Arguments:**
- **`q` (tuple):** The rotation quaternion.
- **`v` (tuple):** The 3-element vector to rotate.
**Returns:**
- **`vector` (tuple):** The rotated 3-element vector.

### quat_rotate_vec3_batch(q, vecs)
Rotates a buffer of 3D vectors by a single quaternion.

**Arguments:**
- **`q` (tuple):** The rotation quaternion.
- **`vecs` (Buffer):** A buffer of 3D vector data.
**Returns:**
- **`data` (bytes):** A new bytes object with the rotated vector data.

### quat_from_to(v1, v2)
Creates a quaternion that represents the shortest rotation from vector `v1` to `v2`.

**Arguments:**
- **`v1`, `v2` (tuple):** 3-element direction vectors.
**Returns:**
- **`quaternion` (tuple):** The resulting rotation.

### quat_get_axis_angle(q)
Decomposes a quaternion into its rotation axis and angle.

**Arguments:**
- **`q` (tuple):** A 4-element quaternion.
**Returns:**
- **`result` (tuple):** A nested tuple `((ax, ay, az), angle_rad)`.

### quat_from_axis_angle(axis, angle)
Creates a quaternion from a rotation axis and an angle in radians.

**Arguments:**
- **`axis` (tuple):** A 3-element rotation axis.
- **`angle` (float):** The angle of rotation in radians.
**Returns:**
- **`quaternion` (tuple):** The resulting 4-element quaternion.

### project(v, mvp, viewport)
Transforms a 3D world-space point to 2D screen-space coordinates.

**Arguments:**
- **`v` (tuple):** A 3-element world position (x, y, z).
- **`mvp` (tuple):** A 16-element Model-View-Projection matrix.
- **`viewport` (tuple):** A 4-element tuple `(x, y, width, height)` of the screen viewport.
**Returns:**
- **`vector` (tuple):** A 3-element tuple `(screen_x, screen_y, depth)`.

### unproject(v, mvp, viewport)
Transforms a 2D screen-space point back into a 3D world-space point.

**Arguments:**
- **`v` (tuple):** A 3-element screen position `(x, y, depth)`, where depth is in [0, 1].
- **`mvp` (tuple):** A 16-element Model-View-Projection matrix.
- **`viewport` (tuple):** A 4-element tuple `(x, y, width, height)`.
**Returns:**
- **`vector` (tuple):** The resulting 3-element world position.

### intersect_ray_plane(ray_origin, ray_dir, plane_pos, plane_norm)
Calculates the intersection point of a ray and an infinite plane.

**Arguments:**
- **`ray_origin`, `ray_dir` (tuple):** 3D vectors defining the ray.
- **`plane_pos`, `plane_norm` (tuple):** A point on the plane and the plane's normal vector.
**Returns:**
- **`result` (tuple):** A tuple `(hit, distance, point)`. `hit` is boolean, `distance` is float, `point` is a 3-element tuple or `None`.

### mat44_identity()
Returns a 4x4 identity matrix.

**Returns:**
- **`matrix` (tuple):** A 16-element identity matrix.

### mat44_get_translation(mat)
Extracts the translation component from a 4x4 matrix.

**Arguments:**
- **`mat` (tuple):** A 16-element matrix.
**Returns:**
- **`vector` (tuple):** A 3-element position vector (x, y, z).

### mat44_get_rotation(mat)
Extracts the rotation component from a 4x4 matrix as a quaternion.

**Arguments:**
- **`mat` (tuple):** A 16-element matrix.
**Returns:**
- **`quaternion` (tuple):** A 4-element quaternion (x, y, z, w).


## class Ship

### set_input(...)

Updates the driving commands for the native ship controller.

**Arguments:**
- **`forward` (float):** Acceleration intent ranging from `-1.0` (Full Reverse) to `1.0` (Full Forward).
- **`right` (float):** Steering intent ranging from `-1.0` (Hard Left) to `1.0` (Hard Right).

**Operational Features:**
- **Zero-Latency Atomics:** Unlike standard body methods, `set_input` is 100% lock-free. It writes directly to internal atomic variables, bypassing the `shadow_lock` and `is_stepping` checks entirely.
- **High-Frequency Execution:** The inputs are picked up by the ship's native **OnStep Listener** which runs at the solver's frequency. This ensures that movement and stabilization forces are always perfectly synchronized with the simulation's current state.
- **Drivetrain Logic:** Steering directly overrides the Y-axis angular velocity while maintaining the PD-controller's damping on the X and Z axes to keep the ship stable during turns.
- **Automatic Activation:** Setting a non-zero input automatically wakes the ship's body in Jolt if it has fallen asleep.
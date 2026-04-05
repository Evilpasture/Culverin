# Culverin Physics Engine - Method Documentation

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

---

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

---

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

Enables or disables Continuous Collision Detection for a body.
Prevents fast-moving small objects (like bullets) from tunneling through walls.


### raycast(...)

Casts a line through the world and returns the first hit.

**Returns:** `(Handle, Fraction, Normal)` or `None`.


### raycast_batch(...)

Executes thousands of raycasts simultaneously using a C-level buffer.
Highly optimized for AI sightlines or sensor sweeps.


### shapecast(...)

Sweeps a 3D shape (e.g., Sphere, Box) along a vector to find collisions.
Useful for thick projectiles or clearance testing.


### overlap_sphere(...)

Finds all bodies overlapping a defined sphere in world space.

**Returns:** List of handles.


### overlap_aabb(...)

Finds all bodies overlapping an Axis-Aligned Bounding Box.


### get_index(...)

Returns the internal ECS (Entity Component System) array index of a body.
Useful for matching bodies to direct memoryview data.


### is_alive(...)

Checks if a uint64 handle points to a valid, un-destroyed body.


### is_active(...)

Checks if a body is currently active (awake) in the solver.


### get_active_indices(...)

Returns a bytes object of active body indices.


### shape_count (property)

Number of unique shapes in cache.


### is_step_pending (property)

Whether a physics step is currently in progress. If True, structural changes are blocked.


### max_bodies (property)

The hard limit of bodies set at init.


### remaining_capacity (property)

Number of slots available before world.step() is required.


### positions (property)

Raw float64 positions buffer for all bodies (memoryview compatible).


### rotations (property)

Raw float32 quaternion rotations buffer for all bodies (memoryview compatible).


### velocities (property)

Raw float64 linear velocities buffer for all bodies (memoryview compatible).


### angular_velocities (property)

Raw float64 angular velocities buffer for all bodies (memoryview compatible).


### count (property)

Current number of bodies in the simulation.


### time (property)

Total simulation time elapsed (in seconds).


### user_data (property)

Raw uint64 user data buffer for all bodies (memoryview compatible).


### get_render_state(...)

Calculates smooth interpolated transforms for rendering.

**Arguments:**
- `alpha (float)`: Interpolation factor between the previous and current physics state (0.0 to 1.0).

**Returns:** `bytes` - Packed float32 array of [x, y, z, rx, ry, rz, rw].


### get_debug_data(...)

Extracts geometry data representing the current state of the physics shapes.

**Returns:** `(lines_bytes, triangles_bytes)`. Each vertex is 16 bytes: [x, y, z, color_u32].


### get_body_stats(...)

Returns a tuple containing current exact physics data for a body.

**Returns:** `((pos_x, pos_y, pos_z), (rot_x, rot_y, rot_z, rot_w), (vel_x, vel_y, vel_z))`


### get_user_data(...)

Retrieves the custom 64-bit integer attached to a body.


### set_user_data(...)

Attaches a custom 64-bit integer (like an entity ID or pointer) to a body.


### get_contact_events(...)

Retrieves a basic list of collision events that occurred during the last step.


### get_contact_events_ex(...)

Retrieves detailed collision events.
Returns a list of dictionaries containing involved bodies, normals, impulses, and contact points.


### get_contact_events_raw(...)

Returns a zero-copy memoryview of the contact buffer.
Highly optimized for extracting massive amounts of collision data in NumPy.

**Format** `'{<QQfffffffffIIII}'`:
- `body1, body2 (uint64)`: Colliding bodies
- `px, py, pz (float32)`: Contact point
- `nx, ny, nz (float32)`: Contact normal
- `impulse, sliding_speed_sq (float32)`: Impact data
- `mat1, mat2, type, _pad (uint32)`: Material and event info


### save_state(...)

Serializes the entire state of the physics world.

**Returns:** `bytes` - Useful for quicksaves or networking rollbacks.


### load_state(...)

Restores the physics world to a previously saved state.
Warning: This immediately overwrites the current simulation.


### create_character(...)

Creates a Kinematic Character Controller (Virtual Character).
Ideal for FPS or RPG player movement.


### _benchmark_parse(...)

Internal macro overhead benchmark. (For engine developers).


### _benchmark_build(...)

Internal tuple generation benchmark. (For engine developers).


## class Character

### handle (property)

The unique physics handle for this character.


### move(...)

Moves the character controller using a requested velocity vector.
The controller automatically handles stepping up stairs, sliding along walls, and gravity.

**Arguments:**
- `velocity (tuple)`: (x, y, z) movement vector.
- `dt (float)`: Time delta.


### get_position(...)

Retrieves the character's current world position.


### set_position(...)

Teleports the character to a new world position.


### set_rotation(...)

Sets the character's rotation quaternion.


### is_grounded(...)

Returns True if the character is currently standing on a floor.


### set_strength(...)

Sets the maximum force the character can exert when pushing dynamic objects.


### get_render_transform(...)

Returns the character's interpolated position and rotation for smooth rendering.


## class Vehicle

### wheel_count (property)

Number of wheels attached to this vehicle.


### set_input(...)

Applies driver inputs to a wheeled vehicle.

**Arguments:**
- `forward (float)`: Acceleration (-1.0 to 1.0).
- `right (float)`: Steering (-1.0 to 1.0).
- `brake (float)`: Braking force (0.0 to 1.0).
- `handbrake (float)`: Handbrake override (0.0 to 1.0).


### set_tank_input(...)

Applies driver inputs to a tracked vehicle.

**Arguments:**
- `left (float)`: Left track power (-1.0 to 1.0).
- `right (float)`: Right track power (-1.0 to 1.0).
- `brake (float)`: Overall braking force.


### get_wheel_transform(...)

Returns the world-space (position, rotation) of a specific wheel for rendering.


### get_wheel_local_transform(...)

Returns the local chassis-space transform of a specific wheel (useful for suspension animation).


### destroy(...)

Manually unregisters and cleans up the vehicle controller.


### get_debug_state(...)

Prints internal drivetrain, suspension, and tire slip data to stderr.


## class Skeleton

### add_joint(...)

Adds a new bone/joint to the skeleton definition.

**Arguments:**
- `name (str)`: The name of the joint.
- `parent_index (int)`: Index of the parent joint (-1 for root).


### get_joint_index(...)

Returns the integer index of a joint by its string name.


### finalize(...)

Bakes the skeleton hierarchy. Must be called before creating ragdoll settings.


## class Ragdoll

### drive_to_pose(...)

Applies motor forces to drive the ragdoll's limbs toward a target animated pose.
Often called "Physical Animation."

**Arguments:**
- `root_pos`, `root_rot`: World transform of the character.
- `matrices (bytes)`: Array of 4x4 float32 matrices representing local bone transforms.


### get_body_handles(...)

Returns a list of handles corresponding to the individual physics bodies making up the ragdoll.


### get_debug_info(...)

Returns a list of dictionaries detailing the position, velocity, and state of every ragdoll limb.


## class RagdollSettings

### add_part(...)

Assigns a physics collision shape to a specific skeleton joint.

**Arguments:**
- `joint_index (int)`: The target joint.
- `shape_type (int)`: culverin.SHAPE_* type.
- `size (tuple)`: Dimensions of the shape.
- `parent_index (int)`: The parent joint it connects to via constraint.


### stabilize(...)

Analyzes the ragdoll hierarchy and automatically disables collisions between connected limbs to prevent jitter.


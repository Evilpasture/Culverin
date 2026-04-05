# Culverin Physics Engine - Method Documentation

## class PhysicsWorld

### step(...)

Advances the physics simulation by 'dt' seconds.
This method flushes all pending command buffers (creations, destructions, impulses)
before running the Jolt Physics solver.

**Arguments:**
- `dt (float)`: The time step to simulate (e.g., 1/60.0). Pass 0.0 to flush commands without advancing time.


### create_body(...)

Creates a new rigid body in the physics world.
Returns a unique uint64 handle used to reference the body in future operations.

**Arguments:**
- `pos (tuple)`: (x, y, z) starting position.
- `rot (tuple)`: (x, y, z, w) starting quaternion rotation (default: identity).
- `shape (int)`: A culverin.SHAPE_* constant.
- `size (tuple)`: Dimensions of the shape.
- `mass (float)`: Mass in kilograms (default: 1.0).
- `motion (int)`: A culverin.MOTION_* constant (STATIC, KINEMATIC, DYNAMIC).


### create_bodies_batch(...)

High-performance batch creation of multiple bodies.

**Arguments:**
- `positions (list)`: List of (x,y,z) tuples.
- `sizes (list)`: List of size tuples.
- `shape (int)`: Shape type applied to all bodies.
- `motion (int)`: Motion type applied to all bodies.

**Returns:** `list[int]` - List of uint64 handles.


### destroy_body(...)

Safely removes a body from the simulation and frees its resources.
The destruction is queued and executed on the next step().


### destroy_bodies_batch(...)

Efficiently destroys multiple bodies at once.

**Arguments:**
- `handles (list[int])`: List of body handles to destroy.


### create_mesh_body(...)

Creates a static body from a complex 3D triangle mesh.
Typically used for static level geometry.

**Arguments:**
- `vertices (bytes)`: Flat float32 array of vertex positions.
- `indices (bytes)`: Flat uint32 array of triangle indices.


### create_constraint(...)

Creates a mechanical joint (Hinge, Slider, Fixed, etc.) between two bodies.

**Arguments:**
- `type (int)`: A culverin.CONSTRAINT_* constant.
- `body1 (int)`: Handle to the first body.
- `body2 (int)`: Handle to the second body.
- `params (tuple)`: Joint-specific configuration parameters.


### destroy_constraint(...)

Removes and frees a constraint by its handle.


### get_constraint_type(...)

Returns the integer type (culverin.CONSTRAINT_*) of a given constraint handle.


### create_vehicle(...)

Creates a wheeled vehicle constraint (Cars, Trucks).

**Arguments:**
- `chassis (int)`: Handle to the chassis body.
- `wheels (list[dict])`: Configuration for each wheel.
- `drive (str)`: "AWD", "RWD", or "FWD".


### create_tracked_vehicle(...)

Creates a tracked vehicle constraint (Tanks, Excavators).

**Arguments:**
- `chassis (int)`: Handle to the chassis body.
- `wheels (list[dict])`: Configuration for inner road wheels.
- `tracks (list[dict])`: Configuration linking wheels into tracks.


### create_ragdoll_settings(...)

Initializes the configuration object required to build a ragdoll.

**Arguments:**
- `skeleton (Skeleton)`: The skeleton hierarchy.


### create_ragdoll(...)

Instantiates a multi-body ragdoll into the physics world.

**Returns:** A Ragdoll object.


### create_heightfield(...)

Creates a static terrain body from a 2D grid of height values.

**Arguments:**
- `heights (bytes)`: 2D float32 array of height map data.
- `grid_size (int)`: Resolution of the grid.


### create_convex_hull(...)

Creates a collision shape by wrapping a point cloud in a convex shell.

**Arguments:**
- `points (bytes)`: Flat float32 array of (x,y,z) points.


### create_compound_body(...)

Creates a single rigid body composed of multiple distinct primitive shapes.

**Arguments:**
- `parts (list)`: List of sub-shape configurations: ((pos), (rot), shape_type, size).


### apply_impulse(...)

Applies an instantaneous change in linear momentum to a body's center of mass.
Best for sudden impacts (explosions, jumping, projectiles).


### apply_angular_impulse(...)

Applies an instantaneous change in rotational momentum.


### apply_impulse_at(...)

Applies an impulse at a specific world-space point on the body, 
resulting in both linear and angular velocity changes.


### apply_force(...)

Applies a continuous linear force (Newtons) to a body. 
Should be called every frame for continuous effects like wind or thrusters.


### apply_torque(...)

Applies a continuous twisting force to a body.


### set_gravity(...)

Sets the global gravity vector for the simulation.
Default is usually (0, -9.81, 0).


### apply_buoyancy(...)

Simulates fluid dynamics on a single body.

**Arguments:**
- `handle (int)`: The body to float.
- `surface_y (float)`: Height of the fluid surface.
- `buoyancy (float)`: Upward force multiplier.


### apply_buoyancy_batch(...)

High-performance batch fluid dynamics for multiple bodies.


### set_position(...)

Teleports a body to a new world position.
Note: Teleporting dynamic bodies can cause physics instability if they overlap others.


### set_rotation(...)

Instantly sets the rotation (quaternion x,y,z,w) of a body.


### set_linear_velocity(...)

Directly overrides the current linear velocity (meters per second) of a body.


### set_angular_velocity(...)

Directly overrides the current angular velocity (radians per second) of a body.


### set_transform(...)

Atomically sets both position and rotation of a body.


### set_collision_filter(...)

Dynamically updates the collision category and mask for an existing body.


### register_material(...)

Defines physics material properties.

**Arguments:**
- `id (int)`: Material ID (0-255).
- `friction (float)`: Surface friction.
- `restitution (float)`: Bounciness.


### set_constraint_target(...)

Drives a motorized constraint (like a hinge) to a target angle or position.


### get_motion_type(...)

Returns the motion type (STATIC, KINEMATIC, DYNAMIC) of the given body.


### set_motion_type(...)

Dynamically changes a body's mobility (e.g., locking a DYNAMIC body to STATIC).


### activate(...)

Forces a sleeping body to wake up and resume simulation.


### deactivate(...)

Forces an active body to go to sleep, saving CPU cycles.


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


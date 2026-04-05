# Culverin Physics Engine - Method Documentation

## PhysicsWorld_step

Advances the physics simulation by 'dt' seconds.
This method flushes all pending command buffers (creations, destructions, impulses)
before running the Jolt Physics solver.

**Arguments:**
- `dt (float)`: The time step to simulate (e.g., 1/60.0). Pass 0.0 to flush commands without advancing time.

## PhysicsWorld_create_body

Creates a new rigid body in the physics world.
Returns a unique uint64 handle used to reference the body in future operations.

**Arguments:**
- `pos (tuple)`: (x, y, z) starting position.
- `rot (tuple)`: (x, y, z, w) starting quaternion rotation (default: identity).
- `shape (int)`: A culverin.SHAPE_* constant.
- `size (tuple)`: Dimensions of the shape.
- `mass (float)`: Mass in kilograms (default: 1.0).
- `motion (int)`: A culverin.MOTION_* constant (STATIC, KINEMATIC, DYNAMIC).

## PhysicsWorld_create_bodies_batch

High-performance batch creation of multiple bodies.

**Arguments:**
- `positions (list)`: List of (x,y,z) tuples.
- `sizes (list)`: List of size tuples.
- `shape (int)`: Shape type applied to all bodies.
- `motion (int)`: Motion type applied to all bodies.

**Returns:** `list[int]` - List of uint64 handles.

## PhysicsWorld_destroy_body

Safely removes a body from the simulation and frees its resources.
The destruction is queued and executed on the next step().

## PhysicsWorld_destroy_bodies_batch

Efficiently destroys multiple bodies at once.

**Arguments:**
- `handles (list[int])`: List of body handles to destroy.

## PhysicsWorld_create_mesh_body

Creates a static body from a complex 3D triangle mesh.
Typically used for static level geometry.

**Arguments:**
- `vertices (bytes)`: Flat float32 array of vertex positions.
- `indices (bytes)`: Flat uint32 array of triangle indices.

## PhysicsWorld_create_constraint

Creates a mechanical joint (Hinge, Slider, Fixed, etc.) between two bodies.

**Arguments:**
- `type (int)`: A culverin.CONSTRAINT_* constant.
- `body1 (int)`: Handle to the first body.
- `body2 (int)`: Handle to the second body.
- `params (tuple)`: Joint-specific configuration parameters.

## PhysicsWorld_destroy_constraint

Removes and frees a constraint by its handle.

## PhysicsWorld_get_constraint_type

Returns the integer type (culverin.CONSTRAINT_*) of a given constraint handle.

## PhysicsWorld_create_vehicle

Creates a wheeled vehicle constraint (Cars, Trucks).

**Arguments:**
- `chassis (int)`: Handle to the chassis body.
- `wheels (list[dict])`: Configuration for each wheel.
- `drive (str)`: "AWD", "RWD", or "FWD".

## PhysicsWorld_create_tracked_vehicle

Creates a tracked vehicle constraint (Tanks, Excavators).

**Arguments:**
- `chassis (int)`: Handle to the chassis body.
- `wheels (list[dict])`: Configuration for inner road wheels.
- `tracks (list[dict])`: Configuration linking wheels into tracks.

## PhysicsWorld_create_ragdoll_settings

Initializes the configuration object required to build a ragdoll.

**Arguments:**
- `skeleton (Skeleton)`: The skeleton hierarchy.

## PhysicsWorld_create_ragdoll

Instantiates a multi-body ragdoll into the physics world.

**Returns:** A Ragdoll object.

## PhysicsWorld_create_heightfield

Creates a static terrain body from a 2D grid of height values.

**Arguments:**
- `heights (bytes)`: 2D float32 array of height map data.
- `grid_size (int)`: Resolution of the grid.

## PhysicsWorld_create_convex_hull

Creates a collision shape by wrapping a point cloud in a convex shell.

**Arguments:**
- `points (bytes)`: Flat float32 array of (x,y,z) points.

## PhysicsWorld_create_compound_body

Creates a single rigid body composed of multiple distinct primitive shapes.

**Arguments:**
- `parts (list)`: List of sub-shape configurations: ((pos), (rot), shape_type, size).

## PhysicsWorld_apply_impulse

Applies an instantaneous change in linear momentum to a body's center of mass.
Best for sudden impacts (explosions, jumping, projectiles).

## PhysicsWorld_apply_angular_impulse

Applies an instantaneous change in rotational momentum.

## PhysicsWorld_apply_impulse_at

Applies an impulse at a specific world-space point on the body, 
resulting in both linear and angular velocity changes.

## PhysicsWorld_apply_force

Applies a continuous linear force (Newtons) to a body. 
Should be called every frame for continuous effects like wind or thrusters.

## PhysicsWorld_apply_torque

Applies a continuous twisting force to a body.

## PhysicsWorld_set_gravity

Sets the global gravity vector for the simulation.
Default is usually (0, -9.81, 0).

## PhysicsWorld_apply_buoyancy

Simulates fluid dynamics on a single body.

**Arguments:**
- `handle (int)`: The body to float.
- `surface_y (float)`: Height of the fluid surface.
- `buoyancy (float)`: Upward force multiplier.

## PhysicsWorld_apply_buoyancy_batch

High-performance batch fluid dynamics for multiple bodies.

## PhysicsWorld_set_position

Teleports a body to a new world position.
Note: Teleporting dynamic bodies can cause physics instability if they overlap others.

## PhysicsWorld_set_rotation

Instantly sets the rotation (quaternion x,y,z,w) of a body.

## PhysicsWorld_set_linear_velocity

Directly overrides the current linear velocity (meters per second) of a body.

## PhysicsWorld_set_angular_velocity

Directly overrides the current angular velocity (radians per second) of a body.

## PhysicsWorld_set_transform

Atomically sets both position and rotation of a body.

## PhysicsWorld_set_collision_filter

Dynamically updates the collision category and mask for an existing body.

## PhysicsWorld_register_material

Defines physics material properties.

**Arguments:**
- `id (int)`: Material ID (0-255).
- `friction (float)`: Surface friction.
- `restitution (float)`: Bounciness.

## PhysicsWorld_set_constraint_target

Drives a motorized constraint (like a hinge) to a target angle or position.

## PhysicsWorld_get_motion_type

Returns the motion type (STATIC, KINEMATIC, DYNAMIC) of the given body.

## PhysicsWorld_set_motion_type

Dynamically changes a body's mobility (e.g., locking a DYNAMIC body to STATIC).

## PhysicsWorld_activate

Forces a sleeping body to wake up and resume simulation.

## PhysicsWorld_deactivate

Forces an active body to go to sleep, saving CPU cycles.

## PhysicsWorld_set_ccd

Enables or disables Continuous Collision Detection for a body.
Prevents fast-moving small objects (like bullets) from tunneling through walls.

## PhysicsWorld_raycast

Casts a line through the world and returns the first hit.

**Returns:** `(Handle, Fraction, Normal)` or `None`.

## PhysicsWorld_raycast_batch

Executes thousands of raycasts simultaneously using a C-level buffer.
Highly optimized for AI sightlines or sensor sweeps.

## PhysicsWorld_shapecast

Sweeps a 3D shape (e.g., Sphere, Box) along a vector to find collisions.
Useful for thick projectiles or clearance testing.

## PhysicsWorld_overlap_sphere

Finds all bodies overlapping a defined sphere in world space.

**Returns:** List of handles.

## PhysicsWorld_overlap_aabb

Finds all bodies overlapping an Axis-Aligned Bounding Box.

## PhysicsWorld_get_index

Returns the internal ECS (Entity Component System) array index of a body.
Useful for matching bodies to direct memoryview data.

## PhysicsWorld_is_alive

Checks if a uint64 handle points to a valid, un-destroyed body.

## PhysicsWorld_is_active

Checks if a body is currently active (awake) in the solver.

## PhysicsWorld_get_active_indices

Returns a packed bytes object of uint32 indices for all currently active bodies.

## PhysicsWorld_get_render_state

Calculates smooth interpolated transforms for rendering.

**Arguments:**
- `alpha (float)`: Interpolation factor between the previous and current physics state (0.0 to 1.0).

**Returns:** `bytes` - Packed float32 array of [x, y, z, rx, ry, rz, rw].

## PhysicsWorld_get_debug_data

Extracts geometry data representing the current state of the physics shapes.

**Returns:** `(lines_bytes, triangles_bytes)`. Each vertex is 16 bytes: [x, y, z, color_u32].

## PhysicsWorld_get_body_stats

Returns a tuple containing current exact physics data for a body.

**Returns:** `((pos_x, pos_y, pos_z), (rot_x, rot_y, rot_z, rot_w), (vel_x, vel_y, vel_z))`

## PhysicsWorld_get_user_data

Retrieves the custom 64-bit integer attached to a body.

## PhysicsWorld_set_user_data

Attaches a custom 64-bit integer (like an entity ID or pointer) to a body.

## PhysicsWorld_get_contact_events

Retrieves a basic list of collision events that occurred during the last step.

## PhysicsWorld_get_contact_events_ex

Retrieves detailed collision events.
Returns a list of dictionaries containing involved bodies, normals, impulses, and contact points.

## PhysicsWorld_get_contact_events_raw

Returns a zero-copy memoryview of the contact buffer.
Highly optimized for extracting massive amounts of collision data in NumPy.

**Format** `'{<QQfffffffffIIII}'`:
- `body1, body2 (uint64)`: Colliding bodies
- `px, py, pz (float32)`: Contact point
- `nx, ny, nz (float32)`: Contact normal
- `impulse, sliding_speed_sq (float32)`: Impact data
- `mat1, mat2, type, _pad (uint32)`: Material and event info

## PhysicsWorld_save_state

Serializes the entire state of the physics world.

**Returns:** `bytes` - Useful for quicksaves or networking rollbacks.

## PhysicsWorld_load_state

Restores the physics world to a previously saved state.
Warning: This immediately overwrites the current simulation.

## PhysicsWorld_create_character

Creates a Kinematic Character Controller (Virtual Character).
Ideal for FPS or RPG player movement.

## PhysicsWorld__benchmark_parse

Internal macro overhead benchmark. (For engine developers).

## PhysicsWorld__benchmark_build

Internal tuple generation benchmark. (For engine developers).

## Character_move

Moves the character controller using a requested velocity vector.
The controller automatically handles stepping up stairs, sliding along walls, and gravity.

**Arguments:**
- `velocity (tuple)`: (x, y, z) movement vector.
- `dt (float)`: Time delta.

## Character_get_position

Retrieves the character's current world position.

## Character_set_position

Teleports the character to a new world position.

## Character_set_rotation

Sets the character's rotation quaternion.

## Character_is_grounded

Returns True if the character is currently standing on a floor.

## Character_set_strength

Sets the maximum force the character can exert when pushing dynamic objects.

## Character_get_render_transform

Returns the character's interpolated position and rotation for smooth rendering.

## Vehicle_set_input

Applies driver inputs to a wheeled vehicle.

**Arguments:**
- `forward (float)`: Acceleration (-1.0 to 1.0).
- `right (float)`: Steering (-1.0 to 1.0).
- `brake (float)`: Braking force (0.0 to 1.0).
- `handbrake (float)`: Handbrake override (0.0 to 1.0).

## Vehicle_set_tank_input

Applies driver inputs to a tracked vehicle.

**Arguments:**
- `left (float)`: Left track power (-1.0 to 1.0).
- `right (float)`: Right track power (-1.0 to 1.0).
- `brake (float)`: Overall braking force.

## Vehicle_get_wheel_transform

Returns the world-space (position, rotation) of a specific wheel for rendering.

## Vehicle_get_wheel_local_transform

Returns the local chassis-space transform of a specific wheel (useful for suspension animation).

## Vehicle_destroy

Manually unregisters and cleans up the vehicle controller.

## Vehicle_get_debug_state

Prints internal drivetrain, suspension, and tire slip data to stderr.

## Skeleton_add_joint

Adds a new bone/joint to the skeleton definition.

**Arguments:**
- `name (str)`: The name of the joint.
- `parent_index (int)`: Index of the parent joint (-1 for root).

## Skeleton_get_joint_index

Returns the integer index of a joint by its string name.

## Skeleton_finalize

Bakes the skeleton hierarchy. Must be called before creating ragdoll settings.

## Ragdoll_drive_to_pose

Applies motor forces to drive the ragdoll's limbs toward a target animated pose.
Often called "Physical Animation."

**Arguments:**
- `root_pos`, `root_rot`: World transform of the character.
- `matrices (bytes)`: Array of 4x4 float32 matrices representing local bone transforms.

## Ragdoll_get_body_handles

Returns a list of handles corresponding to the individual physics bodies making up the ragdoll.

## Ragdoll_get_debug_info

Returns a list of dictionaries detailing the position, velocity, and state of every ragdoll limb.

## RagdollSettings_add_part

Assigns a physics collision shape to a specific skeleton joint.

**Arguments:**
- `joint_index (int)`: The target joint.
- `shape_type (int)`: culverin.SHAPE_* type.
- `size (tuple)`: Dimensions of the shape.
- `parent_index (int)`: The parent joint it connects to via constraint.

## RagdollSettings_stabilize

Analyzes the ragdoll hierarchy and automatically disables collisions between connected limbs to prevent jitter.

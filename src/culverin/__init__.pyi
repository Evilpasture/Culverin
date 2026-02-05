"""
Culverin Physics Engine
High-performance Python bindings for Jolt Physics using Shadow Buffers and Generational Handles.
"""

from typing import Tuple, List, Optional, TypedDict, Union, Any, Dict, Sequence
from . import _culverin_c

# Semantic Types
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Handle = int 
HandleBuffer = Union[bytes, memoryview, "Sequence[int]"]

# --- Constants ---
SHAPE_BOX: int = 0
SHAPE_SPHERE: int = 1
SHAPE_CAPSULE: int = 2
SHAPE_CYLINDER: int = 3
SHAPE_PLANE: int = 4
SHAPE_MESH: int = 5
SHAPE_HEIGHTFIELD: int = 6
SHAPE_CONVEX_HULL: int = 7

MOTION_STATIC: int = 0
MOTION_KINEMATIC: int = 1
MOTION_DYNAMIC: int = 2

CONSTRAINT_FIXED: int = 0
CONSTRAINT_POINT: int = 1
CONSTRAINT_HINGE: int = 2
CONSTRAINT_SLIDER: int = 3
CONSTRAINT_DISTANCE: int = 4
CONSTRAINT_CONE: int = 5

# Contact Event Types
EVENT_ADDED: int = 0
EVENT_PERSISTED: int = 1
EVENT_REMOVED: int = 2

class BodyConfig(TypedDict, total=False):
    pos: Vec3
    rot: Quat
    shape: int
    size: Union[float, Tuple[float], Tuple[float, float], Vec3, Quat]
    mass: float
    user_data: int
    motion: int
    is_sensor: bool
    ccd: bool

class WorldSettings(TypedDict, total=False):
    gravity: Vec3
    penetration_slop: float
    max_bodies: int
    max_pairs: int

class WheelConfig(TypedDict):
    pos: Vec3
    radius: float
    width: float

class TrackConfig(TypedDict):
    indices: List[int] # Indices of wheels belonging to this track
    driven_wheel: int  # Index of the sprocket wheel within the global wheels list

class Character:
    def move(self, velocity: Vec3, dt: float) -> None:
        """Move the virtual character, resolve collisions and stair climbing."""
    def get_position(self) -> Vec3:
        """Get high-precision world position."""
    def set_position(self, pos: Vec3) -> None:
        """Teleport character to world position."""
    def set_rotation(self, rot: Quat) -> None:
        """Set character world rotation."""
    def is_grounded(self) -> bool:
        """Returns True if the character is standing on a surface."""
    def set_strength(self, strength: float) -> None:
        """Set the maximum force used to push dynamic bodies."""
    def get_render_transform(self, alpha: float) -> Tuple[Vec3, Quat]:
        """Returns LERP/NLERP interpolated transform for rendering."""
    @property
    def handle(self) -> Handle:
        """Unique physics handle for this character."""

class Skeleton:
    def add_joint(self, name: str, parent_index: int = -1) -> int:
        """Add a joint to the hierarchy. Parent must exist before children."""
    def finalize(self) -> None:
        """Bakes hierarchy and calculates parent indices. Call before RagdollSettings."""

class RagdollSettings:
    def add_part(
        self, joint_index: int, shape_type: int, size: Any, mass: float, parent_index: int, 
        twist_min: float = -0.1, twist_max: float = 0.1, cone_angle: float = 0.0, 
        axis: Vec3 = (1,0,0), normal: Vec3 = (0,1,0), pos: Vec3 = (0,0,0)
    ) -> None:
        """Configure a limb part."""
    def stabilize(self) -> bool:
        """Attempts to remove initial interpenetrations in the pose."""

class Ragdoll:
    def drive_to_pose(self, root_pos: Vec3, root_rot: Quat, matrices: bytes) -> None:
        """Sets the target pose for the ragdoll motors via packed Model-Space matrices."""
    def get_body_handles(self) -> List[Handle]:
        """Returns a list of physics handles for every limb in the ragdoll."""
    def get_debug_info(self) -> List[Dict[str, Any]]:
        """Returns diagnostic position/velocity data for every limb."""

class Engine:
    def __init__(self, max_torque: float = 500.0, max_rpm: float = 7000.0, min_rpm: float = 1000.0, inertia: float = 0.5): ...

class Automatic:
    def __init__(self, gears: Union[int, List[float]] = 5, clutch_strength: float = 2000.0, shift_up_rpm: float = 5000.0, shift_down_rpm: float = 2000.0): ...

class Manual:
    def __init__(self, gears: Union[int, List[float]] = 5, clutch_strength: float = 5000.0): ...

class Vehicle:
    def set_input(self, forward: float = 0.0, right: float = 0.0, brake: float = 0.0, handbrake: float = 0.0) -> None:
        """Set driver inputs (throttles, steering, braking)."""
    def set_tank_input(self, left: float, right: float, brake: float = 0.0) -> None:
        """
        Set inputs for tracked vehicles. 
        Corrects engine RPM for pivot turns and reverses track ratios when backing up.
        """
    def get_wheel_transform(self, index: int) -> Tuple[Vec3, Quat]:
        """Get world-space wheel transform."""
    def get_wheel_local_transform(self, index: int) -> Tuple[Vec3, Quat]:
        """Get wheel transform relative to chassis."""
    def get_debug_state(self) -> None:
        """Print detailed drivetrain and wheel status to stderr."""
    def destroy(self) -> None:
        """Manually remove the vehicle constraint from simulation."""
    @property
    def wheel_count(self) -> int: ...

class PhysicsWorld:
    def __init__(self, settings: Optional[WorldSettings] = None, bodies: Optional[List[BodyConfig]] = None) -> None:
        """Initialize the physics system. 'bodies' can be pre-baked for speed."""
    def step(self, dt: float = 1.0/60.0) -> None:
        """Advance simulation and sync shadow buffers."""
    
    def create_body(
        self, 
        pos: Vec3 = (0, 0, 0),
        rot: Quat = (0, 0, 0, 1),
        size: Union[float, Vec3, Tuple[float, ...]] = (1, 1, 1),
        shape: int = SHAPE_BOX,
        motion: int = MOTION_DYNAMIC,
        user_data: int = 0,
        is_sensor: bool = False,
        mass: float = -1.0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        friction: float = 0.2,
        restitution: float = 0.0,
        material_id: int = 0,
        ccd: bool = False
    ) -> Handle:
        """
        Queue creation of a standard rigid body.
        Args:
            category: Bitmask for 'What Am I'.
            mask: Bitmask for 'What I Collide With'.
            material_id: User-defined ID for audio/VFX lookup.
        """
        ...
        
    def create_mesh_body(self, pos: Vec3, rot: Quat, vertices: bytes, indices: bytes, user_data: int = 0, category: int = 0xFFFF, mask: int = 0xFFFF) -> Handle:
        """Queue creation of a static triangle mesh body."""
        
    def create_character(self, pos: Vec3, height: float = 1.8, radius: float = 0.4, step_height: float = 0.4, max_slope: float = 45.0) -> Character:
        """Create a virtual character controller."""
        
    def create_vehicle(self, chassis: Handle, wheels: Sequence[WheelConfig], drive: str = "RWD", engine: Optional[Engine] = None, transmission: Optional[Union[Automatic, Manual]] = None) -> Vehicle:
        """Combine bodies into a wheeled vehicle system."""
        
    def create_tracked_vehicle(
        self, 
        chassis: Handle, 
        wheels: Sequence[WheelConfig], 
        tracks: Sequence[TrackConfig], 
        max_torque: float = 5000.0, 
        max_rpm: float = 6000.0
    ) -> Vehicle:
        """Create a native Jolt tracked vehicle (tank) with physical treads."""
        ...

    def create_ragdoll_settings(self, skeleton: Skeleton) -> RagdollSettings:
        """Create a ragdoll configuration template."""
        
    def create_ragdoll(
        self, 
        settings: RagdollSettings, 
        pos: Vec3, 
        rot: Quat = (0, 0, 0, 1), 
        user_data: int = 0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        material_id: int = 0
    ) -> Ragdoll:
        """Instantiate a ragdoll into the world."""

    def create_heightfield(
        self, pos: Vec3, rot: Quat, scale: Vec3, heights: bytes, grid_size: int, 
        user_data: int = 0, category: int = 0xFFFF, mask: int = 0xFFFF,
        material_id: int = 0, friction: float = 0.5, restitution: float = 0.0
    ) -> Handle:
        """
        Create a static terrain from a square grid of height values.
        
        Args:
            pos: World position of the (0,0) corner of the heightfield.
            rot: World rotation.
            scale: (x_scale, y_scale, z_scale). X/Z are grid spacing, Y is height multiplier.
            heights: Packed float32 buffer of size (grid_size * grid_size).
            grid_size: The width/depth of the grid (number of samples along one axis).
            material_id: ID for terrain surface type (e.g., Grass, Mud).
        """
        ...

    def create_convex_hull(
        self,
        pos: Vec3,
        rot: Quat,
        points: bytes,
        motion: int = MOTION_DYNAMIC,
        mass: float = -1.0,
        user_data: int = 0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        material_id: int = 0,
        friction: float = 0.2,
        restitution: float = 0.0,
        ccd: bool = False
    ) -> Handle:
        """
        Create a body from a point cloud. Jolt will generate a tight convex wrapper.
        
        Args:
            pos: World position.
            rot: World rotation.
            points: Packed float32 bytes of vertices (3 floats per point).
            motion: MOTION_DYNAMIC (2), MOTION_KINEMATIC (1), or MOTION_STATIC (0).
            mass: If > 0, overrides calculated mass.
            ccd: Enable Continuous Collision Detection.
        """
        ...

    def create_compound_body(
        self,
        pos: Vec3,
        rot: Quat,
        parts: List[Tuple[Vec3, Quat, int, Any]],
        motion: int = MOTION_DYNAMIC,
        mass: float = -1.0,
        user_data: int = 0,
        is_sensor: bool = False,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        material_id: int = 0,
        friction: float = 0.2,
        restitution: float = 0.0,
        ccd: bool = False
    ) -> Handle:
        """
        Create a single rigid body from multiple shapes.
        
        Args:
            parts: List of (local_pos, local_rot, shape_type, size).
                   Example: [((0,1,0), (0,0,0,1), SHAPE_BOX, (0.5,0.5,0.5))]
        """
        ...
        
    def destroy_body(self, handle: Handle) -> None:
        """Queue destruction of a body. Handle becomes invalid immediately."""
        
    def create_constraint(self, type: int, body1: Handle, body2: Handle, params: Optional[Any] = None) -> int:
        """Create a joint between two bodies."""
        
    def destroy_constraint(self, handle: int) -> None: ...
    
    def apply_impulse(self, handle: Handle, x: float, y: float, z: float) -> None: ...

    def apply_impulse_at(self, handle: Handle, ix: float, iy: float, iz: float, px: float, py: float, pz: float) -> None:
        """
        Apply a linear impulse at a specific world-space position to generate torque.
        Args:
            ix, iy, iz: Impulse vector.
            px, py, pz: World-space application point.
        """
        ...
    
    def apply_buoyancy(
        self, handle: Handle, surface_y: float, buoyancy: float = 1.0, 
        linear_drag: float = 0.5, angular_drag: float = 0.5, dt: float = 1.0/60.0,
        fluid_velocity: Vec3 = (0, 0, 0)
    ) -> bool:
        """Apply Archimedes' principle fluid forces."""
        ...

    def apply_buoyancy_batch(
        self,
        handles: HandleBuffer,
        surface_y: float = 0.0,
        buoyancy: float = 1.0,
        linear_drag: float = 0.5,
        angular_drag: float = 0.5,
        dt: float = 1.0/60.0,
        fluid_velocity: Vec3 = (0, 0, 0)
    ) -> None:
        """
        Applies Archimedes' principle fluid forces to a batch of bodies.

        :param handles: Contiguous buffer of uint64 physics handles (e.g., np.array(..., dtype=np.uint64)).
        :param surface_y: The world Y-coordinate of the fluid surface.
        :param buoyancy: The ratio of fluid density to body density (1.0 = neutral buoyancy).
        :param linear_drag: Linear drag coefficient.
        :param angular_drag: Angular drag coefficient.
        :param dt: The simulation time step (typically 1/60.0).
        :param fluid_velocity: The velocity vector (vx, vy, vz) of the fluid flow.
        :return: None (Forces are applied, result per body is not returned in batch mode).
        """
        ...

    def register_material(self, id: int, friction: float = 0.5, restitution: float = 0.0) -> None:
        """Define physical properties for a material ID lookup."""
        ...
        
    def set_position(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_rotation(self, handle: Handle, x: float, y: float, z: float, w: float) -> None: ...
    def set_transform(self, handle: Handle, pos: Vec3, rot: Quat) -> None: ...
    def set_linear_velocity(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_angular_velocity(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_ccd(self, handle: Handle, enabled: bool) -> None:
        """Enable/Disable Continuous Collision Detection for a body."""
    def set_collision_filter(self, handle: Handle, category: int, mask: int) -> None: ...
    
    def activate(self, handle: Handle) -> None: ...
    def deactivate(self, handle: Handle) -> None: ...
    def get_motion_type(self, handle: Handle) -> int: ...
    def set_motion_type(self, handle: Handle, motion: int) -> None: ...
    def set_user_data(self, handle: Handle, data: int) -> None: ...
    def get_user_data(self, handle: Handle) -> int: ...
    
    def raycast(self, start: Vec3, direction: Vec3, max_dist: float = 1000.0, ignore: Union[Handle, Character] = 0) -> Optional[Tuple[Handle, float, Vec3]]: ...
    def raycast_batch(self, starts: bytes, directions: bytes, max_dist: float = 1000.0) -> bytes:
        """Execute multiple raycasts efficiently (GIL-released)."""
        ...
    def shapecast(self, shape: int, pos: Vec3, rot: Quat, dir: Vec3, size: Any, ignore: Union[Handle, Character] = 0) -> Optional[Tuple[Handle, float, Vec3, Vec3]]: ...
    def overlap_sphere(self, center: Vec3, radius: float) -> List[Handle]: ...
    def overlap_aabb(self, min: Vec3, max: Vec3) -> List[Handle]: ...
    
    def get_contact_events(self) -> List[Tuple[Handle, Handle]]: ...
    def get_contact_events_ex(self) -> List[Dict[str, Any]]:
        """Returns details including pos, normal, impulse, and EVENT_TYPE."""
    def get_contact_events_raw(self) -> memoryview: 
        """
        Returns a read-only memoryview of packed ContactEvent structs (64 bytes each).
        Fields (Little-Endian):
          uint64 body1, body2
          float32 px, py, pz, nx, ny, nz
          float32 impulse, sliding_speed_sq
          uint32 mat1, mat2
          uint32 type
        """
        ...
    def get_debug_data(self, shapes: bool = True, constraints: bool = True, bbox: bool = False, centers: bool = False, wireframe: bool = True) -> Tuple[bytes, bytes]:
        """Returns (line_verts, triangle_verts) as packed byte buffers."""
        ...
    def get_index(self, handle: Handle) -> Optional[int]:
        """Map a handle to the current dense array index (changes when bodies are deleted)."""
    def get_active_indices(self) -> bytes:
        """Returns bytes containing uint32 indices of all active bodies."""
    def is_alive(self, handle: Handle) -> bool:
        """Check if a handle is still valid."""
    def save_state(self) -> bytes:
        """Serialize current world shadow state."""
    def load_state(self, state: bytes) -> None:
        """Restore world state and synchronize Jolt bodies."""
    def get_render_state(self, alpha: float) -> bytes: 
        """Returns a packed float32 buffer of [pos.xyz, rot.xyzw] for all bodies (28 bytes per body)."""

    @property
    def positions(self) -> memoryview:
        """Float32 memoryview [Count * 4] (x, y, z, pad)."""
    @property
    def rotations(self) -> memoryview:
        """Float32 memoryview [Count * 4] (x, y, z, w)."""
    @property
    def velocities(self) -> memoryview:
        """Float32 memoryview [Count * 4] (x, y, z, pad)."""
    @property
    def angular_velocities(self) -> memoryview:
        """Float32 memoryview [Count * 4] (x, y, z, pad)."""
    @property
    def user_data(self) -> memoryview:
        """Uint64 memoryview [Count]."""
    @property
    def count(self) -> int:
        """Number of active bodies in the dense arrays."""
    @property
    def time(self) -> float:
        """Total simulation time."""

__all__ = [
    "PhysicsWorld", "Character", "Vehicle", "Skeleton", "RagdollSettings", "Ragdoll", 
    "BodyConfig", "WorldSettings", "WheelConfig", "Handle", 
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", 
    "CONSTRAINT_SLIDER", "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE",
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED"
]
"""
Culverin Physics Engine
High-performance Python bindings for Jolt Physics using Shadow Buffers and Generational Handles.
"""

from typing import Tuple, List, Optional, TypedDict, Union, Any, Dict, Sequence

# Semantic Types
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Handle = int 

# --- Constants ---
SHAPE_BOX: int = 0
SHAPE_SPHERE: int = 1
SHAPE_CAPSULE: int = 2
SHAPE_CYLINDER: int = 3
SHAPE_PLANE: int = 4
SHAPE_MESH: int = 5

MOTION_STATIC: int = 0
MOTION_KINEMATIC: int = 1
MOTION_DYNAMIC: int = 2

CONSTRAINT_FIXED: int = 0
CONSTRAINT_POINT: int = 1
CONSTRAINT_HINGE: int = 2
CONSTRAINT_SLIDER: int = 3
CONSTRAINT_DISTANCE: int = 4
CONSTRAINT_CONE: int = 5

class BodyConfig(TypedDict, total=False):
    pos: Vec3
    rot: Quat
    shape: int
    size: Union[float, Tuple[float], Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float, float]]
    mass: float
    user_data: int
    motion: int
    is_sensor: bool

class WorldSettings(TypedDict, total=False):
    gravity: Vec3
    penetration_slop: float
    max_bodies: int
    max_pairs: int

class WheelConfig(TypedDict):
    pos: Vec3
    radius: float
    width: float

class Character:
    def move(self, velocity: Vec3, dt: float) -> None:
        """Move the virtual character. resolve collisions and stair climbing."""
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
        """
        Configure a limb part.
        Args:
            pos: The local offset in Bind Pose relative to the parent joint.
            axis: The twist axis for the SwingTwist constraint.
            normal: The plane normal for the SwingTwist constraint.
        """
    def stabilize(self) -> bool:
        """Attempts to remove initial interpenetrations in the pose."""

class Ragdoll:
    def drive_to_pose(self, root_pos: Vec3, root_rot: Quat, matrices: bytes) -> None:
        """
        Sets the target pose for the ragdoll motors.
        Args:
            matrices: Packed Float32 array of 4x4 Model-Space matrices (one per joint).
        """
    def get_body_handles(self) -> List[Handle]:
        """Returns a list of physics handles for every limb in the ragdoll."""
    def get_debug_info(self) -> List[Dict[str, Any]]:
        """Returns position and velocity data for every limb for diagnostic use."""

class Engine:
    def __init__(self, max_torque: float = 500.0, max_rpm: float = 7000.0, min_rpm: float = 1000.0, inertia: float = 0.5): ...

class Automatic:
    def __init__(self, gears: Union[int, List[float]] = 5, clutch_strength: float = 2000.0, shift_up_rpm: float = 5000.0, shift_down_rpm: float = 2000.0): ...

class Manual:
    def __init__(self, gears: Union[int, List[float]] = 5, clutch_strength: float = 5000.0): ...

class Vehicle:
    def set_input(self, forward: float = 0.0, right: float = 0.0, brake: float = 0.0, handbrake: float = 0.0) -> None:
        """Set driver inputs. Steering is usually automatically mapped to wheels with MaxSteerAngle."""
    def get_wheel_transform(self, index: int) -> Tuple[Vec3, Quat]:
        """Get world-space wheel transform."""
    def get_wheel_local_transform(self, index: int) -> Tuple[Vec3, Quat]:
        """Get wheel transform relative to chassis COM."""
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
        """Advance simulation. Flushes command queue and syncs shadow buffers."""
    def create_body(
        self, 
        pos: Vec3 = (0, 0, 0),
        rot: Quat = (0, 0, 0, 1),
        size: Union[float, Vec3, Tuple[float, ...]] = (1, 1, 1),
        shape: int = SHAPE_BOX,
        motion: int = MOTION_DYNAMIC,
        user_data: int = 0,
        is_sensor: bool = False,
        mass: float = -1.0  # Added
    ) -> Handle:
        """
        Queue creation of a standard rigid body.
        If mass > 0, it overrides the default mass calculated from shape density.
        """
        ...
    def create_mesh_body(self, pos: Vec3, rot: Quat, vertices: Any, indices: Any, user_data: int = 0) -> Handle:
        """Queue creation of a static triangle mesh body."""
    def create_character(self, pos: Vec3, height: float = 1.8, radius: float = 0.4, step_height: float = 0.4, max_slope: float = 45.0) -> Character:
        """Create a virtual character controller."""
    def create_vehicle(self, chassis: Handle, wheels: Sequence[WheelConfig], drive: str = "RWD", engine: Optional[Engine] = None, transmission: Optional[Union[Automatic, Manual]] = None) -> Vehicle:
        """Combine bodies into a wheeled vehicle system."""
    def create_ragdoll_settings(self, skeleton: Skeleton) -> RagdollSettings:
        """Create a ragdoll configuration template."""
    def create_ragdoll(self, settings: RagdollSettings, pos: Vec3, rot: Quat = (0, 0, 0, 1), user_data: int = 0) -> Ragdoll:
        """Instantiate a ragdoll into the world."""
    def destroy_body(self, handle: Handle) -> None:
        """Queue destruction of a body. Handle becomes invalid immediately."""
    def create_constraint(self, type: int, body1: Handle, body2: Handle, params: Optional[Any] = None) -> int:
        """Create a joint between two bodies."""
    def destroy_constraint(self, handle: int) -> None: ...
    def apply_impulse(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def apply_buoyancy(
        self, 
        handle: Handle, 
        surface_y: float, 
        buoyancy: float = 1.0, 
        linear_drag: float = 0.5, 
        angular_drag: float = 0.5, 
        dt: float = 1.0/60.0,
        fluid_velocity: Vec3 = (0, 0, 0)
    ) -> bool:
        """
        Calculates and applies buoyancy and fluid drag to a body.
        
        Args:
            handle: The body to affect.
            surface_y: The world-space height of the fluid surface.
            buoyancy: Scale of the upward force (1.0 matches gravity for body density).
            linear_drag: Resistance to movement through fluid.
            angular_drag: Resistance to rotation through fluid.
            dt: The timestep (usually matches your world.step call).
            fluid_velocity: World-space velocity of the fluid (for currents/rivers).
            
        Returns:
            True if the body is at least partially submerged.
        """
        ...
    def set_position(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_rotation(self, handle: Handle, x: float, y: float, z: float, w: float) -> None: ...
    def set_transform(self, handle: Handle, pos: Vec3, rot: Quat) -> None: ...
    def set_linear_velocity(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_angular_velocity(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def activate(self, handle: Handle) -> None: ...
    def deactivate(self, handle: Handle) -> None: ...
    def get_motion_type(self, handle: Handle) -> int: ...
    def set_motion_type(self, handle: Handle, motion: int) -> None: ...
    def set_user_data(self, handle: Handle, data: int) -> None: ...
    def get_user_data(self, handle: Handle) -> int: ...
    def raycast(self, start: Vec3, direction: Vec3, max_dist: float = 1000.0, ignore: Union[Handle, Character] = 0) -> Optional[Tuple[Handle, float, Vec3]]: ...
    def shapecast(self, shape: int, pos: Vec3, rot: Quat, dir: Vec3, size: Any, ignore: Union[Handle, Character] = 0) -> Optional[Tuple[Handle, float, Vec3, Vec3]]: ...
    def overlap_sphere(self, center: Vec3, radius: float) -> List[Handle]: ...
    def overlap_aabb(self, min: Vec3, max: Vec3) -> List[Handle]: ...
    def get_contact_events(self) -> List[Tuple[Handle, Handle]]:
        """Returns basic (ID1, ID2) collision pairs for the last frame."""
    def get_contact_events_ex(self) -> List[Dict[str, Any]]:
        """Returns detailed dictionaries including position, normal, and strength."""
    def get_contact_events_raw(self) -> memoryview: 
        """
        Returns a read-only memoryview of packed ContactEvent structs (48 bytes each).
        """
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
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", "SHAPE_MESH",
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", 
    "CONSTRAINT_SLIDER", "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE"
]
from typing import Tuple, List, Optional, TypedDict, Union, Any
from . import _culverin_c

# Semantic Types
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Handle = int 

# Constants
SHAPE_BOX: int = 0
SHAPE_SPHERE: int = 1
SHAPE_CAPSULE: int = 2
SHAPE_CYLINDER: int = 3
SHAPE_PLANE: int = 4
SHAPE_MESH: int = 5

MOTION_STATIC: int = 0
MOTION_KINEMATIC: int = 1
MOTION_DYNAMIC: int = 2

class BodyConfig(TypedDict, total=False):
    pos: Vec3
    rot: Quat
    shape: int
    size: Union[Tuple[float], Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float, float]]
    mass: float

class WorldSettings(TypedDict, total=False):
    gravity: Vec3
    penetration_slop: float
    max_bodies: int
    max_pairs: int

class PhysicsWorld(_culverin_c.PhysicsWorld):
    def __init__(
        self, 
        settings: Optional[WorldSettings] = None, 
        bodies: Optional[List[BodyConfig]] = None
    ) -> None: ...

    def step(self, dt: float = 1.0/60.0) -> None: ...

    def create_body(
        self, 
        pos: Vec3 = (0, 0, 0),
        rot: Quat = (0, 0, 0, 1),
        size: Union[float, Vec3, Tuple[float, ...]] = (1, 1, 1),
        shape: int = SHAPE_BOX,
        motion: int = MOTION_DYNAMIC
    ) -> Handle: ...

    def create_mesh_body(
        self,
        pos: Vec3,
        rot: Quat,
        vertices: Any, # numpy.ndarray (float32) or bytes
        indices: Any    # numpy.ndarray (uint32) or bytes
    ) -> Handle:
        """
        Creates a static mesh collider from vertex and index buffers.
        Buffers should be float32 [x,y,z...] and uint32 [i1,i2,i3...].
        """
        ...

    def destroy_body(self, handle: Handle) -> None: ...
    
    def apply_impulse(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_position(self, handle: Handle, x: float, y: float, z: float) -> None: ...
    def set_rotation(self, handle: Handle, x: float, y: float, z: float, w: float) -> None: ...
    def set_transform(self, handle: Handle, pos: Vec3, rot: Quat) -> None: ...
    
    def raycast(self, start: Vec3, direction: Vec3, max_dist: float = 1000.0) -> Optional[Tuple[Handle, float]]: ...
    def overlap_sphere(self, center: Vec3, radius: float) -> List[Handle]: ...
    def overlap_aabb(self, min: Vec3, max: Vec3) -> List[Handle]: ...

    def get_index(self, handle: Handle) -> Optional[int]: ...
    def is_alive(self, handle: Handle) -> bool: ...

    def get_motion_type(self, handle: Handle) -> int: 
        """Returns 0 (Static), 1 (Kinematic), or 2 (Dynamic)."""
        ...
        
    def set_motion_type(self, handle: Handle, motion: int) -> None: 
        """Sets motion type. 0=Static, 1=Kinematic, 2=Dynamic."""
        ...

    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...

    @property
    def positions(self) -> memoryview: ...
    @property
    def rotations(self) -> memoryview: ...
    @property
    def velocities(self) -> memoryview: ...
    @property
    def angular_velocities(self) -> memoryview: ...
    @property
    def count(self) -> int: ...
    @property
    def time(self) -> float: ...

__all__ = [
    "PhysicsWorld", "BodyConfig", "WorldSettings", "Handle", 
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", "SHAPE_MESH",
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC"
]
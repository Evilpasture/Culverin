import sys
from typing import Tuple, Optional, Any, List

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

class PhysicsWorld:
    def __init__(
        self, 
        settings: Optional[dict] = None, 
        bodies: Optional[list[dict]] = None
    ) -> None: ...

    def step(self, dt: float = 1.0/60.0) -> None: ...

    # --- Runtime Lifecycle ---
    def create_body(
        self, 
        pos: Tuple[float, float, float] = (0, 0, 0),
        rot: Tuple[float, float, float, float] = (0, 0, 0, 1),
        size: Any = (1, 1, 1), # Can be tuple of len 1, 2, 3, or 4
        shape: int = 0,
        motion: int = 2
    ) -> int: ...

    def create_mesh_body(
        self,
        pos: Tuple[float, float, float],
        rot: Tuple[float, float, float, float],
        vertices: bytes, # Raw buffer of float32
        indices: bytes   # Raw buffer of uint32
    ) -> int: ...

    def destroy_body(self, handle: int) -> None: ...
    
    # --- Interaction ---
    def apply_impulse(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_position(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_rotation(self, handle: int, x: float, y: float, z: float, w: float) -> None: ...
    def set_transform(self, handle: int, pos: Tuple[float, float, float], rot: Tuple[float, float, float, float]) -> None: ...
    def set_linear_velocity(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_angular_velocity(self, handle: int, x: float, y: float, z: float) -> None: ...

    # --- Queries ---
    def raycast(
        self, 
        start: Tuple[float, float, float], 
        direction: Tuple[float, float, float], 
        max_dist: float = 1000.0
    ) -> Optional[Tuple[int, float]]: ...

    def overlap_sphere(self, center: Tuple[float, float, float], radius: float) -> List[int]: ...
    def overlap_aabb(self, min: Tuple[float, float, float], max: Tuple[float, float, float]) -> List[int]: ...

    # --- Utils ---
    def get_index(self, handle: int) -> Optional[int]: ...
    def is_alive(self, handle: int) -> bool: ...
    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...

    # --- Properties ---
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
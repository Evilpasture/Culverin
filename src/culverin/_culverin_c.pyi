import sys
from typing import Tuple, Optional, Any, List

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
        size: Tuple[float, float, float] = (1, 1, 1),
        shape: int = 0,    # 0=Box, 1=Sphere, 2=Capsule
        motion: int = 2    # 0=Static, 1=Kinematic, 2=Dynamic
    ) -> int: ...  # Returns Handle

    def destroy_body(self, handle: int) -> None: ...
    
    # --- Interaction ---
    def apply_impulse(self, handle: int, x: float, y: float, z: float) -> None: ...

    def raycast(
        self, 
        start: Tuple[float, float, float], 
        direction: Tuple[float, float, float], 
        max_dist: float = 1000.0
    ) -> Optional[Tuple[int, float]]: ... # Returns (Handle, Fraction)

    # --- Queries ---
    def overlap_sphere(
        self, 
        center: Tuple[float, float, float], 
        radius: float
    ) -> List[int]: ... # Returns list of Handles

    def overlap_aabb(
        self, 
        min_point: Tuple[float, float, float], 
        max_point: Tuple[float, float, float]
    ) -> List[int]: ... # Returns list of Handles

    # --- Utils ---
    def get_index(self, handle: int) -> Optional[int]: ...
    def is_alive(self, handle: int) -> bool: ...

    # --- Memory Views ---
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
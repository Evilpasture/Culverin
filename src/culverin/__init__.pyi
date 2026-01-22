from typing import Tuple, List, Optional, TypedDict, Union
from ._culverin_c import PhysicsWorld as PhysicsWorld

# Semantic Types
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

class BodyConfig(TypedDict, total=False):
    pos: Vec3          # (x, y, z)
    rot: Quat          # (w, x, y, z)
    size: Vec3         # (half_x, half_y, half_z)
    mass: float        # 0.0 = Static, >0.0 = Dynamic
    restitution: float
    friction: float

class WorldSettings(TypedDict, total=False):
    gravity: Vec3
    penetration_slop: float
    max_bodies: int
    max_pairs: int

# We re-export PhysicsWorld with better type hints for the constructor
class PhysicsWorld:
    def __init__(
        self, 
        settings: Optional[WorldSettings] = None, 
        bodies: Optional[List[BodyConfig]] = None
    ) -> None: ...

    def step(self, dt: float = 1.0/60.0) -> None: ...

    def create_box(self, pos: Vec3, size: Vec3) -> int: ...

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

__all__ = ["PhysicsWorld"]
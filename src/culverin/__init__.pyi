from typing import Tuple, List, Optional, TypedDict, Union, overload
# We don't import the class directly to allow for re-definition with better types
from . import _culverin_c

# Semantic Types
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

class BodyConfig(TypedDict, total=False):
    pos: Vec3          # (x, y, z)
    rot: Quat          # (x, y, z, w)
    shape: int         # 0=Box, 1=Sphere, 2=Capsule
    size: Vec3         # (half_x, half_y, half_z) or (radius, 0, 0)
    mass: float        # 0.0 = Static, >0.0 = Dynamic

class WorldSettings(TypedDict, total=False):
    gravity: Vec3
    penetration_slop: float
    max_bodies: int
    max_pairs: int

# We define this class to shadow the C one with richer types
class PhysicsWorld(_culverin_c.PhysicsWorld):
    def __init__(
        self, 
        settings: Optional[WorldSettings] = None, 
        bodies: Optional[List[BodyConfig]] = None
    ) -> None: ...

    def step(self, dt: float = 1.0/60.0) -> None: ...

    def apply_impulse(self, index: int, x: float, y: float, z: float) -> None: ...

    def raycast(
        self, 
        start: Vec3, 
        direction: Vec3, 
        max_dist: float = 1000.0
    ) -> Optional[Tuple[int, float]]: 
        """
        Casts a ray into the scene.
        
        Args:
            start: (x, y, z) origin
            direction: (x, y, z) direction vector
            max_dist: Maximum ray length (default 1000.0)
            
        Returns:
            None if miss.
            (body_index, fraction) if hit. 
            Hit Position = start + (direction_normalized * max_dist * fraction)
        """
        ...

    @property
    def positions(self) -> memoryview: 
        """Flat array of floats (count * 4). Layout: [x, y, z, pad, ...]"""
        ...
    
    @property
    def rotations(self) -> memoryview: 
        """Flat array of floats (count * 4). Layout: [x, y, z, w, ...]"""
        ...

    @property
    def velocities(self) -> memoryview: ...
    
    @property
    def angular_velocities(self) -> memoryview: ...
    
    @property
    def count(self) -> int: ...

    @property
    def time(self) -> float: ...

__all__ = ["PhysicsWorld", "BodyConfig", "WorldSettings"]
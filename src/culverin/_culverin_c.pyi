import sys
from typing import Tuple, Optional, Any, overload

# Raw C types are simple
class PhysicsWorld:
    def __init__(
        self, 
        settings: Optional[dict] = None, 
        bodies: Optional[list[dict]] = None
    ) -> None: ...

    def step(self, dt: float = 1.0/60.0) -> None: ...

    def apply_impulse(self, index: int, x: float, y: float, z: float) -> None: ...

    def raycast(
        self, 
        start: Tuple[float, float, float], 
        direction: Tuple[float, float, float], 
        max_dist: float = 1000.0
    ) -> Optional[Tuple[int, float]]: 
        """
        Casts a ray. Returns (body_index, fraction) or None.
        """
        ...

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
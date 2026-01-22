import sys
from typing import Tuple, Optional, Any

# Define the C-level interface
class PhysicsWorld:
    def __init__(
        self, 
        settings: Optional[dict] = None, 
        bodies: Optional[list[dict]] = None
    ) -> None: ...

    def step(self, dt: float = 1.0/60.0) -> None: ...

    def create_box(
        self, 
        pos: Tuple[float, float, float], 
        size: Tuple[float, float, float]
    ) -> int: ...

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
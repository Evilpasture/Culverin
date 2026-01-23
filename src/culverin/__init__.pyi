from typing import Tuple, List, Optional, TypedDict, Union
from . import _culverin_c

# Semantic Types
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Handle = int  # 64-bit Stable Identifier

# Constants for convenience
SHAPE_BOX: int = 0
SHAPE_SPHERE: int = 1
SHAPE_CAPSULE: int = 2

MOTION_STATIC: int = 0
MOTION_KINEMATIC: int = 1
MOTION_DYNAMIC: int = 2

class BodyConfig(TypedDict, total=False):
    pos: Vec3
    rot: Quat
    shape: int
    size: Vec3
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

    def step(self, dt: float = 1.0/60.0) -> None:
        """
        Advances the simulation by 'dt'.
        Flushes any pending create/destroy commands at the start of the step.
        Updates memoryviews (positions, rotations) automatically.
        """
        ...

    def create_body(
        self, 
        pos: Vec3 = (0, 0, 0),
        rot: Quat = (0, 0, 0, 1),
        size: Vec3 = (1, 1, 1),
        shape: int = SHAPE_BOX,
        motion: int = MOTION_DYNAMIC
    ) -> Handle:
        """
        Queues a body for creation. The body will be added to the simulation
        at the beginning of the next step().
        
        Returns:
            A stable Handle (int) used to reference this body.
        """
        ...

    def destroy_body(self, handle: Handle) -> None:
        """
        Queues a body for destruction. The body is removed at the start 
        of the next step().
        """
        ...

    def apply_impulse(self, handle: Handle, x: float, y: float, z: float) -> None:
        """
        Applies an instantaneous force impulse to the body.
        Safe to call during simulation (e.g. inside callbacks) or main loop.
        """
        ...

    def raycast(
        self, 
        start: Vec3, 
        direction: Vec3, 
        max_dist: float = 1000.0
    ) -> Optional[Tuple[Handle, float]]: 
        """
        Casts a ray. Returns (Handle, Fraction) or None if missed.
        """
        ...

    def overlap_sphere(self, center: Vec3, radius: float) -> List[Handle]:
        """Returns a list of Handles for all bodies touching the sphere."""
        ...

    def overlap_aabb(self, min_point: Vec3, max_point: Vec3) -> List[Handle]:
        """Returns a list of Handles for all bodies overlapping the AABB."""
        ...

    def get_index(self, handle: Handle) -> Optional[int]: 
        """
        Converts a stable Handle to a dense Index.
        
        Use this to look up data in the .positions / .rotations memoryviews.
        Note: Indices change when bodies are removed (Swap-and-Pop), 
        Handles do not.
        
        Returns None if the handle is invalid or destroyed.
        """
        ...
    
    def is_alive(self, handle: Handle) -> bool:
        """
        Checks if a handle refers to a valid, non-destroyed body.
        Returns True for bodies created this frame (pending) and active bodies.
        Returns False for stale handles or bodies marked for destruction.
        """
        ...

    # --- Properties ---
    @property
    def positions(self) -> memoryview: 
        """Flat f32 array [count * 4]. Layout: [x,y,z,pad, x,y,z,pad...]"""
        ...
    
    @property
    def rotations(self) -> memoryview: 
        """Flat f32 array [count * 4]. Layout: [x,y,z,w, ...]"""
        ...

    @property
    def velocities(self) -> memoryview: ...
    @property
    def angular_velocities(self) -> memoryview: ...
    @property
    def count(self) -> int: ...
    @property
    def time(self) -> float: ...

__all__ = ["PhysicsWorld", "BodyConfig", "WorldSettings", "Handle", 
           "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE"]
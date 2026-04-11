from typing import TypedDict
__version__: str
# Re-export from the Python helper
from ._culverin import (
    Automatic,
    Engine,
    Manual,
    Transmission,
    euler_to_quat,
    load_urdf,
    parse_urdf,
)

# Re-export from the compiled artifact
from ._culverin_c import (
    CONSTRAINT_CONE,
    CONSTRAINT_DISTANCE,
    CONSTRAINT_FIXED,
    CONSTRAINT_HINGE,
    CONSTRAINT_POINT,
    CONSTRAINT_SLIDER,
    EVENT_ADDED,
    EVENT_PERSISTED,
    EVENT_REMOVED,
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    # Constants
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_CONVEX_HULL,
    SHAPE_CYLINDER,
    SHAPE_HEIGHTFIELD,
    SHAPE_MESH,
    SHAPE_PLANE,
    SHAPE_SPHERE,
    Character,
    # Core Classes
    PhysicsWorld,
    Ragdoll,
    RagdollSettings,
    SoftBodySharedSettings,
    Skeleton,
    Vehicle,
    mutate_tuple
)

class WheelConfig(TypedDict):
    pos: tuple[float, float, float]
    radius: float


class TrackConfig(TypedDict):
    indices: list[int]
    driven_wheel: int

__all__ = [
    "CONSTRAINT_CONE",
    "CONSTRAINT_DISTANCE",
    "CONSTRAINT_FIXED",
    "CONSTRAINT_HINGE",
    "CONSTRAINT_POINT",
    "CONSTRAINT_SLIDER",
    "EVENT_ADDED",
    "EVENT_PERSISTED",
    "EVENT_REMOVED",
    "MOTION_DYNAMIC",
    "MOTION_KINEMATIC",
    "MOTION_STATIC",
    "SHAPE_BOX",
    "SHAPE_CAPSULE",
    "SHAPE_CONVEX_HULL",
    "SHAPE_CYLINDER",
    "SHAPE_HEIGHTFIELD",
    "SHAPE_MESH",
    "SHAPE_PLANE",
    "SHAPE_SPHERE",
    "Automatic",
    "Character",
    "Engine",
    "Manual",
    "PhysicsWorld",
    "Ragdoll",
    "RagdollSettings",
    "SoftBodySharedSettings",
    "Skeleton",
    "Transmission",
    "Vehicle",
    "euler_to_quat",
    "load_urdf",
    "parse_urdf",
    "WheelConfig",
    "TrackConfig",
    "mutate_tuple",
]

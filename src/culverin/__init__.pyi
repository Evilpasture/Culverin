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
    BEND_DIHEDRAL,
    BEND_DISTANCE,
    BEND_NONE,
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
    USE_DOUBLE_PRECISION,
    Character,
    ContactEvent,
    MathService,
    # Core Classes
    PhysicsWorld,
    Ragdoll,
    RagdollSettings,
    Registry,
    Ship,
    Skeleton,
    SoftBodySharedSettings,
    Vehicle,
    WorldSettings,
    BodyDefinition,
    _dump_schema_json,  # type: ignore
    mutate_tuple,
)

class WheelConfig(TypedDict):
    pos: tuple[float, float, float]
    radius: float

class TrackConfig(TypedDict):
    indices: list[int]
    driven_wheel: int

__all__ = [
    "BEND_DIHEDRAL",
    "BEND_DISTANCE",
    "BEND_NONE",
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
    "USE_DOUBLE_PRECISION",
    "Automatic",
    "Character",
    "ContactEvent",
    "Engine",
    "Manual",
    "MathService",
    "PhysicsWorld",
    "Ragdoll",
    "RagdollSettings",
    "Registry",
    "Ship",
    "Skeleton",
    "SoftBodySharedSettings",
    "TrackConfig",
    "Transmission",
    "Vehicle",
    "WheelConfig",
    "WorldSettings",
    "BodyDefinition",
    "_dump_schema_json",
    "euler_to_quat",
    "load_urdf",
    "mutate_tuple",
    "parse_urdf",
]

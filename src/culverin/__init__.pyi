from __future__ import annotations

# Re-export from the compiled artifact
from ._culverin_c import (
    # Core Classes
    PhysicsWorld, Character, Vehicle, Ragdoll, RagdollSettings, Skeleton,

    WheelConfig, TrackConfig,
    
    # Constants
    SHAPE_BOX, SHAPE_SPHERE, SHAPE_CAPSULE, SHAPE_CYLINDER, SHAPE_PLANE, SHAPE_MESH, SHAPE_HEIGHTFIELD, SHAPE_CONVEX_HULL,
    MOTION_STATIC, MOTION_KINEMATIC, MOTION_DYNAMIC,
    CONSTRAINT_FIXED, CONSTRAINT_POINT, CONSTRAINT_HINGE, CONSTRAINT_SLIDER, CONSTRAINT_DISTANCE, CONSTRAINT_CONE,
    EVENT_ADDED, EVENT_PERSISTED, EVENT_REMOVED,
)

# Re-export from the Python helper
from ._culverin import (
    Engine, Transmission, Automatic, Manual,
    WheelConfig, TrackConfig,
)

__all__ = [
    "PhysicsWorld", "Character", "Vehicle", "Ragdoll", "RagdollSettings", "Skeleton",
    "Engine", "Transmission", "Automatic", "Manual",
    "WheelConfig", "TrackConfig",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", "CONSTRAINT_SLIDER", "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE",
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED",
]
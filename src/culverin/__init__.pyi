from typing import Any, List, Tuple, Sequence, Dict

# 1. Import Pure Python Helpers
# (We assume _culverin.py has type hints inline or inferred)
from ._culverin import (
    Engine,
    Transmission,
    Automatic,
    Manual,
    validate_constraint,
    validate_settings,
    bake_scene
)

# 2. Import C-Extension Symbols (from _culverin_c.pyi)
from ._culverin_c import (
    # Core Classes
    PhysicsWorld,
    Character,
    Vehicle,
    Ragdoll,
    RagdollSettings,
    Skeleton,

    # Shape Constants
    SHAPE_BOX,
    SHAPE_SPHERE,
    SHAPE_CAPSULE,
    SHAPE_CYLINDER,
    SHAPE_PLANE,
    SHAPE_MESH,
    SHAPE_HEIGHTFIELD,
    SHAPE_CONVEX_HULL,

    # Motion Constants
    MOTION_STATIC,
    MOTION_KINEMATIC,
    MOTION_DYNAMIC,

    # Constraint Constants
    CONSTRAINT_FIXED,
    CONSTRAINT_POINT,
    CONSTRAINT_HINGE,
    CONSTRAINT_SLIDER,
    CONSTRAINT_DISTANCE,
    CONSTRAINT_CONE,

    # Event Constants
    EVENT_ADDED,
    EVENT_PERSISTED,
    EVENT_REMOVED,
)

# 3. Define the Public API
__all__ = [
    "PhysicsWorld",
    "Character",
    "Vehicle",
    "Ragdoll",
    "RagdollSettings",
    "Skeleton",
    "Engine",
    "Transmission",
    "Automatic",
    "Manual",
    "SHAPE_BOX",
    "SHAPE_SPHERE",
    "SHAPE_CAPSULE",
    "SHAPE_CYLINDER",
    "SHAPE_PLANE",
    "SHAPE_MESH",
    "SHAPE_HEIGHTFIELD",
    "SHAPE_CONVEX_HULL",
    "MOTION_STATIC",
    "MOTION_KINEMATIC",
    "MOTION_DYNAMIC",
    "CONSTRAINT_FIXED",
    "CONSTRAINT_POINT",
    "CONSTRAINT_HINGE",
    "CONSTRAINT_SLIDER",
    "CONSTRAINT_DISTANCE",
    "CONSTRAINT_CONE",
    "EVENT_ADDED",
    "EVENT_PERSISTED",
    "EVENT_REMOVED",
    "validate_constraint",
    "validate_settings",
    "bake_scene",
]
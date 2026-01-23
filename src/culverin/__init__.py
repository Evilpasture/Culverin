from ._culverin_c import (
    PhysicsWorld,
    Character, # Don't forget Character if you haven't added it yet!
    SHAPE_BOX,
    SHAPE_SPHERE,
    SHAPE_CAPSULE,
    SHAPE_CYLINDER,
    SHAPE_PLANE,
    SHAPE_MESH, # <--- Added
    MOTION_STATIC,
    MOTION_KINEMATIC,
    MOTION_DYNAMIC
)

__all__ = [
    "PhysicsWorld", "Character",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", 
    "SHAPE_CYLINDER", "SHAPE_PLANE", "SHAPE_MESH", # <--- Added
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC"
]
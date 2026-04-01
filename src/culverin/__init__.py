import os
import sys

# Windows DLL Resolution Fix
if sys.platform == "win32":
    # 1. Check if we provided a path via environment variable
    clang_path = os.environ.get("CLANG_BIN_PATH")
    if clang_path and os.path.exists(clang_path):
        os.add_dll_directory(clang_path)
    
    # 2. Try to find LLVM via PATH as a fallback
    import shutil
    clang_bin = shutil.which("clang")
    if clang_bin:
        os.add_dll_directory(os.path.dirname(clang_bin))

# 1. Load Pure Python Helpers
from ._culverin import (
    Engine, Transmission, Automatic, Manual,
    WheelConfig, TrackConfig,
    validate_constraint, validate_settings, bake_scene
)

# 2. Load Compiled Extension
# The C extension defines the heavy lifting classes and constants
from ._culverin_c import *

# 3. Explicit Export List
__all__ = [
    # Core Classes (C Extension)
    "PhysicsWorld", 
    "Character", 
    "Vehicle", 
    "Ragdoll", 
    "RagdollSettings", 
    "Skeleton",
    
    # Helper Classes (Python)
    "Engine", 
    "Transmission", 
    "Automatic", 
    "Manual",
        "WheelConfig",
        "TrackConfig",
    
    # Constants (C Extension + Python Mirror)
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", 
    "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", "CONSTRAINT_SLIDER", 
    "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE", 
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED",
    
    # Internal Helpers
    "validate_constraint", "validate_settings", "bake_scene"
]
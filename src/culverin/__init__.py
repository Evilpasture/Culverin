import os
import sys
from . import _culverin_c

# --- Windows DLL Resolution Fix ---
if sys.platform == "win32":
    clang_path = os.environ.get("CLANG_BIN_PATH")
    if clang_path and os.path.exists(clang_path):
        os.add_dll_directory(clang_path)
    import shutil
    clang_bin = shutil.which("clang")
    if clang_bin:
        os.add_dll_directory(os.path.dirname(clang_bin))

# 1. Load Pure Python Configs
from ._culverin import (
    Engine, Transmission, Automatic, Manual,
    WheelConfig, TrackConfig,
)

# 2. DEFINE HELPER FUNCTIONS
# We type-hint 'self' as the C class. 
# We use a string "_culverin_c.PhysicsWorld" to avoid runtime issues.
def get_position(self: "_culverin_c.PhysicsWorld", handle: int) -> tuple[float, float, float]:
    # The linter now knows 'self' has '.get_body_stats' because it checks _culverin_c.pyi
    stats = self.get_body_stats(handle)
    if stats: return stats[0]
    raise ValueError(f"Invalid or stale handle: {handle}")

def get_rotation(self: "_culverin_c.PhysicsWorld", handle: int) -> tuple[float, float, float, float]:
    stats = self.get_body_stats(handle)
    if stats: return stats[1]
    raise ValueError(f"Invalid or stale handle: {handle}")

def get_velocity(self: "_culverin_c.PhysicsWorld", handle: int) -> tuple[float, float, float]:
    stats = self.get_body_stats(handle)
    if stats: return stats[2]
    raise ValueError(f"Invalid or stale handle: {handle}")

def world_repr(self: "_culverin_c.PhysicsWorld") -> str:
    # The linter now knows 'self' has '.count' and '.time'
    return f"<culverin.PhysicsWorld bodies={self.count} time={self.time:.2f}>"

# 3. ATTACH HELPERS
# We use 'type: ignore' here because linters often think C-extension 
# classes are immutable/read-only even when MANAGED_DICT is enabled.
_culverin_c.PhysicsWorld.get_position = get_position # type: ignore
_culverin_c.PhysicsWorld.get_rotation = get_rotation # type: ignore
_culverin_c.PhysicsWorld.get_velocity = get_velocity # type: ignore
_culverin_c.PhysicsWorld.__repr__ = world_repr       # type: ignore

# 4. EXPOSE THE C CLASS
PhysicsWorld = _culverin_c.PhysicsWorld

# 5. Export Constants and other classes
from ._culverin_c import (
    Character, Vehicle, Ragdoll, RagdollSettings, Skeleton,
    MOTION_STATIC, MOTION_KINEMATIC, MOTION_DYNAMIC,
    SHAPE_BOX, SHAPE_SPHERE, SHAPE_CAPSULE, SHAPE_CYLINDER, SHAPE_PLANE, 
    SHAPE_MESH, SHAPE_HEIGHTFIELD, SHAPE_CONVEX_HULL,
    CONSTRAINT_FIXED, CONSTRAINT_POINT, CONSTRAINT_HINGE, CONSTRAINT_SLIDER, 
    CONSTRAINT_DISTANCE, CONSTRAINT_CONE, 
    EVENT_ADDED, EVENT_PERSISTED, EVENT_REMOVED,
)

__all__ = [
    "PhysicsWorld", "Character", "Vehicle", "Ragdoll", "RagdollSettings", "Skeleton",
    "Engine", "Transmission", "Automatic", "Manual", "WheelConfig", "TrackConfig",
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", 
    "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", "CONSTRAINT_SLIDER", 
    "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE", 
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED",
]
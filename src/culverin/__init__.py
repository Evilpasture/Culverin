import os
import sys
from pathlib import Path
from typing import TypedDict

from . import _culverin_c

__version__ = _culverin_c.__version__

# --- Windows DLL Resolution Fix ---
if sys.platform == "win32":
    clang_path = os.environ.get("CLANG_BIN_PATH")
    if clang_path and Path(clang_path).exists():
        os.add_dll_directory(str(Path(clang_path)))
    import shutil

    clang_bin = shutil.which("clang")
    if clang_bin:
        os.add_dll_directory(str(Path(clang_bin).parent))

# 1. Load Pure Python Configs
from ._culverin import (
    Automatic,
    Engine,
    Manual,
    Transmission,
    euler_to_quat,
    load_urdf,
    parse_urdf,
)

class WheelConfig(TypedDict):
    pos: tuple[float, float, float]
    radius: float


class TrackConfig(TypedDict):
    indices: list[int]
    driven_wheel: int

# 2. DEFINE HELPER FUNCTIONS
# We type-hint 'self' as the C class.
# We use a string "_culverin_c.PhysicsWorld" to avoid runtime issues.
def get_position(self: _culverin_c.PhysicsWorld, handle: int) -> tuple[float, float, float] | None:
    """Returns the world position of a body as (x, y, z), or None if the handle is invalid."""
    stats = self.get_body_stats(handle)
    # If stats is None, this returns None naturally
    return stats[0] if stats else None


def get_rotation(
    self: _culverin_c.PhysicsWorld, handle: int
) -> tuple[float, float, float, float] | None:
    """Returns the world rotation of a body as (x, y, z, w), or None if the handle is invalid."""
    stats = self.get_body_stats(handle)
    return stats[1] if stats else None


def get_velocity(self: _culverin_c.PhysicsWorld, handle: int) -> tuple[float, float, float] | None:
    """Returns the world velocity of a body as (x, y, z), or None if the handle is invalid."""
    stats = self.get_body_stats(handle)
    return stats[2] if stats else None


def world_repr(self: _culverin_c.PhysicsWorld) -> str:
    return f"<culverin.PhysicsWorld bodies={self.count} time={self.time:.2f}>"


# 3. ATTACH HELPERS
# We use 'type: ignore' here because linters often think C-extension
# classes are immutable/read-only even when MANAGED_DICT is enabled.
_culverin_c.PhysicsWorld.get_position = get_position  # type: ignore
_culverin_c.PhysicsWorld.get_rotation = get_rotation  # type: ignore
_culverin_c.PhysicsWorld.get_velocity = get_velocity  # type: ignore
_culverin_c.PhysicsWorld.__repr__ = world_repr  # type: ignore


# 4. EXPOSE THE C CLASS
PhysicsWorld = _culverin_c.PhysicsWorld

# 5. Export Constants and other classes
from ._culverin_c import (  # noqa: E402
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
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_CONVEX_HULL,
    SHAPE_CYLINDER,
    SHAPE_HEIGHTFIELD,
    SHAPE_MESH,
    SHAPE_PLANE,
    SHAPE_SPHERE,
    Character,
    Ragdoll,
    RagdollSettings,
    Skeleton,
    Vehicle,
)

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
    "Skeleton",
    "Transmission",
    "Vehicle",
    "euler_to_quat",
    "load_urdf",
    "parse_urdf",
    "WheelConfig",
    "TrackConfig",
]

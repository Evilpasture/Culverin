import os
import shutil
import sys
from pathlib import Path
from typing import TypedDict

def _verify_cpu_requirements():
    """Ensure the CPU meets the x86-64-v3 (AVX2/FMA) requirement."""
    import platform
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return # Skip check on ARM (Mac M1/M2, etc.)

    try:
        # On Linux, check /proc/cpuinfo
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                if "avx2" not in content or "fma" not in content:
                    raise RuntimeError(
                        "Culverin requires a CPU with AVX2/FMA support (x86-64-v3). "
                        "GitHub Actions runner or local CPU is too old."
                    )
        # On Windows, we can use a quick check via ctypes or just let the 
        # DLL loader handle it, but for CI, the Linux check is the priority.
    except Exception:
        # If we can't check, we proceed and let the OS signal the error
        pass

_verify_cpu_requirements()

del _verify_cpu_requirements

def setup_runtime_dlls() -> None:
    if sys.platform != "win32":
        return

    # Priority 1: Explicitly defined runtime directory
    # Good for dev environments where you want to point to a specific build
    env_runtime = os.environ.get("CULVERIN_ASAN_LIB_PATH")
    if env_runtime and Path(env_runtime).exists():
        os.add_dll_directory(env_runtime)
        return

    # Priority 2: Derived from LLVM_INSTALL_DIR
    llvm_root = os.environ.get("LLVM_INSTALL_DIR")

    # Priority 3: Auto-discovery via PATH (find clang, then infer lib path)
    if not llvm_root:
        clang_bin = shutil.which("clang")
        if clang_bin:
            llvm_root = Path(clang_bin).parent.parent

    if llvm_root:
        llvm_root = Path(llvm_root)
        # Check both modern and legacy LLVM layout structures
        potential_paths = [
            llvm_root / "lib" / "clang" / "23" / "lib" / "x86_64-pc-windows-msvc",
            llvm_root / "lib" / "windows",
        ]

        for p in potential_paths:
            if p.exists():
                os.add_dll_directory(str(p))
                return


setup_runtime_dlls()

del setup_runtime_dlls

from . import _culverin_c

__version__ = _culverin_c.__version__


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
    """Returns the world position of a body as (x, y, z)."""
    idx = self.get_index(handle)
    if idx is None:
        return None
    # positions format is 'd' or 'f' automatically based on JPH_DOUBLE_PRECISION
    view = memoryview(self.positions)
    base = idx * 4
    try:
        return (view[base], view[base + 1], view[base + 2])
    except (IndexError, ValueError):
        return None


def get_rotation(
    self: _culverin_c.PhysicsWorld, handle: int
) -> tuple[float, float, float, float] | None:
    """Returns the world rotation of a body as (x, y, z, w)."""
    idx = self.get_index(handle)
    if idx is None:
        return None
    view = memoryview(self.rotations)
    base = idx * 4
    try:
        return (view[base], view[base + 1], view[base + 2], view[base + 3])
    except (IndexError, ValueError):
        return None


def get_velocity(self: _culverin_c.PhysicsWorld, handle: int) -> tuple[float, float, float] | None:
    """Returns the world linear velocity of a body as (x, y, z)."""
    idx = self.get_index(handle)
    if idx is None:
        return None
    view = memoryview(self.velocities)
    base = idx * 4
    try:
        return (view[base], view[base + 1], view[base + 2])
    except (IndexError, ValueError):
        return None


def get_angular_velocity(
    self: _culverin_c.PhysicsWorld, handle: int
) -> tuple[float, float, float] | None:
    """Returns the world angular velocity of a body as (x, y, z)."""
    idx = self.get_index(handle)
    if idx is None:
        return None
    view = memoryview(self.angular_velocities)
    base = idx * 4
    try:
        return (view[base], view[base + 1], view[base + 2])
    except (IndexError, ValueError):
        return None


def world_repr(self: _culverin_c.PhysicsWorld) -> str:
    return f"<culverin.PhysicsWorld bodies={self.count} time={self.time:.2f}>"


# 3. ATTACH HELPERS
_culverin_c.PhysicsWorld.get_position = get_position 
_culverin_c.PhysicsWorld.get_rotation = get_rotation  
_culverin_c.PhysicsWorld.get_velocity = get_velocity  
_culverin_c.PhysicsWorld.get_angular_velocity = get_angular_velocity 
_culverin_c.PhysicsWorld.__repr__ = world_repr 


# 4. EXPOSE THE C CLASS
PhysicsWorld = _culverin_c.PhysicsWorld

# 5. Export Constants and other classes
from ._culverin_c import (  # noqa: E402
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
    MathService,
    Ragdoll,
    RagdollSettings,
    Registry,
    Ship,
    Skeleton,
    SoftBodySharedSettings,
    Vehicle,
    _dump_schema_json,  # type: ignore
    mutate_tuple,
)

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
    "_dump_schema_json",
    "euler_to_quat",
    "load_urdf",
    "mutate_tuple",
    "parse_urdf",
]

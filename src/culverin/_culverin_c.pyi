# Type Stubs for the Culverin C-Extension (_culverin_c).
# Using Python built-in generics for modern DX.

# Note: Some methods compiled with STRICT_HANDLE_ENABLED will crash with ValueError. Otherwise, returns None or silent.

from collections.abc import Buffer, Sequence
from typing import Any, Literal, TypedDict, overload

__version__: str

class ContactEvent(TypedDict):
    bodies: tuple[int, int]
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    impulse: float
    slide_sq: float
    materials: tuple[int, int]
    type: int

class Engine:
    max_torque: float
    max_rpm: float
    min_rpm: float
    inertia: float

class Transmission:
    clutch_strength: float
    differential_ratio: float
    ratios: list[float]
    reverse_ratios: list[float]

class Automatic(Transmission):
    mode: int
    shift_up_rpm: float
    shift_down_rpm: float

class Manual(Transmission):
    mode: int

# --- Constants ---
SHAPE_BOX: int = 0
SHAPE_SPHERE: int = 1
SHAPE_CAPSULE: int = 2
SHAPE_CYLINDER: int = 3
SHAPE_PLANE: int = 4
SHAPE_MESH: int = 5
SHAPE_HEIGHTFIELD: int = 6
SHAPE_CONVEX_HULL: int = 7

MOTION_STATIC: int = 0
MOTION_KINEMATIC: int = 1
MOTION_DYNAMIC: int = 2

CONSTRAINT_FIXED: int = 0
CONSTRAINT_POINT: int = 1
CONSTRAINT_HINGE: int = 2
CONSTRAINT_SLIDER: int = 3
CONSTRAINT_DISTANCE: int = 4
CONSTRAINT_CONE: int = 5

EVENT_ADDED: int = 0
EVENT_PERSISTED: int = 1
EVENT_REMOVED: int = 2

USE_DOUBLE_PRECISION: Literal[0, 1]

BEND_NONE: int = 0
BEND_DISTANCE: int = 1
BEND_DIHEDRAL: int = 2

# --- Type Aliases ---
type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]

class WheelConfig(TypedDict):
    pos: tuple[float, float, float]
    radius: float

class TrackConfig(TypedDict):
    indices: list[int]  # The indices of the wheels this track wraps
    driven_wheel: int  # The index of the wheel providing torque

type ShapeSize = Buffer | Sequence[float] | None

class Character:
    @property
    def handle(self) -> int: ...
    def move(self, velocity: Vec3, dt: float) -> None: ...
    def get_position(self) -> Vec3: ...
    def set_position(self, pos: Vec3) -> None: ...
    def set_rotation(self, rot: Quat) -> None: ...
    def is_grounded(self) -> bool: ...
    def set_strength(self, strength: float) -> None: ...
    def get_render_transform(self, alpha: float) -> tuple[Vec3, Quat]: ...

class Skeleton:
    def __init__(self) -> None: ...
    def add_joint(self, name: str, parent_index: int = -1) -> int: ...
    def finalize(self) -> None: ...
    def get_joint_index(self, name: str) -> int: ...

class RagdollSettings:
    def add_part(
        self,
        joint_index: int,
        shape_type: int,
        size: float | Sequence[float],
        mass: float = 10.0,
        parent_index: int = -1,
        twist_min: float = -0.1,
        twist_max: float = 0.1,
        cone_angle: float = 0.0,
        axis: Vec3 = (1.0, 0.0, 0.0),
        normal: Vec3 = (0.0, 1.0, 0.0),
        pos: Vec3 = (0.0, 0.0, 0.0),
    ) -> None: ...
    def stabilize(self) -> bool: ...

class Ragdoll:
    def drive_to_pose(self, root_pos: Vec3, root_rot: Quat, matrices: Buffer) -> None: ...
    def get_body_handles(self) -> list[int]: ...
    def get_debug_info(self) -> list[dict[str, Any]]: ...

class SoftBodySharedSettings:
    def __init__(self) -> None: ...
    def add_vertex(self, pos: Vec3, inv_mass: float) -> None: ...
    def add_face(self, v1: int, v2: int, v3: int) -> None: ...
    def add_pinned_vertex(self, index: int, /) -> None: ...
    def create_constraints(self, compliance: float, bend_type: int = 1) -> None: ...
    def optimize(self) -> None: ...
    def get_vertex_position(self, index: int, /) -> Vec3: ...

class Vehicle:
    @property
    def wheel_count(self) -> int: ...
    def destroy(self) -> None: ...
    def get_debug_state(self) -> None: ...
    def get_wheel_local_transform(self, index: int) -> tuple[Vec3, Quat]: ...
    def get_wheel_transform(self, index: int) -> tuple[Vec3, Quat]: ...
    def set_input(
        self, forward: float = 0.0, right: float = 0.0, brake: float = 0.0, handbrake: float = 0.0
    ) -> None: ...
    def set_tank_input(self, left: float, right: float, brake: float = 0.0) -> None: ...

class PhysicsWorld:
    # --- Properties (Direct Memory Access) ---
    @property
    def positions(self) -> memoryview: ...
    @property
    def rotations(self) -> memoryview: ...
    @property
    def velocities(self) -> memoryview: ...
    @property
    def angular_velocities(self) -> memoryview: ...
    @property
    def user_data(self) -> memoryview: ...
    @property
    def count(self) -> int: ...
    @property
    def time(self) -> float: ...
    @property
    def shape_count(self) -> int: ...
    @property
    def is_step_pending(self) -> bool: ...
    @property
    def max_bodies(self) -> int: ...
    @property
    def remaining_capacity(self) -> int: ...

    # High-level Python helpers
    def get_position(self, handle: int) -> Vec3: ...
    def get_rotation(self, handle: int) -> Quat: ...
    def get_velocity(self, handle: int) -> Vec3: ...

    # --- Lifecycle ---
    def __init__(
        self, settings: dict[str, Any] | None = None, bodies: list[dict[str, Any]] | None = None
    ) -> None: ...
    def step(self, dt: float = ...) -> None: ...

    # --- Creation ---
    def create_body(
        self,
        pos: Vec3 | None = None,
        rot: Quat | None = None,
        size: float | Sequence[float] | None = None,
        shape: int = 0,
        motion: int = 2,
        user_data: int = 0,
        is_sensor: bool = False,
        mass: float = -1.0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        friction: float = 0.2,
        restitution: float = 0.0,
        material_id: int = 0,
        ccd: bool = False,
    ) -> int: ...
    def create_bodies_batch(
        self, positions: list[Vec3], sizes: list[Any], shape_type: int = 0, motion_type: int = 2
    ) -> list[int]: ...
    def create_mesh_body(
        self,
        pos: Vec3,
        rot: Quat,
        vertices: Buffer,
        indices: Buffer,
        user_data: int = 0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
    ) -> int: ...
    def create_convex_hull(
        self,
        pos: Vec3,
        rot: Quat,
        points: Buffer,
        motion: int = 2,
        mass: float = -1.0,
        user_data: int = 0,
        is_sensor: bool = False,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        material_id: int = 0,
        friction: float = 0.2,
        restitution: float = 0.0,
        ccd: bool = False,
    ) -> int: ...
    def create_compound_body(
        self,
        pos: Vec3,
        rot: Quat,
        parts: Sequence[tuple[Vec3, Quat, int, Any]],
        motion: int = 2,
        mass: float = -1.0,
        user_data: int = 0,
        is_sensor: bool = False,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        material_id: int = 0,
        friction: float = 0.2,
        restitution: float = 0.0,
        ccd: bool = False,
    ) -> int: ...
    def create_heightfield(
        self,
        pos: Vec3,
        rot: Quat,
        scale: Vec3,
        heights: Buffer,
        grid_size: int,
        user_data: int = 0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        material_id: int = 0,
        friction: float = 0.5,
        restitution: float = 0.0,
    ) -> int: ...
    def create_soft_body(
        self,
        shared_settings: SoftBodySharedSettings,
        pos: Vec3,
        rot: Quat,
        user_data: int = 0,
        category: int = 0xFFFF,
        mask: int = 0xFFFF,
        pressure: float = 0.0,
        vertex_radius: float = 0.05,
        linear_damping: float = 0.1,
        num_iterations: int = 10,
        max_linear_velocity: float = 500.0,
        gravity_factor: float = 1.0,
        friction: float = 0.2,
        restitution: float = 0.0,
        make_rotation_identity: bool = False,
    ) -> int: ...
    def create_character(
        self,
        pos: Vec3,
        height: float = 1.8,
        radius: float = 0.4,
        step_height: float = 0.4,
        max_slope: float = 45.0,
    ) -> Character: ...
    def create_vehicle(
        self,
        chassis: int,
        wheels: list[WheelConfig],
        drive: str = "RWD",
        engine: Engine | None = None,
        transmission: Transmission | None = None,
    ) -> Vehicle: ...
    def create_tracked_vehicle(
        self,
        chassis: int,
        wheels: list[WheelConfig],
        tracks: list[TrackConfig],
        max_torque: float = 5000.0,
        max_rpm: float = 6000.0,
    ) -> Vehicle: ...
    def create_ragdoll(
        self, settings: RagdollSettings, pos: Vec3, rot: Quat = (0, 0, 0, 1)
    ) -> Ragdoll: ...
    def create_ragdoll_settings(self, skeleton: Skeleton) -> RagdollSettings: ...

    # --- Destruction ---
    def destroy_body(self, handle: int) -> None: ...
    def destroy_bodies_batch(self, handles: Buffer | list[int]) -> None: ...

    # --- Constraints ---
    def create_constraint(
        self,
        type: int,
        body1: int,
        body2: int,
        params: float | Sequence[float] | Sequence[float | Sequence[float]] | None = None,
        motor: dict[str, float | int] | None = None,
    ) -> int: ...
    def destroy_constraint(self, handle: int) -> None: ...
    def set_constraint_target(self, handle: int, target: float) -> None: ...
    def get_constraint_type(self, handle: int) -> int | None: ...

    # --- Forces ---
    def apply_impulse(self, handle: int, x: float, y: float, z: float) -> None: ...
    def apply_impulse_at(
        self, handle: int, ix: float, iy: float, iz: float, px: float, py: float, pz: float
    ) -> None: ...
    def apply_force(self, handle: int, x: float, y: float, z: float) -> None: ...
    def apply_torque(self, handle: int, x: float, y: float, z: float) -> None: ...
    def apply_angular_impulse(self, handle: int, x: float, y: float, z: float) -> None: ...
    def apply_buoyancy(
        self,
        handle: int,
        surface_y: float,
        buoyancy: float = 1.0,
        linear_drag: float = 0.5,
        angular_drag: float = 0.5,
        dt: float = ...,
        fluid_velocity: Vec3 = (0, 0, 0),
    ) -> bool: ...
    def apply_buoyancy_batch(
        self,
        handles: Buffer | list[int],
        surface_y: float = 0.0,
        buoyancy: float = 1.0,
        linear_drag: float = 0.5,
        angular_drag: float = 0.5,
        dt: float = ...,
        fluid_velocity: Vec3 = (0, 0, 0),
    ) -> None: ...

    # --- Setters ---
    def set_position(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_rotation(self, handle: int, x: float, y: float, z: float, w: float) -> None: ...
    def set_transform(self, handle: int, pos: Vec3, rot: Quat) -> None: ...
    def set_linear_velocity(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_angular_velocity(self, handle: int, x: float, y: float, z: float) -> None: ...
    def set_gravity(self, x: float, y: float, z: float) -> None: ...
    def set_ccd(self, handle: int, enabled: bool) -> None: ...
    def set_motion_type(self, handle: int, motion: int) -> None: ...
    def set_collision_filter(self, handle: int, category: int, mask: int) -> None: ...
    def set_user_data(self, handle: int, data: int) -> None: ...
    def register_material(
        self, id: int, friction: float = 0.5, restitution: float = 0.0
    ) -> None: ...
    def activate(self, handle: int) -> None: ...
    def deactivate(self, handle: int) -> None: ...

    # --- Getters & Queries ---
    def get_body_stats(self, handle: int) -> tuple[Vec3, Quat, Vec3] | None: ...
    def get_index(self, handle: int) -> int | None: ...
    def is_alive(self, handle: int) -> bool: ...
    def is_active(self, handle: int) -> bool: ...
    def get_motion_type(self, handle: int) -> int | None: ...
    def get_user_data(self, handle: int) -> int | None: ...
    def get_soft_body_vertices(self, handle: int) -> memoryview: ...
    def raycast(
        self, start: Vec3, direction: Vec3, max_dist: float = 1000.0, ignore: int = 0
    ) -> tuple[int, float, Vec3] | None: ...
    def raycast_batch(
        self, starts: Buffer, directions: Buffer, max_dist: float = 1000.0
    ) -> bytes: ...
    def shapecast(
        self, shape: int, pos: Vec3, rot: Quat, dir: Vec3, size: ShapeSize = None, ignore: int = 0
    ) -> tuple[int, float, Vec3, Vec3] | None: ...
    def overlap_sphere(self, center: Vec3, radius: float) -> list[int]: ...
    def overlap_aabb(self, min: Vec3, max: Vec3) -> list[int]: ...

    # --- Events and Debug ---
    def get_contact_events(self) -> list[tuple[int, int]]: ...
    def get_contact_events_ex(self) -> list[ContactEvent]: ...
    def get_contact_events_raw(self) -> memoryview: ...
    def get_debug_data(
        self,
        shapes: bool = True,
        constraints: bool = True,
        bbox: bool = False,
        centers: bool = False,
        wireframe: bool = True,
    ) -> tuple[bytes, bytes]: ...
    def get_active_indices(self) -> bytes: ...
    def get_render_state(self, alpha: float) -> bytes: ...
    def save_state(self) -> bytes: ...
    def load_state(self, state: bytes) -> None: ...

    # --- Internal / Benchmarking ---
    def _benchmark_parse(self, *args: object, **kwargs: object) -> None: ...
    def _benchmark_build(
        self,
    ) -> tuple[
        int,
        float,
        float,
        str,
        bool,
        Literal[100],
        float,
        Literal[
            "Lorem ipsum dolor sit amet consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        ],
        Literal[False],
    ]: ...

def _dump_schema_json() -> None: ...

@overload
def mutate_tuple(target: tuple[Any, ...], index: int, value: object) -> int:
    ...

@overload
def mutate_tuple[T: tuple[Any, ...]](target: T, index: int, value: object, registry: dict[Any, T], key: object) -> int:
    ...

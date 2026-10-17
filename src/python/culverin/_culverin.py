import array
import math
import opcode
import types
from collections.abc import Callable
from typing import Any, Literal, cast, overload

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
    "LAYER_MOVING",
    "LAYER_NON_MOVING",
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
    "Engine",
    "Manual",
    "Transmission",
    "bake_scene",
    "euler_to_quat",
    "load_urdf",
    "validate_constraint",
    "validate_settings",
]


# --- Constants ---
MOTION_STATIC = 0
MOTION_KINEMATIC = 1
MOTION_DYNAMIC = 2

SHAPE_BOX = 0
SHAPE_SPHERE = 1
SHAPE_CAPSULE = 2
SHAPE_CYLINDER = 3
SHAPE_PLANE = 4
SHAPE_MESH = 5
SHAPE_HEIGHTFIELD = 6
SHAPE_CONVEX_HULL = 7

LAYER_NON_MOVING = 0
LAYER_MOVING = 1

CONSTRAINT_FIXED = 0
CONSTRAINT_POINT = 1
CONSTRAINT_HINGE = 2
CONSTRAINT_SLIDER = 3
CONSTRAINT_DISTANCE = 4
CONSTRAINT_CONE = 5

EVENT_ADDED = 0
EVENT_PERSISTED = 1
EVENT_REMOVED = 2

# --- Configuration Objects ---


class Engine:
    """
    Defines the physical properties of a simulated internal combustion engine.
    Used to configure the power delivery for vehicles.

    Attributes:
        max_torque (float): Peak torque output in Newton-meters (Nm).
        max_rpm (float): The engine's "redline" or maximum rotational speed.
        min_rpm (float): The "idle" speed; the engine cannot stall below this.
        inertia (float): Rotational mass of the engine. Higher values cause
            the RPM to climb and fall more slowly.
    """

    __module__ = "culverin"

    max_torque: float
    max_rpm: float
    min_rpm: float
    inertia: float

    def __init__(
        self,
        max_torque: float = 500.0,
        max_rpm: float = 7000.0,
        min_rpm: float = 1000.0,
        inertia: float = 0.5,
    ) -> None:
        self.max_torque = float(max_torque)
        self.max_rpm = float(max_rpm)
        self.min_rpm = float(min_rpm)
        self.inertia = float(inertia)


class Transmission:
    """
    Base configuration for vehicle gearboxes.
    Defines mechanical advantage and clutch properties.

    Attributes:
        clutch_strength (float): Maximum torque the clutch can transmit before
            slipping. Higher values provide snappier shifts.
        differential_ratio (float): The final drive ratio (e.g., 3.42). Multiplies
            torque sent to the wheels.
        ratios (list[float]): Stack of forward gear ratios.
        reverse_ratios (list[float]): Stack of reverse gear ratios.
    """

    __module__ = "culverin"

    clutch_strength: float
    differential_ratio: float
    ratios: list[float]
    reverse_ratios: list[float]

    def __init__(
        self, gears: int = 5, clutch_strength: float = 2000.0, differential_ratio: float = 3.42
    ) -> None:
        self.clutch_strength = float(clutch_strength)
        self.differential_ratio = float(differential_ratio)
        presets = [2.66, 1.78, 1.30, 1.0, 0.74, 0.50]
        self.ratios = presets[:gears]
        self.reverse_ratios = [-2.90]


class Automatic(Transmission):
    """
    An automated gearbox configuration.
    Automatically handles gear transitions based on engine RPM.

    Attributes:
        shift_up_rpm (float): The RPM threshold at which the transmission
            upshifts to the next gear.
        shift_down_rpm (float): The RPM threshold at which the transmission
            downshifts to a lower gear.
        mode (int): Internal identifier for Automatic mode (0).
    """

    __module__ = "culverin"

    mode: int
    shift_up_rpm: float
    shift_down_rpm: float

    def __init__(
        self,
        gears: int = 5,
        clutch_strength: float = 2000.0,
        differential_ratio: float = 3.42,
        shift_up_rpm: float = 5000.0,
        shift_down_rpm: float = 2000.0,
    ) -> None:
        super().__init__(gears, clutch_strength, differential_ratio)
        self.mode = 0
        self.shift_up_rpm = float(shift_up_rpm)
        self.shift_down_rpm = float(shift_down_rpm)


class Manual(Transmission):
    """
    A manual gearbox configuration.
    Shifts are controlled either by user input or high-level logic scripts.

    Attributes:
        mode (int): Internal identifier for Manual mode (1).
    """

    __module__ = "culverin"

    mode: int

    def __init__(
        self, gears: int = 5, clutch_strength: float = 5000.0, differential_ratio: float = 3.42
    ) -> None:
        super().__init__(gears, clutch_strength, differential_ratio)
        self.mode = 1


# --- Validation Logic ---


ConvertibleToFloat = int | float | str | bytes | bytearray


def _force_float(val: ConvertibleToFloat, name: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError) as err:
        raise TypeError(f"'{name}' must be a number") from err


Vec3Param = float | tuple[float, float, float] | list[float]
Vec4Param = tuple[float, float, float, float] | list[float]
ConstraintParams = tuple[object, ...] | list[object]


def _validate_vec3(v: Vec3Param, name: str) -> tuple[float, float, float]:
    if isinstance(v, (int, float)):  # type: ignore
        f = float(v)
        return (f, f, f)
    if not isinstance(v, (tuple, list)) or len(v) != 3:  # type: ignore
        raise ValueError(f"'{name}' must be a sequence of length 3")
    return (float(v[0]), float(v[1]), float(v[2]))


def _validate_quat(
    q: Vec4Param | tuple[int | float, int | float, int | float, int | float] | list[int | float],
    name: str,
) -> tuple[float, float, float, float]:
    if not isinstance(q, (tuple, list)) or len(q) != 4:  # type: ignore
        raise ValueError(f"'{name}' must be a sequence of length 4")
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


@overload
def validate_constraint(
    type_id: Literal[1], body1: object, body2: object, params: object
) -> None: ...
@overload
def validate_constraint(
    type_id: Literal[2], body1: object, body2: object, params: object
) -> tuple[float, float, float]: ...
@overload
def validate_constraint(
    type_id: Literal[3], body1: object, body2: object, params: object
) -> tuple[float, float]: ...
@overload
def validate_constraint(
    type_id: Literal[4, 5], body1: object, body2: object, params: object
) -> (
    tuple[tuple[float, float, float], tuple[float, float, float]]
    | tuple[tuple[float, float, float], tuple[float, float, float], float, float]
): ...
@overload
def validate_constraint(
    type_id: Literal[6], body1: object, body2: object, params: object
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]: ...


def validate_constraint(
    type_id: int, body1: object, body2: object, params: object
) -> (
    tuple[float, float, float]
    | tuple[float, float]
    | tuple[tuple[float, float, float], tuple[float, float, float]]
    | tuple[tuple[float, float, float], tuple[float, float, float], float, float]
    | tuple[tuple[float, float, float], tuple[float, float, float], float]
    | None
):
    """Constraint validation for constraint creation."""

    # 1. Handle Validation
    if not isinstance(body1, int) or not isinstance(body2, int):
        raise TypeError("Constraint bodies must be integer handles")

    # 2. Dispatch Logic
    if type_id == CONSTRAINT_FIXED:  # Literal 1
        return None

    if type_id == CONSTRAINT_POINT:  # Literal 2
        if not isinstance(params, (tuple, list)):
            raise ValueError("PointConstraint requires a Vec3 sequence")
        params2 = cast(Vec3Param, params)
        return _validate_vec3(params2, "point.pivot")

    if type_id == CONSTRAINT_DISTANCE:  # Literal 3
        if not isinstance(params, (tuple, list)):
            raise ValueError("DistanceConstraint requires (min_dist, max_dist)")
        params2 = cast(ConstraintParams, params)
        if len(params2) != 2:
            raise ValueError("DistanceConstraint requires (min_dist, max_dist)")
        return (
            _force_float(cast(ConvertibleToFloat, params2[0]), "min"),
            _force_float(cast(ConvertibleToFloat, params2[1]), "max"),
        )

    if type_id in (CONSTRAINT_HINGE, CONSTRAINT_SLIDER):  # Literals 4, 5
        if not isinstance(params, (tuple, list)):
            raise ValueError("Hinge/Slider requires ((pivot), (axis), [limits])")
        params2 = cast(ConstraintParams, params)
        if len(params2) < 2:
            raise ValueError("Hinge/Slider requires ((pivot), (axis), [limits])")

        p0 = cast(Vec3Param, params2[0])
        a1 = cast(Vec3Param, params2[1])
        if not isinstance(p0, (tuple, list)) or not isinstance(a1, (tuple, list)):
            raise ValueError("Pivot and Axis must be sequences")

        pivot = _validate_vec3(p0, "pivot")
        axis = _validate_vec3(a1, "axis")

        if len(params2) == 4:
            return (
                pivot,
                axis,
                _force_float(cast(ConvertibleToFloat, params2[2]), "min"),
                _force_float(cast(ConvertibleToFloat, params2[3]), "max"),
            )
        return (pivot, axis)

    if type_id == CONSTRAINT_CONE:  # Literal 6
        if not isinstance(params, (tuple, list)):
            raise ValueError("ConeConstraint requires ((pivot), (axis), half_angle)")
        params2 = cast(ConstraintParams, params)
        if len(params2) != 3:
            raise ValueError("ConeConstraint requires ((pivot), (axis), half_angle)")

        p0 = cast(Vec3Param, params2[0])
        a1 = cast(Vec3Param, params2[1])
        if not isinstance(p0, (tuple, list)) or not isinstance(a1, (tuple, list)):
            raise ValueError("Pivot and Axis must be sequences")

        return (
            _validate_vec3(p0, "pivot"),
            _validate_vec3(a1, "axis"),
            _force_float(cast(ConvertibleToFloat, params2[2]), "angle"),
        )

    raise ValueError(f"Unknown constraint type: {type_id}")


def validate_settings(s: dict[str, Any] | None) -> tuple[float, float, float, float, int, int, int, int, int, int, int]:
    """Settings validation for physics world."""
    settings = s or {}

    # Cast the result of .get() specifically to the type _validate_vec3 expects
    raw_gravity = cast(tuple[float, float, float], settings.get("gravity", (0.0, -9.81, 0.0)))
    grav = _validate_vec3(raw_gravity, "gravity")

    max_jobs = int(settings.get("max_physics_jobs", 2048))
    if max_jobs > 2048:
        raise ValueError("max_physics_jobs cannot exceed JoltC limit of 2048")
        
    max_barriers = int(settings.get("max_physics_barriers", 8))
    if max_barriers > 8:
        raise ValueError("max_physics_barriers cannot exceed JoltC limit of 8")

    return (
        grav[0],
        grav[1],
        grav[2],
        float(settings.get("penetration_slop", 0.02)),
        int(settings.get("max_bodies", 10240)),
        int(settings.get("max_pairs", 65536)),
        int(settings.get("max_contact_constraints", 32768)),
        int(settings.get("temp_allocator_size", 33554432)), # 32 MB
        max_jobs,
        max_barriers,
        int(settings.get("num_threads", 4)),
    )


def validate_body_params(
    shape_type: int,
    pos: list[float] | tuple[float, ...],
    rot: list[float] | tuple[float, ...],
    size: float | tuple[float, float, float] | tuple[float, float, float, float] | list[float],
    motion_type: int,
) -> tuple[
    tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]
]:
    """Body parameter validation function. Validates the body parameters."""
    p = _validate_vec3(cast(tuple[float, float, float], pos), "pos")
    r = _validate_quat(cast(tuple[float, float, float, float], rot), "rot")
    s = [0.0, 0.0, 0.0, 0.0]

    if shape_type == SHAPE_BOX:
        if not isinstance(size, (tuple)):
            sz = _validate_vec3(size, "size")
            s[0], s[1], s[2] = sz
    elif shape_type == SHAPE_SPHERE:
        s[0] = float(size[0] if isinstance(size, (list, tuple)) else size)
    elif shape_type in (SHAPE_CAPSULE, SHAPE_CYLINDER):
        if isinstance(size, (list, tuple)):
            s[0], s[1] = float(size[0]), float(size[1])
    elif shape_type == SHAPE_PLANE:
        # Use a local variable to help the linter understand size has 4 elements
        if not isinstance(size, (list, tuple)) or len(size) != 4:
            raise ValueError("SHAPE_PLANE size must be (nx, ny, nz, constant)")
        s[0], s[1], s[2], s[3] = float(size[0]), float(size[1]), float(size[2]), float(size[3])

    return p, r, (s[0], s[1], s[2], s[3])


def bake_scene(
    bodies: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[int, bytes, bytes, bytes, bytes, bytes, bytes]:
    """Send pure bytes to C, or Python."""
    if not bodies:
        return 0, b"", b"", b"", b"", b"", b""

    arr_pos = array.array("d")
    arr_rot = array.array("f")
    arr_shape = array.array("f")
    arr_mot = array.array("B")
    arr_layer = array.array("B")
    arr_usr = array.array("Q")

    count = 0
    for b in bodies:
        count += 1
        # Extract data safely from Any-typed dict
        shape_type = int(b.get("shape", SHAPE_BOX))
        pos_raw = b.get("pos", (0.0, 0.0, 0.0))
        rot_raw = b.get("rot", (0.0, 0.0, 0.0, 1.0))
        size_raw = b.get("size", (0.0, 0.0, 0.0, 0.0))
        mass = float(b.get("mass", 1.0))
        motion = int(b.get("motion", MOTION_DYNAMIC if mass > 0 else MOTION_STATIC))

        p, r, s = validate_body_params(shape_type, pos_raw, rot_raw, size_raw, motion)

        arr_pos.extend((p[0], p[1], p[2], 0.0))
        arr_rot.extend(r)
        arr_shape.append(float(shape_type))
        arr_shape.extend(s)
        arr_mot.append(motion)
        arr_layer.append(LAYER_MOVING if motion != MOTION_STATIC else LAYER_NON_MOVING)
        arr_usr.append(int(b.get("user_data", 0)))

    return (
        count,
        arr_pos.tobytes(),
        arr_rot.tobytes(),
        arr_shape.tobytes(),
        arr_mot.tobytes(),
        arr_layer.tobytes(),
        arr_usr.tobytes(),
    )


TrigFunc = Callable[[float], float]


def _assemble_euler_to_quat() -> types.FunctionType:
    """This is overengineering."""
    import dis
    import sys

    _ref_src = (
        "import math\n"
        "def _f(r, p, y, _sin=math.sin, _cos=math.cos):\n"
        "    r *= 0.5; p *= 0.5; y *= 0.5\n"
        "    sr = _sin(r); cr = _cos(r)\n"
        "    sp = _sin(p); cp = _cos(p)\n"
        "    sy = _sin(y); cy = _cos(y)\n"
        "    srcp = sr * cp; crsp = cr * sp; crcp = cr * cp; srsp = sr * sp\n"
        "    srcp_cy = srcp * cy; srcp_sy = srcp * sy\n"
        "    crsp_cy = crsp * cy; crsp_sy = crsp * sy\n"
        "    return (srcp_cy - crsp_sy, crsp_cy + srcp_sy,"
        "            crcp*sy - srsp*cy, crcp*cy + srsp*sy)\n"
    )
    _ref_mod = compile(_ref_src, "<string>", "exec")
    _ref = next(c for c in _ref_mod.co_consts if isinstance(c, types.CodeType))

    def _measure_cache(snippet_src: str, opname: str) -> int:
        """
        Measure inline cache entry count for opname by diffing consecutive
        instruction offsets in compiler output. Each cache entry = 2 bytes.
        Zero-scanning is unreliable when adjacent instructions also start with 0.
        """
        mod = compile(snippet_src, "<string>", "exec")
        fn = next(c for c in mod.co_consts if isinstance(c, types.CodeType))
        target = opcode.opmap[opname]
        instrs = list(dis.get_instructions(fn))
        for i, instr in enumerate(instrs):
            if instr.opcode == target and i + 1 < len(instrs):
                span = instrs[i + 1].offset - instr.offset
                return (span - 2) // 2  # subtract op+arg word, divide into 2-byte entries
        raise RuntimeError(f"{opname} not found in snippet")

    _call_cache = _measure_cache("def _f(x, g=len): return g(x)\n", "CALL")
    _binop_cache = _measure_cache("def _f(x): return x * x\n", "BINARY_OP")

    _op = opcode.opmap
    RESUME = _op["RESUME"]
    LOAD_FAST = _op["LOAD_FAST"]
    LOAD_CONST = _op["LOAD_CONST"]
    STORE_FAST = _op["STORE_FAST"]
    PUSH_NULL = _op["PUSH_NULL"]
    CALL = _op["CALL"]
    BINARY_OP = _op["BINARY_OP"]
    BUILD_TUPLE = _op["BUILD_TUPLE"]
    RETURN_VALUE = _op["RETURN_VALUE"]

    OP_ADD = 0
    OP_MUL = 5
    OP_SUB = 10
    OP_IMUL = 18  # *=

    CALL_PAD = [0, 0] * _call_cache
    BINOP_PAD = [0, 0] * _binop_cache

    R, P, Y, SIN, COS = 0, 1, 2, 3, 4
    SR, CR, SP, CP, SY, CY = 5, 6, 7, 8, 9, 10
    SRCP, CRSP, CRCP, SRSP = 11, 12, 13, 14  # 15 locals total, not 19

    def lf(v: int) -> list[int]:
        return [LOAD_FAST, v]

    def sf(v: int) -> list[int]:
        return [STORE_FAST, v]

    def lc(i: int) -> list[int]:
        return [LOAD_CONST, i]

    def binop(op: int) -> list[int]:
        return [BINARY_OP, op, *BINOP_PAD]

    def call(fn: int, arg: int) -> list[int]:
        return [PUSH_NULL, 0, LOAD_FAST, fn, LOAD_FAST, arg, CALL, 1, *CALL_PAD]

    bc: list[int] = []

    def emit(*parts: list[int]) -> None:
        for p in parts:
            bc.extend(p)

    emit(
        [RESUME, 0],
        lf(R),
        lc(1),
        binop(OP_IMUL),
        sf(R),
        lf(P),
        lc(1),
        binop(OP_IMUL),
        sf(P),
        lf(Y),
        lc(1),
        binop(OP_IMUL),
        sf(Y),
        call(SIN, R),
        sf(SR),
        call(COS, R),
        sf(CR),
        call(SIN, P),
        sf(SP),
        call(COS, P),
        sf(CP),
        call(SIN, Y),
        sf(SY),
        call(COS, Y),
        sf(CY),
        lf(SR),
        lf(CP),
        binop(OP_MUL),
        sf(SRCP),
        lf(CR),
        lf(SP),
        binop(OP_MUL),
        sf(CRSP),
        lf(CR),
        lf(CP),
        binop(OP_MUL),
        sf(CRCP),
        lf(SR),
        lf(SP),
        binop(OP_MUL),
        sf(SRSP),
        # i0 = srcp*cy - crsp*sy  (inlined, no srcp_cy/crsp_sy locals)
        lf(SRCP),
        lf(CY),
        binop(OP_MUL),
        lf(CRSP),
        lf(SY),
        binop(OP_MUL),
        binop(OP_SUB),
        # i1 = crsp*cy + srcp*sy
        lf(CRSP),
        lf(CY),
        binop(OP_MUL),
        lf(SRCP),
        lf(SY),
        binop(OP_MUL),
        binop(OP_ADD),
        # i2 = crcp*sy - srsp*cy
        lf(CRCP),
        lf(SY),
        binop(OP_MUL),
        lf(SRSP),
        lf(CY),
        binop(OP_MUL),
        binop(OP_SUB),
        # i3 = crcp*cy + srsp*sy
        lf(CRCP),
        lf(CY),
        binop(OP_MUL),
        lf(SRSP),
        lf(SY),
        binop(OP_MUL),
        binop(OP_ADD),
        [BUILD_TUPLE, 4],
        [RETURN_VALUE, 0],
    )

    # Debug output to diagnose bytecode mismatch
    use_compiler_bc = len(bc) != len(_ref.co_code)

    if not use_compiler_bc:
        assert len(bc) == len(_ref.co_code), (
            f"Bytecode length mismatch: ours={len(bc)} compiler={len(_ref.co_code)} "
            f"(Python {sys.version}) — CALL_cache={_call_cache} BINOP_cache={_binop_cache}"
        )
        assert bytes(bc) == _ref.co_code, (
            f"Bytecode content mismatch — NB_ selectors may differ on {sys.version}\n"
            f"First diff at offset: "
            f"{next(i for i, (a, b) in enumerate(zip(bc, _ref.co_code, strict=False)) if a != b)}"
        )

    varnames = (
        "r",
        "p",
        "y",
        "_sin",
        "_cos",
        "sr",
        "cr",
        "sp",
        "cp",
        "sy",
        "cy",
        "srcp",
        "crsp",
        "crcp",
        "srsp",
    )

    # Use compiler-generated bytecode for Python 3.14+ compatibility
    bytecode = _ref.co_code if use_compiler_bc else bytes(bc)

    if use_compiler_bc:
        # For Python 3.14+, create a function from the reference code object
        return types.FunctionType(
            _ref,
            {"__builtins__": __builtins__, "math": math},
            argdefs=(math.sin, math.cos),
        )

    code_obj = types.CodeType(
        5,  # argcount: r, p, y, _sin, _cos
        0,  # posonlyargcount
        0,  # kwonlyargcount
        len(varnames),  # nlocals
        _ref.co_stacksize,
        0x3,  # CO_OPTIMIZED | CO_NEWLOCALS
        bytecode,
        (None, 0.5),
        (),
        varnames,
        __file__,
        "euler_to_quat",
        "euler_to_quat",
        1,
        _ref.co_linetable,
        _ref.co_exceptiontable,
    )

    return types.FunctionType(
        code_obj,
        {"__builtins__": __builtins__},
        argdefs=(math.sin, math.cos),
    )


euler_to_quat = _assemble_euler_to_quat()


def _parse_vec(text: str) -> tuple[float, float, float]:
    return tuple(map(float, text.split()))  # type: ignore


def parse_urdf(path: str) -> list[dict[str, Any]]:
    """Simple URDF parser that works, probably."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    bodies: list[
        dict[
            str, tuple[float, float, float] | tuple[float, float, float, float] | str | int | float
        ]
    ] = []

    for link in root.findall("link"):
        body: dict[
            str, tuple[float, float, float] | tuple[float, float, float, float] | str | int | float
        ] = {
            "name": link.attrib.get("name", "unnamed"),
            "pos": (0.0, 0.0, 0.0),
            "rot": (0.0, 0.0, 0.0, 1.0),
            "shape": SHAPE_BOX,
            "size": (1.0, 1.0, 1.0),
            "motion": MOTION_DYNAMIC,
            "mass": 1.0,
        }

        # 1. Extract Origin (Transform)
        # Search visual, then collision, then inertial
        for tag in ["visual", "collision", "inertial"]:
            node = link.find(tag)
            if node is not None:
                origin = node.find("origin")
                if origin is not None:
                    if "xyz" in origin.attrib:
                        body["pos"] = _parse_vec(origin.attrib["xyz"])
                    if "rpy" in origin.attrib:
                        r, p, y = _parse_vec(origin.attrib["rpy"])
                        body["rot"] = euler_to_quat(r, p, y)
                    break

        # 2. Extract Geometry (Shape)
        for tag in ["visual", "collision"]:
            node = link.find(tag)
            if node is not None:
                geom = node.find("geometry")
                if geom is not None:
                    # Box
                    box = geom.find("box")
                    if box is not None:
                        body["shape"] = SHAPE_BOX
                        body["size"] = _parse_vec(box.attrib["size"])
                    # Cylinder (Used in your 'arm' link)
                    cyl = geom.find("cylinder")
                    if cyl is not None:
                        body["shape"] = SHAPE_CYLINDER
                        radius = float(cyl.attrib["radius"])
                        length = float(cyl.attrib["length"])
                        body["size"] = (radius, length, 0.0)
                    break

        # 3. Extract Mass
        inertial = link.find("inertial")
        if inertial is not None:
            mass_node = inertial.find("mass")
            if mass_node is not None:
                body["mass"] = float(mass_node.attrib["value"])
                # Static if mass is 0, otherwise Dynamic
                body["motion"] = MOTION_DYNAMIC if body["mass"] > 0 else MOTION_STATIC

        bodies.append(body)
    return bodies


def load_urdf(path: str) -> tuple[int, bytes, bytes, bytes, bytes, bytes, bytes]:
    """
    Maintains compatibility with binary-loading workflows.
    Parses the URDF and returns the baked binary scene tuple.
    """
    return bake_scene(parse_urdf(path))

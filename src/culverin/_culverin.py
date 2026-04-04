import array
from typing import Any, TypedDict, cast, Callable
import math
import xml.etree.ElementTree as ET

__all__ = [
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", 
    "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "LAYER_NON_MOVING", "LAYER_MOVING",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", "CONSTRAINT_SLIDER", 
    "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE", 
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED",
    "Engine", "Transmission", "Automatic", "Manual",
    "WheelConfig", "TrackConfig",
    "validate_constraint", "validate_settings", "bake_scene", "load_urdf", "euler_to_quat"
]

class WheelConfig(TypedDict):
    pos: tuple[float, float, float]
    radius: float

class TrackConfig(TypedDict):
    indices: list[int]
    driven_wheel: int

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
    max_torque: float
    max_rpm: float
    min_rpm: float
    inertia: float

    def __init__(self, max_torque: float = 500.0, max_rpm: float = 7000.0, min_rpm: float = 1000.0, inertia: float = 0.5):
        self.max_torque = float(max_torque)
        self.max_rpm = float(max_rpm)
        self.min_rpm = float(min_rpm)
        self.inertia = float(inertia)

class Transmission:
    clutch_strength: float
    differential_ratio: float
    ratios: list[float]
    reverse_ratios: list[float]

    def __init__(self, gears: int = 5, clutch_strength: float = 2000.0, differential_ratio: float = 3.42):
        self.clutch_strength = float(clutch_strength)
        self.differential_ratio = float(differential_ratio)
        presets = [2.66, 1.78, 1.30, 1.0, 0.74, 0.50]
        self.ratios = presets[:gears]
        self.reverse_ratios = [-2.90]

class Automatic(Transmission):
    mode: int
    shift_up_rpm: float
    shift_down_rpm: float

    def __init__(self, gears: int = 5, clutch_strength: float = 2000.0, differential_ratio: float = 3.42, 
                 shift_up_rpm: float = 5000.0, shift_down_rpm: float = 2000.0):
        super().__init__(gears, clutch_strength, differential_ratio)
        self.mode = 0
        self.shift_up_rpm = float(shift_up_rpm)
        self.shift_down_rpm = float(shift_down_rpm)

class Manual(Transmission):
    mode: int

    def __init__(self, gears: int = 5, clutch_strength: float = 5000.0, differential_ratio: float = 3.42):
        super().__init__(gears, clutch_strength, differential_ratio)
        self.mode = 1

# --- Validation Logic ---

def _force_float(val: Any, name: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        raise TypeError(f"'{name}' must be a number")

def _validate_vec3(v: float | int | tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if isinstance(v, (int, float)): # type: ignore
        f = float(v)
        return (f, f, f)
    if not isinstance(v, (tuple, list)) or len(v) != 3: # type: ignore
        raise ValueError(f"'{name}' must be a sequence of length 3")
    return (float(v[0]), float(v[1]), float(v[2]))

def _validate_quat(q: tuple[int | float] | list[int | float] | tuple[float, float, float, float], name: str) -> tuple[float, float, float, float]:
    if not isinstance(q, (tuple, list)) or len(q) != 4: # type: ignore
        raise ValueError(f"'{name}' must be a sequence of length 4")
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

def validate_constraint(type_id: int, body1: Any, body2: Any, params: int | float | tuple[int | float]) -> Any:
    # Use Any for body1/body2 so linter doesn't complain about "redundant" isinstance checks
    if not isinstance(body1, int) or not isinstance(body2, int):
        raise TypeError("Constraint bodies must be integer handles")
    
    if type_id == CONSTRAINT_FIXED:
        return None

    if type_id == CONSTRAINT_POINT:
        if not isinstance(params, (tuple, list)):
            return _validate_vec3(params, "point.pivot")

    if type_id == CONSTRAINT_DISTANCE:
        if not isinstance(params, (tuple, list)) or len(params) != 2:
            raise ValueError("DistanceConstraint requires (min_dist, max_dist)")
        return (_force_float(params[0], "min"), _force_float(params[1], "max"))

    if type_id in (CONSTRAINT_HINGE, CONSTRAINT_SLIDER):
        if not isinstance(params, (tuple, list)) or len(params) < 2:
            raise ValueError("Hinge/Slider requires ((pivot), (axis), [limits])")
        pivot = _validate_vec3(params[0], "pivot")
        axis = _validate_vec3(params[1], "axis")
        if len(params) == 4:
            return (pivot, axis, _force_float(params[2], "min"), _force_float(params[3], "max"))
        return (pivot, axis)

    if type_id == CONSTRAINT_CONE:
        if not isinstance(params, (tuple, list)) or len(params) != 3:
            raise ValueError("ConeConstraint requires ((pivot), (axis), half_angle)")
        return (_validate_vec3(params[0], "pivot"), _validate_vec3(params[1], "axis"), _force_float(params[2], "angle"))

    raise ValueError(f"Unknown constraint type: {type_id}")

def validate_settings(s: dict[str, Any] | None) -> tuple[float, float, float, float, int, int]:
    settings = s or {}
    
    # Cast the result of .get() specifically to the type _validate_vec3 expects
    raw_gravity = cast(tuple[float, float, float], settings.get("gravity", (0.0, -9.81, 0.0)))
    grav = _validate_vec3(raw_gravity, "gravity")
    
    return (
        grav[0], grav[1], grav[2], 
        float(settings.get("penetration_slop", 0.02)), 
        int(settings.get("max_bodies", 10240)), 
        int(settings.get("max_pairs", 65536))
    )

def validate_body_params(
    shape_type: int, 
    pos: list[float] | tuple[float, ...], 
    rot: list[float] | tuple[float, ...], 
    size: float | int | tuple[float, float, float] | tuple[float, float, float, float], 
    motion_type: int
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]:
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

def bake_scene(bodies: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[int, bytes, bytes, bytes, bytes, bytes, bytes]:
    if not bodies: return 0, b"", b"", b"", b"", b"", b""
    
    arr_pos = array.array('d')
    arr_rot = array.array('f')
    arr_shape = array.array('f')
    arr_mot = array.array('B')
    arr_layer = array.array('B')
    arr_usr = array.array('Q')
    
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
        
    return count, arr_pos.tobytes(), arr_rot.tobytes(), arr_shape.tobytes(), arr_mot.tobytes(), arr_layer.tobytes(), arr_usr.tobytes()

TrigFunc = Callable[[float], float]

import types
import opcode

def _assemble_euler_to_quat() -> types.FunctionType:
    # Opcodes
    RESUME       = opcode.opmap['RESUME']
    LOAD_FAST    = opcode.opmap['LOAD_FAST']
    LOAD_CONST   = opcode.opmap['LOAD_CONST']
    STORE_FAST   = opcode.opmap['STORE_FAST']
    PUSH_NULL    = opcode.opmap['PUSH_NULL']
    CALL         = opcode.opmap['CALL']
    BINARY_OP    = opcode.opmap['BINARY_OP']
    BUILD_TUPLE  = opcode.opmap['BUILD_TUPLE']
    RETURN_VALUE = opcode.opmap['RETURN_VALUE']

    # BINARY_OP selectors
    OP_ADD  = 0
    OP_MUL  = 5
    OP_SUB  = 10
    OP_IMUL = 18  # *=

    # Inline cache padding (3.12+): CALL needs 3 entries, BINARY_OP needs 1
    # Each entry = 2 zero bytes
    CALL_CACHE  = [0, 0, 0, 0, 0, 0]
    BINOP_CACHE = [0, 0]

    # varname indices
    R, P, Y, SIN, COS = 0, 1, 2, 3, 4
    SR, CR, SP, CP, SY, CY = 5, 6, 7, 8, 9, 10
    SRCP, CRSP, CRCP, SRSP = 11, 12, 13, 14
    SRCP_CY, SRCP_SY, CRSP_CY, CRSP_SY = 15, 16, 17, 18

    def lf(v):    return [LOAD_FAST,  v]
    def sf(v):    return [STORE_FAST, v]
    def lc(i):    return [LOAD_CONST, i]
    def binop(op):return [BINARY_OP,  op] + BINOP_CACHE
    def call(fn, arg):
        return [PUSH_NULL, 0, LOAD_FAST, fn, LOAD_FAST, arg, CALL, 1] + CALL_CACHE

    bc = []
    def emit(*parts):
        for p in parts: bc.extend(p)

    emit(
        [RESUME, 0],

        # r *= 0.5; p *= 0.5; y *= 0.5
        lf(R), lc(1), binop(OP_IMUL), sf(R),
        lf(P), lc(1), binop(OP_IMUL), sf(P),
        lf(Y), lc(1), binop(OP_IMUL), sf(Y),

        # sr,cr,sp,cp,sy,cy
        call(SIN, R), sf(SR),
        call(COS, R), sf(CR),
        call(SIN, P), sf(SP),
        call(COS, P), sf(CP),
        call(SIN, Y), sf(SY),
        call(COS, Y), sf(CY),

        # intermediate products
        lf(SR), lf(CP), binop(OP_MUL), sf(SRCP),
        lf(CR), lf(SP), binop(OP_MUL), sf(CRSP),
        lf(CR), lf(CP), binop(OP_MUL), sf(CRCP),
        lf(SR), lf(SP), binop(OP_MUL), sf(SRSP),

        lf(SRCP), lf(CY), binop(OP_MUL), sf(SRCP_CY),
        lf(SRCP), lf(SY), binop(OP_MUL), sf(SRCP_SY),
        lf(CRSP), lf(CY), binop(OP_MUL), sf(CRSP_CY),
        lf(CRSP), lf(SY), binop(OP_MUL), sf(CRSP_SY),

        # x, y, z, w — left on stack for BUILD_TUPLE
        lf(SRCP_CY), lf(CRSP_SY), binop(OP_SUB),
        lf(CRSP_CY), lf(SRCP_SY), binop(OP_ADD),
        lf(CRCP), lf(SY), binop(OP_MUL),
        lf(SRSP), lf(CY), binop(OP_MUL), binop(OP_SUB),
        lf(CRCP), lf(CY), binop(OP_MUL),
        lf(SRSP), lf(SY), binop(OP_MUL), binop(OP_ADD),

        [BUILD_TUPLE, 4],
        [RETURN_VALUE, 0],
    )

    varnames = (
        'r', 'p', 'y', '_sin', '_cos',
        'sr', 'cr', 'sp', 'cp', 'sy', 'cy',
        'srcp', 'crsp', 'crcp', 'srsp',
        'srcp_cy', 'srcp_sy', 'crsp_cy', 'crsp_sy',
    )

    # Steal linetable + exceptiontable from compiler output of
    # structurally identical source — avoids hand-encoding the 3.12
    # location table format (which is non-trivial and version-specific)
    _ref_src = '''
import math
def _f(r, p, y, _sin=math.sin, _cos=math.cos):
    r *= 0.5; p *= 0.5; y *= 0.5
    sr = _sin(r); cr = _cos(r)
    sp = _sin(p); cp = _cos(p)
    sy = _sin(y); cy = _cos(y)
    srcp = sr * cp; crsp = cr * sp; crcp = cr * cp; srsp = sr * sp
    srcp_cy = srcp * cy; srcp_sy = srcp * sy
    crsp_cy = crsp * cy; crsp_sy = crsp * sy
    return (srcp_cy - crsp_sy, crsp_cy + srcp_sy, crcp*sy - srsp*cy, crcp*cy + srsp*sy)
'''
    _ref_mod = compile(_ref_src, '<string>', 'exec')
    _ref = next(c for c in _ref_mod.co_consts if isinstance(c, types.CodeType))

    code_obj = types.CodeType(
        5,                    # argcount: r, p, y, _sin, _cos
        0,                    # posonlyargcount
        0,                    # kwonlyargcount
        len(varnames),        # nlocals
        _ref.co_stacksize,    # stacksize — verified by compiler
        0x3,                  # CO_OPTIMIZED | CO_NEWLOCALS
        bytes(bc),
        (None, 0.5),          # consts[0]=None (implicit), consts[1]=0.5
        (),                   # names (no globals)
        varnames,
        __file__,
        'euler_to_quat',
        'euler_to_quat',
        1,
        _ref.co_linetable,
        _ref.co_exceptiontable,
    )

    return types.FunctionType(
        code_obj,
        {'__builtins__': __builtins__},
        argdefs=(math.sin, math.cos),  # defaults for _sin, _cos
    )

euler_to_quat = _assemble_euler_to_quat()

def _parse_vec(text: str) -> tuple[float, float, float]:
    return tuple(map(float, text.split())) # type: ignore

def load_urdf(path: str):
    """
    Parses a URDF file and returns the baked scene tuple 
    ready to be loaded by PhysicsWorld_init.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    bodies: list[dict[str, tuple[float, float, float] | tuple[float, float, float, float] | str | int | float]] = []

    for link in root.findall('link'):
        body: dict[str, tuple[float, float, float] | tuple[float, float, float, float] | str | int | float] = {
            'pos': (0.0, 0.0, 0.0),
            'rot': (0.0, 0.0, 0.0, 1.0),
            'shape': 'box',
            'size': (1.0, 1.0, 1.0),
            'motion': MOTION_DYNAMIC,
            'mass': 1.0
        }

        # Visual geometry
        visual = link.find('visual')
        if visual is not None:
            origin = visual.find('origin')
            if origin is not None:
                if 'xyz' in origin.attrib:
                    body['pos'] = _parse_vec(origin.attrib['xyz'])
                if 'rpy' in origin.attrib:
                    r, p, y = _parse_vec(origin.attrib['rpy'])
                    body['rot'] = euler_to_quat(r, p, y)
            
            geom = visual.find('geometry')
            if geom is not None:
                # 1. Box Logic
                box_node = geom.find('box')
                if box_node is not None:
                    body['shape'] = SHAPE_BOX
                    body['size'] = _parse_vec(box_node.attrib['size'])
                
                # 2. Sphere Logic
                sphere_node = geom.find('sphere')
                if sphere_node is not None:
                    body['shape'] = SHAPE_SPHERE
                    # Use a float directly, or a 3-tuple to satisfy the checker
                    radius = float(sphere_node.attrib['radius'])
                    body['size'] = (radius, 0.0, 0.0) 

                # 3. Capsule/Cylinder Logic
                capsule_node = geom.find('capsule') or geom.find('cylinder')
                if capsule_node is not None:
                    body['shape'] = SHAPE_CAPSULE if capsule_node.tag == 'capsule' else SHAPE_CYLINDER
                    r = float(capsule_node.attrib['radius'])
                    l = float(capsule_node.attrib['length'])
                    body['size'] = (r, l, 0.0)
        
        # Inertial mass
        inertial = link.find('inertial')
        if inertial is not None:
            mass_node = inertial.find('mass')
            if mass_node is not None:
                body['mass'] = float(mass_node.attrib['value'])
        
        bodies.append(body)

    return bake_scene(bodies)
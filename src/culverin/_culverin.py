import array
from typing import TypedDict

__all__ = [
    # Constants
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", 
    "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "LAYER_NON_MOVING", "LAYER_MOVING",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", "CONSTRAINT_SLIDER", 
    "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE", 
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED",
    
    # Config Classes
    "Engine", "Transmission", "Automatic", "Manual",
    "WheelConfig", "TrackConfig",
    
    # Internal Helpers
    "validate_constraint", "validate_settings", "bake_scene"
]

class WheelConfig(TypedDict):
    pos: tuple[float, float, float]
    radius: float

class TrackConfig(TypedDict):
    indices: list[int]      # The indices of the wheels this track wraps
    driven_wheel: int       # The index of the wheel providing torque

# --- Constants (Mirrored from C for offline baking support) ---
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
    def __init__(self, max_torque=500.0, max_rpm=7000.0, min_rpm=1000.0, inertia=0.5):
        self.max_torque = float(max_torque)
        self.max_rpm = float(max_rpm)
        self.min_rpm = float(min_rpm)
        self.inertia = float(inertia)

class Transmission:
    def __init__(self, gears=5, clutch_strength=2000.0, differential_ratio=3.42):
        self.clutch_strength = float(clutch_strength)
        self.differential_ratio = float(differential_ratio)
        presets = [2.66, 1.78, 1.30, 1.0, 0.74, 0.50]
        self.ratios = presets[:gears]
        self.reverse_ratios = [-2.90]

class Automatic(Transmission):
    def __init__(self, gears=5, clutch_strength=2000.0, differential_ratio=3.42, 
                 shift_up_rpm=5000.0, shift_down_rpm=2000.0):
        super().__init__(gears, clutch_strength, differential_ratio)
        self.mode = 0 # Auto
        self.shift_up_rpm = float(shift_up_rpm)
        self.shift_down_rpm = float(shift_down_rpm)

class Manual(Transmission):
    def __init__(self, gears=5, clutch_strength=5000.0, differential_ratio=3.42):
        super().__init__(gears, clutch_strength, differential_ratio)
        self.mode = 1 # Manual

# --- Validation Logic ---

def validate_constraint(type_id, body1, body2, params):
    if not isinstance(body1, int) or not isinstance(body2, int):
        raise TypeError("Constraint bodies must be integer handles")
    
    if type_id == CONSTRAINT_FIXED:
        return None

    if type_id == CONSTRAINT_POINT:
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

def _force_float(val, name):
    try:
        return float(val)
    except:
        raise TypeError(f"'{name}' must be a number")

def _validate_vec3(v, name):
    if len(v) != 3: raise ValueError(f"'{name}' must be len 3")
    return (float(v[0]), float(v[1]), float(v[2]))

def _validate_quat(q, name):
    if len(q) != 4: raise ValueError(f"'{name}' must be len 4")
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

def validate_settings(s):
    s = s or {}
    grav = _validate_vec3(s.get("gravity", (0.0, -9.81, 0.0)), "gravity")
    return (grav[0], grav[1], grav[2], float(s.get("penetration_slop", 0.02)), int(s.get("max_bodies", 10240)), int(s.get("max_pairs", 65536)))

def validate_body_params(shape_type, pos, rot, size, motion_type):
    p = _validate_vec3(pos, "pos")
    r = _validate_quat(rot, "rot")
    s = [0.0, 0.0, 0.0, 0.0]
    
    if shape_type == SHAPE_BOX:
        sz = _validate_vec3(size, "size")
        s[0], s[1], s[2] = sz
    elif shape_type == SHAPE_SPHERE:
        s[0] = float(size[0] if isinstance(size, (list, tuple)) else size)
    elif shape_type in (SHAPE_CAPSULE, SHAPE_CYLINDER):
        s[0], s[1] = float(size[0]), float(size[1])
    elif shape_type == SHAPE_PLANE:
        n = _validate_vec3(size[:3], "normal")
        s[0], s[1], s[2], s[3] = n[0], n[1], n[2], float(size[3])
    
    return p, r, tuple(s)

def bake_scene(bodies):
    if not bodies: return 0, b"", b"", b"", b"", b"", b""
    
    arr_pos = array.array('d')
    arr_rot = array.array('f')
    arr_shape = array.array('f')
    arr_mot = array.array('B')
    arr_layer = array.array('B')
    arr_usr = array.array('Q')
    
    for b in bodies:
        motion = b.get("motion", MOTION_DYNAMIC if b.get("mass", 1.0) > 0 else MOTION_STATIC)
        p, r, s = validate_body_params(b.get("shape", SHAPE_BOX), b.get("pos", (0,0,0)), b.get("rot", (0,0,0,1)), b.get("size", (0,0,0,0)), motion)
        
        arr_pos.extend((p[0], p[1], p[2], 0.0))
        arr_rot.extend(r)
        arr_shape.append(float(b.get("shape", SHAPE_BOX)))
        arr_shape.extend(s)
        arr_mot.append(motion)
        arr_layer.append(LAYER_MOVING if motion != MOTION_STATIC else LAYER_NON_MOVING)
        arr_usr.append(int(b.get("user_data", 0)))
        
    return len(bodies), arr_pos.tobytes(), arr_rot.tobytes(), arr_shape.tobytes(), arr_mot.tobytes(), arr_layer.tobytes(), arr_usr.tobytes()
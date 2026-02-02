import math
import array
from typing import Union, List, Tuple, Dict, Any

__all__ = [
    "MOTION_STATIC", "MOTION_KINEMATIC", "MOTION_DYNAMIC",
    "SHAPE_BOX", "SHAPE_SPHERE", "SHAPE_CAPSULE", "SHAPE_CYLINDER", "SHAPE_PLANE", "SHAPE_MESH", "SHAPE_HEIGHTFIELD", "SHAPE_CONVEX_HULL",
    "LAYER_NON_MOVING", "LAYER_MOVING",
    "CONSTRAINT_FIXED", "CONSTRAINT_POINT", "CONSTRAINT_HINGE", "CONSTRAINT_SLIDER", "CONSTRAINT_DISTANCE", "CONSTRAINT_CONE", 
    "EVENT_ADDED", "EVENT_PERSISTED", "EVENT_REMOVED",
    "Engine", "Transmission", "Automatic", "Manual",
    "validate_constraint", "validate_settings", "bake_scene"
]

# --- Constants matching Jolt/JoltC ---
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

class Engine:
    def __init__(self, max_torque=500.0, max_rpm=7000.0, min_rpm=1000.0, inertia=0.5):
        self.max_torque = float(max_torque)
        self.max_rpm = float(max_rpm)
        self.min_rpm = float(min_rpm)
        self.inertia = float(inertia)

class Transmission:
    def __init__(self, gears: Union[int, list] = 5, clutch_strength=2000.0):
        self.clutch_strength = float(clutch_strength)
        # If gears is an int, provide standard ratios. If list, use as provided.
        if isinstance(gears, int):
            # Standard 1st to Nth ratios
            presets = [4.0, 2.5, 1.7, 1.2, 1.0, 0.8, 0.7]
            self.ratios = presets[:gears]
        else:
            self.ratios = [float(g) for g in gears]
        self.reverse_ratios = [-3.0]

class Automatic(Transmission):
    def __init__(self, gears=5, clutch_strength=2000.0, shift_up_rpm=5000.0, shift_down_rpm=2000.0):
        super().__init__(gears, clutch_strength)
        self.mode = 0 # JPH_TransmissionMode_Auto
        self.shift_up_rpm = float(shift_up_rpm)
        self.shift_down_rpm = float(shift_down_rpm)

class Manual(Transmission):
    def __init__(self, gears=5, clutch_strength=5000.0):
        super().__init__(gears, clutch_strength)
        self.mode = 1 # JPH_TransmissionMode_Manual

class Skeleton:
    def __init__(self, joints: List[str]):
        """
        joints: List of parent-ordered joint names.
        Example: ["Root", "Spine", "Head", "L_Arm"] 
        (Assumes index based parenting or simple linear for MVP, 
         better pass List[Tuple[name, parent_idx]])
        """
        # In C, we expose add_joint.
        pass

class RagdollSettings:
    def add_part(self, joint_index: int, shape_type: int, size: tuple, mass: float, parent_index: int,
                 twist_min: float, twist_max: float, cone_angle: float,
                 axis: tuple = (1,0,0), normal: tuple = (0,1,0)):
        pass

class Ragdoll:
    def drive_to_pose(self, root_pos, root_rot, matrices_bytes):
        pass
    def get_body_handles(self) -> List[int]:
        ...

# --- Internal Validation Helpers ---

def validate_constraint(type_id, body1, body2, params):
    """
    Validates arguments for creating a physics constraint.
    Returns the normalized params tuple expected by the C extension.
    """
    # 1. Validate Handles
    if not isinstance(body1, int) or not isinstance(body2, int):
        raise TypeError("Constraint bodies must be integer handles")
    
    # 2. Validate Type
    if not isinstance(type_id, int):
        raise TypeError("Constraint type must be an integer")

    # 3. Validate Params based on type
    
    # --- FIXED: No params ---
    if type_id == CONSTRAINT_FIXED:
        return None

    # --- POINT: (px, py, pz) ---
    if type_id == CONSTRAINT_POINT:
        # C expects: "fff" -> (x, y, z)
        return _validate_vec3(params, "point.pivot")

    # --- DISTANCE: (min, max) ---
    if type_id == CONSTRAINT_DISTANCE:
        # C expects: "ff" -> (min, max)
        if not isinstance(params, (tuple, list)) or len(params) != 2:
            raise ValueError("DistanceConstraint requires params=(min_dist, max_dist)")
        
        mn = _force_float(params[0], "min_dist")
        mx = _force_float(params[1], "max_dist")
        if mn < 0 or mx < 0:
            raise ValueError("Distance constraints cannot be negative")
        if mn > mx:
            raise ValueError(f"Min distance ({mn}) cannot be greater than max distance ({mx})")
            
        return (mn, mx)

    # --- HINGE / SLIDER: ((px,py,pz), (ax,ay,az), [min, max]) ---
    if type_id in (CONSTRAINT_HINGE, CONSTRAINT_SLIDER):
        # C expects: "(fff)(fff)|ff"
        name = "Hinge" if type_id == CONSTRAINT_HINGE else "Slider"
        
        if not isinstance(params, (tuple, list)) or len(params) < 2:
            raise ValueError(f"{name}Constraint requires params=((pivot), (axis), [min, max])")
        
        pivot = _validate_vec3(params[0], f"{name}.pivot")
        axis = _validate_vec3(params[1], f"{name}.axis")
        
        # Axis sanity check (must be non-zero for cross product logic in C)
        if sum(x*x for x in axis) < 1e-9:
            raise ValueError(f"{name}.axis cannot be a zero vector")

        if len(params) == 4:
            mn = _force_float(params[2], f"{name}.min")
            mx = _force_float(params[3], f"{name}.max")
            return (pivot, axis, mn, mx)
        elif len(params) == 2:
            return (pivot, axis)
        else:
            raise ValueError(f"{name} limits must be provided as a pair (min, max)")

    # --- CONE: ((px,py,pz), (ax,ay,az), half_angle) ---
    if type_id == CONSTRAINT_CONE:
        # C expects: "(fff)(fff)f"
        if not isinstance(params, (tuple, list)) or len(params) != 3:
            raise ValueError("ConeConstraint requires params=((pivot), (twist_axis), half_angle)")
        
        pivot = _validate_vec3(params[0], "cone.pivot")
        axis = _validate_vec3(params[1], "cone.twist_axis")
        angle = _force_float(params[2], "cone.half_angle")
        
        if sum(x*x for x in axis) < 1e-9:
            raise ValueError("Cone.twist_axis cannot be a zero vector")
            
        return (pivot, axis, angle)

    raise ValueError(f"Unknown constraint type ID: {type_id}")

def _force_float(val, name):
    try:
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"Physics Error: '{name}' must be a finite number (got {f})")
        return f
    except (ValueError, TypeError):
        raise TypeError(f"Physics Error: '{name}' must be a number, not {type(val).__name__}")

def _validate_vec3(v, name):
    if len(v) != 3:
        raise ValueError(f"Physics Error: '{name}' must have 3 components (x, y, z)")
    return (_force_float(v[0], f"{name}.x"), 
            _force_float(v[1], f"{name}.y"), 
            _force_float(v[2], f"{name}.z"))

def _validate_quat(q, name):
    if len(q) != 4:
        raise ValueError(f"Physics Error: '{name}' must have 4 components (x, y, z, w)")
    # Jolt expects X, Y, Z, W layout
    res = (_force_float(q[0], f"{name}.x"), 
           _force_float(q[1], f"{name}.y"), 
           _force_float(q[2], f"{name}.z"),
           _force_float(q[3], f"{name}.w"))
    
    # Normalization check / Drift Correction
    mag_sq = sum(x*x for x in res)
    if not (0.99 < mag_sq < 1.01):
         # If it's close enough to be a rounding error, normalize it.
         # If it's zero or garbage, let math.sqrt raise or result in NaNs which are caught later
         if mag_sq > 1e-9:
            inv = 1.0 / math.sqrt(mag_sq)
            res = tuple(x * inv for x in res)
    return res

# --- Public API ---

def validate_settings(s):
    """
    Enforces invariants for PhysicsWorld construction.
    """
    s = s or {}
    grav = _validate_vec3(s.get("gravity", (0.0, -9.81, 0.0)), "settings.gravity")
    
    max_bodies = int(s.get("max_bodies", 10240))
    if max_bodies < 1: raise ValueError("max_bodies must be at least 1")
        
    return (
        grav[0], grav[1], grav[2],
        _force_float(s.get("penetration_slop", 0.02), "penetration_slop"),
        max_bodies,
        int(s.get("max_pairs", 65536))
    )

def validate_body_params(shape_type, pos, rot, size, motion_type):
    """
    Validates a single body's parameters.
    Standardizes 'size' into a 4-float tuple for the C ShapeKey.
    """
    # 1. Basic Transform
    p = _validate_vec3(pos, "pos")
    r = _validate_quat(rot, "rot")
    
    # 2. Motion Invariants
    if motion_type not in (MOTION_STATIC, MOTION_KINEMATIC, MOTION_DYNAMIC):
        raise ValueError(f"Invalid motion type: {motion_type}")

    # 3. Shape Specific Invariants
    s = [0.0, 0.0, 0.0, 0.0]
    
    if shape_type == SHAPE_BOX:
        # half-extents must be positive
        s[0], s[1], s[2] = _validate_vec3(size, "box.size")
        if any(x <= 0 for x in s[:3]):
            raise ValueError("Box half-extents must be positive")
            
    elif shape_type == SHAPE_SPHERE:
        s[0] = _force_float(size[0] if isinstance(size, (list, tuple)) else size, "sphere.radius")
        if s[0] <= 0: raise ValueError("Sphere radius must be positive")
        
    elif shape_type == SHAPE_CAPSULE or shape_type == SHAPE_CYLINDER:
        if len(size) < 2: raise ValueError("Capsule/Cylinder requires (half_height, radius)")
        s[0] = _force_float(size[0], "half_height")
        s[1] = _force_float(size[1], "radius")
        if s[0] <= 0 or s[1] <= 0:
            raise ValueError("Capsule/Cylinder dimensions must be positive")
            
    elif shape_type == SHAPE_PLANE:
        if motion_type != MOTION_STATIC:
            raise ValueError("Invariant Violation: SHAPE_PLANE must be MOTION_STATIC")
        if len(size) < 4: raise ValueError("Plane requires (nx, ny, nz, distance)")
        # Normal + Distance
        s[0], s[1], s[2] = _validate_vec3(size[:3], "plane.normal")
        s[3] = _force_float(size[3], "plane.distance")
        # Ensure normal is roughly normalized
        n_mag = math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)
        if not (0.9 < n_mag < 1.1):
            raise ValueError("Plane normal must be a unit vector")
    
    elif shape_type == SHAPE_MESH:
        # Meshes cannot be baked via simple config dicts because they need 
        # pointer data (vertices/indices). They must be created at runtime.
        raise ValueError("SHAPE_MESH cannot be baked via init. Use create_mesh_body() instead.")
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    return p, r, tuple(s)

def bake_scene(bodies):
    """
    Validates a list of bodies and packs them into flat binary buffers.
    Returns a 7-element tuple required by C-init.
    """
    if not isinstance(bodies, list):
        raise TypeError("Bake Error: 'bodies' must be a list of dicts")
        
    count = len(bodies)
    # Return empty buffers if count is 0. Note the 7th empty bytes for user_data
    if count == 0:
        return 0, b"", b"", b"", b"", b"", b""

    # Use 'f' for float32, 'B' for unsigned char
    arr_pos = array.array('f')
    arr_rot = array.array('f')
    arr_shape_data = array.array('f') # [type, p1, p2, p3, p4]
    arr_motion = array.array('B')
    arr_layer = array.array('B')
    arr_user_data = array.array('Q')  # uint64_t for Entity IDs

    for i, b in enumerate(bodies):
        stype = b.get("shape", SHAPE_BOX)
        motion = b.get("motion", MOTION_DYNAMIC if b.get("mass", 1.0) > 0 else MOTION_STATIC)
        
        # Enforce all rules
        # Use default size (0,0,0,0) if missing, validator handles specific requirements
        pos, rot, shape_params = validate_body_params(
            stype, 
            b.get("pos", (0,0,0)), 
            b.get("rot", (0,0,0,1)), 
            b.get("size", (0,0,0,0)), 
            motion
        )

        # Pack Position (with 4th float padding for alignment compatibility)
        arr_pos.append(pos[0])
        arr_pos.append(pos[1])
        arr_pos.append(pos[2])
        arr_pos.append(0.0)
        # Pack Rotation
        arr_rot.append(rot[0])
        arr_rot.append(rot[1])
        arr_rot.append(rot[2])
        arr_rot.append(rot[3])
        # Pack Shape [Type, P1, P2, P3, P4]
        arr_shape_data.append(float(stype))
        arr_shape_data.extend(shape_params)
        
        arr_motion.append(motion)
        arr_layer.append(LAYER_MOVING if motion != MOTION_STATIC else LAYER_NON_MOVING)
        
        # Pack User Data (Entity ID)
        arr_user_data.append(int(b.get("user_data", 0)))

    return (
        count,
        arr_pos.tobytes(),          # 1
        arr_rot.tobytes(),          # 2
        arr_shape_data.tobytes(),   # 3
        arr_motion.tobytes(),       # 4
        arr_layer.tobytes(),        # 5
        arr_user_data.tobytes()     # 6
    )
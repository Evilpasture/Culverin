import math
import array

# Constants matching JPH_MotionType
MOTION_STATIC = 0
MOTION_KINEMATIC = 1
MOTION_DYNAMIC = 2

# Shape Types
SHAPE_BOX = 0
SHAPE_SPHERE = 1
SHAPE_CAPSULE = 2
SHAPE_CYLINDER = 3

# Layers
LAYER_NON_MOVING = 0
LAYER_MOVING = 1

def _as_float(val, name):
    try:
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"'{name}' must be finite")
        return f
    except (ValueError, TypeError):
        raise TypeError(f"'{name}' must be a number")

def _as_vec3(val, name):
    try:
        x, y, z = val
        return (_as_float(x, f"{name}.x"), 
                _as_float(y, f"{name}.y"), 
                _as_float(z, f"{name}.z"))
    except (ValueError, TypeError):
        raise TypeError(f"'{name}' must be a sequence of 3 numbers")

def _as_quat(val, name):
    try:
        w, x, y, z = val
        # Jolt expects X, Y, Z, W layout in memory
        return (_as_float(x, f"{name}.x"), 
                _as_float(y, f"{name}.y"), 
                _as_float(z, f"{name}.z"),
                _as_float(w, f"{name}.w"))
    except (ValueError, TypeError):
        raise TypeError(f"'{name}' must be a sequence of 4 numbers (w, x, y, z)")

def validate_settings(s):
    """
    Returns a tuple of floats/ints matching the C struct order.
    """
    s = s or {}
    grav = _as_vec3(s.get("gravity", (0.0, -9.81, 0.0)), "gravity")
    
    return (
        grav[0], grav[1], grav[2], # Gravity x,y,z
        _as_float(s.get("penetration_slop", 0.02), "penetration_slop"),
        int(s.get("max_bodies", 10240)),
        int(s.get("max_pairs", 65536))
    )

def bake_scene(bodies):
    """
    Converts list of BodyConfig dicts into flat binary buffers.
    """
    if not isinstance(bodies, list): raise TypeError("bodies must be a list")
    count = len(bodies)
    if count == 0: return 0, b"", b"", b"", b"", b""

    arr_pos = array.array('f')
    arr_rot = array.array('f')
    arr_shape_data = array.array('f') # [type, p1, p2, p3]
    arr_motion = array.array('B')     # [motion_type]
    arr_layer = array.array('B')      # [layer_id]

    for i, b in enumerate(bodies):
        # 1. Transform
        arr_pos.extend(b.get("pos", (0, 0, 0)) + (0.0,))
        arr_rot.extend(b.get("rot", (0, 0, 0, 1))) # x, y, z, w

        # 2. Shape Logic
        stype = b.get("shape", SHAPE_BOX)
        size = b.get("size", (0.5, 0.5, 0.5))
        if stype == SHAPE_BOX:
            arr_shape_data.extend([SHAPE_BOX, size[0], size[1], size[2]])
        elif stype == SHAPE_SPHERE:
            arr_shape_data.extend([SHAPE_SPHERE, size[0], 0.0, 0.0]) # radius
        elif stype == SHAPE_CAPSULE:
            arr_shape_data.extend([SHAPE_CAPSULE, size[0], size[1], 0.0]) # half-height, radius
        
        # 3. Dynamics & Layers
        mass = b.get("mass", 1.0)
        is_dynamic = mass > 0.0
        arr_motion.append(2 if is_dynamic else 0) # Dynamic vs Static
        arr_layer.append(LAYER_MOVING if is_dynamic else LAYER_NON_MOVING)

    return (
        count,
        arr_pos.tobytes(),
        arr_rot.tobytes(),
        arr_shape_data.tobytes(),
        arr_motion.tobytes(),
        arr_layer.tobytes()
    )
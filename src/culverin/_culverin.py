import math
import array

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

LAYER_NON_MOVING = 0
LAYER_MOVING = 1

def _force_float(val, name):
    try:
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"Physics Error: '{name}' must be finite (got {f})")
        return f
    except (ValueError, TypeError):
        raise TypeError(f"Physics Error: '{name}' must be a number")

def _validate_vec3(v, name):
    if len(v) != 3:
        raise ValueError(f"Physics Error: '{name}' must have 3 components")
    return (_force_float(v[0], f"{name}.x"), 
            _force_float(v[1], f"{name}.y"), 
            _force_float(v[2], f"{name}.z"))

def _validate_quat(q, name):
    if len(q) != 4:
        raise ValueError(f"Physics Error: '{name}' must have 4 components (x, y, z, w)")
    res = (_force_float(q[0], f"{name}.x"), 
           _force_float(q[1], f"{name}.y"), 
           _force_float(q[2], f"{name}.z"),
           _force_float(q[3], f"{name}.w"))
    mag_sq = sum(x*x for x in res)
    if not (0.99 < mag_sq < 1.01):
         inv = 1.0 / math.sqrt(mag_sq)
         res = tuple(x * inv for x in res)
    return res

def validate_body_params(shape_type, pos, rot, size, motion_type):
    p = _validate_vec3(pos, "pos")
    r = _validate_quat(rot, "rot")
    s = [0.0, 0.0, 0.0, 0.0]
    
    if shape_type == SHAPE_BOX:
        s[0], s[1], s[2] = _validate_vec3(size, "box.size")
    elif shape_type == SHAPE_SPHERE:
        s[0] = _force_float(size[0] if isinstance(size, (list, tuple)) else size, "radius")
    elif shape_type in (SHAPE_CAPSULE, SHAPE_CYLINDER):
        s[0] = _force_float(size[0], "half_height")
        s[1] = _force_float(size[1], "radius")
    elif shape_type == SHAPE_PLANE:
        if motion_type != MOTION_STATIC:
            raise ValueError("Invariant Violation: SHAPE_PLANE must be MOTION_STATIC")
        s[0], s[1], s[2] = _validate_vec3(size[:3], "plane.normal")
        s[3] = _force_float(size[3], "plane.distance")
    elif shape_type == SHAPE_MESH:
        if motion_type != MOTION_STATIC:
             raise ValueError("Invariant Violation: SHAPE_MESH must be MOTION_STATIC")

    return p, r, tuple(s)

def bake_scene(bodies):
    if not isinstance(bodies, list): raise TypeError("Bake Error: 'bodies' must be a list")
    count = len(bodies)
    if count == 0: return 0, b"", b"", b"", b"", b"", b""

    arr_pos = array.array('f')
    arr_rot = array.array('f')
    arr_shape_data = array.array('f') # Stride 5: [type, p1, p2, p3, p4]
    arr_motion = array.array('B')
    arr_layer = array.array('B')
    arr_user_data = array.array('Q')  # uint64_t for Entity IDs

    for b in bodies:
        stype = b.get("shape", SHAPE_BOX)
        motion = b.get("motion", MOTION_DYNAMIC if b.get("mass", 1.0) > 0 else MOTION_STATIC)
        
        pos, rot, shape_params = validate_body_params(
            stype, b.get("pos", (0,0,0)), b.get("rot", (0,0,0,1)), b.get("size", (1,1,1,0)), motion
        )

        # Pack Position with float4 alignment (16 bytes)
        arr_pos.extend([pos[0], pos[1], pos[2], 0.0])
        # Pack Rotation (x,y,z,w)
        arr_rot.extend(rot)
        # Pack Shape (type + 4 params)
        arr_shape_data.append(float(stype))
        arr_shape_data.extend(shape_params)
        
        arr_motion.append(motion)
        arr_layer.append(LAYER_MOVING if motion != MOTION_STATIC else LAYER_NON_MOVING)
        
        # User Data (Entity ID)
        arr_user_data.append(int(b.get("user_data", 0)))

    return (
        count,
        arr_pos.tobytes(),
        arr_rot.tobytes(),
        arr_shape_data.tobytes(),
        arr_motion.tobytes(),
        arr_layer.tobytes(),
        arr_user_data.tobytes() # New 7th element in tuple
    )
import math
import array

# Constants matching JPH_MotionType
MOTION_STATIC = 0
MOTION_KINEMATIC = 1
MOTION_DYNAMIC = 2

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
    if not isinstance(bodies, list):
        raise TypeError("bodies must be a list")

    count = len(bodies)
    if count == 0:
        return 0, b"", b"", b"", b""

    # Arrays for flat C-memory (using 'f' for float)
    # We add padding to vec3 to align to 16 bytes (x, y, z, w) for SIMD in C
    arr_pos = array.array('f')
    arr_rot = array.array('f')
    arr_extent = array.array('f')
    arr_motion = array.array('B') # Unsigned char for motion type

    for i, b in enumerate(bodies):
        # 1. Position (x, y, z) -> Packed as (x, y, z, 0.0)
        p = _as_vec3(b.get("pos", (0,0,0)), f"body[{i}].pos")
        arr_pos.extend(p)
        arr_pos.append(0.0) # Padding

        # 2. Rotation (w, x, y, z)
        r = _as_quat(b.get("rot", (1,0,0,0)), f"body[{i}].rot")
        arr_rot.extend(r)

        # 3. Extents (half_x, half_y, half_z) -> Packed as (x, y, z, 0.0)
        s = _as_vec3(b.get("size", (0.5,0.5,0.5)), f"body[{i}].size")
        arr_extent.extend(s)
        arr_extent.append(0.0) # Padding

        # 4. Motion Type (Derived from mass)
        mass = _as_float(b.get("mass", 0.0), f"body[{i}].mass")
        if mass > 0.0:
            arr_motion.append(MOTION_DYNAMIC)
        else:
            arr_motion.append(MOTION_STATIC)

    return (
        count,
        arr_pos.tobytes(),
        arr_rot.tobytes(),
        arr_extent.tobytes(),
        arr_motion.tobytes()
    )
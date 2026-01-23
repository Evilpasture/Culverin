import math
import array

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

LAYER_NON_MOVING = 0
LAYER_MOVING = 1

# --- Internal Validation Helpers ---

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
    # Basic normalization check (optional but recommended for stability)
    mag_sq = sum(x*x for x in res)
    if not (0.9 < mag_sq < 1.1):
         # We don't throw here to allow slight drift, but we should notify if needed
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
        if motion_type != MOTION_STATIC:
             raise ValueError("Invariant Violation: SHAPE_MESH is only supported as MOTION_STATIC in this engine version")
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    return p, r, tuple(s)

def bake_scene(bodies):
    """
    Validates a list of bodies and packs them into binary buffers.
    """
    if not isinstance(bodies, list):
        raise TypeError("Bake Error: 'bodies' must be a list of dicts")
        
    count = len(bodies)
    if count == 0:
        return 0, b"", b"", b"", b"", b""

    # Use 'f' for float32, 'B' for unsigned char
    arr_pos = array.array('f')
    arr_rot = array.array('f')
    arr_shape_data = array.array('f') # [type, p1, p2, p3, p4]
    arr_motion = array.array('B')
    arr_layer = array.array('B')

    for i, b in enumerate(bodies):
        stype = b.get("shape", SHAPE_BOX)
        motion = b.get("motion", MOTION_DYNAMIC if b.get("mass", 1.0) > 0 else MOTION_STATIC)
        
        # Enforce all rules
        pos, rot, shape_params = validate_body_params(
            stype, 
            b.get("pos", (0,0,0)), 
            b.get("rot", (0,0,0,1)), 
            b.get("size", (1,1,1,0)), 
            motion
        )

        # Pack Position (with 4th float padding for alignment compatibility)
        arr_pos.extend([pos[0], pos[1], pos[2], 0.0])
        # Pack Rotation
        arr_rot.extend(rot)
        # Pack Shape [Type, P1, P2, P3, P4]
        arr_shape_data.append(float(stype))
        arr_shape_data.extend(shape_params)
        
        arr_motion.append(motion)
        arr_layer.append(LAYER_MOVING if motion != MOTION_STATIC else LAYER_NON_MOVING)

    return (
        count,
        arr_pos.tobytes(),
        arr_rot.tobytes(),
        arr_shape_data.tobytes(),
        arr_motion.tobytes(),
        arr_layer.tobytes()
    )
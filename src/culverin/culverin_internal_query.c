
#include "culverin_internal_query.h"

// --- Helper: Shape Caching (Internal) ---
JPH_Shape *find_or_create_shape(PhysicsWorldObject *self, int type,
                                       const float *params) {
  // 1. HARDENED KEY CONSTRUCTION
  // We zero out the key first so that unused parameters for a specific
  // shape type (like p2-p4 for a Sphere) don't cause cache misses.
  ShapeKey key;
  memset(&key, 0, sizeof(ShapeKey));
  key.type = (uint32_t)type;

  switch (type) {
  case 0: // BOX: Uses 3 params (half-extents)
    key.p1 = params[0];
    key.p2 = params[1];
    key.p3 = params[2];
    break;
  case 1: // SPHERE: Uses 1 param (radius)
    key.p1 = params[0];
    break;
  case 2: // CAPSULE: Uses 2 params (half-height, radius)
  case 3: // CYLINDER: Uses 2 params (half-height, radius)
    key.p1 = params[0];
    key.p2 = params[1];
    break;
  case 4: // PLANE: Uses 4 params (nx, ny, nz, d)
    key.p1 = params[0];
    key.p2 = params[1];
    key.p3 = params[2];
    key.p4 = params[3];
    break;
  default:
    break;
  }

  // 2. CACHE LOOKUP
  for (size_t i = 0; i < self->shape_cache_count; i++) {
    ShapeKey *entry_key = &self->shape_cache[i].key;

    // Explicit comparison avoids padding/alignment issues and handles -0.0 vs
    // 0.0 correctly
    if (entry_key->type == key.type && entry_key->p1 == key.p1 &&
        entry_key->p2 == key.p2 && entry_key->p3 == key.p3 &&
        entry_key->p4 == key.p4) {
      return self->shape_cache[i].shape;
    }
  }

  // 3. SHAPE CREATION (Only if not found)
  JPH_Shape *shape = NULL;
  if (type == 0) {
    JPH_Vec3 he = {key.p1, key.p2, key.p3};
    JPH_BoxShapeSettings *s = JPH_BoxShapeSettings_Create(&he, 0.05f);
    shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 1) {
    JPH_SphereShapeSettings *s = JPH_SphereShapeSettings_Create(key.p1);
    shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 2) {
    JPH_CapsuleShapeSettings *s =
        JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
    shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 3) {
    JPH_CylinderShapeSettings *s =
        JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
    shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 4) {
    JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
    // Note: Planes in Jolt often require a half-extent (1000.0f) to define
    // their "active" area
    JPH_PlaneShapeSettings *s =
        JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
    shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  }

  if (!shape) {
    return NULL;
  }

  // 4. CACHE EXPANSION
  if (self->shape_cache_count >= self->shape_cache_capacity) {
    size_t new_cap =
        (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
    // Note: PyMem_RawRealloc is safe here because this is called under
    // SHADOW_LOCK and is not inside the Jolt step.
    void *new_ptr =
        PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
    if (!new_ptr) {
      JPH_Shape_Destroy(shape);
      PyErr_NoMemory();
      return NULL;
    }
    self->shape_cache = (ShapeEntry *)new_ptr;
    self->shape_cache_capacity = new_cap;
  }

  // 5. STORAGE
  self->shape_cache[self->shape_cache_count].key = key;
  self->shape_cache[self->shape_cache_count].shape = shape;
  self->shape_cache_count++;

  return shape;
}
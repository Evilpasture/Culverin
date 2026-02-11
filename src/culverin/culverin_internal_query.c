#include "culverin_internal_query.h"
#include "culverin_filters.h"

// --- Helper: Shape Caching (Internal) ---
// This is called during creation (e.g. create_body). 
// Since creation is low-frequency compared to rays, we handle locking internally here.
JPH_Shape *find_or_create_shape(PhysicsWorldObject *self, int type,
                                const float *params) {
  ShapeKey key;
  memset(&key, 0, sizeof(ShapeKey));
  key.type = (uint32_t)type;

  switch (type) {
    case 0: key.p1 = params[0]; key.p2 = params[1]; key.p3 = params[2]; break;
    case 1: key.p1 = params[0]; break;
    case 2:
    case 3: key.p1 = params[0]; key.p2 = params[1]; break;
    case 4: key.p1 = params[0]; key.p2 = params[1]; key.p3 = params[2]; key.p4 = params[3]; break;
  }

  // Cache lookup (Under shadow_lock, which the caller create_body already holds)
  for (size_t i = 0; i < self->shape_cache_count; i++) {
    ShapeKey *entry_key = &self->shape_cache[i].key;
    if (entry_key->type == key.type && entry_key->p1 == key.p1 &&
        entry_key->p2 == key.p2 && entry_key->p3 == key.p3 &&
        entry_key->p4 == key.p4) {
      return self->shape_cache[i].shape;
    }
  }

  // --- Jolt Shape Creation ---
  // We need the trampoline lock because shape creation uses the global s_TempAllocator
  
  JPH_Shape *shape = NULL;
  Py_BEGIN_ALLOW_THREADS
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
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
    JPH_CapsuleShapeSettings *s = JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
    shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 3) {
    JPH_CylinderShapeSettings *s = JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
    shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 4) {
    JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
    JPH_PlaneShapeSettings *s = JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
    shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  }

  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_END_ALLOW_THREADS

  if (!shape) return NULL;

  // Cache expansion
  if (self->shape_cache_count >= self->shape_cache_capacity) {
    size_t new_cap = (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
    void *new_ptr = PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
    if (!new_ptr) {
      JPH_Shape_Destroy(shape);
      PyErr_NoMemory();
      return NULL;
    }
    self->shape_cache = (ShapeEntry *)new_ptr;
    self->shape_cache_capacity = new_cap;
  }

  self->shape_cache[self->shape_cache_count].key = key;
  self->shape_cache[self->shape_cache_count].shape = shape;
  self->shape_cache_count++;

  return shape;
}

void free_shape_cache(PhysicsWorldObject *self) {
  if (!self->shape_cache) {
    return;
  }

  for (size_t i = 0; i < self->shape_cache_count; i++) {
    if (self->shape_cache[i].shape) {
      JPH_Shape_Destroy(self->shape_cache[i].shape);
    }
  }
  PyMem_RawFree(self->shape_cache);
  self->shape_cache = NULL;
  self->shape_cache_count = 0;
}

// Helper 1: Run the Raycast
// ASSUMPTION: Caller has already acquired g_jph_trampoline_lock and released GIL.
bool execute_raycast_query(PhysicsWorldObject *self, JPH_BodyID ignore_bid,
                           const JPH_RVec3 *origin, const JPH_Vec3 *direction,
                           JPH_RayCastResult *hit) {
  // 1. Filter Setup (Safe, doesn't touch shared Jolt memory yet)
  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide = filter_allow_all_bp};
  JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide = filter_allow_all_obj};
  JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  CastShapeFilter filter_ctx = {.ignore_id = ignore_bid};
  JPH_BodyFilter_Procs filter_procs = {.ShouldCollide = CastShape_BodyFilter};
  JPH_BodyFilter *bf = JPH_BodyFilter_Create(&filter_ctx);
  JPH_BodyFilter_SetProcs(&filter_procs);

  // 2. Execution
  const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
  bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, bp_f, obj_f, bf);

  // 3. Cleanup
  JPH_BodyFilter_Destroy(bf);
  JPH_BroadPhaseLayerFilter_Destroy(bp_f);
  JPH_ObjectLayerFilter_Destroy(obj_f);

  return has_hit;
}

// Helper 2: Extract World Space Normal after hit
// ASSUMPTION: Caller holds the Jolt lock.
void extract_hit_normal(PhysicsWorldObject *self, JPH_BodyID bodyID,
                        JPH_SubShapeID subShapeID2, const JPH_RVec3 *origin,
                        const JPH_Vec3 *ray_dir, float fraction,
                        JPH_Vec3 *normal_out) {
  const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockRead lock;
  JPH_BodyLockInterface_LockRead(lock_iface, bodyID, &lock);

  if (lock.body) {
    JPH_RVec3 hit_p = {origin->x + ray_dir->x * fraction,
                       origin->y + ray_dir->y * fraction,
                       origin->z + ray_dir->z * fraction};
    JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, subShapeID2, &hit_p, normal_out);
  } else {
    *normal_out = (JPH_Vec3){0, 1, 0};
  }
  JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);
}

// Collector for Shapecast
float CastShape_ClosestCollector(void *context, const JPH_ShapeCastResult *result) {
  CastShapeContext *ctx = (CastShapeContext *)context;
  if (result->fraction < ctx->hit.fraction) {
    ctx->hit = *result;
    ctx->has_hit = true;
  }
  return result->fraction;
}

// Helper 3: Internal logic to run the actual query
// ASSUMPTION: Caller has already acquired g_jph_trampoline_lock and released GIL.
void shapecast_execute_internal(PhysicsWorldObject *self,
                                const JPH_Shape *shape,
                                const JPH_RMat4 *transform,
                                const JPH_Vec3 *sweep_dir,
                                JPH_BodyID ignore_bid, CastShapeContext *ctx) {
  JPH_BroadPhaseLayerFilter_Procs bp_p = {.ShouldCollide = filter_allow_all_bp};
  JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_p);

  JPH_ObjectLayerFilter_Procs obj_p = {.ShouldCollide = filter_allow_all_obj};
  JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_p);

  CastShapeFilter filter_ctx = {.ignore_id = ignore_bid};
  JPH_BodyFilter_Procs bf_p = {.ShouldCollide = CastShape_BodyFilter};
  JPH_BodyFilter *bf = JPH_BodyFilter_Create(&filter_ctx);
  JPH_BodyFilter_SetProcs(&bf_p);

  JPH_STACK_ALLOC(JPH_ShapeCastSettings, settings);
  JPH_ShapeCastSettings_Init(settings);
  settings->backFaceModeTriangles = JPH_BackFaceMode_IgnoreBackFaces;
  settings->backFaceModeConvex = JPH_BackFaceMode_IgnoreBackFaces;

  JPH_RVec3 base_offset = {0, 0, 0};
  const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  JPH_NarrowPhaseQuery_CastShape(nq, shape, transform, sweep_dir, settings,
                                 &base_offset, CastShape_ClosestCollector, ctx,
                                 bp_f, obj_f, bf, NULL);

  JPH_BodyFilter_Destroy(bf);
  JPH_BroadPhaseLayerFilter_Destroy(bp_f);
  JPH_ObjectLayerFilter_Destroy(obj_f);
}
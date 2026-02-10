#include "culverin_query_methods.h"


// Unified hit collector for both Broad and Narrow phase overlaps
static void overlap_record_hit(OverlapContext *ctx, JPH_BodyID bid) {
  // 1. Grow buffer if needed (Native memory, no GIL required)
  if (ctx->count >= ctx->capacity) {
    size_t new_cap = (ctx->capacity == 0) ? 32 : ctx->capacity * 2;
    uint64_t *new_ptr = PyMem_RawRealloc(ctx->hits, new_cap * sizeof(uint64_t));
    if (!new_ptr) {
      return; // Drop hit on OOM (safer than crashing)
    }
    ctx->hits = new_ptr;
    ctx->capacity = new_cap;
  }

  // 2. Retrieve the baked Handle from Jolt UserData
  // This handle contains the Generation + Slot at the time of creation.
  ctx->hits[ctx->count++] =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);
}

static float OverlapCallback_Narrow(void *context,
                                    const JPH_CollideShapeResult *result) {
  overlap_record_hit((OverlapContext *)context, result->bodyID2);
  return 1.0f; // Continue looking for more hits
}

static float OverlapCallback_Broad(void *context, const JPH_BodyID result_bid) {
  overlap_record_hit((OverlapContext *)context, result_bid);
  return 1.0f; // Continue
}

PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float radius = 1.0f;
  static char *kwlist[] = {"center", "radius", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &x, &y, &z,
                                   &radius)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  OverlapContext ctx = {.world = self, .hits = NULL, .count = 0, .capacity = 0};

  // Jolt Resources
  JPH_Shape *shape = NULL;
  JPH_BroadPhaseLayerFilter *bp_filter = NULL;
  JPH_ObjectLayerFilter *obj_filter = NULL;
  JPH_BodyFilter *body_filter = NULL;

  // --- 1. PHASE GUARD (Blocking) ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  // Reserve query slot
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. RESOURCE PREP ---
  JPH_SphereShapeSettings *ss = JPH_SphereShapeSettings_Create(radius);
  if (!ss) {
    PyErr_NoMemory();
    goto cleanup;
  }
  shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);
  if (!shape) {
    PyErr_NoMemory();
    goto cleanup;
  }

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)x;
  pos->y = (double)y;
  pos->z = (double)z;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = 0;
  rot->y = 0;
  rot->z = 0;
  rot->w = 1;
  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, rot, pos);
  JPH_STACK_ALLOC(JPH_Vec3, scale);
  scale->x = 1.0f;
  scale->y = 1.0f;
  scale->z = 1.0f;
  JPH_STACK_ALLOC(JPH_RVec3, base_offset);
  base_offset->x = 0;
  base_offset->y = 0;
  base_offset->z = 0;
  JPH_STACK_ALLOC(JPH_CollideShapeSettings, settings);
  JPH_CollideShapeSettings_Init(settings);

  // --- 3. FILTER SETUP & EXECUTION (Serialized) ---
  // We MUST lock the trampoline because SetProcs modifies global state.
  SHADOW_LOCK(&g_jph_trampoline_lock);

  // BroadPhase: Allow All
  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide =
                                                  filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  // ObjectLayer: Allow All
  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide =
                                               filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  // BodyFilter: Default (True)
  JPH_BodyFilter_Procs bf_procs = {.ShouldCollide = filter_true_body};
  body_filter = JPH_BodyFilter_Create(NULL);
  JPH_BodyFilter_SetProcs(&bf_procs);

  const JPH_NarrowPhaseQuery *nq =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  JPH_NarrowPhaseQuery_CollideShape(nq, shape, scale, transform, settings,
                                    base_offset, OverlapCallback_Narrow, &ctx,
                                    bp_filter, obj_filter, body_filter, NULL);

  // Restore Defaults & Unlock
  // (filter_true_body is effectively the default, but we set it explicitly
  // above)
  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  // --- 4. VALIDATION (Locked) ---
  ret_val = PyList_New(0);
  if (!ret_val) {
    goto cleanup;
  }

  SHADOW_LOCK(&self->shadow_lock);
  for (size_t i = 0; i < ctx.count; i++) {
    uint64_t h = ctx.hits[i];
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {
      PyObject *py_h = PyLong_FromUnsignedLongLong(h);
      if (py_h) {
        PyList_Append(ret_val, py_h);
        Py_DECREF(py_h);
      }
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

cleanup:
  // Release query slot
  atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);

  if (shape) {
    JPH_Shape_Destroy(shape);
  }
  if (bp_filter) {
    JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  }
  if (obj_filter) {
    JPH_ObjectLayerFilter_Destroy(obj_filter);
  }
  if (body_filter) {
    JPH_BodyFilter_Destroy(body_filter);
  }

  if (ctx.hits) {
    PyMem_RawFree(ctx.hits);
  }

  return ret_val;
}

PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  float min_x = 0.0f;
  float min_y = 0.0f;
  float min_z = 0.0f;
  float max_x = 0.0f;
  float max_y = 0.0f;
  float max_z = 0.0f;
  static char *kwlist[] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, &min_x,
                                   &min_y, &min_z, &max_x, &max_y, &max_z)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  OverlapContext ctx = {.world = self, .hits = NULL, .count = 0, .capacity = 0};

  JPH_BroadPhaseLayerFilter *bp_filter = NULL;
  JPH_ObjectLayerFilter *obj_filter = NULL;

  // --- 1. PHASE GUARD ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT PREP ---
  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x;
  box->min.y = min_y;
  box->min.z = min_z;
  box->max.x = max_x;
  box->max.y = max_y;
  box->max.z = max_z;

  // --- 3. FILTER & EXECUTION (Serialized) ---
  SHADOW_LOCK(&g_jph_trampoline_lock);

  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide =
                                                  filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide =
                                               filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  const JPH_BroadPhaseQuery *bq =
      JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);
  JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx,
                                   bp_filter, obj_filter);

  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  // --- 4. VALIDATION ---
  ret_val = PyList_New(0);
  if (!ret_val) {
    goto cleanup;
  }

  SHADOW_LOCK(&self->shadow_lock);
  for (size_t i = 0; i < ctx.count; i++) {
    uint64_t h = ctx.hits[i];
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {
      PyObject *py_h = PyLong_FromUnsignedLongLong(h);
      if (py_h) {
        PyList_Append(ret_val, py_h);
        Py_DECREF(py_h);
      }
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

cleanup:
  atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);

  if (bp_filter) {
    JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  }
  if (obj_filter) {
    JPH_ObjectLayerFilter_Destroy(obj_filter);
  }
  if (ctx.hits) {
    PyMem_RawFree(ctx.hits);
  }

  return ret_val;
}
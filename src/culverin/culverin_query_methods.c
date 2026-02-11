#include "culverin_query_methods.h"
#include "culverin_filters.h"
#include "culverin_parsers.h"

// --- Helper: Signal End of Query ---
// This is crucial for the Condition Variable approach. 
// If we are the last query to finish, we must wake up the physics stepper.
static void end_query_scope(PhysicsWorldObject *self) {
  // 1. Lock native mutex first
  NATIVE_MUTEX_LOCK(self->step_sync.mutex);
  
  // 2. Decrement and check
  // We use fetch_sub; if it was 1, it's now 0.
  uint32_t prev = atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_acq_rel);
  
  if (prev == 1) {
    // We were the last one, signal the stepper
    NATIVE_COND_BROADCAST(self->step_sync.cond);
  }
  
  // 3. Unlock
  NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
}

// Unified hit collector for both Broad and Narrow phase overlaps
static void overlap_record_hit(OverlapContext *ctx, JPH_BodyID bid) {
  if (ctx->count >= ctx->capacity) {
    size_t new_cap = (ctx->capacity == 0) ? 32 : ctx->capacity * 2;
    uint64_t *new_ptr = PyMem_RawRealloc(ctx->hits, new_cap * sizeof(uint64_t));
    if (!new_ptr) {
      return; 
    }
    ctx->hits = new_ptr;
    ctx->capacity = new_cap;
  }
  ctx->hits[ctx->count++] =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);
}

static float OverlapCallback_Narrow(void *context,
                                    const JPH_CollideShapeResult *result) {
  overlap_record_hit((OverlapContext *)context, result->bodyID2);
  return 1.0f; 
}

static float OverlapCallback_Broad(void *context, const JPH_BodyID result_bid) {
  overlap_record_hit((OverlapContext *)context, result_bid);
  return 1.0f; 
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self, PyObject *args,
                                      PyObject *kwds) {
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

  JPH_Shape *shape = NULL;
  JPH_BroadPhaseLayerFilter *bp_filter = NULL;
  JPH_ObjectLayerFilter *obj_filter = NULL;
  JPH_BodyFilter *body_filter = NULL;

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

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
  rot->x = 0; rot->y = 0; rot->z = 0; rot->w = 1;
  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, rot, pos);
  JPH_STACK_ALLOC(JPH_Vec3, scale);
  scale->x = 1.0f; scale->y = 1.0f; scale->z = 1.0f;
  JPH_STACK_ALLOC(JPH_RVec3, base_offset);
  base_offset->x = 0; base_offset->y = 0; base_offset->z = 0;
  JPH_STACK_ALLOC(JPH_CollideShapeSettings, settings);
  JPH_CollideShapeSettings_Init(settings);

  // --- EXECUTION ---
  
  Py_BEGIN_ALLOW_THREADS 
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  

  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide = filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide = filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  JPH_BodyFilter_Procs bf_procs = {.ShouldCollide = filter_true_body};
  body_filter = JPH_BodyFilter_Create(NULL);
  JPH_BodyFilter_SetProcs(&bf_procs);

  const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  JPH_NarrowPhaseQuery_CollideShape(nq, shape, scale, transform, settings,
                                    base_offset, OverlapCallback_Narrow, &ctx,
                                    bp_filter, obj_filter, body_filter, NULL);
  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_END_ALLOW_THREADS 
  

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
  // --- SIGNALING CHANGE HERE ---
  end_query_scope(self); 

  if (shape) JPH_Shape_Destroy(shape);
  if (bp_filter) JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  if (obj_filter) JPH_ObjectLayerFilter_Destroy(obj_filter);
  if (body_filter) JPH_BodyFilter_Destroy(body_filter);
  if (ctx.hits) PyMem_RawFree(ctx.hits);

  return ret_val;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self, PyObject *args,
                                    PyObject *kwds) {
  float min_x = 0.0f; float min_y = 0.0f; float min_z = 0.0f;
  float max_x = 0.0f; float max_y = 0.0f; float max_z = 0.0f;
  static char *kwlist[] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, &min_x,
                                   &min_y, &min_z, &max_x, &max_y, &max_z)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  OverlapContext ctx = {.world = self, .hits = NULL, .count = 0, .capacity = 0};

  JPH_BroadPhaseLayerFilter *bp_filter = NULL;
  JPH_ObjectLayerFilter *obj_filter = NULL;

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x; box->min.y = min_y; box->min.z = min_z;
  box->max.x = max_x; box->max.y = max_y; box->max.z = max_z;
  
  Py_BEGIN_ALLOW_THREADS
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  

  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide = filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide = filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  const JPH_BroadPhaseQuery *bq = JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);
  JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx,
                                   bp_filter, obj_filter);

  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_END_ALLOW_THREADS
  

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
  // --- SIGNALING CHANGE HERE ---
  end_query_scope(self);

  if (bp_filter) JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  if (obj_filter) JPH_ObjectLayerFilter_Destroy(obj_filter);
  if (ctx.hits) PyMem_RawFree(ctx.hits);

  return ret_val;
}

PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *args,
                               PyObject *kwds) {
  float sx; float sy; float sz;
  float dx; float dy; float dz;
  float max_dist = 1000.0f;
  uint64_t ignore_h = 0;
  static char *kwlist[] = {"start", "direction", "max_dist", "ignore", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|fK", kwlist, &sx,
                                   &sy, &sz, &dx, &dy, &dz, &max_dist,
                                   &ignore_h)) {
    return NULL;
  }

  PyObject *result = NULL;
  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f) {
    Py_RETURN_NONE;
  }
  float mag = sqrtf(mag_sq);
  float scale = max_dist / mag;

  JPH_STACK_ALLOC(JPH_RVec3, origin);
  origin->x = sx; origin->y = sy; origin->z = sz;
  JPH_STACK_ALLOC(JPH_Vec3, direction);
  direction->x = dx * scale; direction->y = dy * scale; direction->z = dz * scale;
  JPH_STACK_ALLOC(JPH_RayCastResult, hit);
  memset(hit, 0, sizeof(JPH_RayCastResult));

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);

  JPH_BodyID ignore_bid = 0;
  uint32_t ignore_slot = 0;
  if (ignore_h != 0 && unpack_handle(self, ignore_h, &ignore_slot)) {
    ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  bool has_hit = false;
  JPH_Vec3 normal = {0, 0, 0};
  Py_BEGIN_ALLOW_THREADS
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  
  

  // These two functions call Jolt internals. They MUST be inside the JPH lock.
  has_hit = execute_raycast_query(self, ignore_bid, origin, direction, hit);
  
  if (has_hit) {
    extract_hit_normal(self, hit->bodyID, hit->subShapeID2, origin, direction,
                       hit->fraction, &normal);
  }

  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_END_ALLOW_THREADS
  

  if (!has_hit) {
    goto exit;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BodyHandle handle = (BodyHandle)JPH_BodyInterface_GetUserData(
      self->body_interface, hit->bodyID);
  uint32_t slot = (uint32_t)(handle & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(handle >> 32);

  if (slot < self->slot_capacity && self->generations[slot] == gen &&
      self->slot_states[slot] == SLOT_ALIVE) {
    result = Py_BuildValue("Kf(fff)", handle, hit->fraction, normal.x, normal.y,
                           normal.z);
  }
  SHADOW_UNLOCK(&self->shadow_lock);

exit:
  // --- SIGNALING CHANGE HERE ---
  end_query_scope(self);

  return result ? result : Py_None;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static PyObject *PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                            PyObject *const *args, 
                                            size_t nargsf, 
                                            PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    Py_buffer b_starts = {0}, b_dirs = {0};
    float max_dist = 1000.0f;

    // --- 1. THE FAST PATH: Positional Only (99% of calls) ---
    if (LIKELY(kwnames == NULL && (nargs == 2 || nargs == 3))) {
        if (UNLIKELY(PyObject_GetBuffer(args[0], &b_starts, PyBUF_SIMPLE) < 0)) return NULL;
        if (UNLIKELY(PyObject_GetBuffer(args[1], &b_dirs, PyBUF_SIMPLE) < 0)) {
            PyBuffer_Release(&b_starts); return NULL;
        }
        if (nargs == 3) {
            max_dist = (float)PyFloat_AsDouble(args[2]);
            if (UNLIKELY(PyErr_Occurred())) goto fail_buffers;
        }
    } 
    else {
        // --- 2. THE SLOW PATH: Keywords or Count Mismatch ---
        // We fallback to the standard public API. This is only called once per batch.
        static char *kwlist[] = {"starts", "directions", "max_dist", NULL};
        
        // Pack positional args into a tuple for the old API
        PyObject *temp_tuple = PyTuple_New(nargs);
        if (!temp_tuple) return NULL;
        for (Py_ssize_t i = 0; i < nargs; i++) {
            Py_INCREF(args[i]);
            PyTuple_SET_ITEM(temp_tuple, i, args[i]);
        }

        // Convert kwnames (tuple) to a temporary dict (required by ParseTupleAndKeywords)
        PyObject *temp_dict = NULL;
        if (kwnames) {
            temp_dict = PyDict_New();
            Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
            for (Py_ssize_t i = 0; i < nkw; i++) {
                PyDict_SetItem(temp_dict, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }

        int ok = PyArg_ParseTupleAndKeywords(temp_tuple, temp_dict, "y*y*|f", kwlist, 
                                             &b_starts, &b_dirs, &max_dist);
        Py_XDECREF(temp_dict);
        Py_DECREF(temp_tuple);
        if (!ok) return NULL;
    }

    // --- 3. PRE-COMPUTATION & SIMD HOISTING ---
    if (UNLIKELY(b_starts.len != b_dirs.len || (b_starts.len % 12 != 0))) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        goto fail_buffers;
    }

    size_t count = b_starts.len / 12;
    PyObject *result_bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * sizeof(RayCastBatchResult)));
    if (UNLIKELY(!result_bytes)) goto fail_buffers;

    const float *CULV_RESTRICT f_starts = (const float *)b_starts.buf;
    const float *CULV_RESTRICT f_dirs   = (const float *)b_dirs.buf;
    RayCastBatchResult *CULV_RESTRICT results = (RayCastBatchResult *)PyBytes_AsString(result_bytes);
    
    const float inv_eps = 1e-12f;
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);

    // --- 4. LOCK & PHASE GUARD ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_IF_STEP_PENDING(self);
    atomic_fetch_add(&self->active_queries, 1);
    SHADOW_UNLOCK(&self->shadow_lock);

    Py_BEGIN_ALLOW_THREADS
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    // Filter allocation (reusable across batch)
    JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(NULL);
    JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(NULL);
    JPH_BodyFilter *body_f = JPH_BodyFilter_Create(NULL);

    // --- 5. THE HOT LOOP ---
    for (size_t i = 0; i < count; i++) {
        results[i].handle = 0; // Faster than memset(results[i], 0, 48)

        size_t off = i * 3;
        float dx = f_dirs[off], dy = f_dirs[off+1], dz = f_dirs[off+2];
        float mag_sq = dx*dx + dy*dy + dz*dz;

        if (mag_sq < inv_eps) continue;

        // Vectorizable block (sqrt + mul)
        float scale = max_dist / sqrtf(mag_sq);
        JPH_Vec3 v_dir = { dx * scale, dy * scale, dz * scale };
        JPH_RVec3 v_ori = { (double)f_starts[off], (double)f_starts[off+1], (double)f_starts[off+2] };

        JPH_RayCastResult hit;
        if (JPH_NarrowPhaseQuery_CastRay(query, &v_ori, &v_dir, &hit, bp_f, obj_f, body_f)) {
            RayCastBatchResult *res = &results[i];
            res->handle = JPH_BodyInterface_GetUserData(self->body_interface, hit.bodyID);
            res->fraction = hit.fraction;
            res->subShapeID = hit.subShapeID2;

            // Direct Material Lookup (Bypasses Python handle logic)
            uint32_t slot = (uint32_t)(res->handle & 0xFFFFFFFF);
            res->material_id = self->material_ids[self->slot_to_dense[slot]];

            // Normal Extraction (Requires JPH Body Lock)
            JPH_BodyLockRead lock;
            JPH_BodyLockInterface_LockRead(lock_iface, hit.bodyID, &lock);
            if (lock.body) {
                JPH_RVec3 hit_p = { v_ori.x + (double)v_dir.x * (double)hit.fraction, 
                                    v_ori.y + (double)v_dir.y * (double)hit.fraction, 
                                    v_ori.z + (double)v_dir.z * (double)hit.fraction };
                JPH_Vec3 norm;
                JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, hit.subShapeID2, &hit_p, &norm);
                res->nx = norm.x; res->ny = norm.y; res->nz = norm.z;
                res->px = (float)hit_p.x; res->py = (float)hit_p.y; res->pz = (float)hit_p.z;
            }
            JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);
        }
    }

    JPH_BodyFilter_Destroy(body_f);
    JPH_BroadPhaseLayerFilter_Destroy(bp_f);
    JPH_ObjectLayerFilter_Destroy(obj_f);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

    // --- 6. CLEANUP ---
    end_query_scope(self);
    PyBuffer_Release(&b_starts);
    PyBuffer_Release(&b_dirs);
    return result_bytes;

fail_buffers:
    if (b_starts.obj) PyBuffer_Release(&b_starts);
    if (b_dirs.obj) PyBuffer_Release(&b_dirs);
    return NULL;
}

PyObject *PhysicsWorld_shapecast(PhysicsWorldObject *self, PyObject *args,
                                 PyObject *kwds) {
  int shape_type = 0;
  float px; float py; float pz;
  float rx; float ry; float rz; float rw;
  float dx; float dy; float dz = 0.0f;
  PyObject *py_size = NULL;
  uint64_t ignore_h = 0;
  static char *kwlist[] = {"shape", "pos",    "rot", "dir",
                           "size",  "ignore", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i(fff)(ffff)(fff)O|K", kwlist,
                                   &shape_type, &px, &py, &pz, &rx, &ry, &rz,
                                   &rw, &dx, &dy, &dz, &py_size, &ignore_h)) {
    return NULL;
  }

  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f) {
    Py_RETURN_NONE;
  }
  float s[4];
  parse_shape_params(py_size, s);

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  JPH_Shape *shape = find_or_create_shape(self, shape_type, s);
  if (!shape) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters");
  }

  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  JPH_BodyID ignore_bid = 0;
  uint32_t ignore_slot;
  if (ignore_h && unpack_handle(self, ignore_h, &ignore_slot)) {
    ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- PREP VECTORS ---
  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, &(JPH_Quat){rx, ry, rz, rw},
                                &(JPH_RVec3){px, py, pz});
  JPH_Vec3 sweep_dir = {dx, dy, dz};

  CastShapeContext ctx = {.has_hit = false};
  ctx.hit.fraction = 1.0f;

  
  Py_BEGIN_ALLOW_THREADS
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

  shapecast_execute_internal(self, shape, transform, &sweep_dir, ignore_bid,
                             &ctx);
  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_END_ALLOW_THREADS
  
  // --- GIL RE-ACQUIRED ---

  PyObject *result = NULL;
  if (ctx.has_hit) {
    float nx = -ctx.hit.penetrationAxis.x;
    float ny = -ctx.hit.penetrationAxis.y;
    float nz = -ctx.hit.penetrationAxis.z;
    float n_len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (n_len > 1e-6f) {
      nx /= n_len; ny /= n_len; nz /= n_len;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BodyHandle h = (BodyHandle)JPH_BodyInterface_GetUserData(
        self->body_interface, ctx.hit.bodyID2);
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    if (slot < self->slot_capacity &&
        self->generations[slot] == (uint32_t)(h >> 32) &&
        self->slot_states[slot] == SLOT_ALIVE) {
      result = Py_BuildValue(
          "Kf(fff)(fff)", h, ctx.hit.fraction, ctx.hit.contactPointOn2.x,
          ctx.hit.contactPointOn2.y, ctx.hit.contactPointOn2.z, nx, ny, nz);
    }
    SHADOW_UNLOCK(&self->shadow_lock);
  }

  // --- SIGNALING CHANGE HERE ---
  end_query_scope(self);
  
  return result ? result : Py_None;
}
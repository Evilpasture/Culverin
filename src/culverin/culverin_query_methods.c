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
  float sx, sy, sz;
  float dx, dy, dz;
  float max_dist = 1000.0f;
  uint64_t ignore_h = 0;
  static char *kwlist[] = {"start", "direction", "max_dist", "ignore", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|fK", kwlist, 
                                   &sx, &sy, &sz, &dx, &dy, &dz, 
                                   &max_dist, &ignore_h)) {
    return NULL;
  }

  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f) {
    Py_RETURN_NONE;
  }
  
  float mag = sqrtf(mag_sq);
  float scale = max_dist / mag;

  JPH_STACK_ALLOC(JPH_RVec3, origin);
  origin->x = sx; origin->y = sy; origin->z = sz;
  
  JPH_STACK_ALLOC(JPH_Vec3, direction);
  direction->x = dx * scale; 
  direction->y = dy * scale; 
  direction->z = dz * scale;
  
  JPH_STACK_ALLOC(JPH_RayCastResult, hit);
  hit->bodyID = JPH_INVALID_BODY_ID;
  hit->fraction = 1.0f;
  hit->subShapeID2 = 0;

  // Resolve ignore body under lock
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);

  JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
  if (ignore_h != 0) {
      uint32_t ignore_slot;
      if ((int)unpack_handle(self, ignore_h, &ignore_slot) && 
          self->slot_states[ignore_slot] == SLOT_ALIVE) {
          uint32_t dense = self->slot_to_dense[ignore_slot];
          
          // Bounds check against CAPACITY, not count
          if (dense < self->capacity) {
              ignore_bid = self->body_ids[dense];
          }
      }
  }

  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
  SHADOW_UNLOCK(&self->shadow_lock);
  // Execute raycast (GIL released, Jolt lock held)
  bool has_hit = false;
  JPH_Vec3 normal = {0, 0, 0};
  BodyHandle hit_handle = 0;
  float hit_fraction = 0.0f;
  
  Py_BEGIN_ALLOW_THREADS
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

  has_hit = execute_raycast_query(self, ignore_bid, origin, direction, hit);
  
  if (has_hit) {
    extract_hit_normal(self, hit->bodyID, hit->subShapeID2, origin, direction,
                       hit->fraction, &normal);
    // Get handle while holding Jolt lock
    hit_handle = (BodyHandle)JPH_BodyInterface_GetUserData(
        self->body_interface, hit->bodyID);
    hit_fraction = hit->fraction;
  }

  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  
  // Decrement query counter before re-acquiring GIL
  end_query_scope(self);
  
  Py_END_ALLOW_THREADS

  // Build result (GIL held)
  if (!has_hit) {
    Py_RETURN_NONE;
  }

  PyObject *result = NULL;
  SHADOW_LOCK(&self->shadow_lock);
  
  uint32_t slot = (uint32_t)(hit_handle & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(hit_handle >> 32);

  if (slot < self->slot_capacity && 
      self->generations[slot] == gen &&
      self->slot_states[slot] == SLOT_ALIVE) {
    
    result = Py_BuildValue("Kf(fff)", 
                           hit_handle, 
                           hit_fraction, 
                           normal.x, 
                           normal.y,
                           normal.z);
  }
  
  SHADOW_UNLOCK(&self->shadow_lock);

  if (!result) {
    Py_RETURN_NONE;
  }
  return result;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                            PyObject *const *args, 
                                            size_t nargsf, 
                                            PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    Py_buffer b_starts = {0}, b_dirs = {0};
    float max_dist = 1000.0f;

    // --- 1. ARGUMENT PARSING ---
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
        static char *kwlist[] = {"starts", "directions", "max_dist", NULL};
        PyObject *temp_tuple = PyTuple_New(nargs);
        if (!temp_tuple) return NULL;
        for (Py_ssize_t i = 0; i < nargs; i++) {
            Py_INCREF(args[i]);
            PyTuple_SET_ITEM(temp_tuple, i, args[i]);
        }
        PyObject *temp_dict = NULL;
        if (kwnames) {
            temp_dict = PyDict_New();
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); i++) {
                PyDict_SetItem(temp_dict, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }
        int ok = PyArg_ParseTupleAndKeywords(temp_tuple, temp_dict, "y*y*|f", kwlist, 
                                             &b_starts, &b_dirs, &max_dist);
        Py_XDECREF(temp_dict); Py_DECREF(temp_tuple);
        if (!ok) return NULL;
    }

    if (UNLIKELY(b_starts.len != b_dirs.len || (b_starts.len % 12 != 0))) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        goto fail_buffers;
    }

    size_t count = b_starts.len / 12;
    
    // Prevent overflow and excessive memory usage
    if (count > 10000000) {  // 10M rays max (480MB result buffer)
        PyErr_SetString(PyExc_ValueError, "Batch size exceeds 10M rays");
        goto fail_buffers;
    }
    
    PyObject *result_bytes = PyBytes_FromStringAndSize(
        NULL, 
        (Py_ssize_t)(count * sizeof(RayCastBatchResult))
    );
    if (UNLIKELY(!result_bytes)) goto fail_buffers;

    // --- 2. PHASE GUARD & SNAPSHOT ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_IF_STEP_PENDING(self);

    // SAFETY: We hold g_jph_trampoline_lock during execution, which prevents
    // flush_commands_internal from reallocating these arrays.
    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;
    const uint32_t *CULV_RESTRICT mats = self->material_ids;
    const size_t slot_cap = self->slot_capacity;
    const size_t body_cap = self->capacity;  // Use capacity, not count!

    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 3. JOLT EXECUTION ---
    const float *CULV_RESTRICT f_starts = (const float *)b_starts.buf;
    const float *CULV_RESTRICT f_dirs   = (const float *)b_dirs.buf;
    RayCastBatchResult *CULV_RESTRICT results = (RayCastBatchResult *)PyBytes_AsString(result_bytes);
    
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyInterface *bi = self->body_interface;

    Py_BEGIN_ALLOW_THREADS
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    for (size_t i = 0; i < count; i++) {
        results[i].handle = 0; 
        size_t off = i * 3;
        float dx = f_dirs[off], dy = f_dirs[off+1], dz = f_dirs[off+2];
        float mag_sq = dx*dx + dy*dy + dz*dz;

        if (mag_sq < 1e-12f) continue;

        float scale = max_dist / sqrtf(mag_sq);
        JPH_Vec3 v_dir = { dx * scale, dy * scale, dz * scale };
        JPH_RVec3 v_ori = { 
            (double)f_starts[off], 
            (double)f_starts[off+1], 
            (double)f_starts[off+2] 
        };

        JPH_RayCastResult hit;
        hit.bodyID = JPH_INVALID_BODY_ID;
        hit.fraction = 1.0f;
        hit.subShapeID2 = 0;

        if (JPH_NarrowPhaseQuery_CastRay(query, &v_ori, &v_dir, &hit, NULL, NULL, NULL)) {
            uint64_t h = JPH_BodyInterface_GetUserData(bi, hit.bodyID);
            if (h != 0) {
                RayCastBatchResult *res = &results[i];
                res->handle = h;
                res->fraction = hit.fraction;
                res->subShapeID = hit.subShapeID2;

                // Note: We don't validate generation for performance.
                // Stale handles will fail later validation.
                uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
                if (slot < slot_cap) {
                    uint32_t dense = s2d[slot];
                    if (dense < body_cap && mats) {  // Check capacity, not count!
                        res->material_id = mats[dense];
                    }
                }

                JPH_BodyLockRead j_lock;
                JPH_BodyLockInterface_LockRead(lock_iface, hit.bodyID, &j_lock);
                if (j_lock.body) {
                    JPH_RVec3 hit_p = { 
                        v_ori.x + (double)v_dir.x * (double)hit.fraction, 
                        v_ori.y + (double)v_dir.y * (double)hit.fraction, 
                        v_ori.z + (double)v_dir.z * (double)hit.fraction 
                    };
                    JPH_Vec3 norm;
                    JPH_Body_GetWorldSpaceSurfaceNormal(j_lock.body, hit.subShapeID2, &hit_p, &norm);
                    res->nx = norm.x; res->ny = norm.y; res->nz = norm.z;
                    res->px = (float)hit_p.x; res->py = (float)hit_p.y; res->pz = (float)hit_p.z;
                }
                JPH_BodyLockInterface_UnlockRead(lock_iface, &j_lock);
            }
        }
    }

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    
    // Decrement query counter BEFORE re-acquiring GIL
    end_query_scope(self);
    
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&b_starts); 
    PyBuffer_Release(&b_dirs);
    return result_bytes;

fail_buffers:
    // Note: We only reach here before incrementing active_queries,
    // so we don't need to call end_query_scope.
    if (b_starts.obj) PyBuffer_Release(&b_starts);
    if (b_dirs.obj) PyBuffer_Release(&b_dirs);
    return NULL;
}

PyObject *PhysicsWorld_shapecast(PhysicsWorldObject *self, PyObject *args,
                                 PyObject *kwds) {
  int shape_type = 0;
  JPH_Real px, py, pz;
  float rx, ry, rz, rw;
  float dx, dy, dz;
  PyObject *py_size = NULL;
  uint64_t ignore_h = 0;
  
  static char *kwlist[] = {"shape", "pos", "rot", "dir", "size", "ignore", NULL};

  // 1. ARGUMENT PARSING (GIL HELD)
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i(ddd)(ffff)(fff)O|K", kwlist,
                                   &shape_type, &px, &py, &pz, &rx, &ry, &rz, &rw,
                                   &dx, &dy, &dz, &py_size, &ignore_h)) {
    return NULL;
  }

  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f) {
    Py_RETURN_NONE;
  }
  
  float s[4];
  parse_body_size(py_size, s);
  
  CastShapeContext ctx = {0};
  uint64_t hit_handle = 0;
  bool has_valid_hit = false;
  
  // 2. QUERY SETUP AND EXECUTION (GIL Released, Jolt Lock Held)
  Py_BEGIN_ALLOW_THREADS

  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  SHADOW_LOCK(&self->shadow_lock);

  JPH_Shape *shape = find_or_create_shape_locked(self, shape_type, s);
  
  JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
  if (ignore_h) {
      uint32_t slot;
      if ((int)unpack_handle(self, ignore_h, &slot) && 
          self->slot_states[slot] == SLOT_ALIVE) {
          ignore_bid = self->body_ids[self->slot_to_dense[slot]];
      }
  }

  if (!shape) {
      SHADOW_UNLOCK(&self->shadow_lock);
      NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
      Py_BLOCK_THREADS;
      return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters");
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  // Increment query counter AFTER validation
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);

  // 3. EXECUTE QUERY
  JPH_RMat4 transform;
  JPH_RVec3 pos = {px, py, pz};
  JPH_Quat rot = {rx, ry, rz, rw};
  JPH_RMat4_RotationTranslation(&transform, &rot, &pos);
  
  JPH_Vec3 sweep_dir = {dx, dy, dz};
  ctx.has_hit = false;
  ctx.hit.fraction = 1.0f;
  
  shapecast_execute_internal(self, shape, &transform, &sweep_dir, ignore_bid, &ctx);

  // 4. RESOLVE HIT HANDLE (While holding Jolt lock)
  if (ctx.has_hit) {
      uint64_t h_raw = JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
      hit_handle = h_raw;
      has_valid_hit = true;
  }

  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

  // 5. SIGNAL QUERY END (Before re-acquiring GIL)
  int prev = atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);
  if (prev == 1) {
      NATIVE_MUTEX_LOCK(self->step_sync.mutex);
      NATIVE_COND_BROADCAST(self->step_sync.cond);
      NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
  }

  Py_END_ALLOW_THREADS

  // 6. BUILD RESULT (GIL Re-acquired)
  PyObject *result = NULL;
  
  if (has_valid_hit) {
      // Normalize normal
      float nx = -ctx.hit.penetrationAxis.x;
      float ny = -ctx.hit.penetrationAxis.y;
      float nz = -ctx.hit.penetrationAxis.z;
      
      float n_len = sqrtf(nx*nx + ny*ny + nz*nz);
      if (n_len > 1e-6f) {
          float inv_len = 1.0f / n_len;
          nx *= inv_len; ny *= inv_len; nz *= inv_len;
      }

      // Check shadow data under lock
      SHADOW_LOCK(&self->shadow_lock);
      
      uint32_t slot = (uint32_t)(hit_handle & 0xFFFFFFFF);
      uint32_t gen = (uint32_t)(hit_handle >> 32);

      if (slot < self->slot_capacity && 
          self->generations[slot] == gen && 
          self->slot_states[slot] == SLOT_ALIVE) {
          
          // Create all atomic objects first
          PyObject *handle_obj = PyLong_FromUnsignedLongLong(hit_handle);
          PyObject *fraction_obj = PyFloat_FromDouble(ctx.hit.fraction);
          PyObject *px = PyFloat_FromDouble(ctx.hit.contactPointOn2.x);
          PyObject *py = PyFloat_FromDouble(ctx.hit.contactPointOn2.y);
          PyObject *pz = PyFloat_FromDouble(ctx.hit.contactPointOn2.z);
          PyObject *nx_obj = PyFloat_FromDouble(nx);
          PyObject *ny_obj = PyFloat_FromDouble(ny);
          PyObject *nz_obj = PyFloat_FromDouble(nz);
          
          // Check for allocation failures
          if (!handle_obj || !fraction_obj || !px || !py || !pz || 
              !nx_obj || !ny_obj || !nz_obj) {
              Py_XDECREF(handle_obj); Py_XDECREF(fraction_obj);
              Py_XDECREF(px); Py_XDECREF(py); Py_XDECREF(pz);
              Py_XDECREF(nx_obj); Py_XDECREF(ny_obj); Py_XDECREF(nz_obj);
              SHADOW_UNLOCK(&self->shadow_lock);
              return NULL;
          }
          
          PyObject *pos_tup = PyTuple_Pack(3, px, py, pz);
          PyObject *norm_tup = PyTuple_Pack(3, nx_obj, ny_obj, nz_obj);
          
          // Decrement the float refs (tuple now owns them)
          Py_DECREF(px); Py_DECREF(py); Py_DECREF(pz);
          Py_DECREF(nx_obj); Py_DECREF(ny_obj); Py_DECREF(nz_obj);
          
          if (!pos_tup || !norm_tup) {
              Py_XDECREF(pos_tup); Py_XDECREF(norm_tup);
              Py_DECREF(handle_obj); Py_DECREF(fraction_obj);
              SHADOW_UNLOCK(&self->shadow_lock);
              return NULL;
          }
          
          result = PyTuple_Pack(4, handle_obj, fraction_obj, pos_tup, norm_tup);
          
          // Decrement all intermediate objects
          Py_DECREF(handle_obj);
          Py_DECREF(fraction_obj);
          Py_DECREF(pos_tup);
          Py_DECREF(norm_tup);
          
          if (!result) {
              SHADOW_UNLOCK(&self->shadow_lock);
              return NULL;
          }
      }
      SHADOW_UNLOCK(&self->shadow_lock);
  }

  if (!result) {
      Py_RETURN_NONE;
  }
  return result;
}
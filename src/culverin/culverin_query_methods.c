#include "culverin_query_methods.h"
#include "culverin_filters.h"
#include "culverin_parsers.h"
#include "culverin_compiler_specifics.h"

// --- Helper: Signal End of Query ---
// This is crucial for the Condition Variable approach.
// If we are the last query to finish, we must wake up the physics stepper.
static void end_query_scope(PhysicsWorldObject *self) {
  // 1. Lock native mutex first
  NATIVE_MUTEX_LOCK(self->step_sync.mutex);

  // 2. Decrement and check
  // We use fetch_sub; if it was 1, it's now 0.
  uint32_t prev =
      atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_acq_rel);

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

  // --- EXECUTION ---

  Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide =
                                                  filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide =
                                               filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  JPH_BodyFilter_Procs bf_procs = {.ShouldCollide = filter_true_body};
  body_filter = JPH_BodyFilter_Create(NULL);
  JPH_BodyFilter_SetProcs(&bf_procs);

  const JPH_NarrowPhaseQuery *nq =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self, PyObject *args,
                                    PyObject *kwds) {
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

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x;
  box->min.y = min_y;
  box->min.z = min_z;
  box->max.x = max_x;
  box->max.y = max_y;
  box->max.z = max_z;

  Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

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

PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self,
                               PyObject *const *args, size_t nargsf,
                               PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_start = NULL, *o_dir = NULL;
    float max_dist = 1000.0f;
    uint64_t ignore_h = 0;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[Raycast_COUNT];
    targets[IDX_RAY_START] = &o_start;
    targets[IDX_RAY_DIR]   = &o_dir;
    targets[IDX_RAY_DIST]  = &max_dist;
    targets[IDX_RAY_IGN]   = &ignore_h;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &RaycastParser, targets)) {
        return NULL;
    }

    // 3. VECTOR EXTRACTION (Precision Safe - Outside Lock)
    JPH_Real sx, sy, sz; // World Pos (Double/Float)
    float dx, dy, dz;    // Direction (Float)
    
    if (!parse_vec3_direct(o_start, &sx, &sy, &sz)) return NULL;
    if (!parse_vec3_direct(o_dir, &dx, &dy, &dz))   return NULL;

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < 1e-9f)) {
        Py_RETURN_NONE;
    }

    // Prepare Jolt Stack structures
    float mag = sqrtf(mag_sq);
    float scale = max_dist / mag;

    JPH_STACK_ALLOC(JPH_RVec3, origin);
    *origin = (JPH_RVec3){sx, sy, sz};

    JPH_STACK_ALLOC(JPH_Vec3, direction);
    *direction = (JPH_Vec3){dx * scale, dy * scale, dz * scale};

    JPH_STACK_ALLOC(JPH_RayCastResult, hit);
    hit->bodyID = JPH_INVALID_BODY_ID;
    hit->fraction = 1.0f;
    hit->subShapeID2 = 0;

    // 4. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    
    // Priority Guard: Raycasts must wait if a Step is running or about to start
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_IF_STEP_PENDING(self);

    JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
    if (ignore_h != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, (BodyHandle)ignore_h, &ignore_slot) &&
            self->slot_states[ignore_slot] == SLOT_ALIVE) {
            ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
        }
    }

    // Mark query as active to prevent resize/dealloc during execution
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    
    // Release shadow lock early! Jolt core handles its own internal locking.
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (No GIL, Jolt Trampoline Lock)
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
        // Get our Python-side BodyHandle from Jolt's UserData
        hit_handle = (BodyHandle)JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
        hit_fraction = hit->fraction;
    }

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    
    // Unmark query
    int prev = atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);
    if (prev == 1) {
        NATIVE_MUTEX_LOCK(self->step_sync.mutex);
        NATIVE_COND_BROADCAST(self->step_sync.cond);
        NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
    }
    Py_END_ALLOW_THREADS

    // 6. RESULT PHASE (Re-verify handle integrity)
    if (!has_hit) {
        Py_RETURN_NONE;
    }

    PyObject *result = NULL;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = (uint32_t)(hit_handle & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(hit_handle >> 32);

    // Final safety: Ensure the hit body wasn't recycled while we were building result
    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {

        result = Py_BuildValue("Kf(fff)", hit_handle, hit_fraction, 
                               normal.x, normal.y, normal.z);
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    return result ? result : Py_None;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                     PyObject *const *args, size_t nargsf,
                                     PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_starts = NULL;
    PyObject *o_dirs = NULL;
    float max_dist = 1000.0f;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[RayBatch_COUNT];
    targets[IDX_RB_STARTS] = &o_starts;
    targets[IDX_RB_DIRS]   = &o_dirs;
    targets[IDX_RB_DIST]   = &max_dist;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &RayBatchParser, targets)) {
        return NULL;
    }

    // 3. BUFFER EXTRACTION & VALIDATION (Outside Lock)
    Py_buffer b_starts = {0};
    Py_buffer b_dirs = {0};

    if (UNLIKELY(PyObject_GetBuffer(o_starts, &b_starts, PyBUF_SIMPLE) < 0)) return NULL;
    if (UNLIKELY(PyObject_GetBuffer(o_dirs, &b_dirs, PyBUF_SIMPLE) < 0)) {
        PyBuffer_Release(&b_starts);
        return NULL;
    }

    if (UNLIKELY(b_starts.len != b_dirs.len || (b_starts.len % 12 != 0))) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch: expected float32 triples of equal length");
        goto fail_buffers;
    }

    size_t count = b_starts.len / 12;
    if (count == 0) {
        PyBuffer_Release(&b_starts); PyBuffer_Release(&b_dirs);
        return PyBytes_FromStringAndSize(NULL, 0);
    }

    if (UNLIKELY(count > 10000000)) { // 10M Limit
        PyErr_SetString(PyExc_ValueError, "Batch size exceeds 10M rays");
        goto fail_buffers;
    }

    PyObject *result_bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * sizeof(RayCastBatchResult)));
    if (UNLIKELY(!result_bytes)) goto fail_buffers;

    // 4. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    
    // Priority Guard: Wait if physics is stepping or about to step
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_IF_STEP_PENDING(self);

    // Snapshot pointers for thread-safe narrow-phase lookup
    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;
    const uint32_t *CULV_RESTRICT mats = self->material_ids;
    const size_t slot_cap = self->slot_capacity;
    const size_t body_cap = self->capacity;

    // Mark query as active
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (No GIL, Jolt Trampoline Lock)
    const float *CULV_RESTRICT f_starts = (const float *)b_starts.buf;
    const float *CULV_RESTRICT f_dirs = (const float *)b_dirs.buf;
    RayCastBatchResult *CULV_RESTRICT results = (RayCastBatchResult *)PyBytes_AsString(result_bytes);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyInterface *bi = self->body_interface;

    Py_BEGIN_ALLOW_THREADS 
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    for (size_t i = 0; i < count; i++) {
        size_t off = i * 3;
        results[i].handle = 0; // Initialize as "no hit"

        float dx = f_dirs[off], dy = f_dirs[off + 1], dz = f_dirs[off + 2];
        float mag_sq = dx * dx + dy * dy + dz * dz;
        if (mag_sq < 1e-12f) continue;

        float scale = max_dist / sqrtf(mag_sq);
        JPH_Vec3 v_dir = {dx * scale, dy * scale, dz * scale};
        JPH_RVec3 v_ori = {(double)f_starts[off], (double)f_starts[off + 1], (double)f_starts[off + 2]};

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

                // Lookup Material ID from Shadow Buffer snapshot
                uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
                if (slot < slot_cap) {
                    uint32_t dense = s2d[slot];
                    if (dense < body_cap && mats) res->material_id = mats[dense];
                }

                // Normal & Position Extraction (Requires Body Lock)
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
    // Decrement and check if we were the last active query
    int prev_queries = atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);
    
    if (prev_queries == 1) {
        // We were the last ones! We MUST wake up the world.step() thread.
        NATIVE_MUTEX_LOCK(self->step_sync.mutex);
        NATIVE_COND_BROADCAST(self->step_sync.cond);
        NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
    }
    Py_END_ALLOW_THREADS

    // 6. CLEANUP
    PyBuffer_Release(&b_starts);
    PyBuffer_Release(&b_dirs);
    return result_bytes;

fail_buffers:
    if (b_starts.obj) PyBuffer_Release(&b_starts);
    if (b_dirs.obj) PyBuffer_Release(&b_dirs);
    return NULL;
}

PyObject *PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                 PyObject *const *args, size_t nargsf,
                                 PyObject *kwnames) {
    // 1. DEFAULT VALUES
    int shape_type = 0;
    PyObject *o_pos = NULL, *o_rot = NULL, *o_dir = NULL, *o_size = NULL;
    uint64_t ignore_h = 0;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[Shapecast_COUNT];
    targets[IDX_SC_SHAPE]  = &shape_type;
    targets[IDX_SC_POS]    = &o_pos;
    targets[IDX_SC_ROT]    = &o_rot;
    targets[IDX_SC_DIR]    = &o_dir;
    targets[IDX_SC_SIZE]   = &o_size;
    targets[IDX_SC_IGNORE] = &ignore_h;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &ShapecastParser, targets)) {
        return NULL;
    }

    // 3. EXTRACTION (Outside Lock)
    JPH_Real px, py, pz;
    float rx, ry, rz, rw;
    float dx, dy, dz;
    float s[4];

    if (!parse_vec3_direct(o_pos, &px, &py, &pz)) return NULL;
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) return NULL;
    if (!parse_vec3_direct(o_dir, &dx, &dy, &dz)) return NULL;
    parse_body_size(o_size, s);

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < 1e-9f)) Py_RETURN_NONE;

    CastShapeContext ctx = {0};
    uint64_t hit_handle = 0;
    bool has_valid_hit = false;

    // 4. JOLT RESOLUTION & EXECUTION (No GIL)
    Py_BEGIN_ALLOW_THREADS
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    SHADOW_LOCK(&self->shadow_lock);

    // Block if the world is currently stepping
    BLOCK_UNTIL_NOT_STEPPING(self);

    JPH_Shape *shape = find_or_create_shape_locked(self, shape_type, s);
    JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
    if (ignore_h) {
        uint32_t slot;
        if (unpack_handle(self, (BodyHandle)ignore_h, &slot) && 
            self->slot_states[slot] == SLOT_ALIVE) {
            ignore_bid = self->body_ids[self->slot_to_dense[slot]];
        }
    }

    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
        Py_BLOCK_THREADS; // Re-acquire GIL before returning error
        return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters");
    }
    
    // Mark query active to prevent world destruction/resize
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // Setup Sweep
    JPH_RMat4 transform;
    JPH_RVec3 v_pos = {px, py, pz};
    JPH_Quat v_rot = {rx, ry, rz, rw};
    JPH_RMat4_RotationTranslation(&transform, &v_rot, &v_pos);
    JPH_Vec3 sweep_dir = {dx, dy, dz};

    ctx.has_hit = false;
    ctx.hit.fraction = 1.0f;

    shapecast_execute_internal(self, shape, &transform, &sweep_dir, ignore_bid, &ctx);

    if (ctx.has_hit) {
        hit_handle = JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
        has_valid_hit = (hit_handle != 0);
    }

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    
    // Unmark query and signal end
    int prev =
      atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);
    if (prev == 1) {
      NATIVE_MUTEX_LOCK(self->step_sync.mutex);
      NATIVE_COND_BROADCAST(self->step_sync.cond);
      NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
    }
    Py_END_ALLOW_THREADS

    // 5. RESULT CONSTRUCTION (GIL Held)
    if (!has_valid_hit) Py_RETURN_NONE;

    // Normal calculation (Geometric processing)
    float nx = -ctx.hit.penetrationAxis.x, ny = -ctx.hit.penetrationAxis.y, nz = -ctx.hit.penetrationAxis.z;
    float n_len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (n_len > 1e-6f) {
        float inv = 1.0f / n_len; nx *= inv; ny *= inv; nz *= inv;
    }

    PyObject *result = NULL;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = (uint32_t)(hit_handle & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(hit_handle >> 32);

    // Verify handle is still valid in our shadow system
    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {

        /**
         * Py_BuildValue is much safer and faster than manual packing.
         * Format: 
         * K: unsigned long long (handle)
         * f: float (fraction)
         * (fff): tuple of 3 floats (position)
         * (fff): tuple of 3 floats (normal)
         */
        result = Py_BuildValue("Kf(fff)(fff)", 
            hit_handle, 
            (double)ctx.hit.fraction,
            (double)ctx.hit.contactPointOn2.x, (double)ctx.hit.contactPointOn2.y, (double)ctx.hit.contactPointOn2.z,
            (double)nx, (double)ny, (double)nz
        );
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    return result ? result : Py_None;
}
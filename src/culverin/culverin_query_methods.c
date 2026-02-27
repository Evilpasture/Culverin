#include "culverin_query_methods.h"
#include "culverin_arg_indices.h"
#include "culverin_compiler_specifics.h"
#include "culverin_filters.h"
#include "culverin_math.h"
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
        size_t new_cap = ctx->capacity * 2;
        uint64_t *new_ptr;

        if (ctx->is_on_stack) {
            new_ptr = CULV_RAW_MALLOC(new_cap * sizeof(uint64_t));
            if (new_ptr) {
                memcpy(new_ptr, ctx->hits, ctx->count * sizeof(uint64_t));
            }
            ctx->is_on_stack = false;
        } else {
            new_ptr = CULV_RAW_REALLOC(ctx->hits, new_cap * sizeof(uint64_t));
        }

        if (!new_ptr) {
            return;
        }
        ctx->hits     = new_ptr;
        ctx->capacity = new_cap;
    }
    ctx->hits[ctx->count++] = JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);
}

static float OverlapCallback_Narrow(void *context, const JPH_CollideShapeResult *result) {
    overlap_record_hit((OverlapContext *)context, result->bodyID2);
    return 1.0f;
}

static float OverlapCallback_Broad(void *context, const JPH_BodyID result_bid) {
    overlap_record_hit((OverlapContext *)context, result_bid);
    return 1.0f;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_sphere(PhysicsWorldObject *self, PyObject *const *args,
                                      size_t nargsf, PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_center = NULL;
    float radius       = 1.0f;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[OverlapSphere_COUNT];
    targets[IDX_OS_CENTER] = (void *)&o_center;
    targets[IDX_OS_RADIUS] = (void *)&radius;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &OverlapSphereParser, targets)) {
        return NULL;
    }

    // 3. VECTOR EXTRACTION (Outside Lock)
    JPH_Real cx;
    JPH_Real cy;
    JPH_Real cz;
    if (!parse_vec3_direct(o_center, &cx, &cy, &cz)) {
        return NULL;
    }

    PyObject *ret_val = NULL;
    uint64_t small_hit_stack[STACK_ALLOCATE_HITS]; // Pre-allocate 64 hits on the stack
    OverlapContext ctx = {
        .world       = self,
        .hits        = small_hit_stack,
        .count       = 0,
        .capacity    = STACK_ALLOCATE_HITS,
        .is_on_stack = true // Add this flag to your struct
    };

    JPH_Shape *shape                     = NULL;
    JPH_BroadPhaseLayerFilter *bp_filter = NULL;
    JPH_ObjectLayerFilter *obj_filter    = NULL;
    JPH_BodyFilter *body_filter          = NULL;

    // 4. RESOURCE RESOLUTION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    // OPTIMIZATION: Use the shape cache instead of creating a new shape every call
    float s_params[4] = {radius, 0, 0, 0};
    shape             = find_or_create_shape_locked(self, CULV_SHAPE_SPHERE, s_params);

    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Failed to resolve sphere shape");
    }

    // Mark query active
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION (No GIL, No Trampoline Lock)
    Py_BEGIN_ALLOW_THREADS
        // REMOVED: NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

        // Prepare Jolt Stack structures inside ALLOW_THREADS
        JPH_RMat4 transform;
    JPH_RVec3 v_pos = {cx, cy, cz};
    JPH_Quat v_rot  = {0, 0, 0, 1};
    JPH_RMat4_RotationTranslation(&transform, &v_rot, &v_pos);

    JPH_Vec3 scale        = {1.0f, 1.0f, 1.0f};
    JPH_RVec3 base_offset = {0, 0, 0};

    JPH_CollideShapeSettings settings;
    JPH_CollideShapeSettings_Init(&settings);

    bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);

    obj_filter = JPH_ObjectLayerFilter_Create(NULL);

    body_filter = JPH_BodyFilter_Create(NULL);

    const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

    JPH_NarrowPhaseQuery_CollideShape(nq, shape, &scale, &transform, &settings, &base_offset,
                                      OverlapCallback_Narrow, &ctx, bp_filter, obj_filter,
                                      body_filter, NULL);

    // REMOVED: NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // Signal end of query
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (GIL Held)
        ret_val = PyList_New(0);
    if (ret_val) {
        SHADOW_LOCK(&self->shadow_lock);
        for (size_t i = 0; i < ctx.count; i++) {
            uint64_t h = ctx.hits[i];
            uint32_t slot;
            if ((int)unpack_handle(self, h, &slot) && self->slot_states[slot] == SLOT_ALIVE) {
                PyObject *py_h = PyLong_FromUnsignedLongLong(h);
                if (py_h) {
                    PyList_Append(ret_val, py_h);
                    Py_DECREF(py_h);
                }
            }
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }

    // Cleanup resources
    if (bp_filter) {
        JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    }
    if (obj_filter) {
        JPH_ObjectLayerFilter_Destroy(obj_filter);
    }
    if (body_filter) {
        JPH_BodyFilter_Destroy(body_filter);
    }
    if (ctx.hits && !ctx.is_on_stack) {
        CULV_RAW_FREE(ctx.hits);
    }

    return ret_val;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_aabb(PhysicsWorldObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_min = NULL;
    PyObject *o_max = NULL;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[OverlapAABB_COUNT];
    targets[IDX_OA_MIN] = &o_min;
    targets[IDX_OA_MAX] = &o_max;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &OverlapAABBParser, targets)) {
        return NULL;
    }

    // 3. VECTOR EXTRACTION (Outside Lock)
    JPH_Real mix, miy, miz, max, may, maz;
    if (!parse_vec3_direct(o_min, &mix, &miy, &miz))
        return NULL;
    if (!parse_vec3_direct(o_max, &max, &may, &maz))
        return NULL;

    PyObject *ret_val = NULL;
    uint64_t small_hit_stack[STACK_ALLOCATE_HITS]; // Pre-allocate 64 hits on the stack
    OverlapContext ctx = {
        .world       = self,
        .hits        = small_hit_stack,
        .count       = 0,
        .capacity    = STACK_ALLOCATE_HITS,
        .is_on_stack = true // Add this flag to your struct
    };
    JPH_BroadPhaseLayerFilter *bp_filter = NULL;
    JPH_ObjectLayerFilter *obj_filter    = NULL;

    // 4. PHASE GUARD (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    // Mark query active to prevent world mutation during execution
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // Prepare Jolt AABox
    JPH_STACK_ALLOC(JPH_AABox, box);
    box->min = (JPH_Vec3){(float)mix, (float)miy, (float)miz};
    box->max = (JPH_Vec3){(float)max, (float)may, (float)maz};

    // 5. EXECUTION (No GIL, No Trampoline Lock)
    Py_BEGIN_ALLOW_THREADS
        // REMOVED: NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
        bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);

    obj_filter = JPH_ObjectLayerFilter_Create(NULL);

    const JPH_BroadPhaseQuery *bq = JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);

    // Jolt Broadphase queries are thread-safe while the world is not stepping
    JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx, bp_filter, obj_filter);

    // REMOVED: NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // Unmark query and signal the world.step() thread if it's waiting for us
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (GIL Held)
        ret_val = PyList_New(0);
    if (!ret_val) {
        goto query_cleanup;
    }

    SHADOW_LOCK(&self->shadow_lock);
    for (size_t i = 0; i < ctx.count; i++) {
        uint64_t h = ctx.hits[i];
        uint32_t slot;
        // Verify handle remains valid in our shadow registry
        if (unpack_handle(self, h, &slot) && self->slot_states[slot] == SLOT_ALIVE) {
            PyObject *py_h = PyLong_FromUnsignedLongLong(h);
            if (py_h) {
                PyList_Append(ret_val, py_h);
                Py_DECREF(py_h);
            }
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);

query_cleanup:
    // Filter cleanup
    if (bp_filter)
        JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    if (obj_filter)
        JPH_ObjectLayerFilter_Destroy(obj_filter);

    // ctx.hits was allocated using CULV_RAW_REALLOC, which is GIL-safe
    if (ctx.hits && !ctx.is_on_stack) {
        CULV_RAW_FREE(ctx.hits);
    }

    return ret_val;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *const *args, size_t nargsf,
                               PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_start = NULL;
    PyObject *o_dir   = NULL;
    float max_dist    = 1000.0f;
    uint64_t ignore_h = 0;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[Raycast_COUNT];
    targets[IDX_RAY_START] = (void *)&o_start;
    targets[IDX_RAY_DIR]   = (void *)&o_dir;
    targets[IDX_RAY_DIST]  = (void *)&max_dist;
    targets[IDX_RAY_IGN]   = (void *)&ignore_h;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &RaycastParser, targets)) {
        return NULL;
    }

    // 3. VECTOR EXTRACTION (Outside Lock)
    JPH_Real sx, sy, sz;
    float dx, dy, dz;

    if (!parse_vec3_direct(o_start, &sx, &sy, &sz))
        return NULL;
    if (!parse_vec3_direct(o_dir, &dx, &dy, &dz))
        return NULL;

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < 1e-12f)) {
        Py_RETURN_NONE;
    }

    // MATH OPTIMIZATION: Avoid sqrtf/division for normalized vectors
    float scale;
    if (fabsf(mag_sq - 1.0f) < 1e-4f) {
        scale = max_dist;
    } else {
        scale = max_dist * culverin_fast_rsqrt(mag_sq);
    }

    JPH_STACK_ALLOC(JPH_RVec3, origin);
    *origin = (JPH_RVec3){sx, sy, sz};

    JPH_STACK_ALLOC(JPH_Vec3, direction);
    *direction = (JPH_Vec3){dx * scale, dy * scale, dz * scale};

    JPH_STACK_ALLOC(JPH_RayCastResult, hit);
    hit->bodyID      = JPH_INVALID_BODY_ID;
    hit->fraction    = 1.0f;
    hit->subShapeID2 = 0;

    // 4. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
    if (ignore_h != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, (BodyHandle)ignore_h, &ignore_slot) &&
            self->slot_states[ignore_slot] == SLOT_ALIVE) {
            ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
        }
    }

    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (Completely Lockless)
    bool has_hit          = false;
    JPH_Vec3 normal       = {0, 0, 0};
    BodyHandle hit_handle = 0;
    float hit_fraction    = 0.0f;

    Py_BEGIN_ALLOW_THREADS
        // REMOVED: NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

        JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(NULL);
    JPH_ObjectLayerFilter *obj_f        = JPH_ObjectLayerFilter_Create(NULL);
    CastShapeFilter filter_ctx          = {.ignore_id = ignore_bid};
    JPH_BodyFilter *bf                  = JPH_BodyFilter_Create(&filter_ctx);

    // 5a. Cast the ray
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    // Use NULL for filters as we handle simple ignore logic via broadphase or specific ID
    has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, bp_f, obj_f, bf);

    // Filter the 'ignore_bid' manually if hit
    if (has_hit && hit->bodyID == ignore_bid) {
        has_hit = false; // Simple ignore logic
    }

    if (has_hit) {
        hit_handle   = (BodyHandle)JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
        hit_fraction = hit->fraction;

        // 5b. Extract Normal using NoLock interface
        const JPH_BodyLockInterface *li =
            JPH_PhysicsSystem_GetBodyLockInterfaceNoLock(self->system);
        JPH_BodyLockRead j_lock;
        JPH_BodyLockInterface_LockRead(li, hit->bodyID, &j_lock);
        if (j_lock.body) {
            JPH_RVec3 hit_p = {origin->x + (double)direction->x * (double)hit->fraction,
                               origin->y + (double)direction->y * (double)hit->fraction,
                               origin->z + (double)direction->z * (double)hit->fraction};
            JPH_Body_GetWorldSpaceSurfaceNormal(j_lock.body, hit->subShapeID2, &hit_p, &normal);
        }
        JPH_BodyLockInterface_UnlockRead(li, &j_lock);
    }

    JPH_BodyFilter_Destroy(bf);
    JPH_BroadPhaseLayerFilter_Destroy(bp_f);
    JPH_ObjectLayerFilter_Destroy(obj_f);

    // REMOVED: NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // Unmark query and signal stepper if necessary
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT PHASE (Re-verify handle integrity)
        if (!has_hit || hit_handle == 0) {
        Py_RETURN_NONE;
    }

    PyObject *result = NULL;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = (uint32_t)(hit_handle & 0xFFFFFFFF);
    uint32_t gen  = (uint32_t)(hit_handle >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {
        result = Py_BuildValue("Kf(fff)", hit_handle, (double)hit_fraction, (double)normal.x,
                               (double)normal.y, (double)normal.z);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    return result ? result : Py_None;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast_batch(PhysicsWorldObject *self, PyObject *const *args, size_t nargsf,
                                     PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_starts = NULL;
    PyObject *o_dirs   = NULL;
    float max_dist     = 1000.0f;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[RayBatch_COUNT];
    targets[IDX_RB_STARTS] = (void *)&o_starts;
    targets[IDX_RB_DIRS]   = (void *)&o_dirs;
    targets[IDX_RB_DIST]   = (void *)&max_dist;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &RayBatchParser, targets)) {
        return NULL;
    }

    // 3. BUFFER EXTRACTION & VALIDATION (Outside Lock)
    Py_buffer b_starts = {0};
    Py_buffer b_dirs   = {0};

    if (UNLIKELY(PyObject_GetBuffer(o_starts, &b_starts, PyBUF_SIMPLE) < 0)) {
        return NULL;
    }
    if (UNLIKELY(PyObject_GetBuffer(o_dirs, &b_dirs, PyBUF_SIMPLE) < 0)) {
        PyBuffer_Release(&b_starts);
        return NULL;
    }

    if (UNLIKELY(b_starts.len != b_dirs.len || (b_starts.len % 12 != 0))) {
        PyErr_SetString(PyExc_ValueError,
                        "Buffer size mismatch: expected float32 triples of equal length");
        goto fail_buffers;
    }

    size_t count = b_starts.len / 12;
    if (count == 0) {
        PyBuffer_Release(&b_starts);
        PyBuffer_Release(&b_dirs);
        return PyBytes_FromStringAndSize(NULL, 0);
    }

    if (UNLIKELY(count > 10000000)) { // 10M Limit
        PyErr_SetString(PyExc_ValueError, "Batch size exceeds 10M rays");
        goto fail_buffers;
    }

    PyObject *result_bytes =
        PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(count * sizeof(RayCastBatchResult)));
    if (UNLIKELY(!result_bytes)) {
        goto fail_buffers;
    }

    // 4. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);

    // Priority Guard: Wait if physics is stepping or about to step
    BLOCK_UNTIL_CAN_QUERY(self);

    // Snapshot pointers for thread-safe narrow-phase lookup
    const uint32_t *CULV_RESTRICT s2d  = self->slot_to_dense;
    const uint32_t *CULV_RESTRICT mats = self->material_ids;
    const size_t slot_cap              = self->slot_capacity;
    const size_t body_cap              = self->capacity;

    // Mark query as active
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (No GIL, completely lockless!)
    const float *CULV_RESTRICT f_starts = (const float *)b_starts.buf;
    const float *CULV_RESTRICT f_dirs   = (const float *)b_dirs.buf;
    RayCastBatchResult *CULV_RESTRICT results =
        (RayCastBatchResult *)PyBytes_AsString(result_bytes);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

    // OPTIMIZATION: Use the NoLock interface. Since we know the world isn't stepping,
    // we don't need Jolt to perform atomic locks on individual bodies to read their normals.
    const JPH_BodyLockInterface *lock_iface =
        JPH_PhysicsSystem_GetBodyLockInterfaceNoLock(self->system);

    JPH_BodyInterface *bi = self->body_interface;

    Py_BEGIN_ALLOW_THREADS

        // REMOVED: NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

        for (size_t i = 0; i < count; i++) {
        size_t off        = i * 3;
        results[i].handle = 0; // Initialize as "no hit"

        float dx     = f_dirs[off];
        float dy     = f_dirs[off + 1];
        float dz     = f_dirs[off + 2];
        float mag_sq = dx * dx + dy * dy + dz * dz;

        if (mag_sq < 1e-12f) {
            continue;
        }

        float scale;

        // OPTIMIZATION 1: The Fast Path
        // If the vector is already normalized (mag_sq is ~1.0), skip all complex math.
        // fabsf is highly optimized by compilers (often a single bitwise AND instruction).
        if (fabsf(mag_sq - 1.0f) < 1e-4f) {
            scale = max_dist;
        }
        // OPTIMIZATION 2: Fast Inverse Square Root
        else {
            scale = max_dist * culverin_fast_rsqrt(mag_sq);
        }

        JPH_Vec3 v_dir  = {dx * scale, dy * scale, dz * scale};
        JPH_RVec3 v_ori = {(double)f_starts[off], (double)f_starts[off + 1],
                           (double)f_starts[off + 2]};

        JPH_RayCastResult hit;
        hit.bodyID      = JPH_INVALID_BODY_ID;
        hit.fraction    = 1.0f;
        hit.subShapeID2 = 0;

        if (JPH_NarrowPhaseQuery_CastRay(query, &v_ori, &v_dir, &hit, NULL, NULL, NULL)) {
            uint64_t h = JPH_BodyInterface_GetUserData(bi, hit.bodyID);
            if (h != 0) {
                RayCastBatchResult *res = &results[i];
                res->handle             = h;
                res->fraction           = hit.fraction;
                res->subShapeID         = hit.subShapeID2;

                uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
                if (slot < slot_cap) {
                    uint32_t dense = s2d[slot];
                    if (dense < body_cap && mats) {
                        res->material_id = mats[dense];
                    }
                }

                // This is now virtually instantaneous (No atomics/mutexes inside)
                JPH_BodyLockRead j_lock;
                JPH_BodyLockInterface_LockRead(lock_iface, hit.bodyID, &j_lock);
                if (j_lock.body) {
                    JPH_RVec3 hit_p = {v_ori.x + (double)v_dir.x * (double)hit.fraction,
                                       v_ori.y + (double)v_dir.y * (double)hit.fraction,
                                       v_ori.z + (double)v_dir.z * (double)hit.fraction};
                    JPH_Vec3 norm;
                    JPH_Body_GetWorldSpaceSurfaceNormal(j_lock.body, hit.subShapeID2, &hit_p,
                                                        &norm);
                    res->nx = norm.x;
                    res->ny = norm.y;
                    res->nz = norm.z;
                    res->px = (float)hit_p.x;
                    res->py = (float)hit_p.y;
                    res->pz = (float)hit_p.z;
                }
                JPH_BodyLockInterface_UnlockRead(lock_iface, &j_lock);
            }
        }
    }

    // REMOVED: NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // Decrement and check if we were the last active query
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. CLEANUP
        PyBuffer_Release(&b_starts);
    PyBuffer_Release(&b_dirs);
    return result_bytes;

fail_buffers:
    if (b_starts.obj) {
        PyBuffer_Release(&b_starts);
    }
    if (b_dirs.obj) {
        PyBuffer_Release(&b_dirs);
    }
    return NULL;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_shapecast(PhysicsWorldObject *self, PyObject *const *args, size_t nargsf,
                                 PyObject *kwnames) {
    // 1. DEFAULT VALUES
    int shape_type    = 0;
    PyObject *o_pos   = NULL;
    PyObject *o_rot   = NULL;
    PyObject *o_dir   = NULL;
    PyObject *o_size  = NULL;
    uint64_t ignore_h = 0;

    // 2. FAST PARSE
    void *targets[Shapecast_COUNT];
    targets[IDX_SC_SHAPE]  = (void *)&shape_type;
    targets[IDX_SC_POS]    = (void *)&o_pos;
    targets[IDX_SC_ROT]    = (void *)&o_rot;
    targets[IDX_SC_DIR]    = (void *)&o_dir;
    targets[IDX_SC_SIZE]   = (void *)&o_size;
    targets[IDX_SC_IGNORE] = (void *)&ignore_h;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &ShapecastParser, targets)) {
        return NULL;
    }

    // 3. EXTRACTION (Outside Lock)
    JPH_Real px, py, pz;
    float rx, ry, rz, rw;
    float dx, dy, dz;
    float s[4];

    if (!parse_vec3_direct(o_pos, &px, &py, &pz))
        return NULL;
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw))
        return NULL;
    if (!parse_vec3_direct(o_dir, &dx, &dy, &dz))
        return NULL;
    parse_body_size(o_size, s);

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < 1e-12f)) {
        Py_RETURN_NONE;
    }

    // 4. RESOURCE RESOLUTION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Wait if world is updating
    BLOCK_UNTIL_CAN_QUERY(self);

    // Look up shape in our internal cache (needs shadow_lock)
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
        return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters or cache failure");
    }

    // Mark query as active so the world doesn't change under us
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (No GIL, No Trampoline Lock)
    CastShapeContext ctx = {0};
    ctx.has_hit          = false;
    ctx.hit.fraction     = 1.0f;
    uint64_t hit_handle  = 0;

    Py_BEGIN_ALLOW_THREADS
        // REMOVED: NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

        JPH_RMat4 transform;
    JPH_RVec3 v_pos = {px, py, pz};
    JPH_Quat v_rot  = {rx, ry, rz, rw};
    JPH_RMat4_RotationTranslation(&transform, &v_rot, &v_pos);
    JPH_Vec3 sweep_dir = {dx, dy, dz};

    // Execute the sweep
    shapecast_execute_internal(self, shape, &transform, &sweep_dir, ignore_bid, &ctx);

    if (ctx.has_hit) {
        // Jolt UserData access is thread-safe/atomic
        hit_handle = JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
    }

    // REMOVED: NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // Signal end of query
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION
        if (!ctx.has_hit || hit_handle == 0) {
        Py_RETURN_NONE;
    }

    // Normal logic: Shapecast returns a penetration axis.
    // We invert it to get the surface normal and normalize it.
    float nx       = -ctx.hit.penetrationAxis.x;
    float ny       = -ctx.hit.penetrationAxis.y;
    float nz       = -ctx.hit.penetrationAxis.z;
    float n_mag_sq = nx * nx + ny * ny + nz * nz;

    if (n_mag_sq > 1e-12f) {
        float inv_n = culverin_fast_rsqrt(n_mag_sq);
        nx *= inv_n;
        ny *= inv_n;
        nz *= inv_n;
    }

    PyObject *result = NULL;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = (uint32_t)(hit_handle & 0xFFFFFFFF);
    uint32_t gen  = (uint32_t)(hit_handle >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {

        result =
            Py_BuildValue("Kd(ddd)(ddd)", hit_handle, (double)ctx.hit.fraction,
                          (double)ctx.hit.contactPointOn2.x, (double)ctx.hit.contactPointOn2.y,
                          (double)ctx.hit.contactPointOn2.z, (double)nx, (double)ny, (double)nz);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    return result ? result : Py_None;
}

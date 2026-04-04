#include "culverin_query_methods.h"
#include "culverin_arg_indices.h"
#include "culverin_compiler_specifics.h"
#include "culverin_fast_build.h"
#include "culverin_math.h"
#include "culverin_parsers.h"

static constexpr float CQM_DEFAULT_MAX_DIST = 1000.0f;
static constexpr float CQM_EPSILON_SMALL = 1e-12f;
static constexpr float CQM_EPSILON_RELATIVE = 1e-4f;
static constexpr int CQM_VEC3_STRIDE = 12; // bytes for 3 floats

// --- Helper: Signal End of Query ---
// This is crucial for the Condition Variable approach.
// If we are the last query to finish, we must wake up the physics stepper.
void end_query_scope(PhysicsWorldObject *self) {
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
PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES (Unchanged)
    PyObject *o_center = nullptr;
    float radius       = 1.0f;

    // 2. FAST PARSE (Unchanged)
    void *targets[OverlapSphere_COUNT];
    targets[IDX_OS_CENTER] = (void *)&o_center;
    targets[IDX_OS_RADIUS] = (void *)&radius;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.OverlapSphereParser, targets)) {
        return nullptr;
    }

    // 3. VECTOR EXTRACTION (Unchanged)
    JPH_Real cx;
    JPH_Real cy;
    JPH_Real cz;
    if (!parse_vec3_direct(o_center, &cx, &cy, &cz)) {
        return nullptr;
    }

    PyObject *ret_val = nullptr;
    uint64_t small_hit_stack[STACK_ALLOCATE_HITS];
    OverlapContext ctx = {.world       = self,
                          .hits        = small_hit_stack,
                          .count       = 0,
                          .capacity    = STACK_ALLOCATE_HITS,
                          .is_on_stack = true};

    JPH_Shape *shape                     = nullptr;
    JPH_BroadPhaseLayerFilter *bp_filter = nullptr;
    JPH_ObjectLayerFilter *obj_filter    = nullptr;
    JPH_BodyFilter *body_filter          = nullptr;

    // 4. RESOURCE RESOLUTION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    float s_params[4] = {radius, 0, 0, 0};
    shape             = find_or_create_shape_locked(self, CULV_SHAPE_SPHERE, s_params);

    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Failed to resolve sphere shape");
    }

    // Mark query active so the Stepper thread waits for us
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION (Released GIL)
    Py_BEGIN_ALLOW_THREADS JPH_RMat4 transform;
    JPH_RVec3 v_pos = {cx, cy, cz};
    JPH_Quat v_rot  = {0, 0, 0, 1};
    JPH_RMat4_RotationTranslation(&transform, &v_rot, &v_pos);

    JPH_Vec3 scale        = {1.0f, 1.0f, 1.0f};
    JPH_RVec3 base_offset = {0, 0, 0};

    JPH_CollideShapeSettings settings;
    JPH_CollideShapeSettings_Init(&settings);

    bp_filter   = JPH_BroadPhaseLayerFilter_Create(nullptr);
    obj_filter  = JPH_ObjectLayerFilter_Create(nullptr);
    body_filter = JPH_BodyFilter_Create(nullptr);

    const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

    JPH_NarrowPhaseQuery_CollideShape(nq, shape, &scale, &transform, &settings, &base_offset,
                                      OverlapCallback_Narrow, &ctx, bp_filter, obj_filter,
                                      body_filter, nullptr);

    // Unmark query and signal condition variable
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (ATOMIC REFACTOR)
        ret_val = PyList_New(0);
    if (ret_val) {
        SHADOW_LOCK(&self->shadow_lock);
        for (size_t i = 0; i < ctx.count; i++) {
            uint64_t raw_h = ctx.hits[i];
            uint32_t slot;

            // TSan Fix: In C23, casting a value to an _Atomic type is a valid
            // way to pass it to a function expecting an atomic temporary.
            if (unpack_handle(self, (BodyHandle)raw_h, &slot)) {

                // TSan Fix: Atomic load of state to verify body is still valid
                uint8_t state =
                    atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

                if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
                    PyObject *py_h = PyLong_FromUnsignedLongLong(raw_h);
                    if (py_h) {
                        PyList_Append(ret_val, py_h);
                        Py_DECREF(py_h);
                    }
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
PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES (Unchanged)
    PyObject *o_min = nullptr;
    PyObject *o_max = nullptr;

    // 2. FAST PARSE (Unchanged)
    void *targets[OverlapAABB_COUNT];
    targets[IDX_OA_MIN] = (void *)&o_min;
    targets[IDX_OA_MAX] = (void *)&o_max;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.OverlapAABBParser, targets)) {
        return nullptr;
    }

    // 3. VECTOR EXTRACTION (Unchanged)
    JPH_Real mix;
    JPH_Real miy;
    JPH_Real miz;
    JPH_Real max;
    JPH_Real may;
    JPH_Real maz;
    if (!parse_vec3_direct(o_min, &mix, &miy, &miz)) {
        return nullptr;
    }
    if (!parse_vec3_direct(o_max, &max, &may, &maz)) {
        return nullptr;
    }

    PyObject *ret_val = nullptr;
    uint64_t small_hit_stack[STACK_ALLOCATE_HITS];
    OverlapContext ctx                   = {.world       = self,
                                            .hits        = small_hit_stack,
                                            .count       = 0,
                                            .capacity    = STACK_ALLOCATE_HITS,
                                            .is_on_stack = true};
    JPH_BroadPhaseLayerFilter *bp_filter = nullptr;
    JPH_ObjectLayerFilter *obj_filter    = nullptr;

    // 4. PHASE GUARD (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    // Mark query active so the Stepper waits for completion
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // Prepare Jolt AABox
    JPH_STACK_ALLOC(JPH_AABox, box);
    box->min = (JPH_Vec3){(float)mix, (float)miy, (float)miz};
    box->max = (JPH_Vec3){(float)max, (float)may, (float)maz};

    // 5. EXECUTION (No GIL)
    Py_BEGIN_ALLOW_THREADS bp_filter = JPH_BroadPhaseLayerFilter_Create(nullptr);
    obj_filter                       = JPH_ObjectLayerFilter_Create(nullptr);

    const JPH_BroadPhaseQuery *bq = JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);

    JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx, bp_filter, obj_filter);

    // Signal query finish
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (ATOMIC REFACTOR)
        ret_val = PyList_New(0);
    if (!ret_val) {
        goto query_cleanup;
    }

    SHADOW_LOCK(&self->shadow_lock);
    for (size_t i = 0; i < ctx.count; i++) {
        uint64_t raw_h = ctx.hits[i];
        uint32_t slot;

        // TSan Fix: Initialize atomic temporary for handle verification
        if (unpack_handle(self, (BodyHandle)raw_h, &slot)) {

            // TSan Fix: Atomic load of state (Acquire matches release in op_CREATE_BODY)
            uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

            if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
                PyObject *py_h = PyLong_FromUnsignedLongLong(raw_h);
                if (py_h) {
                    PyList_Append(ret_val, py_h);
                    Py_DECREF(py_h);
                }
            }
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);

query_cleanup:
    if (bp_filter) {
        JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    }
    if (obj_filter) {
        JPH_ObjectLayerFilter_Destroy(obj_filter);
    }
    if (ctx.hits && !ctx.is_on_stack) {
        CULV_RAW_FREE(ctx.hits);
    }

    return ret_val;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES (Unchanged)
    PyObject *o_start     = nullptr;
    PyObject *o_dir       = nullptr;
    float max_dist        = CQM_DEFAULT_MAX_DIST;
    uint64_t ignore_h_raw = 0;

    // 2. FAST PARSE (Unchanged)
    void *targets[Raycast_COUNT];
    targets[IDX_RAY_START] = (void *)&o_start;
    targets[IDX_RAY_DIR]   = (void *)&o_dir;
    targets[IDX_RAY_DIST]  = (void *)&max_dist;
    targets[IDX_RAY_IGN]   = (void *)&ignore_h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RaycastParser, targets)) {
        return nullptr;
    }

    // 3. VECTOR EXTRACTION (Unchanged)
    JPH_Real sx;
    JPH_Real sy;
    JPH_Real sz;
    float dx;
    float dy;
    float dz;
    if (!parse_vec3_direct(o_start, &sx, &sy, &sz)) {
        return nullptr;
    }
    if (!parse_vec3_direct(o_dir, &dx, &dy, &dz)) {
        return nullptr;
    }

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < CQM_EPSILON_SMALL)) {
        Py_RETURN_NONE;
    }

    float scale = (fabsf(mag_sq - 1.0f) < CQM_EPSILON_RELATIVE) ? max_dist
                                                                 : max_dist * culverin_fast_rsqrt(mag_sq);

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
    if (ignore_h_raw != 0) {
        uint32_t ignore_slot;
        // TSan Fix: Cast to BodyHandle for atomic parameter unpacking
        if (unpack_handle(self, (BodyHandle)ignore_h_raw, &ignore_slot)) {
            // TSan Fix: Atomic check of liveness
            uint8_t state =
                atomic_load_explicit(&self->slot_states[ignore_slot], memory_order_acquire);
            if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
                ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
            }
        }
    }

    // Mark query active
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (Lockless)
    bool has_hit            = false;
    JPH_Vec3 normal         = {0, 0, 0};
    uint64_t hit_handle_raw = 0;
    float hit_fraction      = 0.0f;

    Py_BEGIN_ALLOW_THREADS JPH_BroadPhaseLayerFilter *bp_f =
        JPH_BroadPhaseLayerFilter_Create(nullptr);
    JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(nullptr);
    CastShapeFilter filter_ctx   = {.ignore_id = ignore_bid};
    JPH_BodyFilter *bf           = JPH_BodyFilter_Create(&filter_ctx);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, bp_f, obj_f, bf);

    if (has_hit) {
        // JPH_BodyInterface_GetUserData returns uint64_t natively
        hit_handle_raw = JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
        hit_fraction   = hit->fraction;

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

    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT PHASE (ATOMIC REFACTOR)
        if (!has_hit || hit_handle_raw == 0) {
        Py_RETURN_NONE;
    }

    PyObject *result = nullptr;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = (uint32_t)(hit_handle_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(hit_handle_raw >> HANDLE_INDEX_BITS);

    if (slot < self->slot_capacity) {
        // TSan Fix: Atomic load of metadata for verification
        uint32_t cur_gen = atomic_load_explicit(&self->generations[slot], memory_order_acquire);
        uint8_t state    = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

        if (cur_gen == gen && (state == SLOT_ALIVE || state == SLOT_CHARACTER)) {
            result = FastBuild_Tuple(hit_handle_raw, hit_fraction,
                                     FastBuild_Tuple(normal.x, normal.y, normal.z));
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    if (!result) {
        return PyErr_Occurred() ? nullptr : Py_None;
    }
    return result;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES (Unchanged)
    PyObject *o_starts = nullptr;
    PyObject *o_dirs   = nullptr;
    float max_dist     = CQM_DEFAULT_MAX_DIST;

    // 2. FAST PARSE (Unchanged)
    void *targets[RayBatch_COUNT];
    targets[IDX_RB_STARTS] = (void *)&o_starts;
    targets[IDX_RB_DIRS]   = (void *)&o_dirs;
    targets[IDX_RB_DIST]   = (void *)&max_dist;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RayBatchParser, targets)) {
        return nullptr;
    }

    // 3. BUFFER EXTRACTION & VALIDATION (Unchanged)
    Py_buffer b_starts = {};
    Py_buffer b_dirs   = {};
    if (UNLIKELY(PyObject_GetBuffer(o_starts, &b_starts, PyBUF_SIMPLE) < 0)) {
        return nullptr;
    }
    if (UNLIKELY(PyObject_GetBuffer(o_dirs, &b_dirs, PyBUF_SIMPLE) < 0)) {
        PyBuffer_Release(&b_starts);
        return nullptr;
    }

    if (UNLIKELY(b_starts.len != b_dirs.len || (b_starts.len % CQM_VEC3_STRIDE != 0))) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        goto fail_buffers;
    }

    size_t count = b_starts.len / CQM_VEC3_STRIDE;
    if (count == 0) {
        PyBuffer_Release(&b_starts);
        PyBuffer_Release(&b_dirs);
        return PyBytes_FromStringAndSize(nullptr, 0);
    }

    PyObject *result_bytes =
        PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)(count * sizeof(RayCastBatchResult)));
    if (UNLIKELY(!result_bytes)) {
        goto fail_buffers;
    }

    // 4. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    // Snapshot pointers (Stable because PhysicsWorld_resize is blocked by
    // is_stepping/active_queries)
    const uint32_t *CULV_RESTRICT s2d  = self->slot_to_dense;
    const uint32_t *CULV_RESTRICT mats = self->material_ids;
    const size_t slot_cap              = self->slot_capacity;
    const size_t body_cap              = atomic_load_explicit(&self->count, memory_order_relaxed);

    // Register active query to block structural flushes
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (Lockless Batch Processing)
    const float *CULV_RESTRICT f_starts = (const float *)b_starts.buf;
    const float *CULV_RESTRICT f_dirs   = (const float *)b_dirs.buf;
    RayCastBatchResult *CULV_RESTRICT results =
        (RayCastBatchResult *)PyBytes_AsString(result_bytes);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    const JPH_BodyLockInterface *lock_iface =
        JPH_PhysicsSystem_GetBodyLockInterfaceNoLock(self->system);
    JPH_BodyInterface *bi = self->body_interface;

    Py_BEGIN_ALLOW_THREADS for (size_t i = 0; i < count; i++) {
        size_t off        = i * 3;
        results[i].handle = 0; // Pre-init as "no hit"

        float dx     = f_dirs[off];
        float dy     = f_dirs[off + 1];
        float dz     = f_dirs[off + 2];
        float mag_sq = dx * dx + dy * dy + dz * dz;
        if (mag_sq < CQM_EPSILON_SMALL) {
            continue;
        }

        float scale = (fabsf(mag_sq - 1.0f) < CQM_EPSILON_RELATIVE) ? max_dist
                                                                      : max_dist * culverin_fast_rsqrt(mag_sq);

        JPH_Vec3 v_dir  = {dx * scale, dy * scale, dz * scale};
        JPH_RVec3 v_ori = {(double)f_starts[off], (double)f_starts[off + 1],
                           (double)f_starts[off + 2]};

        JPH_RayCastResult hit;
        hit.bodyID      = JPH_INVALID_BODY_ID;
        hit.fraction    = 1.0f;
        hit.subShapeID2 = 0;

        if (JPH_NarrowPhaseQuery_CastRay(query, &v_ori, &v_dir, &hit, nullptr, nullptr, nullptr)) {
            uint64_t h_raw = JPH_BodyInterface_GetUserData(bi, hit.bodyID);
            if (h_raw != 0) {
                uint32_t slot = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
                uint32_t gen  = (uint32_t)(h_raw >> HANDLE_INDEX_BITS);

                if (LIKELY(slot < slot_cap)) {
                    // TSan Fix: Atomic verify hit integrity (Acquire ensures visibility of
                    // creator's writes)
                    uint32_t cur_gen =
                        atomic_load_explicit(&self->generations[slot], memory_order_acquire);
                    uint8_t state =
                        atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

                    if (cur_gen == gen && (state == SLOT_ALIVE || state == SLOT_CHARACTER)) {
                        RayCastBatchResult *res = &results[i];
                        res->handle             = h_raw;
                        res->fraction           = hit.fraction;
                        res->subShapeID         = hit.subShapeID2;

                        uint32_t dense = s2d[slot];
                        if (dense < body_cap) {
                            res->material_id = mats[dense];
                        }

                        JPH_BodyLockRead j_lock;
                        JPH_BodyLockInterface_LockRead(lock_iface, hit.bodyID, &j_lock);
                        if (j_lock.body) {
                            JPH_RVec3 hit_p = {v_ori.x + (double)v_dir.x * (double)hit.fraction,
                                               v_ori.y + (double)v_dir.y * (double)hit.fraction,
                                               v_ori.z + (double)v_dir.z * (double)hit.fraction};
                            JPH_Vec3 norm;
                            JPH_Body_GetWorldSpaceSurfaceNormal(j_lock.body, hit.subShapeID2,
                                                                &hit_p, &norm);
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
        }
    }

    end_query_scope(self);
    Py_END_ALLOW_THREADS

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
    return nullptr;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES (Unchanged)
    int shape_type        = 0;
    PyObject *o_pos       = nullptr;
    PyObject *o_rot       = nullptr;
    PyObject *o_dir       = nullptr;
    PyObject *o_size      = nullptr;
    uint64_t ignore_h_raw = 0;

    // 2. FAST PARSE (Unchanged)
    void *targets[Shapecast_COUNT];
    targets[IDX_SC_SHAPE]  = (void *)&shape_type;
    targets[IDX_SC_POS]    = (void *)&o_pos;
    targets[IDX_SC_ROT]    = (void *)&o_rot;
    targets[IDX_SC_DIR]    = (void *)&o_dir;
    targets[IDX_SC_SIZE]   = (void *)&o_size;
    targets[IDX_SC_IGNORE] = (void *)&ignore_h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ShapecastParser, targets)) {
        return nullptr;
    }

    // 3. EXTRACTION (Unchanged)
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    float dx;
    float dy;
    float dz;
    float s[4];
    if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
        return nullptr;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }
    if (!parse_vec3_direct(o_dir, &dx, &dy, &dz)) {
        return nullptr;
    }
    parse_body_size(o_size, s);

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < CQM_EPSILON_SMALL)) {
        Py_RETURN_NONE;
    }

    // 4. RESOURCE RESOLUTION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    JPH_Shape *shape = find_or_create_shape_locked(self, shape_type, s);

    JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
    if (ignore_h_raw != 0) {
        uint32_t ignore_slot;
        // TSan Fix: Cast raw uint64 to BodyHandle for atomic check
        if (unpack_handle(self, (BodyHandle)ignore_h_raw, &ignore_slot)) {
            // TSan Fix: Atomic load of state (Acquire matches release in creators)
            uint8_t state =
                atomic_load_explicit(&self->slot_states[ignore_slot], memory_order_acquire);
            if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
                ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
            }
        }
    }

    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters or cache failure");
    }

    // Mark query active so the Stepper thread waits for us
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION PHASE (Released GIL)
    CastShapeContext ctx    = {0};
    ctx.has_hit             = false;
    ctx.hit.fraction        = 1.0f;
    uint64_t hit_handle_raw = 0;

    Py_BEGIN_ALLOW_THREADS JPH_RMat4 transform;
    JPH_RVec3 v_pos = {px, py, pz};
    JPH_Quat v_rot  = {rx, ry, rz, rw};
    JPH_RMat4_RotationTranslation(&transform, &v_rot, &v_pos);
    JPH_Vec3 sweep_dir = {dx, dy, dz};

    shapecast_execute_internal(self, shape, &transform, &sweep_dir, ignore_bid, &ctx);

    if (ctx.has_hit) {
        // Jolt UserData is uint64_t natively
        hit_handle_raw = JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
    }

    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (ATOMIC REFACTOR)
        if (!ctx.has_hit || hit_handle_raw == 0) {
        Py_RETURN_NONE;
    }

    float nx       = -ctx.hit.penetrationAxis.x;
    float ny       = -ctx.hit.penetrationAxis.y;
    float nz       = -ctx.hit.penetrationAxis.z;
    float n_mag_sq = nx * nx + ny * ny + nz * nz;

    if (n_mag_sq > CQM_EPSILON_SMALL) {
        float inv_n = culverin_fast_rsqrt(n_mag_sq);
        nx *= inv_n;
        ny *= inv_n;
        nz *= inv_n;
    }

    PyObject *result = nullptr;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = (uint32_t)(hit_handle_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(hit_handle_raw >> HANDLE_INDEX_BITS);

    if (slot < self->slot_capacity) {
        // TSan Fix: Atomic loads to verify handle is still valid after lockless execution
        uint32_t cur_gen = atomic_load_explicit(&self->generations[slot], memory_order_acquire);
        uint8_t state    = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

        if (cur_gen == gen && (state == SLOT_ALIVE || state == SLOT_CHARACTER)) {
            result = FastBuild_Tuple(hit_handle_raw, ctx.hit.fraction,
                                     FastBuild_Tuple(ctx.hit.contactPointOn2.x,
                                                     ctx.hit.contactPointOn2.y,
                                                     ctx.hit.contactPointOn2.z),
                                     FastBuild_Tuple(nx, ny, nz));
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    if (!result) {
        return PyErr_Occurred() ? nullptr : Py_None;
    }
    return result;
}
#include "culverin_query_methods.h"
#include "culverin_arg_indices.h"
#include "culverin_compiler_specifics.h"
#include "culverin_fast_build.h"
#include "culverin_math.h"
#include "culverin_parsers.h"
#include "culverin_physics_sync.h"

static constexpr float CQM_DEFAULT_MAX_DIST = 1000.0f;
static constexpr float CQM_EPSILON_SMALL    = 1e-12f;
static constexpr float CQM_EPSILON_RELATIVE = 1e-4f;
static constexpr int CQM_VEC3_STRIDE        = 12; // bytes for 3 floats

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

    // 1. FAST PARSE
    PyObject *o_center                 = nullptr;
    float radius                       = 1.0f;
    void *targets[OverlapSphere_COUNT] = {
        [IDX_OS_CENTER] = (void *)&o_center,
        [IDX_OS_RADIUS] = (void *)&radius,
    };

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.OverlapSphereParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION
    JPH_Real cx;
    JPH_Real cy;
    JPH_Real cz;
    if (!parse_vec3_direct(o_center, &cx, &cy, &cz)) {
        return nullptr;
    }

    // 3. CONTEXT SETUP
    uint64_t small_hit_stack[STACK_ALLOCATE_HITS];
    OverlapContext ctx = {.world       = self,
                          .hits        = small_hit_stack,
                          .count       = 0,
                          .capacity    = STACK_ALLOCATE_HITS,
                          .is_on_stack = true};

    // 4. RESOURCE RESOLUTION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    float s_params[4] = {radius, 0, 0, 0};
    JPH_Shape *shape  = find_or_create_shape_locked(self, CULV_SHAPE_SPHERE, s_params);

    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Failed to resolve sphere shape");
    }

    // Mark query active so the Stepper thread waits for us
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. EXECUTION (Released GIL)
    JPH_BroadPhaseLayerFilter *bp_filter = nullptr;
    JPH_ObjectLayerFilter *obj_filter    = nullptr;
    JPH_BodyFilter *body_filter          = nullptr;

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

    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (Refactored with SlotPredicate)
        PyObject *ret_list = PyList_New(0);
    if (ret_list) {
        SHADOW_LOCK(&self->shadow_lock);
        for (size_t i = 0; i < ctx.count; i++) {
            uint64_t raw_h = ctx.hits[i];
            uint32_t slot;

            if (unpack_handle(self, (BodyHandle)raw_h, &slot)) {
                const uint8_t state =
                    atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

                // Use standard mask: Alive, Character, and SoftBody are valid results for a sphere
                // overlap
                const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

                if (pred.is_immediate) {
                    PyObject *py_h = PyLong_FromUnsignedLongLong(raw_h);
                    if (py_h) {
                        PyList_Append(ret_list, py_h);
                        Py_DECREF(py_h);
                    }
                }
            }
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }

    // 7. CLEANUP
    if (bp_filter) {
        JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    }
    if (obj_filter) {
        JPH_ObjectLayerFilter_Destroy(obj_filter);
    }
    if (body_filter) {
        JPH_BodyFilter_Destroy(body_filter);
    }
    if (!ctx.is_on_stack) {
        CULV_RAW_FREE(ctx.hits);
    }

    return ret_list;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_min                  = nullptr;
    PyObject *o_max                  = nullptr;
    void *targets[OverlapAABB_COUNT] = {
        [IDX_OA_MIN] = (void *)&o_min,
        [IDX_OA_MAX] = (void *)&o_max,
    };

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.OverlapAABBParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION
    JPH_Real mix;
    JPH_Real miy;
    JPH_Real miz;
    JPH_Real max;
    JPH_Real may;
    JPH_Real maz;
    if (!parse_vec3_direct(o_min, &mix, &miy, &miz) ||
        !parse_vec3_direct(o_max, &max, &may, &maz)) {
        return nullptr;
    }

    // 3. CONTEXT SETUP
    uint64_t small_hit_stack[STACK_ALLOCATE_HITS];
    OverlapContext ctx = {.world       = self,
                          .hits        = small_hit_stack,
                          .count       = 0,
                          .capacity    = STACK_ALLOCATE_HITS,
                          .is_on_stack = true};

    // 4. CONCURRENCY GUARD
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);
    // Increment active_queries so the stepping thread waits for us
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. JOLT EXECUTION (GIL Released)
    JPH_BroadPhaseLayerFilter *bp_filter = nullptr;
    JPH_ObjectLayerFilter *obj_filter    = nullptr;

    Py_BEGIN_ALLOW_THREADS JPH_STACK_ALLOC(JPH_AABox, box);
    box->min = (JPH_Vec3){(float)mix, (float)miy, (float)miz};
    box->max = (JPH_Vec3){(float)max, (float)may, (float)maz};

    bp_filter  = JPH_BroadPhaseLayerFilter_Create(nullptr);
    obj_filter = JPH_ObjectLayerFilter_Create(nullptr);

    const JPH_BroadPhaseQuery *bq = JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);
    JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx, bp_filter, obj_filter);

    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 6. RESULT CONSTRUCTION (Refactored with SlotPredicate)
        PyObject *ret_list = PyList_New(0);
    if (ret_list) {
        SHADOW_LOCK(&self->shadow_lock);
        for (size_t i = 0; i < ctx.count; i++) {
            uint64_t raw_h = ctx.hits[i];
            uint32_t slot;

            if (unpack_handle(self, (BodyHandle)raw_h, &slot)) {
                const uint8_t state =
                    atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

                // Use the standard mask: returns True for ALIVE, CHARACTER, and SOFT_BODY
                const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

                if (pred.is_immediate) {
                    PyObject *py_h = PyLong_FromUnsignedLongLong(raw_h);
                    if (py_h) {
                        PyList_Append(ret_list, py_h);
                        Py_DECREF(py_h);
                    }
                }
            }
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }

    // 7. CLEANUP
    if (bp_filter) {
        JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    }
    if (obj_filter) {
        JPH_ObjectLayerFilter_Destroy(obj_filter);
    }
    if (!ctx.is_on_stack) {
        CULV_RAW_FREE(ctx.hits);
    }

    return ret_list;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_start     = nullptr;
    PyObject *o_dir       = nullptr;
    float max_dist        = CQM_DEFAULT_MAX_DIST;
    uint64_t ignore_h_raw = 0;

    void *targets[Raycast_COUNT] = {
        [IDX_RAY_START] = (void *)&o_start,
        [IDX_RAY_DIR]   = (void *)&o_dir,
        [IDX_RAY_DIST]  = (void *)&max_dist,
        [IDX_RAY_IGN]   = (void *)&ignore_h_raw,
    };

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.RaycastParser,
                           targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION & MATH
    JPH_Real sx;
    JPH_Real sy;
    JPH_Real sz;
    float dx;
    float dy;
    float dz;
    if (!parse_vec3_direct(o_start, &sx, &sy, &sz) || !parse_vec3_direct(o_dir, &dx, &dy, &dz)) {
        return nullptr;
    }

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < CQM_EPSILON_SMALL)) {
        Py_RETURN_NONE;
    }

    auto scale = (fabsf(mag_sq - 1.0f) < CQM_EPSILON_RELATIVE)
                      ? max_dist
                      : max_dist * culverin_fast_rsqrt(mag_sq);

    // 3. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
    if (ignore_h_raw != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, (BodyHandle)ignore_h_raw, &ignore_slot)) {
            const uint8_t state =
                atomic_load_explicit(&self->slot_states[ignore_slot], memory_order_acquire);
            const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

            if (pred.is_immediate) {
                ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
            }
        }
    }

    // Mark query active so Stepper waits for completion
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. EXECUTION PHASE (Lockless)
    bool has_hit            = false;
    JPH_Vec3 normal         = {};
    uint64_t hit_handle_raw = 0;
    float hit_fraction      = 0.0f;

    Py_BEGIN_ALLOW_THREADS JPH_STACK_ALLOC(JPH_RVec3, origin);
    *origin = (JPH_RVec3){sx, sy, sz};
    JPH_STACK_ALLOC(JPH_Vec3, direction);
    *direction = (JPH_Vec3){dx * scale, dy * scale, dz * scale};

    JPH_STACK_ALLOC(JPH_RayCastResult, hit);
    hit->bodyID      = JPH_INVALID_BODY_ID;
    hit->fraction    = 1.0f;
    hit->subShapeID2 = 0;

    JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(nullptr);
    JPH_ObjectLayerFilter *obj_f    = JPH_ObjectLayerFilter_Create(nullptr);
    CastShapeFilter filter_ctx      = {.ignore_id = ignore_bid};
    JPH_BodyFilter *bf              = JPH_BodyFilter_Create(&filter_ctx);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, bp_f, obj_f, bf);

    if (has_hit) {
        hit_handle_raw = JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
        hit_fraction   = hit->fraction;

        const auto *li =
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

        // 5. RESULT CONSTRUCTION (Predicate Validation)
        if (!has_hit || hit_handle_raw == 0) {
        Py_RETURN_NONE;
    }

    PyObject *result = nullptr;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot;
    if (unpack_handle(self, (BodyHandle)hit_handle_raw, &slot)) {
        const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

        // Raycast hits must be on bodies existing in the simulation (Standard Mask)
        const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

        if (pred.is_immediate) {
            result = FastBuild_Tuple(hit_handle_raw, hit_fraction,
                                     FastBuild_Tuple(normal.x, normal.y, normal.z));
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    if (result) {
        return result;
    }

    Py_RETURN_NONE;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_starts = nullptr;
    PyObject *o_dirs   = nullptr;
    float max_dist     = CQM_DEFAULT_MAX_DIST;

    void *targets[RayBatch_COUNT] = {
        [IDX_RB_STARTS] = (void *)&o_starts,
        [IDX_RB_DIRS]   = (void *)&o_dirs,
        [IDX_RB_DIST]   = (void *)&max_dist,
    };

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.RayBatchParser,
                           targets)) {
        return nullptr;
    }

    // 2. BUFFER EXTRACTION & VALIDATION
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
        PyBuffer_Release(&b_starts);
        PyBuffer_Release(&b_dirs);
        return nullptr;
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
        PyBuffer_Release(&b_starts);
        PyBuffer_Release(&b_dirs);
        return nullptr;
    }

    // 3. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    // Snapshot pointers for the loop
    const uint32_t *CULV_RESTRICT s2d  = self->slot_to_dense;
    const uint32_t *CULV_RESTRICT mats = self->material_ids;
    const size_t body_cap              = atomic_load_explicit(&self->count, memory_order_relaxed);

    // Register active query to block structural flushes/resizes
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. EXECUTION PHASE (Lockless Batch Processing)
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

        float scale = (fabsf(mag_sq - 1.0f) < CQM_EPSILON_RELATIVE)
                          ? max_dist
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
            uint32_t slot;

            // Use the standard handle unpacking and predicate check
            if (unpack_handle(self, (BodyHandle)h_raw, &slot)) {
                const uint8_t state =
                    atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
                const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

                if (pred.is_immediate) {
                    RayCastBatchResult *res = &results[i];
                    res->handle             = h_raw;
                    res->fraction           = hit.fraction;
                    res->subShapeID         = hit.subShapeID2;

                    uint32_t dense = s2d[slot];
                    if (dense < body_cap) {
                        res->material_id = mats[dense];
                    }

                    // Extract normal and position data
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
    }

    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 5. CLEANUP
        PyBuffer_Release(&b_starts);
    PyBuffer_Release(&b_dirs);

    return result_bytes;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    int shape_type        = 0;
    PyObject *o_pos       = nullptr;
    PyObject *o_rot       = nullptr;
    PyObject *o_dir       = nullptr;
    PyObject *o_size      = nullptr;
    uint64_t ignore_h_raw = 0;

    void *targets[Shapecast_COUNT] = {
        [IDX_SC_SHAPE] = (void *)&shape_type, [IDX_SC_POS] = (void *)&o_pos,
        [IDX_SC_ROT] = (void *)&o_rot,        [IDX_SC_DIR] = (void *)&o_dir,
        [IDX_SC_SIZE] = (void *)&o_size,      [IDX_SC_IGNORE] = (void *)&ignore_h_raw,
    };

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ShapecastParser,
                           targets)) {
        return nullptr;
    }

    // 2. EXTRACTION
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
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw) ||
        !parse_vec3_direct(o_dir, &dx, &dy, &dz)) {
        return nullptr;
    }
    parse_body_size(o_size, s);

    float mag_sq = dx * dx + dy * dy + dz * dz;
    if (UNLIKELY(mag_sq < CQM_EPSILON_SMALL)) {
        Py_RETURN_NONE;
    }

    // 3. RESOURCE RESOLUTION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_CAN_QUERY(self);

    JPH_Shape *shape = find_or_create_shape_locked(self, shape_type, s);
    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters or cache failure");
    }

    JPH_BodyID ignore_bid = JPH_INVALID_BODY_ID;
    if (ignore_h_raw != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, (BodyHandle)ignore_h_raw, &ignore_slot)) {
            const uint8_t state =
                atomic_load_explicit(&self->slot_states[ignore_slot], memory_order_acquire);
            const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);
            if (pred.is_immediate) {
                ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
            }
        }
    }

    // Mark query active so the Stepper thread waits for us
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. EXECUTION PHASE (Released GIL)
    CastShapeContext ctx    = {.has_hit = false};
    ctx.hit.fraction        = 1.0f;
    uint64_t hit_handle_raw = 0;

    Py_BEGIN_ALLOW_THREADS JPH_RMat4 transform;
    JPH_RVec3 v_pos = {px, py, pz};
    JPH_Quat v_rot  = {rx, ry, rz, rw};
    JPH_RMat4_RotationTranslation(&transform, &v_rot, &v_pos);
    JPH_Vec3 sweep_dir = {dx, dy, dz};

    shapecast_execute_internal(self, shape, &transform, &sweep_dir, ignore_bid, &ctx);

    if (ctx.has_hit) {
        hit_handle_raw = JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
    }

    end_query_scope(self);
    Py_END_ALLOW_THREADS

        // 5. RESULT CONSTRUCTION (Predicate Validation)
        if (!ctx.has_hit || hit_handle_raw == 0) {
        Py_RETURN_NONE;
    }

    PyObject *result = nullptr;
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot;
    if (unpack_handle(self, (BodyHandle)hit_handle_raw, &slot)) {
        const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

        // Use standard mask: Alive, Character, and SoftBody are valid results for a shapecast
        const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

        if (pred.is_immediate) {
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

            result = FastBuild_Tuple(hit_handle_raw, ctx.hit.fraction,
                                     FastBuild_Tuple(ctx.hit.contactPointOn2.x,
                                                     ctx.hit.contactPointOn2.y,
                                                     ctx.hit.contactPointOn2.z),
                                     FastBuild_Tuple(nx, ny, nz));
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    if (result) {
        return result;
    }

    Py_RETURN_NONE;
}
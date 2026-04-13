#include "culverin_filters.h"
#include "culverin_handler.h"
#if !defined(_CRT_SECURE_NO_WARNINGS)
#    define _CRT_SECURE_NO_WARNINGS
#endif
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_character.h"
#include "culverin_compiler_specifics.h"
#include "culverin_constraint.h"
#include "culverin_contact_listener.h"
#include "culverin_fast_build.h"
#include "culverin_fast_parse.h"
#include "culverin_getters.h"
#include "culverin_parsers.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"
#include "culverin_query_methods.h"
#include "culverin_ragdoll.h"
#include "culverin_shadow_sync.h"
#include "culverin_soft_body.h"
#include "culverin_vehicle.h"
#include "joltc.h"
#include <stdatomic.h>

// ============================================================================
// Semantic Constants - Magic Number Replacements
// ============================================================================

// Memory and Alignment
static constexpr size_t INITIAL_BODY_CAPACITY = 1024;

// Physics Simulation
static constexpr float DEFAULT_FRAME_TIME    = 1.0f / 60.0f;
static constexpr float DEFAULT_LINEAR_DRAG   = 0.5f;
static constexpr float DEFAULT_ANGULAR_DRAG  = 0.5f;
static constexpr float DEFAULT_FRICTION      = 0.2f;
static constexpr float CONVEX_HULL_TOLERANCE = 0.05f;

// Collision Filtering
static constexpr uint32_t COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
static constexpr uint32_t COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

// Numerical Tolerances
static constexpr float EPSILON_FLOAT                    = 1e-6f;
static constexpr float EPSILON_QUATERNION_NORMALIZATION = 0.000001f;

// Array Indices and Counts
static constexpr int QUATERNION_INTERPOLATION_Z_INDEX = 5;
static constexpr int QUATERNION_INTERPOLATION_W_INDEX = 6;
static constexpr int INERTIA_MATRIX_COMPONENT_COUNT   = 3;
static constexpr size_t FLOATS_PER_INTERPOLATED_BODY  = 7;    // 3 position + 4 quaternion
static constexpr float RESTITUTION_BUFFER             = 0.5f; // Default restitution/bounce
static constexpr size_t VERTEX_STRIDE_BYTES           = 12;   // 3 floats (x, y, z) * 4 bytes
static constexpr size_t INITIAL_MATERIAL_CAPACITY     = 16;   // Initial material data capacity
static constexpr float DEFAULT_BODY_SIZE              = 0.5f;

// Global lock for JPH callbacks
NativeMutex g_jph_trampoline_lock; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// --- Lifecycle: Deallocation ---
PyType_DeclareSlot_Status PhysicsWorld_traverse(PhysicsWorldObject *self, visitproc visit,
                                                void *arg) {
    // 1. Visit the type itself (Required for all heap types)
    Py_VISIT(Py_TYPE(self));

    // 2. If you add any other PyObject* members to your struct in the future,
    // you MUST visit them here.

    return 0;
}

PyType_DeclareSlot_Status PhysicsWorld_clear(CULV_MAYBE_UNUSED PhysicsWorldObject *self) {
    // Currently nothing to clear.
    return 0;
}

PyType_DeclareSlot_Void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
    PyTypeObject *tp = Py_TYPE(self);

    atomic_store_explicit(&self->is_deallocating, true, memory_order_release);

    // 1. The GC "Safety Shield"
    // Use the check-then-untrack to avoid the 'already untracked' abort
    if (PyObject_GC_IsTracked((PyObject *)self)) {
        PyObject_GC_UnTrack(self);
    }

    // 2. Weakref cleanup
    if (self->weakreflist != nullptr) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }

    // 3. The "Manual" work
    PhysicsWorld_free_members(self);

    // 4. Final destruction
    // For Heap Types, we use the type's free function
    tp->tp_free((PyObject *)self);

    // 5. Release the type itself
    Py_DECREF(tp);
}
// --- Lifecycle: Initialization ---

// Orchestrator function
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_Status PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                                            PyObject *kwds) {
    if (self->system != nullptr) {
        // Please don't call __init__() again.
        PyErr_SetString(PyExc_RuntimeError,
                        "PhysicsWorld instance has already been initialized and "
                        "cannot be re-initialized.");
        return -1;
    }
    PyObject *settings_dict = nullptr;
    PyObject *bodies_list   = nullptr;
    PyObject *baked         = nullptr;
    float gx;
    float gy;
    float gz;
    int max_bodies;
    int max_pairs;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", (char *[]){"settings", "bodies", nullptr},
                                     &settings_dict, &bodies_list)) {
        return -1;
    }

    // 1. Initial State
    // 1.1 Jolt Core Pointers
    self->system               = nullptr;
    self->char_vs_char_manager = nullptr;
    self->body_interface       = nullptr;
    self->job_system           = nullptr;
    self->bp_interface         = nullptr;
    self->pair_filter          = nullptr;
    self->bp_filter            = nullptr;
    self->contact_listener     = nullptr;

    // 1.2. Hot Sync Shadow Buffers
    self->positions          = nullptr;
    self->prev_positions     = nullptr;
    self->rotations          = nullptr;
    self->prev_rotations     = nullptr;
    self->linear_velocities  = nullptr;
    self->angular_velocities = nullptr;
    self->body_ids           = nullptr;
    self->user_data          = nullptr;
    self->soft_shadows       = nullptr;
    self->material_ids       = nullptr;

    // 1.3. Data Buffers & Mapping Tables
    self->contact_events         = nullptr;
    self->contact_buffer         = nullptr;
    self->materials              = nullptr;
    self->command_queue          = nullptr;
    self->command_queue_spare    = nullptr;
    self->shape_cache            = nullptr;
    self->id_to_handle_map       = nullptr;
    self->constraints            = nullptr;
    self->categories             = nullptr;
    self->masks                  = nullptr;
    self->generations            = nullptr;
    self->slot_to_dense          = nullptr;
    self->dense_to_slot          = nullptr;
    self->free_slots             = nullptr;
    self->constraint_generations = nullptr;
    self->free_constraint_slots  = nullptr;
    self->slot_states            = nullptr;
    self->constraint_states      = nullptr;

    // 1.4. Counters & Simulation State
    self->contact_count        = 0;
    self->contact_capacity     = 0;
    self->contact_max_capacity = 0;
    atomic_init(&self->contact_atomic_idx, 0);

    self->material_count    = 0;
    self->material_capacity = 0;
    atomic_init(&self->free_count, 0);
    self->slot_capacity        = 0;
    self->command_count        = 0;
    self->command_capacity     = 0;
    self->spare_capacity       = 0;
    self->shape_cache_count    = 0;
    self->shape_cache_capacity = 0;
    atomic_init(&self->count, 0);
    self->capacity              = 0;
    self->constraint_count      = 0;
    self->constraint_capacity   = 0;
    self->free_constraint_count = 0;
    self->time                  = 0.0;

    // 1.5. Query & Sync State
    self->max_jolt_bodies = 0;
    atomic_init(&self->active_queries, 0);
    atomic_init(&self->view_export_count, 0);
#if !defined(Py_GIL_DISABLED)
    atomic_init(&self->waiting_threads, 0);
#endif
    atomic_init(&self->step_requested, false);
    atomic_init(&self->is_stepping, false);
    self->needs_optimization = false;

    // 1.6. Complex Structs (Safe to zero these individually)
    memset(&self->step_sync, 0, sizeof(ShadowSync));
    // Note: INIT_LOCK(self->shadow_lock) handles its own initialization

    // 1.7. View Metadata
    self->view_shape[0]   = 0;
    self->view_shape[1]   = 0;
    self->view_strides[0] = 0;
    self->view_strides[1] = 0;

    // 1.8. Debug Renderer
    self->debug_renderer = nullptr;
    memset(&self->debug_lines, 0, sizeof(DebugBuffer));
    memset(&self->debug_triangles, 0, sizeof(DebugBuffer));
    INIT_LOCK(self->shadow_lock);
    self->debug_renderer = JPH_DebugRenderer_Create(self);
    atomic_init(&self->is_stepping, false);

    INIT_NATIVE_MUTEX(self->step_sync.mutex);
    INIT_NATIVE_COND(self->step_sync.cond);

    // 2. Settings & Jolt Init
    if (init_settings(self, settings_dict, &gx, &gy, &gz, &max_bodies, &max_pairs) < 0) {
        goto fail;
    }
    WorldLimits limits    = {max_bodies, max_pairs};
    GravityVector gravity = {gx, gy, gz};
    if (init_jolt_core(self, limits, gravity) < 0) {
        goto fail;
    }

    if (verify_abi_alignment(self->body_interface) < 0) {
        goto fail;
    }

    self->contact_max_capacity = CONTACT_MAX_CAPACITY;
    self->contact_buffer       = CULV_RAW_MALLOC(CONTACT_MAX_CAPACITY * sizeof(ContactEvent));
    atomic_init(&self->contact_atomic_idx, 0);
    self->contact_listener = JPH_ContactListener_Create(self);
    JPH_PhysicsSystem_SetContactListener(self->system, self->contact_listener);

    // 3. Bake & Buffers
    if (bodies_list && bodies_list != Py_None) {
        PyObject *st_helper = get_culverin_state(PyType_GetModule(Py_TYPE(self)))->helper;
        PyObject *bake_func = PyObject_GetAttrString(st_helper, "bake_scene");
        baked               = PyObject_CallFunctionObjArgs(bake_func, bodies_list, nullptr);
        Py_XDECREF(bake_func);
        if (!baked) {
            goto fail;
        }
        atomic_store_explicit(&self->count, PyLong_AsSize_t(PyTuple_GetItem(baked, 0)),
                              memory_order_relaxed);
    }

    if (allocate_buffers(self, max_bodies) < 0) {
        goto fail;
    }

    // 4. Constraints & Data Loading
    constexpr uint32_t CONSTRAINT_INITIAL_CAPACITY = 256;
    self->constraint_capacity                      = CONSTRAINT_INITIAL_CAPACITY;
    self->constraints =
        (JPH_Constraint **)CULV_RAW_CALLOC(CONSTRAINT_INITIAL_CAPACITY, sizeof(JPH_Constraint *));
    self->constraint_generations = CULV_RAW_CALLOC(CONSTRAINT_INITIAL_CAPACITY, sizeof(uint32_t));
    self->free_constraint_slots  = CULV_RAW_MALLOC(CONSTRAINT_INITIAL_CAPACITY * sizeof(uint32_t));
    self->constraint_states      = CULV_RAW_CALLOC(CONSTRAINT_INITIAL_CAPACITY, sizeof(uint8_t));
    if (!self->constraints || !self->free_constraint_slots) {
        goto fail;
    }

    for (uint32_t i = 0; i < CONSTRAINT_INITIAL_CAPACITY; i++) {
        self->constraint_generations[i] = 1;
        self->free_constraint_slots[i]  = i;
    }
    self->free_constraint_count = CONSTRAINT_INITIAL_CAPACITY;

    if (baked && load_baked_scene(self, baked) < 0) {
        goto fail;
    }
    Py_XDECREF(baked);

    // 1. Load the current count (from baked scene or 0)
    size_t start_idx = atomic_load_explicit(&self->count, memory_order_relaxed);

    for (uint32_t i = (uint32_t)start_idx; i < (uint32_t)self->slot_capacity; i++) {
        // 2. Initialize the atomic generation for this slot
        atomic_init(&self->generations[i], 1);

        // 3. Atomically increment free_count and get the index to store the slot
        size_t f_idx = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
        self->free_slots[f_idx] = i;
    }
    SHADOW_LOCK(&self->shadow_lock);
    culverin_sync_shadow_buffers(self);
    SHADOW_UNLOCK(&self->shadow_lock);
    return 0;

fail:
    Py_XDECREF(baked);
    PhysicsWorld_free_members(self);
    return -1;
}

/**
 * HELPER: physics_world_commit_create_locked
 * Encapsulates slot acquisition, handle generation, and command queuing.
 */
static uint64_t physics_world_commit_create_locked(PhysicsWorldObject *self,
                                                   JPH_BodyCreationSettings *settings,
                                                   uint32_t slot_state) {
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);

    // 1. Boundary Check: Hard Jolt Limit
    if (UNLIKELY(current_count >= self->max_jolt_bodies)) {
        PyErr_Format(PyExc_RuntimeError, "PhysicsWorld limit reached: %u bodies",
                     self->max_jolt_bodies);
        return 0;
    }

    // 2. Resource Management: Resize if empty
    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        size_t next_cap = (self->capacity == 0) ? INITIAL_BODY_CAPACITY : self->capacity * 2;
        if (next_cap > self->max_jolt_bodies) {
            next_cap = self->max_jolt_bodies;
        }

        // PhysicsWorld_resize returns -1 if someone is holding a memoryview (BufferError)
        if (PhysicsWorld_resize(self, next_cap) < 0) {
            return 0;
        }

        // Re-fetch counts after potentially successful resize
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);

        // CRITICAL FIX: If we still have no slots after resize attempt, we are at the limit
        if (UNLIKELY(available == 0)) {
            PyErr_SetString(PyExc_RuntimeError, "World is at maximum capacity (Limit hit)");
            return 0;
        }
    }

    // 3. Command Buffer Capacity
    if (UNLIKELY(!ensure_command_capacity(self))) {
        return 0;
    }

    // 4. Atomic Popping from Free Stack
    // We now know available > 0
    uint32_t slot  = self->free_slots[--available];
    uint32_t dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

    // Commit the new free count
    atomic_store_explicit(&self->free_count, available, memory_order_release);

    // 5. Handle and Metadata Mappings
    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);

    JPH_BodyCreationSettings_SetUserData(settings, raw_h);

    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;
    self->body_ids[dense]      = JPH_INVALID_BODY_ID;

    // 6. Slot State Publishing
    atomic_store_explicit(&self->slot_states[slot], slot_state, memory_order_release);

    return raw_h;
}

// Helper: Apply mass, sensor, CCD, and sleeping settings to the creation
// struct
static void configure_body_settings(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                                    BodyConfig cfg) {
    // Use the members of the struct instead of loose variables
    if (cfg.is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }

    if (cfg.use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }

    if (cfg.motion_type == 2) { // MOTION_DYNAMIC
        JPH_BodyCreationSettings_SetAllowSleeping(settings, true);
    }

    JPH_BodyCreationSettings_SetFriction(settings, cfg.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, cfg.restitution);

    if (cfg.mass > 0.0f) {
        JPH_MassProperties mp;
        JPH_Shape_GetMassProperties(shape, &mp);
        float scale = cfg.mass / fmaxf(mp.mass, EPSILON_FLOAT);
        mp.mass     = cfg.mass;
        for (int i = 0; i < 3; i++) {
            mp.inertia.column[i].x *= scale;
            mp.inertia.column[i].y *= scale;
            mp.inertia.column[i].z *= scale;
        }
        JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
        JPH_BodyCreationSettings_SetOverrideMassProperties(
            settings, JPH_OverrideMassProperties_CalculateInertia);
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t h_raw;
    float x;
    float y;
    float z;
    void *targets[Vec3_COUNT] = {
        [IDX_V3_H] = &h_raw, [IDX_V3_X] = &x, [IDX_V3_Y] = &y, [IDX_V3_Z] = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ImpulseParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Impulse");

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state      = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // SPECULATIVE QUEUE WRITE
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_APPLY_IMPULSE, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;
    self->command_count += pred.is_deferred;

    const JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

    if (pred.is_immediate) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
        JPH_BodyInterface_AddImpulse(self->body_interface, bid, &imp);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS Py_RETURN_NONE;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_apply_impulse_at(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t h_raw;
    float ix;
    float iy;
    float iz;
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    void *targets[ImpAt_COUNT] = {
        [IDX_IMPAT_H] = (void *)&h_raw, [IDX_IMPAT_IX] = (void *)&ix, [IDX_IMPAT_IY] = (void *)&iy,
        [IDX_IMPAT_IZ] = (void *)&iz,   [IDX_IMPAT_PX] = (void *)&px, [IDX_IMPAT_PY] = (void *)&py,
        [IDX_IMPAT_PZ] = (void *)&pz};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ImpulseAtParser,
                           targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(ix, iy, iz, "Impulse");
    VALIDATE_FINITE_VEC3(px, py, pz, "Impulse position");

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state      = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // SPECULATIVE QUEUE WRITE
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_APPLY_IMPULSE_AT, slot);
    cmd->impulse_at.ix  = ix;
    cmd->impulse_at.iy  = iy;
    cmd->impulse_at.iz  = iz;
    cmd->impulse_at.px  = px;
    cmd->impulse_at.py  = py;
    cmd->impulse_at.pz  = pz;
    self->command_count += pred.is_deferred;

    const JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

    if (pred.is_immediate) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {ix, iy, iz};
        JPH_RVec3 v_pos                     = {px, py, pz};
        JPH_BodyInterface_AddImpulse2(self->body_interface, bid, &imp, &v_pos);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS Py_RETURN_NONE;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}
PyCFunction_DeclareMethod PhysicsWorld_apply_angular_impulse(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t h_raw;
    float x;
    float y;
    float z;
    void *targets[Vec3_COUNT] = {
        [IDX_V3_H] = &h_raw, [IDX_V3_X] = &x, [IDX_V3_Y] = &y, [IDX_V3_Z] = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.AngImpulseParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Angular impulse");

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    // CALCULATE PREDICATES (Passing MASK_IMM_STRICT to keep logic 100% identical)
    const uint8_t state      = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STRICT);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // SPECULATIVE WRITE
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_APPLY_ANG_IMPULSE, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    self->command_count += pred.is_deferred;

    // DATA RESOLUTION (Safe to fetch regardless of state)
    const JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

    if (pred.is_immediate) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
        JPH_BodyInterface_AddAngularImpulse(self->body_interface, bid, &imp);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS Py_RETURN_NONE;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_apply_force(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    void *targets[Vec3_COUNT] = {
        [IDX_V3_H] = &h_raw, [IDX_V3_X] = &x, [IDX_V3_Y] = &y, [IDX_V3_Z] = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ForceParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Force");

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state      = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We always write the command. We only increment the count if it's deferred.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_APPLY_FORCE, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    self->command_count += pred.is_deferred;

    // 5. UNCONDITIONAL JOLT DATA LOOKUP
    const uint32_t dense_idx = self->slot_to_dense[slot];
    const JPH_BodyID bid     = self->body_ids[dense_idx];

    // 6. FINAL EXECUTION DISPATCH
    if (pred.is_immediate) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS JPH_Vec3 force_vec = {x, y, z};
        JPH_BodyInterface_AddForce(self->body_interface, bid, &force_vec);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS Py_RETURN_NONE;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // Identity check: Returns None for success (Deferred), raises Error for INVALID.
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_apply_torque(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    void *targets[Vec3_COUNT] = {
        [IDX_V3_H] = &h_raw, [IDX_V3_X] = &x, [IDX_V3_Y] = &y, [IDX_V3_Z] = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.TorqueParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Torque");

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Torque is typically for Rigid Bodies (ALIVE), but we use STANDARD (ALIVE | CHARACTER)
    // to match the flexibility of the Force/Impulse API.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We write to the current index; we only "commit" it by incrementing count if deferred.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_APPLY_TORQUE, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    self->command_count += pred.is_deferred;

    // 5. UNCONDITIONAL JOLT DATA LOOKUP
    // Safe to fetch because the handle check passed.
    const JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

    // 6. FINAL EXECUTION DISPATCH
    if (pred.is_immediate) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS JPH_Vec3 torque_vec = {x, y, z};
        JPH_BodyInterface_AddTorque(self->body_interface, bid, &torque_vec);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS Py_RETURN_NONE;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // If it was PENDING_CREATE, we return success (it's queued).
    // If it was anything else (DEAD/DESTROYING), we raise an error.
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_gravity(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Unchanged)
    float x;
    float y;
    float z;

    void *targets[XYZ_COUNT] = {
        [IDX_XYZ_X] = (void *)&x,
        [IDX_XYZ_Y] = (void *)&y,
        [IDX_XYZ_Z] = (void *)&z,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.GravityParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Gravity");

    // 2. CRITICAL SECTION (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);

    // Global properties require the engine to be idle
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Load atomic count safely.
    // Acquire ensures we see the final state of all shadow buffer mappings.
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    // Jolt Interaction
    JPH_Vec3 g = {x, y, z};
    JPH_PhysicsSystem_SetGravity(self->system, &g);

    // Safety check for body count overflow using atomic snapshot
    if (UNLIKELY(current_count > UINT32_MAX)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_OverflowError, "Body count exceeds Jolt limit");
        return nullptr;
    }

    // Immediate reaction: Wake up all bodies so they react to new gravity direction.
    // We use the snapshot count to ensure we don't read out-of-bounds of body_ids.
    if (current_count > 0) {
        JPH_BodyInterface_ActivateBodies(self->body_interface, self->body_ids,
                                         (uint32_t)current_count);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}


PyCFunction_DeclareMethod PhysicsWorld_get_gravity(PhysicsWorldObject *self, PyObject *Py_UNUSED(ignored)) {
    // We acquire the shadow_lock to ensure we don't read gravity 
    // mid-update if the simulation is currently swapping buffers.
    SHADOW_LOCK(&self->shadow_lock);
    
    JPH_Vec3 g;
    JPH_PhysicsSystem_GetGravity(self->system, &g);
    
    SHADOW_UNLOCK(&self->shadow_lock);

    return FastBuild_Tuple(
        FastBuild_Value(g.x), 
        FastBuild_Value(g.y), 
        FastBuild_Value(g.z)
    );
}

PyCFunction_DeclareMethod PhysicsWorld_get_body_stats(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.HOnlyParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Stats are only valid for bodies in simulation (Alive or Character).
    // MASK_IMM_STANDARD is perfect for this.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (pred.is_immediate) {
        // Shadow buffers are stable here (protected by SHADOW_LOCK + BLOCK_UNTIL_NOT_STEPPING)
        uint32_t i = self->slot_to_dense[slot];

        // Snapshot values while holding the lock
        PosStride p = ((PosStride *)self->positions)[i];
        AuxStride r = ((AuxStride *)self->rotations)[i];
        AuxStride v = ((AuxStride *)self->linear_velocities)[i];

        SHADOW_UNLOCK(&self->shadow_lock);

        // 4. RESULT CONSTRUCTION
        // Nested tuples: ((px, py, pz), (rx, ry, rz, rw), (vx, vy, vz))
        return FastBuild_Tuple(FastBuild_Tuple(p.x, p.y, p.z), FastBuild_Tuple(r.x, r.y, r.z, r.w),
                               FastBuild_Tuple(v.x, v.y, v.z));
    }

    // 5. ERROR FALLBACK
    SHADOW_UNLOCK(&self->shadow_lock);
    RAISE_STALE_HANDLE();
}
PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    double surface_y;
    float buoyancy  = 1.0f;
    float lin_drag  = DEFAULT_LINEAR_DRAG;
    float ang_drag  = DEFAULT_ANGULAR_DRAG;
    float dt        = DEFAULT_FRAME_TIME;
    PyObject *o_vel = nullptr;

    void *targets[Buoy_COUNT] = {
        [IDX_BUOY_HANDLE] = (void *)&h_raw,      [IDX_BUOY_SURFACE_Y] = (void *)&surface_y,
        [IDX_BUOY_BUOYANCY] = (void *)&buoyancy, [IDX_BUOY_LIN_DRAG] = (void *)&lin_drag,
        [IDX_BUOY_ANG_DRAG] = (void *)&ang_drag, [IDX_BUOY_DT] = (void *)&dt,
        [IDX_BUOY_VEL] = (void *)&o_vel};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.BuoyParser,
                           targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_FLOAT(buoyancy, "buoyancy");
    VALIDATE_FINITE_FLOAT(lin_drag, "linear drag");
    VALIDATE_FINITE_FLOAT(ang_drag, "angular drag");
    VALIDATE_FINITE_FLOAT(dt, "dt");

    float vx = 0;
    float vy = 0;
    float vz = 0;
    if (o_vel && o_vel != Py_None) {
        if (!parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
            return nullptr;
        }
        VALIDATE_FINITE_VEC3(vx, vy, vz, "fluid velocity");
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_FALSE;
    }

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Buoyancy is only valid for bodies actually in simulation (MASK_IMM_STRICT = SLOT_ALIVE).
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STRICT);

    if (pred.is_immediate) {
        const uint32_t dense = self->slot_to_dense[slot];
        const JPH_BodyID bid = self->body_ids[dense];

        // Register active query so Stepper doesn't destroy this body during calculation
        atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);

        SHADOW_UNLOCK(&self->shadow_lock);

        // 4. EXECUTION PHASE (Unlocked & GIL-Friendly)
        bool submerged                               = false;
        Py_BEGIN_ALLOW_THREADS JPH_BodyInterface *bi = self->body_interface;
        JPH_PhysicsSystem *sys                       = self->system;

        JPH_BodyInterface_ActivateBody(bi, bid);

        JPH_Vec3 gravity;
        JPH_PhysicsSystem_GetGravity(sys, &gravity);

        JPH_STACK_ALLOC(JPH_RVec3, surf_pos);
        *surf_pos = (JPH_RVec3){0, (JPH_Real)surface_y, 0};

        JPH_STACK_ALLOC(JPH_Vec3, surf_norm);
        *surf_norm = (JPH_Vec3){0, 1.0f, 0};

        JPH_STACK_ALLOC(JPH_Vec3, fluid_vel);
        *fluid_vel = (JPH_Vec3){vx, vy, vz};

        submerged = JPH_BodyInterface_ApplyBuoyancyImpulse(
            bi, bid, surf_pos, surf_norm, buoyancy, lin_drag, ang_drag, fluid_vel, &gravity, dt);

        // Signal completion to the Stepper thread
        end_query_scope(self);
        Py_END_ALLOW_THREADS

            return PyBool_FromLong((int)submerged);
    }

    // 5. FALLBACK
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    PyObject *o_handles = nullptr;
    PyObject *o_vel     = nullptr;
    JPH_Real surface_y  = 0.0;
    float buoyancy      = 1.0f;
    float lin_drag      = DEFAULT_LINEAR_DRAG;
    float ang_drag      = DEFAULT_ANGULAR_DRAG;
    float dt            = DEFAULT_FRAME_TIME;

    void *targets[BatchBuoy_COUNT] = {
        [IDX_BBUOY_HANDLES] = (void *)&o_handles, [IDX_BBUOY_SURFACE_Y] = (void *)&surface_y,
        [IDX_BBUOY_BUOYANCY] = (void *)&buoyancy, [IDX_BBUOY_LIN_DRAG] = (void *)&lin_drag,
        [IDX_BBUOY_ANG_DRAG] = (void *)&ang_drag, [IDX_BBUOY_DT] = (void *)&dt,
        [IDX_BBUOY_VEL] = (void *)&o_vel};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.BatchBuoyParser,
                           targets)) {
        return nullptr;
    }

    // 2. BUFFER & VELOCITY EXTRACTION
    Py_buffer h_view;
    if (PyObject_GetBuffer(o_handles, &h_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    if (UNLIKELY(h_view.itemsize != 8 && h_view.len % 8 != 0)) {
        PyBuffer_Release(&h_view);
        PyErr_SetString(PyExc_ValueError, "Handle buffer must be uint64 array");
        return nullptr;
    }

    float vx = 0;
    float vy = 0;
    float vz = 0;
    if (o_vel && o_vel != Py_None) {
        if (!parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
            PyBuffer_Release(&h_view);
            return nullptr;
        }
    }

    const size_t handle_count = (size_t)h_view.len / 8;
    if (handle_count == 0) {
        PyBuffer_Release(&h_view);
        Py_RETURN_NONE;
    }

    // 3. RESOLUTION PHASE (Critical Section)
    JPH_BodyID *ids = (JPH_BodyID *)CULV_RAW_MALLOC(handle_count * sizeof(JPH_BodyID));
    if (!ids) {
        PyBuffer_Release(&h_view);
        return PyErr_NoMemory();
    }

    uint64_t *handles_raw = (uint64_t *)h_view.buf;
    size_t valid_count    = 0;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    for (size_t i = 0; i < handle_count; i++) {
        uint32_t slot = 0;
        if (unpack_handle(self, (BodyHandle)handles_raw[i], &slot)) {
            const uint8_t state =
                atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

            // Buoyancy batching only supports bodies already in Jolt.
            const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STRICT);

            if (pred.is_immediate) {
                ids[valid_count++] = self->body_ids[self->slot_to_dense[slot]];
            }
        }
    }

    if (valid_count > 0) {
        // Guard the batch against destruction by the Stepper thread
        atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&h_view);

    // 4. EXECUTION PHASE (Lockless Batch)
    if (valid_count > 0) {
        Py_BEGIN_ALLOW_THREADS JPH_BodyInterface *bi = self->body_interface;
        JPH_Vec3 gravity;
        JPH_PhysicsSystem_GetGravity(self->system, &gravity);

        JPH_STACK_ALLOC(JPH_RVec3, surf_pos);
        *surf_pos = (JPH_RVec3){0, surface_y, 0};

        JPH_STACK_ALLOC(JPH_Vec3, surf_norm);
        *surf_norm = (JPH_Vec3){0, 1.0f, 0};

        JPH_STACK_ALLOC(JPH_Vec3, fluid_vel);
        *fluid_vel = (JPH_Vec3){vx, vy, vz};

        for (size_t i = 0; i < valid_count; i++) {
            JPH_BodyInterface_ActivateBody(bi, ids[i]);
            JPH_BodyInterface_ApplyBuoyancyImpulse(bi, ids[i], surf_pos, surf_norm, buoyancy,
                                                   lin_drag, ang_drag, fluid_vel, &gravity, dt);
        }

        end_query_scope(self);
        Py_END_ALLOW_THREADS
    }

    CULV_RAW_FREE(ids);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_save_state(PhysicsWorldObject *self,
                                                  PyObject *Py_UNUSED(unused)) {
    SHADOW_LOCK(&self->shadow_lock);

    // Ensure state is static
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // TSan Fix: Load atomic count and snapshot for consistent size calculation
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    size_t slot_cap      = self->slot_capacity;
    double current_time  = self->time;

    // 1. Size Calculation
    constexpr size_t HEADER_SIZE = sizeof(size_t) + sizeof(double) + sizeof(size_t);

    size_t pos_size_total = current_count * sizeof(PosStride);
    size_t aux_size_total = current_count * sizeof(AuxStride);

    // Mappings: gen(u32), s2d(u32), d2s(u32), state(u8)
    size_t mapping_size = slot_cap * (sizeof(uint32_t) * 3 + sizeof(uint8_t));

    size_t total_size = HEADER_SIZE + pos_size_total + (3 * aux_size_total) + mapping_size;

    PyObject *bytes = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)total_size);
    if (!bytes) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return nullptr;
    }

    char *ptr = PyBytes_AsString(bytes);

    // 2. Encode Header (Using snapshot values)
    memcpy(ptr, &current_count, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, &current_time, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &slot_cap, sizeof(size_t));
    ptr += sizeof(size_t);

    // 3. Encode Dense Buffers (Non-atomic, safe to memcpy under lock)
    memcpy(ptr, self->positions, pos_size_total);
    ptr += pos_size_total;

    memcpy(ptr, self->rotations, aux_size_total);
    ptr += aux_size_total;

    memcpy(ptr, self->linear_velocities, aux_size_total);
    ptr += aux_size_total;

    memcpy(ptr, self->angular_velocities, aux_size_total);
    ptr += aux_size_total;

    // 4. Encode Mapping Tables (ATOMIC REFACTOR)

    // Generations (Atomic uint32_t)
    uint32_t *out_gens = (uint32_t *)ptr;
    for (size_t i = 0; i < slot_cap; i++) {
        out_gens[i] = atomic_load_explicit(&self->generations[i], memory_order_relaxed);
    }
    ptr += (slot_cap * sizeof(uint32_t));

    // slot_to_dense (Non-atomic)
    memcpy(ptr, self->slot_to_dense, slot_cap * sizeof(uint32_t));
    ptr += (slot_cap * sizeof(uint32_t));

    // dense_to_slot (Non-atomic)
    memcpy(ptr, self->dense_to_slot, slot_cap * sizeof(uint32_t));
    ptr += (slot_cap * sizeof(uint32_t));

    // slot_states (Atomic uint8_t)
    uint8_t *out_stats = (uint8_t *)ptr;
    for (size_t i = 0; i < slot_cap; i++) {
        out_stats[i] = atomic_load_explicit(&self->slot_states[i], memory_order_relaxed);
    }
    // ptr increment not needed for the last block

    SHADOW_UNLOCK(&self->shadow_lock);
    return bytes;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_load_state(PhysicsWorldObject *self, PyObject *const *args,
                                                  Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st              = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *state_obj            = nullptr;
    void *targets[LoadState_COUNT] = {
        [IDX_LS_STATE] = (void *)&state_obj,
    };

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.LoadStateParser, targets)) {
        return nullptr;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(state_obj, &view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    // 1. Snapshot raw buffer (GIL held)
    void *local_state_copy = CULV_RAW_MALLOC(view.len);
    if (!local_state_copy) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    memcpy(local_state_copy, view.buf, view.len);
    size_t total_len = (size_t)view.len;
    PyBuffer_Release(&view);

    SHADOW_LOCK(&self->shadow_lock);

    // 2. Concurrency Guard
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // 3. Header Extraction (Using standard types, matching save_state format)
    char *ptr                    = (char *)local_state_copy;
    constexpr size_t HEADER_SIZE = sizeof(size_t) + sizeof(double) + sizeof(size_t);

    if (total_len < HEADER_SIZE) {
        goto size_fail;
    }

    size_t saved_count = 0;
    double saved_time  = 0.0;
    size_t saved_cap   = 0;

    memcpy(&saved_count, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(&saved_time, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&saved_cap, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    if (saved_cap != self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        CULV_RAW_FREE(local_state_copy);
        PyErr_Format(PyExc_ValueError, "Capacity mismatch: World is %zu, Snapshot is %zu",
                     self->slot_capacity, saved_cap);
        return nullptr;
    }

    // 4. Size Validation
    size_t pos_bytes     = saved_count * sizeof(PosStride);
    size_t aux_bytes     = saved_count * sizeof(AuxStride);
    size_t mapping_bytes = saved_cap * (sizeof(uint32_t) * 3 + sizeof(uint8_t));

    if (UNLIKELY(total_len != (HEADER_SIZE + pos_bytes + (aux_bytes * 3) + mapping_bytes))) {
        goto size_fail;
    }

    // 5. Restore World Context
    atomic_store_explicit(&self->count, saved_count, memory_order_relaxed);
    self->time          = saved_time;
    self->view_shape[0] = (Py_ssize_t)saved_count;

    // 6. Restore Shadow Buffers (Non-atomic, safe to memcpy under lock)
    memcpy(self->positions, ptr, pos_bytes);
    ptr += pos_bytes;
    memcpy(self->rotations, ptr, aux_bytes);
    ptr += aux_bytes;
    memcpy(self->linear_velocities, ptr, aux_bytes);
    ptr += aux_bytes;
    memcpy(self->angular_velocities, ptr, aux_bytes);
    ptr += aux_bytes;

    // 7. Restore Atomic Mappings (Loop required)
    uint32_t *in_gens = (uint32_t *)ptr;
    for (size_t i = 0; i < saved_cap; i++) {
        atomic_store_explicit(&self->generations[i], in_gens[i], memory_order_relaxed);
    }
    ptr += (saved_cap * sizeof(uint32_t));

    memcpy(self->slot_to_dense, ptr, saved_cap * sizeof(uint32_t));
    ptr += (saved_cap * sizeof(uint32_t));
    memcpy(self->dense_to_slot, ptr, saved_cap * sizeof(uint32_t));
    ptr += (saved_cap * sizeof(uint32_t));

    uint8_t *in_stats = (uint8_t *)ptr;
    for (size_t i = 0; i < saved_cap; i++) {
        atomic_store_explicit(&self->slot_states[i], in_stats[i], memory_order_relaxed);
    }

    // 8. Rebuild Free List
    size_t local_free_count = 0;
    for (uint32_t i = 0; i < saved_cap; i++) {
        if (atomic_load_explicit(&self->slot_states[i], memory_order_relaxed) == SLOT_EMPTY) {
            self->free_slots[local_free_count++] = i;
        }
    }
    atomic_store_explicit(&self->free_count, local_free_count, memory_order_release);

    // 9. JPH SYNC (Bridges Shadow to C++)
    JPH_BodyID *bids      = self->body_ids;
    JPH_BodyInterface *bi = self->body_interface;
    auto *shadow_pos      = (PosStride *)self->positions;
    auto *shadow_rot      = (AuxStride *)self->rotations;
    auto *shadow_lvel     = (AuxStride *)self->linear_velocities;
    auto *shadow_avel     = (AuxStride *)self->angular_velocities;

    SHADOW_UNLOCK(&self->shadow_lock);

    for (size_t i = 0; i < saved_count; i++) {
        JPH_BodyID bid = bids[i];
        if (bid == JPH_INVALID_BODY_ID) {
            continue;
        }

        JPH_RVec3 p = {shadow_pos[i].x, shadow_pos[i].y, shadow_pos[i].z};
        JPH_Quat q  = {shadow_rot[i].x, shadow_rot[i].y, shadow_rot[i].z, shadow_rot[i].w};
        JPH_Vec3 lv = {shadow_lvel[i].x, shadow_lvel[i].y, shadow_lvel[i].z};
        JPH_Vec3 av = {shadow_avel[i].x, shadow_avel[i].y, shadow_avel[i].z};

        JPH_BodyInterface_SetPositionAndRotation(bi, bid, &p, &q, JPH_Activation_Activate);
        JPH_BodyInterface_SetLinearVelocity(bi, bid, &lv);
        JPH_BodyInterface_SetAngularVelocity(bi, bid, &av);

        uint32_t slot = self->dense_to_slot[i];
        uint32_t gen  = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);

        BodyHandle h   = make_handle(slot, gen);
        uint64_t raw_h = atomic_load_explicit(&h, memory_order_relaxed);

        JPH_BodyInterface_SetUserData(bi, bid, raw_h);
        uint32_t j_idx = JPH_ID_TO_INDEX(bid);
        if (self->id_to_handle_map && j_idx <= self->max_jolt_bodies) {
            atomic_store_explicit(&self->id_to_handle_map[j_idx], raw_h, memory_order_relaxed);
        }
    }

    CULV_RAW_FREE(local_state_copy);
    Py_RETURN_NONE;

size_fail:
    SHADOW_UNLOCK(&self->shadow_lock);
    CULV_RAW_FREE(local_state_copy);
    PyErr_SetString(PyExc_ValueError, "Snapshot buffer truncated or stride mismatch");
    return nullptr;
}
[[gnu::flatten]]
PyCFunction_DeclareMethod PhysicsWorld_step(PhysicsWorldObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
    CulverinState *st         = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    float dt                  = DEFAULT_FRAME_TIME;
    void *targets[Step_COUNT] = {
        [IDX_STEP_DT] = (void *)&dt,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.StepParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_FLOAT(dt, "dt");

    // Check death flag
    if (atomic_load_explicit(&self->is_deallocating, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot step: World is deallocating");
        return nullptr;
    }

    // --- PHASE 0: RE-ENTRANCY GUARD ---
    if (atomic_load_explicit(&self->is_stepping, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "Concurrent step detected.");
        return nullptr;
    }

    // --- PHASE 1: SHADOW STATE LOCK-DOWN ---
    SHADOW_LOCK(&self->shadow_lock);

    // I have no idea why I should even need this, but the GIL is giving me trouble, so this will
    // do.
    // TODO: investigate how the hell does this work
#if !defined(Py_GIL_DISABLED)
    // ANTI-STARVATION: Yield to waiting Python threads (Getters/Mutators)

    while (atomic_load_explicit(&self->waiting_threads, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS culverin_yield();
        Py_END_ALLOW_THREADS SHADOW_LOCK(&self->shadow_lock);
    }
#endif

    // Raise flags
    atomic_store_explicit(&self->is_stepping, true, memory_order_relaxed);
    atomic_store_explicit(&self->step_requested, true, memory_order_relaxed);

    // Drain in-flight queries
    if (atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(self->step_sync.mutex);
        while (atomic_load_explicit(&self->active_queries, memory_order_relaxed) > 0) {
            NATIVE_COND_WAIT(self->step_sync.cond, self->step_sync.mutex);
        }
        NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
        Py_END_ALLOW_THREADS SHADOW_LOCK(&self->shadow_lock);
    }

    // Command Queue Swap (Safe non-atomic logic under SHADOW_LOCK)
    PhysicsCommand *captured_queue = self->command_queue;
    size_t captured_count          = self->command_count;

    self->command_queue       = self->command_queue_spare;
    self->command_queue_spare = captured_queue;
    self->command_count       = 0;

    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- PHASE 2: JOLT CRUNCH (GIL Released) ---
    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    CULV_PROFILE_BEGIN(jolt_step);

    // 1. Process Batch Mutations (Shadow-to-Jolt)
    if (captured_count > 0) {
        flush_commands_internal(self, captured_queue, captured_count);
        self->needs_optimization = true;
    }

    // 2. Simulation Step
    if (dt <= 0.0f) {
        JPH_PhysicsSystem_OptimizeBroadPhase(self->system);
        self->needs_optimization = false;
    } else {
        JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);
        if (self->needs_optimization) {
            JPH_PhysicsSystem_OptimizeBroadPhase(self->system);
            self->needs_optimization = false;
        }
    }

    // SYNC HANDOVER: Wait for Numpy/Proxies to finish
    // Note: We use the dedicated step_sync.mutex, NOT the shadow_lock
    NATIVE_MUTEX_LOCK(self->step_sync.mutex);
    while (atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
        NATIVE_COND_WAIT(self->step_sync.cond, self->step_sync.mutex);
    }

    // 3. Jolt-to-Shadow Buffer Sync
    culverin_sync_shadow_buffers(self);

    NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);

    CULV_PROFILE_END(jolt_step, "Jolt Physics Crunch", (unsigned int)captured_count);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        // --- PHASE 3: FINALIZATION ---
        SHADOW_LOCK(&self->shadow_lock);

    // We no longer need to cleanup buffer. The internal lifecycle handles it.

    // Metadata Updates
    size_t c_idx        = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    self->contact_count = (c_idx > self->contact_max_capacity) ? self->contact_max_capacity : c_idx;

    // TSan Fix: Snapshot final atomic body count to update Python memoryview layout
    size_t final_count  = atomic_load_explicit(&self->count, memory_order_acquire);
    self->view_shape[0] = (Py_ssize_t)final_count;

    self->time += (double)dt;

    // Fence Release
    atomic_store_explicit(&self->is_stepping, false, memory_order_release);
    atomic_store_explicit(&self->step_requested, false, memory_order_release);

    NATIVE_MUTEX_LOCK(self->step_sync.mutex);
    NATIVE_COND_BROADCAST(self->step_sync.cond);
    NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);

    SHADOW_UNLOCK(&self->shadow_lock);

    Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_create_convex_hull(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_points   = nullptr;
    int motion_type      = 2;
    float mass           = -1.0f;
    float friction       = DEFAULT_FRICTION;
    float restitution    = 0.0f;
    uint64_t user_data   = 0;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;

    void *targets[HC_COUNT] = {
        [IDX_HC_POS] = (void *)&o_pos,        [IDX_HC_ROT] = (void *)&o_rot,
        [IDX_HC_DATA] = (void *)&o_points,    [IDX_HC_MOTION] = (void *)&motion_type,
        [IDX_HC_MASS] = (void *)&mass,        [IDX_HC_USER_DATA] = (void *)&user_data,
        [IDX_HC_SENSOR] = (void *)&is_sensor, [IDX_HC_CAT] = (void *)&category,
        [IDX_HC_MASK] = (void *)&mask,        [IDX_HC_MAT_ID] = (void *)&material_id,
        [IDX_HC_FRIC] = (void *)&friction,    [IDX_HC_REST] = (void *)&restitution,
        [IDX_HC_CCD] = (void *)&use_ccd};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ConvexHullParser,
                           targets)) {
        return nullptr;
    }

    // 2. VECTOR/POINT EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");

    Py_buffer points_view;
    if (PyObject_GetBuffer(o_points, &points_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    size_t num_points = points_view.len / VERTEX_STRIDE_BYTES;
    if (UNLIKELY(num_points < 3)) {
        PyBuffer_Release(&points_view);
        return PyErr_Format(PyExc_ValueError, "Need >= 3 points");
    }

    // 3. JOLT SHAPE BUILD (No GIL)
    JPH_Shape *shape = nullptr;
    Py_BEGIN_ALLOW_THREADS;
    auto *jolt_points = (JPH_Vec3 *)CULV_RAW_MALLOC(num_points * sizeof(JPH_Vec3));
    float *raw        = (float *)points_view.buf;
    for (size_t i = 0; i < num_points; i++) {
        jolt_points[i] = (JPH_Vec3){raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]};
    }

    auto *hull_settings = JPH_ConvexHullShapeSettings_Create(jolt_points, (uint32_t)num_points,
                                                             CONVEX_HULL_TOLERANCE);
    CULV_RAW_FREE(jolt_points);
    if (hull_settings) {
        shape = (JPH_Shape *)JPH_ConvexHullShapeSettings_CreateShape(hull_settings);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hull_settings);
    }
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&points_view);

    if (!shape) {
        return PyErr_Format(PyExc_RuntimeError, "Convex Hull build failed");
    }

    // 4. SETTINGS PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, (JPH_MotionType)motion_type,
        (motion_type == MOTION_STATIC ? OBJECT_LAYER_STATIC : OBJECT_LAYER_DYNAMIC));

    BodyConfig config = {mass, friction, restitution, (int)is_sensor, (int)use_ccd, motion_type};
    configure_body_settings(settings, shape, config);

    // 5. COMMIT PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint64_t raw_h = physics_world_commit_create_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(shape);
        return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
    }

    // 6. SHADOW BUFFER UPDATE
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->categories[dense]               = category;
    self->masks[dense]                    = mask;
    self->material_ids[dense]             = material_id;

    // 7. QUEUE COMMAND
    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings    = settings;
    cmd->create.user_data   = user_data;
    cmd->create.category    = category;
    cmd->create.mask        = mask;
    cmd->create.material_id = material_id;

    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_Shape_Destroy(shape);

    return PyLong_FromUnsignedLongLong(raw_h);
}

// Helper 1: Build the Jolt Compound Shape from the Python parts list
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static JPH_Shape *init_compound_shape(PhysicsWorldObject *self, PyObject *parts) {
    if (!PyList_Check(parts)) {
        PyErr_SetString(PyExc_TypeError, "Compound parts must be a list");
        return nullptr;
    }

    Py_ssize_t num_parts = PyList_Size(parts);
    if (num_parts == 0) {
        PyErr_SetString(PyExc_ValueError, "Compound shape must have at least one part");
        return nullptr;
    }

    // --- 1. PARSE PHASE (GIL Held) ---
    // Allocate temp buffer to store parsed data so we can release GIL later
    CompoundPart *buffer = CULV_RAW_MALLOC(sizeof(CompoundPart) * num_parts);
    if (!buffer) {
        PyErr_NoMemory();
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < num_parts; i++) {
        PyObject *item = PyList_GetItem(parts, i);
        // Expecting tuple: (pos, rot, type, size_params)
        if (!PyTuple_Check(item) || PyTuple_Size(item) < 4) {
            CULV_RAW_FREE(buffer);
            PyErr_Format(PyExc_ValueError, "Part %zd must be a tuple(pos, rot, type, size)", i);
            return nullptr;
        }

        PyObject *p_pos  = PyTuple_GetItem(item, 0);
        PyObject *p_rot  = PyTuple_GetItem(item, 1);
        long type_l      = PyLong_AsLong(PyTuple_GetItem(item, 2));
        PyObject *p_size = PyTuple_GetItem(item, 3);

        if (PyErr_Occurred()) {
            CULV_RAW_FREE(buffer);
            return nullptr;
        }

        buffer[i].type = (int)type_l;
        memset(buffer[i].params, 0, sizeof(float) * 4);

        // Parse Position
        if (PyTuple_Check(p_pos) && PyTuple_Size(p_pos) == 3) {
            buffer[i].local_p.x = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 0));
            buffer[i].local_p.y = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 1));
            buffer[i].local_p.z = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 2));
        } else {
            buffer[i].local_p = (JPH_Vec3){0, 0, 0};
        }

        // Parse Rotation
        if (PyTuple_Check(p_rot) && PyTuple_Size(p_rot) == 4) {
            buffer[i].local_q.x = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 0));
            buffer[i].local_q.y = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 1));
            buffer[i].local_q.z = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 2));
            buffer[i].local_q.w = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 3));
        } else {
            buffer[i].local_q = (JPH_Quat){0, 0, 0, 1};
        }

        // Parse Size Params
        if (PyTuple_Check(p_size)) {
            Py_ssize_t sz = PyTuple_Size(p_size);
            for (int j = 0; j < 4 && j < sz; j++) {
                buffer[i].params[j] = (float)PyFloat_AsDouble(PyTuple_GetItem(p_size, j));
            }
        } else if (PyFloat_Check(p_size) || PyLong_Check(p_size)) {
            buffer[i].params[0] = (float)PyFloat_AsDouble(p_size);
        }
    }

    if (PyErr_Occurred()) {
        CULV_RAW_FREE(buffer);
        return nullptr;
    }

    // --- 2. JOLT EXECUTION PHASE (Release GIL, Acquire Jolt Lock) ---
    JPH_Shape *final_shape = nullptr;
    Py_BEGIN_ALLOW_THREADS

        NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    JPH_StaticCompoundShapeSettings *compound_settings = JPH_StaticCompoundShapeSettings_Create();

    // We iterate through our C buffer, acquiring Shadow Lock briefly for each
    // shape lookup
    for (Py_ssize_t i = 0; i < num_parts; i++) {
        SHADOW_LOCK(&self->shadow_lock);
        JPH_Shape *sub_shape = find_or_create_shape_locked(self, buffer[i].type, buffer[i].params);
        SHADOW_UNLOCK(&self->shadow_lock);

        if (sub_shape) {
            JPH_CompoundShapeSettings_AddShape2((JPH_CompoundShapeSettings *)compound_settings,
                                                &buffer[i].local_p, &buffer[i].local_q, sub_shape,
                                                0);
        }
    }

    final_shape = (JPH_Shape *)JPH_StaticCompoundShape_Create(compound_settings);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)compound_settings);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        // --- 3. CLEANUP ---
        CULV_RAW_FREE(buffer);

    if (!final_shape) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create compound shape");
        return nullptr;
    }

    return final_shape;
}

// Helper 2: Apply physics properties (mass, friction, etc) to creation settings
static void apply_body_creation_props(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                                      BodyCreationProps props) {
    if (props.mass > 0.0f) {
        JPH_MassProperties mp;
        JPH_Shape_GetMassProperties(shape, &mp);
        if (mp.mass > EPSILON_FLOAT) {
            float scale = props.mass / mp.mass;
            mp.mass     = props.mass;
            for (int i = 0; i < INERTIA_MATRIX_COMPONENT_COUNT; i++) {
                mp.inertia.column[i].x *= scale;
                mp.inertia.column[i].y *= scale;
                mp.inertia.column[i].z *= scale;
            }
            JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
            JPH_BodyCreationSettings_SetOverrideMassProperties(
                settings, JPH_OverrideMassProperties_CalculateInertia);
        }
    }

    if (props.is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }

    if (props.use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }

    JPH_BodyCreationSettings_SetAllowSleeping(settings, true);

    JPH_BodyCreationSettings_SetFriction(settings, props.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, props.restitution);
}

// Orchestrator
PyCFunction_DeclareMethod PhysicsWorld_create_compound_body(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_parts    = nullptr;
    int motion_type      = 2;
    float mass           = -1.0f;
    float friction       = DEFAULT_FRICTION;
    float restitution    = 0.0f;
    uint64_t user_data   = 0;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;

    void *targets[HC_COUNT] = {
        [IDX_HC_POS] = (void *)&o_pos,        [IDX_HC_ROT] = (void *)&o_rot,
        [IDX_HC_DATA] = (void *)&o_parts,     [IDX_HC_MOTION] = (void *)&motion_type,
        [IDX_HC_MASS] = (void *)&mass,        [IDX_HC_USER_DATA] = (void *)&user_data,
        [IDX_HC_SENSOR] = (void *)&is_sensor, [IDX_HC_CAT] = (void *)&category,
        [IDX_HC_MASK] = (void *)&mask,        [IDX_HC_MAT_ID] = (void *)&material_id,
        [IDX_HC_FRIC] = (void *)&friction,    [IDX_HC_REST] = (void *)&restitution,
        [IDX_HC_CCD] = (void *)&use_ccd};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.CompoundParser,
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
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }
    if (UNLIKELY(!PyList_Check(o_parts))) {
        PyErr_SetString(PyExc_TypeError, "'parts' must be list");
        return nullptr;
    }

    // 3. SHAPE BUILD (Heavy lifting)
    JPH_Shape *final_shape = init_compound_shape(self, o_parts);
    if (!final_shape) {
        return nullptr;
    }

    // 4. JOLT PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        final_shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw},
        (JPH_MotionType)motion_type,
        (motion_type == MOTION_STATIC) ? OBJECT_LAYER_STATIC : OBJECT_LAYER_DYNAMIC);

    BodyCreationProps props = {mass, friction, restitution, (int)is_sensor, (int)use_ccd};
    apply_body_creation_props(settings, final_shape, props);

    // 5. COMMIT PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint64_t raw_h = physics_world_commit_create_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(final_shape);
        return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
    }

    // 6. SHADOW BUFFER UPDATE
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense]          = (PosStride){px, py, pz, 0.0};
    ((PosStride *)self->prev_positions)[dense]     = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense]          = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->prev_rotations)[dense]     = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->linear_velocities)[dense]  = (AuxStride){0};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){0};

    self->categories[dense]   = category;
    self->masks[dense]        = mask;
    self->material_ids[dense] = material_id;
    self->user_data[dense]    = user_data;

    // 7. QUEUE COMMAND
    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings    = settings;
    cmd->create.user_data   = user_data;
    cmd->create.category    = category;
    cmd->create.mask        = mask;
    cmd->create.material_id = material_id;

    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_Shape_Destroy(final_shape);

    return PyLong_FromUnsignedLongLong(raw_h);
}

// Helper 1: Resolve material properties based on ID and explicit overrides
static MaterialSettings resolve_material_params(PhysicsWorldObject *self, uint32_t material_id,
                                                MaterialSettings input) {
    // 1. Start with Jolt Defaults
    float f = DEFAULT_FRICTION;
    float r = 0.0f;

    // 2. Lookup Registry Defaults
    if (material_id > 0) {
        SHADOW_LOCK(&self->shadow_lock);
        for (size_t i = 0; i < self->material_count; i++) {
            if (self->materials[i].id == material_id) {
                f = self->materials[i].friction;
                r = self->materials[i].restitution;
                break;
            }
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }

    // 3. Apply Overrides (if input values are non-negative)
    MaterialSettings resolved;
    resolved.friction    = (input.friction >= 0.0f) ? input.friction : f;
    resolved.restitution = (input.restitution >= 0.0f) ? input.restitution : r;

    return resolved;
}

// Main Orchestrator
PyCFunction_DeclareMethod PhysicsWorld_create_body(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    auto nargs        = PyVectorcall_NARGS(nargsf);

    // 1. DEFAULT VALUES
    JPH_Real px          = 0.0;
    JPH_Real py          = 0.0;
    JPH_Real pz          = 0.0;
    float rx             = 0.0f;
    float ry             = 0.0f;
    float rz             = 0.0f;
    float rw             = 1.0f;
    float mass           = -1.0f;
    float friction       = -1.0f;
    float restitution    = -1.0f;
    int shape_type       = 0;
    int motion_type      = 2;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    uint64_t user_data   = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_size     = nullptr;

    // 2. TARGET MAPPING (Explicitly mapped via Enum)
    // Using explicit indices [IDX_...] makes this reorder-proof.
    void *targets[Body_COUNT] = {
        [IDX_POS] = (void *)&o_pos,          [IDX_ROT] = (void *)&o_rot,
        [IDX_SIZE] = (void *)&o_size,        [IDX_SHAPE] = (void *)&shape_type,
        [IDX_MOTION] = (void *)&motion_type, [IDX_USER_DATA] = (void *)&user_data,
        [IDX_SENSOR] = (void *)&is_sensor,   [IDX_MASS] = (void *)&mass,
        [IDX_CAT] = (void *)&category,       [IDX_MASK] = (void *)&mask,
        [IDX_FRIC] = (void *)&friction,      [IDX_REST] = (void *)&restitution,
        [IDX_MAT] = (void *)&material_id,    [IDX_CCD] = (void *)&use_ccd,
    };

    // 3. THE FAST PARSE
    // We pass the global 'BodyParser' defined in the God Init helper
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.BodyParser, targets)) {
        return nullptr;
    }

    // 4. CONVERT COMPLEX TYPES
    if (o_pos && o_pos != Py_None) {
        if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
            return nullptr; // PyErr already set
        }
        // GUARD: This is the exact point test_numerical_stability checks
        VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    }

    if (o_rot && o_rot != Py_None) {
        if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
            return nullptr;
        }
        VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");
    }

    // Validation
    if (shape_type == 4 && motion_type != 0) {
        return PyErr_Format(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
    }

    // GUARD: Floats (Only check if they aren't the 'unset' -1.0 default)
    if (mass != -1.0f) {
        VALIDATE_FINITE_FLOAT(mass, "mass");
    }
    if (friction != -1.0f) {
        VALIDATE_FINITE_FLOAT(friction, "friction");
    }
    if (restitution != -1.0f) {
        VALIDATE_FINITE_FLOAT(restitution, "restitution");
    }

    // Handle Material & Size
    MaterialSettings mat_in = {friction, restitution};
    MaterialSettings mat    = resolve_material_params(self, material_id, mat_in);
    float s[4]              = {DEFAULT_BODY_SIZE, DEFAULT_BODY_SIZE, DEFAULT_BODY_SIZE,
                               0.0f}; // <--- Initialize with defaults
    parse_body_size(o_size, s);
    // New Guard for Size components
    VALIDATE_FINITE_VEC4(s[0], s[1], s[2], s[3], "Shape size");

    // JOLT PREP (GIL RELEASED)
    JPH_Shape *shape                   = nullptr;
    JPH_BodyCreationSettings *settings = nullptr;

    Py_BEGIN_ALLOW_THREADS;
    SHADOW_LOCK(&self->shadow_lock);
    shape = find_or_create_shape_locked(self, shape_type, s);
    SHADOW_UNLOCK(&self->shadow_lock);

    if (shape) {
        JPH_RVec3 j_pos = {px, py, pz};
        JPH_Quat j_rot  = {rx, ry, rz, rw};
        settings        = JPH_BodyCreationSettings_Create3(
            shape, &j_pos, &j_rot, (JPH_MotionType)motion_type,
            (motion_type == MOTION_KINEMATIC || motion_type == MOTION_STATIC)
                       ? OBJECT_LAYER_STATIC
                       : OBJECT_LAYER_DYNAMIC);
        if (settings) {
            BodyConfig config = {mass,           mat.friction, mat.restitution,
                                 (int)is_sensor, (int)use_ccd, motion_type};
            configure_body_settings(settings, shape, config);
        }
    }
    Py_END_ALLOW_THREADS;

    if (!settings) {
        return PyErr_Format(PyExc_RuntimeError, "Failed to create BodySettings");
    }

    // COMMIT PHASE (Inside Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint64_t raw_h = physics_world_commit_create_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
    }

    // SHADOW BUFFER INITIALIZATION
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense]          = (PosStride){px, py, pz, 0.0};
    ((PosStride *)self->prev_positions)[dense]     = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense]          = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->prev_rotations)[dense]     = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->linear_velocities)[dense]  = (AuxStride){0};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){0};

    self->categories[dense]   = category;
    self->masks[dense]        = mask;
    self->material_ids[dense] = material_id;
    self->user_data[dense]    = user_data;

    // FINALIZE VISIBILITY
    // Update view shape only AFTER all buffer data is stable
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    // QUEUE COMMAND
    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings    = settings;
    cmd->create.user_data   = user_data;
    cmd->create.category    = category;
    cmd->create.mask        = mask;
    cmd->create.material_id = material_id;

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(raw_h);
}

PyCFunction_DeclareMethod PhysicsWorld_create_bodies_batch(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE (Zero Lock Contention)
    PyObject *py_pos   = nullptr;
    PyObject *py_sizes = nullptr;
    int shape_type     = 0;
    int motion_type    = 2;

    void *targets[BatchCreate_COUNT] = {[IDX_BC_POSITIONS] = (void *)&py_pos,
                                        [IDX_BC_SIZES]     = (void *)&py_sizes,
                                        [IDX_BC_SHAPE]     = (void *)&shape_type,
                                        [IDX_BC_MOTION]    = (void *)&motion_type};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.BatchCreateParser, targets)) {
        return nullptr;
    }

    if (!PyList_Check(py_pos) || !PyList_Check(py_sizes)) {
        return PyErr_Format(PyExc_TypeError, "Inputs must be lists");
    }

    const Py_ssize_t batch_count = PyList_GET_SIZE(py_pos);
    if (PyList_GET_SIZE(py_sizes) != batch_count) {
        return PyErr_Format(PyExc_ValueError, "List length mismatch");
    }
    if (batch_count == 0) {
        return PyList_New(0);
    }

    // 2. ARENA ALLOCATION (Include space for PyObject* pointers)
    // We allocate: PosStride + ShapeParams + SettingsPtr + HandleUint64 + ResultPyObjectPtr
    size_t arena_size =
        batch_count * (sizeof(PosStride) + sizeof(ShapeParams) +
                       sizeof(JPH_BodyCreationSettings *) + sizeof(uint64_t) + sizeof(PyObject *));
    void *arena = CULV_RAW_MALLOC(arena_size);
    if (UNLIKELY(!arena)) {
        return PyErr_NoMemory();
    }

    auto *pos_buf       = (PosStride *)arena;
    auto *size_buf      = (ShapeParams *)(pos_buf + batch_count);
    auto **settings_buf = (JPH_BodyCreationSettings **)(size_buf + batch_count);
    auto *handles_out   = (uint64_t *)(settings_buf + batch_count);
    auto **py_results   = (PyObject **)(handles_out + batch_count);

    memset((void *)settings_buf, 0, batch_count * sizeof(void *));

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        parse_py_vec3(PyList_GET_ITEM(py_pos, i), &pos_buf[i]);
        parse_body_size(PyList_GET_ITEM(py_sizes, i), size_buf[i].p);
    }

    // 3. JOLT PREP (No GIL, Inline Shape Caching)
    Py_BEGIN_ALLOW_THREADS;
    SHADOW_LOCK(&self->shadow_lock);
    JPH_Shape *last_shape = nullptr;
    ShapeParams last_size = {-1.0f, -1.0f, -1.0f, -1.0f};

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        JPH_Shape *shape    = last_shape;
        const float *curr_p = size_buf[i].p;
        const float *last_p = last_size.p;

        // Manual logical comparison: Correct float semantics and better optimization
        if (curr_p[0] != last_p[0] || curr_p[1] != last_p[1] || curr_p[2] != last_p[2] ||
            curr_p[3] != last_p[3]) {

            shape      = find_or_create_shape_locked(self, shape_type, curr_p);
            last_shape = shape;

            // Struct assignment is faster and safer than memcpy in C23
            last_size = size_buf[i];
        }

        if (shape) {
            JPH_RVec3 j_p   = {pos_buf[i].x, pos_buf[i].y, pos_buf[i].z};
            JPH_Quat j_r    = {0, 0, 0, 1};
            settings_buf[i] = JPH_BodyCreationSettings_Create3(
                shape, &j_p, &j_r, (JPH_MotionType)motion_type,
                (motion_type == MOTION_STATIC) ? OBJECT_LAYER_STATIC : OBJECT_LAYER_DYNAMIC);
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_END_ALLOW_THREADS;

    // 4. BULK COMMIT PHASE (Inside Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);

    // Expand if necessary
    if (UNLIKELY(available < (size_t)batch_count ||
                 (current_count + batch_count) > self->capacity)) {
        size_t needed = current_count + batch_count + INITIAL_BODY_CAPACITY;
        if (needed > self->max_jolt_bodies) {
            needed = self->max_jolt_bodies;
        }
        if (PhysicsWorld_resize(self, needed) < 0) {
            goto fail_locked;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    if (UNLIKELY(current_count + batch_count > self->max_jolt_bodies)) {
        PyErr_Format(PyExc_RuntimeError, "Batch exceeds world limit");
        goto fail_locked;
    }

    if (UNLIKELY(!ensure_command_bulk_capacity(self, (size_t)batch_count))) {
        // If we can't grow the command queue, we can't proceed.
        // Jump to cleanup to destroy JPH settings and free the arena.
        goto fail_locked;
    }

    uint32_t base_dense =
        (uint32_t)atomic_fetch_add_explicit(&self->count, batch_count, memory_order_relaxed);
    size_t free_head = available - batch_count;
    atomic_store_explicit(&self->free_count, free_head, memory_order_release);

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (UNLIKELY(!settings_buf[i])) {
            handles_out[i] = 0;
            continue;
        }

        uint32_t slot  = self->free_slots[free_head + i];
        uint32_t dense = base_dense + (uint32_t)i;
        uint32_t gen   = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
        uint64_t raw_h = ((uint64_t)gen << HANDLE_INDEX_BITS) | slot;

        JPH_BodyCreationSettings_SetUserData(settings_buf[i], raw_h);

        self->slot_to_dense[slot]  = dense;
        self->dense_to_slot[dense] = slot;
        self->body_ids[dense]      = JPH_INVALID_BODY_ID;

        PosStride p                                = pos_buf[i];
        p.w                                        = 0.0;
        ((PosStride *)self->positions)[dense]      = p;
        ((PosStride *)self->prev_positions)[dense] = p;
        ((AuxStride *)self->rotations)[dense]      = (AuxStride){0, 0, 0, 1};
        ((AuxStride *)self->prev_rotations)[dense] = (AuxStride){0, 0, 0, 1};

        atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);

        PhysicsCommand *cmd  = &self->command_queue[self->command_count++];
        cmd->header          = CMD_HEADER(CMD_CREATE_BODY, slot);
        cmd->create.settings = settings_buf[i];

        handles_out[i] = raw_h;
    }
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. FAST BUILD PHASE (Outside Lock)
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (handles_out[i] != 0) {
            // Converts uint64_t to PyLong object
            py_results[i] = FastBuild_Value(handles_out[i]);
        } else {
            // Correctly handles Py_None as a "failure to create" entry
            py_results[i] = FastBuild_Value(nullptr);
        }
    }

    // Pack the pointer array into a Python List
    // This function handles cleanup/decref internally if any py_results[i] is NULL
    PyObject *res_list = fb_pack_list((size_t)batch_count, py_results);

    CULV_RAW_FREE(arena);
    return res_list;

fail_locked:
    SHADOW_UNLOCK(&self->shadow_lock);
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (settings_buf[i]) {
            JPH_BodyCreationSettings_Destroy(settings_buf[i]);
        }
    }
    CULV_RAW_FREE(arena);
    return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
}

/**
 * Helper 1: Build the Jolt triangle array while verifying index bounds.
 */
static JPH_IndexedTriangle *build_mesh_triangles(const uint32_t *raw, MeshBounds bounds) {
    auto *jolt_tris =
        (JPH_IndexedTriangle *)CULV_RAW_MALLOC(bounds.tri_count * sizeof(JPH_IndexedTriangle));
    if (!jolt_tris) {
        PyErr_NoMemory();
        return nullptr;
    }

    for (uint32_t t = 0; t < bounds.tri_count; t++) {
        uint32_t i1 = raw[t * 3 + 0];
        uint32_t i2 = raw[t * 3 + 1];
        uint32_t i3 = raw[t * 3 + 2];

        if (i1 >= bounds.vertex_count || i2 >= bounds.vertex_count || i3 >= bounds.vertex_count) {
            CULV_RAW_FREE(jolt_tris);
            PyErr_Format(PyExc_ValueError, "Mesh index out of range: %u/%u/%u >= %u", i1, i2, i3,
                         bounds.vertex_count);
            return nullptr;
        }

        jolt_tris[t].i1            = i1;
        jolt_tris[t].i2            = i2;
        jolt_tris[t].i3            = i3;
        jolt_tris[t].materialIndex = 0;
        jolt_tris[t].userData      = 0;
    }
    return jolt_tris;
}

/**
 * Helper 2: Encapsulate Jolt Mesh creation (Settings -> BVH build -> Shape).
 */
static JPH_Shape *build_mesh_shape(const void *v_data, MeshBounds bounds,
                                   JPH_IndexedTriangle *tris) {
    JPH_MeshShapeSettings *mss = JPH_MeshShapeSettings_Create2(
        (JPH_Vec3 *)v_data, bounds.vertex_count, tris, bounds.tri_count);
    if (!mss) {
        PyErr_SetString(PyExc_RuntimeError, "Jolt MeshSettings allocation failed");
        return nullptr;
    }

    JPH_Shape *shape = (JPH_Shape *)JPH_MeshShapeSettings_CreateShape(mss);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)mss);

    if (!shape) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Jolt Mesh BVH build failed (Triangle data degenerate?)");
    }
    return shape;
}

/**
 * Main Orchestrator
 */

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    PyObject *o_pos     = nullptr;
    PyObject *o_rot     = nullptr;
    PyObject *o_verts   = nullptr;
    PyObject *o_indices = nullptr;
    uint64_t user_data  = 0;
    uint32_t cat        = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask       = COLLISION_FILTER_ALL_MASKS;

    void *targets[Mesh_COUNT] = {[IDX_MSH_POS]       = (void *)&o_pos,
                                 [IDX_MSH_ROT]       = (void *)&o_rot,
                                 [IDX_MSH_VERTS]     = (void *)&o_verts,
                                 [IDX_MSH_INDICES]   = (void *)&o_indices,
                                 [IDX_MSH_USER_DATA] = (void *)&user_data,
                                 [IDX_MSH_CAT]       = (void *)&cat,
                                 [IDX_MSH_MASK]      = (void *)&mask};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.MeshParser,
                           targets)) {
        return nullptr;
    }

    // 2. VECTOR/QUAT EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    Py_buffer v_view;
    Py_buffer i_view;
    if (PyObject_GetBuffer(o_verts, &v_view, PyBUF_SIMPLE) != 0 ||
        PyObject_GetBuffer(o_indices, &i_view, PyBUF_SIMPLE) != 0) {
        if (v_view.buf) {
            PyBuffer_Release(&v_view);
        }
        return nullptr;
    }

    if (UNLIKELY(v_view.len % VERTEX_STRIDE_BYTES != 0 || i_view.len % VERTEX_STRIDE_BYTES != 0)) {
        PyBuffer_Release(&v_view);
        PyBuffer_Release(&i_view);
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        return nullptr;
    }

    // 3. JOLT SHAPE BUILD (No GIL)
    JPH_Shape *shape  = nullptr;
    MeshBounds bounds = {(uint32_t)(i_view.len / VERTEX_STRIDE_BYTES),
                         (uint32_t)(v_view.len / VERTEX_STRIDE_BYTES)};

    Py_BEGIN_ALLOW_THREADS;
    JPH_IndexedTriangle *tris = build_mesh_triangles((uint32_t *)i_view.buf, bounds);
    if (tris) {
        shape = build_mesh_shape(v_view.buf, bounds, tris);
        CULV_RAW_FREE(tris);
    }
    Py_END_ALLOW_THREADS;

    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    if (!shape) {
        return nullptr;
    }

    // 4. SETTINGS PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static,
        OBJECT_LAYER_STATIC);

    if (!settings) {
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

    // 5. COMMIT PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint64_t raw_h = physics_world_commit_create_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(shape);
        return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
    }

    // 6. SHADOW BUFFER UPDATE
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->categories[dense]               = cat;
    self->masks[dense]                    = mask;
    self->user_data[dense]                = user_data;

    // 7. QUEUE COMMAND
    PhysicsCommand *cmd   = &self->command_queue[self->command_count++];
    cmd->header           = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings  = settings;
    cmd->create.user_data = user_data;
    cmd->create.category  = cat;
    cmd->create.mask      = mask;

    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_Shape_Destroy(shape);
    return PyLong_FromUnsignedLongLong(raw_h);
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_body(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.DestroyParser,
                           targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Predicate: Can we destroy this?
    static constexpr uint8_t BITMASK_CLAMP = 7;
    const uint32_t is_destructible = !!((1u << (state & BITMASK_CLAMP)) & MASK_DESTRUCTIBLE);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // SPECULATIVE WRITE: Always write the command.
    // If is_destructible is 0, command_count doesn't advance and this is overwritten.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_DESTROY_BODY, slot);

    self->command_count += is_destructible;

    if (is_destructible) {
        // Immediate state transition to prevent further mutations
        atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_DESTROY, memory_order_release);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_bodies_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st                 = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *py_handles_in           = nullptr;
    void *targets[BatchDestroy_COUNT] = {[IDX_BD_HANDLES] = (void *)&py_handles_in};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.BatchDestroyParser, targets)) {
        return nullptr;
    }

    PyObject *py_seq = PySequence_Fast(py_handles_in, "handles must be a sequence");
    if (UNLIKELY(!py_seq)) {
        return nullptr;
    }

    const Py_ssize_t batch_count = PySequence_Fast_GET_SIZE(py_seq);
    if (batch_count <= 0) {
        Py_DECREF(py_seq);
        Py_RETURN_NONE;
    }

    // 1. EXTRACTION PHASE (Outside Lock)
    // Extract Python handles into a raw C array while we don't care about the physics lock.
    uint64_t *handle_cache = (uint64_t *)CULV_RAW_MALLOC(batch_count * sizeof(uint64_t));
    if (UNLIKELY(!handle_cache)) {
        Py_DECREF(py_seq);
        return PyErr_NoMemory();
    }

    PyObject **items         = PySequence_Fast_ITEMS(py_seq);
    size_t actual_work_count = 0;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        uint64_t h = PyLong_AsUnsignedLongLong(items[i]);
        if (UNLIKELY(PyErr_Occurred())) {
            PyErr_Clear();
            continue; // Skip invalid handles
        }
        handle_cache[actual_work_count++] = h;
    }
    Py_DECREF(py_seq);

    if (actual_work_count == 0) {
        CULV_RAW_FREE(handle_cache);
        Py_RETURN_NONE;
    }

    // 2. COMMIT PHASE (Inside Lock - Ultra Fast)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    if (UNLIKELY(!ensure_command_bulk_capacity(self, actual_work_count))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        CULV_RAW_FREE(handle_cache);
        return PyErr_NoMemory();
    }

    // Hoist pointers for maximum loop speed
    PhysicsCommand *cmd_q   = self->command_queue;
    size_t cmd_idx          = self->command_count;
    _Atomic uint8_t *states = self->slot_states;

    for (size_t i = 0; i < actual_work_count; i++) {
        uint32_t slot = 0;
        if (unpack_handle(self, (BodyHandle)handle_cache[i], &slot)) {
            const uint8_t state = atomic_load_explicit(&states[slot], memory_order_acquire);

            // Branchless liveness check using the mask
            static constexpr uint8_t BITMASK_CLAMP = 7;
            const uint32_t is_destructible =
                !!((1u << (state & BITMASK_CLAMP)) & MASK_DESTRUCTIBLE);

            // Commit command
            cmd_q[cmd_idx].header = CMD_HEADER(CMD_DESTROY_BODY, slot);
            cmd_idx += is_destructible;

            if (is_destructible) {
                // Immediate transition blocks other mutators from using this slot
                atomic_store_explicit(&states[slot], SLOT_PENDING_DESTROY, memory_order_release);
            }
        }
    }

    self->command_count = cmd_idx;
    SHADOW_UNLOCK(&self->shadow_lock);

    CULV_RAW_FREE(handle_cache);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_position(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    void *targets[SetPos_COUNT] = {[IDX_SETPOS_HANDLE] = &h_raw,
                                   [IDX_SETPOS_X]      = &x,
                                   [IDX_SETPOS_Y]      = &y,
                                   [IDX_SETPOS_Z]      = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.SetPosParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "SetPosition");

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Position updates are valid for Alive bodies, Characters, and Pending creations.
    // We use MASK_IMM_STANDARD to identify bodies already in the simulation.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We always write the command. We advance count if the body is executable (Alive or Pending).
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_SET_POS, slot);
    cmd->pos.x          = x;
    cmd->pos.y          = y;
    cmd->pos.z          = z;

    self->command_count += pred.is_executable;

    // 5. SHADOW BUFFER MIRROR (Zero-Streak Reset)
    // We update the shadow buffers immediately so that Python-side getters
    // see the change before the next step, and interpolation is reset.
    if (pred.is_executable) {
        uint32_t dense  = self->slot_to_dense[slot];
        PosStride p_val = {x, y, z, 0.0};

        // Update both CURRENT and PREVIOUS to "reset" interpolation streak.
        ((PosStride *)self->positions)[dense]      = p_val;
        ((PosStride *)self->prev_positions)[dense] = p_val;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 6. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}
PyCFunction_DeclareMethod PhysicsWorld_set_rotation(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    float w;
    void *targets[SetRot_COUNT] = {[IDX_SETROT_H] = &h_raw,
                                   [IDX_SETROT_X] = &x,
                                   [IDX_SETROT_Y] = &y,
                                   [IDX_SETROT_Z] = &z,
                                   [IDX_SETROT_W] = &w};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.SetRotParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_QUAT(x, y, z, w, "SetRotation");

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Rotation is valid for Alive bodies, Characters, and Pending creations.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // Advance count only if valid (Alive or Pending).
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_SET_ROT, slot);
    cmd->quat.x         = x;
    cmd->quat.y         = y;
    cmd->quat.z         = z;
    cmd->quat.w         = w;

    self->command_count += pred.is_executable;

    // 5. SHADOW BUFFER MIRROR (Zero-Streak Reset)
    // We update both rotations and prev_rotations to ensure NLERP
    // results in exactly this rotation immediately.
    if (pred.is_executable) {
        uint32_t dense    = self->slot_to_dense[slot];
        AuxStride rot_val = {x, y, z, w};

        ((AuxStride *)self->rotations)[dense]      = rot_val;
        ((AuxStride *)self->prev_rotations)[dense] = rot_val;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 6. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    // If NOT executable (e.g. PENDING_DESTROY), use the shim macro.
    // This will return None in release mode, or raise ValueError in strict mode.
    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    void *targets[Vec3_COUNT] = {
        [IDX_V3_H] = &h_raw, [IDX_V3_X] = &x, [IDX_V3_Y] = &y, [IDX_V3_Z] = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.SetLinVelParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "LinearVelocity");

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and virtual characters (MASK_IMM_STANDARD) + Pending bodies
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We write the command immediately and commit it by incrementing count if valid.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_SET_LINVEL, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    self->command_count += pred.is_executable;

    // 5. CAUSAL CONSISTENCY MIRROR
    // Update the shadow buffer immediately so getters see the new velocity in the same frame.
    if (pred.is_executable) {
        uint32_t dense                                = self->slot_to_dense[slot];
        ((AuxStride *)self->linear_velocities)[dense] = (AuxStride){x, y, z, 0.0f};
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 6. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    void *targets[Vec3_COUNT] = {
        [IDX_V3_H] = &h_raw, [IDX_V3_X] = &x, [IDX_V3_Y] = &y, [IDX_V3_Z] = &z};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.SetAngVelParser,
                           targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "AngularVelocity");

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Angular velocity is valid for Rigid Bodies (SLOT_ALIVE) and Pending creations.
    // MASK_IMM_STRICT ensures characters don't receive angular velocity commands.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STRICT);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // Write immediately; increment count only if body is Alive or Pending.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_SET_ANGVEL, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    self->command_count += pred.is_executable;

    // 5. CAUSAL CONSISTENCY MIRROR
    // Update the shadow buffer so Python-side reads are consistent with this write.
    if (pred.is_executable) {
        uint32_t dense                                 = self->slot_to_dense[slot];
        ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){x, y, z, 0.0f};
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 6. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.GetMotionParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Queries are only valid for bodies already in Jolt (Alive or Character).
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (pred.is_immediate) {
        // Safe to read non-atomic shadow buffers while holding SHADOW_LOCK and sim is blocked.
        const uint32_t dense = self->slot_to_dense[slot];
        const JPH_BodyID bid = self->body_ids[dense];

        SHADOW_UNLOCK(&self->shadow_lock);

        // 4. JOLT INTERACTION (Outside Shadow Lock)
        JPH_MotionType mt = JPH_BodyInterface_GetMotionType(self->body_interface, bid);
        return PyLong_FromLong((long)mt);
    }

    // 5. ERROR FALLBACK
    SHADOW_UNLOCK(&self->shadow_lock);

    if (state == SLOT_PENDING_CREATE) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Cannot query motion type of a body that has not been flushed to Jolt yet");
        return nullptr;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    int motion_type;
    void *targets[SetMotion_COUNT] = {[IDX_SM_H] = &h_raw, [IDX_SM_M] = &motion_type};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.SetMotionParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Motion types are only valid for rigid bodies (SLOT_ALIVE) and pending ones.
    // MASK_IMM_STRICT excludes SLOT_CHARACTER.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STRICT);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We write the command immediately; it is committed if the state is executable.
    PhysicsCommand *cmd     = &self->command_queue[self->command_count];
    cmd->header             = CMD_HEADER(CMD_SET_MOTION, slot);
    cmd->motion.motion_type = motion_type;

    self->command_count += pred.is_executable;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    uint64_t data_raw;
    void *targets[SetUserData_COUNT] = {[IDX_SUD_H] = &h_raw, [IDX_SUD_D] = &data_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.SetUserDataParser, targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // User data is valid for Rigid Bodies, Characters (MASK_IMM_STANDARD), and Pending creations.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // Advance count if body is Alive, Character, or Pending.
    PhysicsCommand *cmd          = &self->command_queue[self->command_count];
    cmd->header                  = CMD_HEADER(CMD_SET_USER_DATA, slot);
    cmd->user_data.user_data_val = data_raw;

    self->command_count += pred.is_executable;

    // 5. CAUSAL CONSISTENCY MIRROR (Shadow Buffer)
    // Update the shadow buffer immediately so getters like `get_user_data`
    // see the change in the same frame.
    if (pred.is_executable) {
        uint32_t dense         = self->slot_to_dense[slot];
        self->user_data[dense] = data_raw;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 6. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_get_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.GetUserDataParser, targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // User data reads are valid for Alive, Character, and Pending bodies.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    // 4. IMMEDIATE SHADOW BUFFER READ
    // Shadow buffers are populated immediately during creation, so they are safe
    // to read for is_executable (Immediate + Deferred) states.
    if (pred.is_executable) {
        uint64_t val = self->user_data[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyLong_FromUnsignedLongLong(val);
    }

    // 5. ERROR FALLBACK
    SHADOW_UNLOCK(&self->shadow_lock);
    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_activate(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ActivateParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Activation is valid for Alive bodies, Characters (MASK_IMM_STANDARD), and Pending creations.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We write the command immediately and increment the counter only if executable.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_ACTIVATE, slot);

    self->command_count += pred.is_executable;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_deactivate(PhysicsWorldObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    // Note: Reusing ActivateParser as the schema is identical (single handle)
    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ActivateParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Deactivation is valid for Alive bodies, Characters (MASK_IMM_STANDARD), and Pending
    // creations.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We write the command immediately and increment the counter only if executable.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_DEACTIVATE, slot);

    self->command_count += pred.is_executable;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_transform(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    PyObject *o_pos              = nullptr;
    PyObject *o_rot              = nullptr;
    void *targets[SetTrns_COUNT] = {[IDX_ST_HANDLE] = (void *)&h_raw,
                                    [IDX_ST_POS]    = (void *)&o_pos,
                                    [IDX_ST_ROT]    = (void *)&o_rot};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.SetTrnsParser,
                           targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION (Outside Lock)
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
        return nullptr;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(px, py, pz, "SetTransform position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "SetTransform rotation");

    // 3. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 4. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Transform updates are valid for Alive, Character, and Pending bodies.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 5. SPECULATIVE COMMAND QUEUE WRITE
    // Advance count if the body state is executable.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_SET_TRNS, slot);
    cmd->transform.px   = px;
    cmd->transform.py   = py;
    cmd->transform.pz   = pz;
    cmd->transform.rx   = rx;
    cmd->transform.ry   = ry;
    cmd->transform.rz   = rz;
    cmd->transform.rw   = rw;

    self->command_count += pred.is_executable;

    // 6. SHADOW BUFFER MIRROR (Zero-Streak Reset)
    // Synchronize both current and previous buffers to avoid interpolation artifacts.
    if (pred.is_executable) {
        uint32_t dense  = self->slot_to_dense[slot];
        PosStride p_val = {px, py, pz, 0.0};
        AuxStride r_val = {rx, ry, rz, rw};

        ((PosStride *)self->positions)[dense]      = p_val;
        ((PosStride *)self->prev_positions)[dense] = p_val;
        ((AuxStride *)self->rotations)[dense]      = r_val;
        ((AuxStride *)self->prev_rotations)[dense] = r_val;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 7. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_set_ccd(PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    bool enabled;
    void *targets[CCD_COUNT] = {[IDX_CCD_H] = &h_raw, [IDX_CCD_E] = &enabled};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.CCDParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // CCD updates are valid for Alive bodies, Characters, and Pending creations.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 4. SPECULATIVE COMMAND QUEUE WRITE
    // We write to the current index; we only advance count if the state is executable.
    PhysicsCommand *cmd = &self->command_queue[self->command_count];
    cmd->header         = CMD_HEADER(CMD_SET_CCD, slot);

    // In Jolt logic: 1 = LinearCast (CCD), 0 = Discrete
    cmd->motion.motion_type = (int)enabled;

    self->command_count += pred.is_executable;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. FINAL DISPATCH
    if (LIKELY(pred.is_executable)) {
        Py_RETURN_NONE;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_get_index(PhysicsWorldObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ActivateParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Valid for bodies currently in simulation (Alive or Character).
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (pred.is_immediate) {
        // Shadow mappings are non-atomic; safe to access under shadow_lock + block_stepping.
        uint32_t idx = self->slot_to_dense[slot];
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyLong_FromUnsignedLong(idx);
    }

    // 4. ERROR FALLBACK
    SHADOW_UNLOCK(&self->shadow_lock);
    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_is_alive(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ActivateParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    bool result   = false;

    if (unpack_handle(self, (BodyHandle)h_raw, &slot)) {
        const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

        // Use standard mask (Alive/Character) + Deferred (Pending) via is_executable.
        const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);
        result                   = (bool)pred.is_executable;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. RETURN RESULT
    if (result) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod PhysicsWorld_is_active(PhysicsWorldObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &h_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ActivateParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Only bodies actually in the Jolt system can be "active" or "sleeping".
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (pred.is_immediate) {
        // Shadow buffers and BodyIDs are stable under shadow_lock + block_stepping.
        const uint32_t dense = self->slot_to_dense[slot];
        const JPH_BodyID bid = self->body_ids[dense];

        SHADOW_UNLOCK(&self->shadow_lock);

        // 4. JOLT INTERACTION (Outside Shadow Lock)
        // JPH_BodyInterface_IsActive returns true if the body is currently simulating.
        bool active = JPH_BodyInterface_IsActive(self->body_interface, bid);

        if (active) {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    }

    // 5. FALLBACK
    // Pending bodies (pred.is_deferred) are not yet in the simulation, so they are not "active".
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_active_indices(PhysicsWorldObject *self,
                                                          PyObject *Py_UNUSED(args)) {
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Load count atomically to guarantee consistency with the Stepper thread.
    size_t count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyBytes_FromStringAndSize(nullptr, 0);
    }

    // 1. Snapshot the BodyIDs while locked (Fast)
    // We only access self->body_ids while holding the SHADOW_LOCK and
    // after BLOCK_UNTIL_NOT_STEPPING, ensuring the Stepper thread is idle.
    JPH_BodyID *id_scratch = (JPH_BodyID *)CULV_RAW_MALLOC(count * sizeof(JPH_BodyID));
    if (!id_scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }
    memcpy(id_scratch, self->body_ids, count * sizeof(JPH_BodyID));
    SHADOW_UNLOCK(&self->shadow_lock);

    // 2. Query activity state WHILE UNLOCKED (Deadlock safe)
    // Jolt's IsActive check is thread-safe for reading.
    auto *results = (uint32_t *)CULV_RAW_MALLOC(count * sizeof(uint32_t));
    if (!results) {
        CULV_RAW_FREE(id_scratch);
        return PyErr_NoMemory();
    }

    size_t active_count   = 0;
    JPH_BodyInterface *bi = self->body_interface;

    for (size_t i = 0; i < count; i++) {
        if (id_scratch[i] != JPH_INVALID_BODY_ID &&
            (int)JPH_BodyInterface_IsActive(bi, id_scratch[i])) {
            results[active_count++] = (uint32_t)i;
        }
    }

    // 3. Construct Python object and cleanup
    PyObject *bytes_obj =
        PyBytes_FromStringAndSize((char *)results, (Py_ssize_t)(active_count * sizeof(uint32_t)));
    CULV_RAW_FREE(id_scratch);
    CULV_RAW_FREE(results);
    return bytes_obj;
}

PyCFunction_DeclareMethod PhysicsWorld_get_render_state(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Unchanged)
    float alpha;
    void *targets[Render_COUNT] = {
        [IDX_RND_ALPHA] = (void *)&alpha,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RenderParser, targets)) {
        return nullptr;
    }

    alpha        = fmaxf(0.0f, fminf(1.0f, alpha));
    auto d_alpha = (double)alpha;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Atomic load of count to ensure consistent loop bounds
    size_t count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (UNLIKELY(count == 0)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyBytes_FromStringAndSize(nullptr, 0);
    }

    size_t total_bytes  = count * FLOATS_PER_INTERPOLATED_BODY * sizeof(float);
    PyObject *bytes_obj = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)total_bytes);
    if (!bytes_obj) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    float *out = (float *)PyBytes_AsString(bytes_obj);

    // Map shadow buffers (These are stable while holding SHADOW_LOCK + BLOCK_UNTIL_NOT_STEPPING)
    auto *curr_p = (PosStride *)self->positions;
    auto *prev_p = (PosStride *)self->prev_positions;
    auto *curr_r = (AuxStride *)self->rotations;
    auto *prev_r = (AuxStride *)self->prev_rotations;

    // 2. MATH & INTERPOLATION
    for (size_t i = 0; i < count; i++) {
        size_t dst = i * FLOATS_PER_INTERPOLATED_BODY;

        // Position Lerp (Double precision)
        JPH_Real px = prev_p[i].x + (curr_p[i].x - prev_p[i].x) * d_alpha;
        JPH_Real py = prev_p[i].y + (curr_p[i].y - prev_p[i].y) * d_alpha;
        JPH_Real pz = prev_p[i].z + (curr_p[i].z - prev_p[i].z) * d_alpha;

        out[dst + 0] = (float)px;
        out[dst + 1] = (float)py;
        out[dst + 2] = (float)pz;

        // Rotation NLerp (Float precision)
        float q1x = prev_r[i].x;
        float q1y = prev_r[i].y;
        float q1z = prev_r[i].z;
        float q1w = prev_r[i].w;
        float q2x = curr_r[i].x;
        float q2y = curr_r[i].y;
        float q2z = curr_r[i].z;
        float q2w = curr_r[i].w;

        float dot = q1x * q2x + q1y * q2y + q1z * q2z + q1w * q2w;
        if (dot < 0.0f) {
            q2x = -q2x;
            q2y = -q2y;
            q2z = -q2z;
            q2w = -q2w;
        }

        float rx = q1x + (q2x - q1x) * alpha;
        float ry = q1y + (q2y - q1y) * alpha;
        float rz = q1z + (q2z - q1z) * alpha;
        float rw = q1w + (q2w - q1w) * alpha;

        float mag_sq  = rx * rx + ry * ry + rz * rz + rw * rw;
        float inv_len = (mag_sq > EPSILON_QUATERNION_NORMALIZATION) ? 1.0f / sqrtf(mag_sq) : 1.0f;

        out[dst + 3]                                = rx * inv_len;
        out[dst + 4]                                = ry * inv_len;
        out[dst + QUATERNION_INTERPOLATION_Z_INDEX] = rz * inv_len;
        out[dst + QUATERNION_INTERPOLATION_W_INDEX] = rw * inv_len;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    return bytes_obj;
}

PyCFunction_DeclareMethod PhysicsWorld_set_collision_filter(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    uint32_t category;
    uint32_t mask;
    void *targets[ColFilter_COUNT] = {
        [IDX_CF_H] = &h_raw, [IDX_CF_C] = &category, [IDX_CF_M] = &mask};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.ColFilterParser,
                           targets)) {
        return nullptr;
    }

    // 2. CONCURRENCY CONTROL
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes (like collision filters) must block for both simulation and queries
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // 3. RESOLUTION & PREDICATE CALCULATION
    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Collision filters are valid for Alive bodies and Characters.
    const SlotPredicate pred = get_slot_predicate(state, MASK_IMM_STANDARD);

    if (pred.is_immediate) {
        // Since we hold the SHADOW_LOCK and the simulation is idle,
        // these arrays are stable and non-atomic writes are safe.
        const uint32_t dense = self->slot_to_dense[slot];

        self->categories[dense] = category;
        self->masks[dense]      = mask;

        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    // 4. ERROR FALLBACK
    SHADOW_UNLOCK(&self->shadow_lock);

    // If pending, we usually expect the user to have set filters during creation.
    // Late-setting filters for pending bodies is not supported in the immediate shadow path.
    if (state == SLOT_PENDING_CREATE) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Cannot update collision filters of a body until it has been flushed to Jolt");
        return nullptr;
    }

    RAISE_STALE_HANDLE();
}

PyCFunction_DeclareMethod PhysicsWorld_register_material(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES
    uint32_t id;
    float friction    = RESTITUTION_BUFFER;
    float restitution = 0.0f;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[RegMat_COUNT] = {
        [IDX_RM_ID]   = (void *)&id,
        [IDX_RM_FRIC] = (void *)&friction,
        [IDX_RM_REST] = (void *)&restitution,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.RegMatParser, targets)) {
        return nullptr;
    }

    // 3. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes (registry expansion) require global world idleness
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // Update existing material if ID is already registered
    // Safe: SHADOW_LOCK + BLOCK_UNTIL_NOT_QUERYING ensures no callbacks are iterating this
    for (size_t i = 0; i < self->material_count; i++) {
        if (self->materials[i].id == id) {
            self->materials[i].friction    = friction;
            self->materials[i].restitution = restitution;
            SHADOW_UNLOCK(&self->shadow_lock);
            Py_RETURN_NONE;
        }
    }

    // Grow capacity if needed
    if (self->material_count >= self->material_capacity) {
        size_t new_cap = (self->material_capacity == 0) ? INITIAL_MATERIAL_CAPACITY
                                                        : self->material_capacity * 2;
        auto *new_ptr =
            (MaterialData *)CULV_RAW_REALLOC(self->materials, new_cap * sizeof(MaterialData));
        if (UNLIKELY(!new_ptr)) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }
        self->materials         = new_ptr;
        self->material_capacity = new_cap;
    }

    // Add new material entry
    self->materials[self->material_count].id          = id;
    self->materials[self->material_count].friction    = friction;
    self->materials[self->material_count].restitution = restitution;
    self->material_count++;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_create_heightfield(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE & VALIDATION
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_scale    = nullptr;
    PyObject *o_heights  = nullptr;
    int grid_size        = 0;
    uint64_t user_data   = 0;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    float friction       = DEFAULT_FRICTION;
    float restitution    = 0.0f;

    void *targets[Heightfield_COUNT] = {
        [IDX_HF_POS] = (void *)&o_pos,           [IDX_HF_ROT] = (void *)&o_rot,
        [IDX_HF_SCALE] = (void *)&o_scale,       [IDX_HF_HEIGHTS] = (void *)&o_heights,
        [IDX_HF_GRID_SIZE] = (void *)&grid_size, [IDX_HF_USER_DATA] = (void *)&user_data,
        [IDX_HF_CAT] = (void *)&category,        [IDX_HF_MASK] = (void *)&mask,
        [IDX_HF_MAT_ID] = (void *)&material_id,  [IDX_HF_FRIC] = (void *)&friction,
        [IDX_HF_REST] = (void *)&restitution};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.HeightfieldParser, targets)) {
        return nullptr;
    }

    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    float sx;
    float sy;
    float sz;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw) ||
        !parse_vec3_direct(o_scale, &sx, &sy, &sz)) {
        return nullptr;
    }

    Py_buffer h_view;
    if (PyObject_GetBuffer(o_heights, &h_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (UNLIKELY(h_view.len != (Py_ssize_t)((Py_ssize_t)grid_size * grid_size * sizeof(float)))) {
        PyBuffer_Release(&h_view);
        return PyErr_Format(PyExc_ValueError, "Height buffer size mismatch");
    }

    // 2. SHAPE CREATION (No GIL)
    JPH_Shape *shape = nullptr;
    Py_BEGIN_ALLOW_THREADS;
    JPH_Vec3 offset                  = {};
    JPH_Vec3 scale                   = {sx, sy, sz};
    JPH_HeightFieldShapeSettings *hf = JPH_HeightFieldShapeSettings_Create(
        (float *)h_view.buf, &offset, &scale, (uint32_t)grid_size, nullptr);
    if (hf) {
        shape = (JPH_Shape *)JPH_HeightFieldShapeSettings_CreateShape(hf);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hf);
    }
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&h_view);
    if (!shape) {
        return PyErr_Format(PyExc_RuntimeError, "HeightField build failed");
    }

    // 3. SETTINGS PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static,
        OBJECT_LAYER_STATIC);
    JPH_BodyCreationSettings_SetFriction(settings, friction);
    JPH_BodyCreationSettings_SetRestitution(settings, restitution);

    // 4. COMMIT PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint64_t raw_h = physics_world_commit_create_locked(self, settings, SLOT_PENDING_CREATE);

    if (UNLIKELY(!raw_h)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(shape);
        return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();
    }

    // 5. SHADOW BUFFER UPDATE
    uint32_t slot  = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense = self->slot_to_dense[slot];

    ((PosStride *)self->positions)[dense]      = (PosStride){px, py, pz, 0.0};
    ((PosStride *)self->prev_positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense]      = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->prev_rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->categories[dense]                    = category;
    self->masks[dense]                         = mask;
    self->material_ids[dense]                  = material_id;
    self->user_data[dense]                     = user_data;

    // 6. QUEUE COMMAND
    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings    = settings;
    cmd->create.user_data   = user_data;
    cmd->create.category    = category;
    cmd->create.mask        = mask;
    cmd->create.material_id = material_id;

    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_Shape_Destroy(shape);
    return PyLong_FromUnsignedLongLong(raw_h);
}

PyCFunction_DeclareMethod PhysicsWorld_get_debug_data(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES
    bool draw_shapes       = true;
    bool draw_constraints  = true;
    bool draw_bounding_box = false;
    bool draw_centers      = false;
    bool wireframe         = true;

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[DebugData_COUNT] = {
        [IDX_DD_SHAPES]      = (void *)&draw_shapes,
        [IDX_DD_CONSTRAINTS] = (void *)&draw_constraints,
        [IDX_DD_BBOX]        = (void *)&draw_bounding_box,
        [IDX_DD_CENTERS]     = (void *)&draw_centers,
        [IDX_DD_WIREFRAME]   = (void *)&wireframe,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.DebugDataParser, targets)) {
        return nullptr;
    }

    // 3. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Debug rendering iterates Jolt bodies; ensure simulation is idle
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Reset Buffer Counts (Reuses existing raw memory in debug_lines/triangles)
    self->debug_lines.count     = 0;
    self->debug_triangles.count = 0;

    // 4. JOLT INTERACTION
    JPH_DrawSettings settings;
    JPH_DrawSettings_InitDefault(&settings);
    settings.drawShape                 = draw_shapes;
    settings.drawShapeWireframe        = wireframe;
    settings.drawBoundingBox           = draw_bounding_box;
    settings.drawCenterOfMassTransform = draw_centers;

    // Draw Bodies into our internal DebugBuffers
    if ((int)draw_shapes || (int)draw_bounding_box || (int)draw_centers) {
        JPH_PhysicsSystem_DrawBodies(self->system, &settings, self->debug_renderer, nullptr);
    }

    // Draw Constraints
    if (draw_constraints) {
        JPH_PhysicsSystem_DrawConstraints(self->system, self->debug_renderer);
        JPH_PhysicsSystem_DrawConstraintLimits(self->system, self->debug_renderer);
    }

    // 5. EXPORT TO PYTHON BYTES
    // We snapshot the raw C-arrays into immutable Python bytes objects
    PyObject *lines_bytes =
        PyBytes_FromStringAndSize((const char *)self->debug_lines.data,
                                  (Py_ssize_t)(self->debug_lines.count * sizeof(DebugVertex)));

    PyObject *tris_bytes =
        PyBytes_FromStringAndSize((const char *)self->debug_triangles.data,
                                  (Py_ssize_t)(self->debug_triangles.count * sizeof(DebugVertex)));

    SHADOW_UNLOCK(&self->shadow_lock);

    if (UNLIKELY(!lines_bytes || !tris_bytes)) {
        Py_XDECREF(lines_bytes);
        Py_XDECREF(tris_bytes);
        return PyErr_NoMemory();
    }

    // Return as (lines, triangles)
    PyObject *ret = PyTuple_Pack(2, lines_bytes, tris_bytes);
    Py_DECREF(lines_bytes);
    Py_DECREF(tris_bytes);
    return ret;
}

PyCFunction_DeclareMethod culv_dump_schema_json(PyObject *self, PyObject *Py_UNUSED(args)) {
    // self is the module object
    CulverinState *st = get_culverin_state(self);

    const char *filename = "culverin_schema.json";
    FILE *f              = fopen(filename, "w");
    if (!f) {
        return PyErr_SetFromErrno(PyExc_IOError);
    }

    // Pass the pointer to the parser struct and the file handle
    fp_dump_schemas_json(&st->parsers, f);

    fclose(f);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_benchmark_parse(CULV_MAYBE_UNUSED PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    CulverinState *st         = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    constexpr size_t NUM_ARGS = 64;
    uint64_t values[NUM_ARGS] = {};
    void *targets[NUM_ARGS];

    // Map targets to values array (Loop is efficient here, unrolled by compiler)
    for (size_t i = 0; i < NUM_ARGS; ++i) {
        targets[i] = &values[i];
    }

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (UNLIKELY(
            !FastParse_Unified(args, nargs, kwnames, &st->parsers.StressTestParser, targets))) {
        return nullptr;
    }

    Py_RETURN_NONE;
}

/**
 * Benchmark for FastBuild_Tuple using METH_NOARGS.
 * This eliminates all argument parsing overhead to isolate the
 * performance of the Culverin Fast Build engine.
 */
PyCFunction_DeclareMethod PhysicsWorld_benchmark_build(CULV_MAYBE_UNUSED PyObject *self,
                                                       CULV_MAYBE_UNUSED PyObject *args) {
    // 1. Define dummy C data to be "built" into Python
    constexpr int trash         = 42;
    constexpr float trashpi     = 3.14f;
    constexpr double trasheuler = 2.718281828;
    int i_val                   = trash;
    float f_val                 = trashpi;
    double d_val                = trasheuler;
    const char *s_val =
        "Pater noster qui es in caelis, sanctificetur nomen tuum. Adveniat regnum tuum. Fiat "
        "voluntas tua, sicut in caelo et in terra. Panem nostrum quotidianum da nobis hodie, et "
        "dimitte nobis debita nostra sicut et nos dimittimus debitoribus nostris. Et ne nos "
        "inducas in tentationem, sed libera nos a malo. Amen.";
    bool b_val = true;

    // 2. Execute the Build
    // FastBuild_Tuple uses FB_VAL to route types at compile-time.
    // It then uses fb_pack_tuple to perform O(1) allocation and
    // O(N) reference stealing.
    constexpr float stfu = 200.0f;
    PyObject *result =
        FastBuild_Tuple(i_val, f_val, d_val, s_val, b_val, (int)100, (float)stfu,
                        "Lorem ipsum dolor sit amet consectetur adipiscing elit, sed do eiusmod "
                        "tempor incididunt ut labore et dolore magna aliqua.",
                        false);

    // 3. Error handling
    if (UNLIKELY(!result)) {
        // fb_pack_tuple handles internal cleanup on failure
        return nullptr;
    }

    // Return the new reference to the caller
    return result;
}

#include <stdatomic.h>
#include <tupleobject.h>

PyCFunction_DeclareMethod culv_mutate_tuple(CULV_MAYBE_UNUSED PyObject *self, PyObject *const *args,
                                            Py_ssize_t nargsf) {
    // --- 1. DECLARATIONS (C++ / goto safety) ---
    auto nargs              = PyVectorcall_NARGS(nargsf);
    constexpr auto MIN_ARGS = 3;
    constexpr auto MAX_ARGS = 5;

    PyObject *target   = nullptr;
    PyObject *new_val  = nullptr;
    PyObject *registry = nullptr;
    PyObject *key      = nullptr;
    PyObject *old_val  = nullptr;
    PyObject *stored   = nullptr;

    Py_ssize_t index     = 0;
    Py_hash_t final_hash = -1;
    int success          = 0;

    // --- 2. PRE-FLIGHT CHECKS ---
    if (nargs != MIN_ARGS && nargs != MAX_ARGS) {
        PyErr_Format(PyExc_TypeError, "mutate_tuple() takes 3 or 5 arguments (%zd given)", nargs);
        return nullptr;
    }

    target  = args[0];
    new_val = args[2];
    if (!PyTuple_Check(target)) {
        PyErr_SetString(PyExc_TypeError, "arg 0 must be a tuple");
        return nullptr;
    }

    index = PyLong_AsSsize_t(args[1]);
    if (index == -1 && PyErr_Occurred()) {
        return nullptr;
    }

    Py_ssize_t tuple_len = Py_SIZE(target);
    if (index < 0) {
        index += tuple_len;
    }
    if (index < 0 || index >= tuple_len) {
        PyErr_Format(PyExc_IndexError, "tuple index %zd out of range", index);
        return nullptr;
    }

    if (nargs == MAX_ARGS) {
        registry = args[3];
        key      = args[4];
        if (!PyDict_Check(registry)) {
            PyErr_SetString(PyExc_TypeError, "arg 3 must be a dict");
            return nullptr;
        }
    }

    // --- 3. CRITICAL SECTION ---
#if defined(Py_BEGIN_CRITICAL_SECTION)
    Py_BEGIN_CRITICAL_SECTION(target);
#endif

    // Step A: Registry Removal
    if (registry) {
        stored = PyDict_GetItemWithError(registry, key);
        if (!stored) {
            if (!PyErr_Occurred()) {
                PyErr_SetObject(PyExc_KeyError, key);
            }
            goto exit_critical;
        }
        if (stored != target) {
            PyErr_SetString(PyExc_ValueError, "registry[key] is not the same object as target");
            goto exit_critical;
        }
        Py_INCREF(target);
        if (PyDict_DelItem(registry, key) < 0) {
            Py_DECREF(target);
            goto exit_critical;
        }
    }

    // Step B: The Mutation (Atomic Swap)
    {
        PyTupleObject *t = (PyTupleObject *)target;
        Py_INCREF(new_val);

        // Cast the item slot to an atomic pointer.
        // This prevents TSan "Data Race" warnings and ensures memory visibility.
        _Atomic(PyObject *) *atomic_slot = (_Atomic(PyObject *) *)&(t->ob_item[index]);
        old_val = atomic_exchange_explicit(atomic_slot, new_val, memory_order_acq_rel);

        // Step C: Rehash Cache Bust (Atomic)
#if PY_VERSION_HEX >= 0x030E0000
        _Atomic(Py_hash_t) *atomic_hash = (_Atomic(Py_hash_t) *)&(t->ob_hash);
        atomic_store_explicit(atomic_hash, -1, memory_order_relaxed);
#endif
    }

    final_hash = PyObject_Hash(target);

    if (final_hash == -1) {
        // Rollback (Atomic)
        _Atomic(PyObject *) *atomic_slot =
            (_Atomic(PyObject *) *)&((PyTupleObject *)target)->ob_item[index];
        atomic_store_explicit(atomic_slot, old_val, memory_order_relaxed);
        Py_DECREF(new_val);
        if (registry) {
            Py_DECREF(target);
        }
        goto exit_critical;
    }

    // Step D: Registry Re-insertion
    if (registry) {
        if (PyDict_SetItem(registry, key, target) < 0) {
            // Rollback (Atomic)
            _Atomic(PyObject *) *atomic_slot =
                (_Atomic(PyObject *) *)&((PyTupleObject *)target)->ob_item[index];
            atomic_store_explicit(atomic_slot, old_val, memory_order_relaxed);
            Py_DECREF(new_val);
            Py_DECREF(target);
            final_hash = -1;
            goto exit_critical;
        }
        Py_DECREF(target);
    }

    Py_DECREF(old_val);
    success = 1;

exit_critical:
#if defined(Py_BEGIN_CRITICAL_SECTION)
    Py_END_CRITICAL_SECTION();
#endif

    if (!success) {
        return nullptr;
    }
    return PyLong_FromSsize_t(final_hash);
}

// --- The Documentation System ---

#ifndef CULVERIN_DOCS_PATH
#    define CULVERIN_DOCS_PATH "../../docs/DOCS.md"
#endif

static const char ALL_DOCS[] = {
// NOLINTNEXTLINE(readability-magic-numbers)
#embed CULVERIN_DOCS_PATH suffix(, 0)
};

// Global flag to ensure we only stitch once (important for subinterpreters)
static atomic_int docs_status = 0;

static const char *COMMENT_MARKER = "<!--";

static char *allocate_docstring(const char *start, size_t length) {
    char *doc = (char *)PyMem_RawMalloc(length + 1);
    if (doc == nullptr) {
        return nullptr;
    }
    memcpy(doc, start, length);
    doc[length] = '\0';
    return doc;
}

// Master Docstring Extractor (Handles Nested ## class -> ### method)
static char *extract_docstring(const char *class_name, const char *method_name) {
    static constexpr size_t DOC_BUFFER = 128;
    char class_key[DOC_BUFFER];
    snprintf(class_key, sizeof(class_key), "## class %s", class_name);

    // 1. Find the Class boundary
    char *class_start = strstr(ALL_DOCS, class_key);
    if (!class_start) {
        return nullptr;
    }

    // The scope of this class ends when the next class begins
    char *class_end = strstr(class_start + 1, "## class ");

    // 2. Find the Method boundary within this class
    char method_key[DOC_BUFFER];
    snprintf(method_key, sizeof(method_key), "### %s", method_name);

    char *method_start = strstr(class_start, method_key);

    // Ensure we found it AND it belongs to THIS class AND isn't a substring (like finding "step_up"
    // for "step")
    while (method_start && (class_end == nullptr || method_start < class_end)) {
        char c = *(method_start + strlen(method_key));
        // We allow '(', '\r', '\n', ' ', or '\0' after the method name
        if (c == '(' || c == '\r' || c == '\n' || c == ' ' || c == '\0') {
            break; // Valid match
        }
        method_start = strstr(method_start + 1, method_key); // Keep searching
    }

    if (!method_start || (class_end != nullptr && method_start >= class_end)) {
        return nullptr;
    }

    // 3. Move past the "### method(...)\n" header
    char *doc_start = method_start;
    while (*doc_start != '\0' && *doc_start != '\n' && *doc_start != '\r') {
        doc_start++;
    }
    while (*doc_start == '\n' || *doc_start == '\r') {
        doc_start++;
    }

    // 4. Skip HTML Comments if present
    if (strncmp(doc_start, COMMENT_MARKER, 4) == 0) {
        char *comment_end = strstr(doc_start, "-->");
        if (comment_end) {
            doc_start = comment_end + 3; // Skip "-->"
            while (*doc_start == '\n' || *doc_start == '\r' || *doc_start == ' ') {
                doc_start++;
            }
        }
    }

    // 5. Find the end of this method's docs (next ### or ##)
    char *doc_end = doc_start;
    while (*doc_end != '\0') {
        if (strncmp(doc_end, "### ", 4) == 0 || strncmp(doc_end, "## ", 3) == 0) {
            break;
        }
        doc_end++;
    }

    // 6. Trim trailing whitespace
    if (doc_end > doc_start) {
        doc_end--;
        while (doc_end > doc_start &&
               (*doc_end == '\n' || *doc_end == '\r' || *doc_end == ' ' || *doc_end == '\t')) {
            doc_end--;
        }
    }

    if (doc_end >= doc_start) {
        return allocate_docstring(doc_start, (size_t)(doc_end - doc_start + 1));
    }

    return nullptr;
}

// Pass 1: Stitch docstrings to PyMethodDefs
static void stitch_docs(PyMethodDef *methods, const char *class_name) {
    if (methods == nullptr) {
        return;
    }
    for (PyMethodDef *m = methods; m->ml_name != nullptr; m++) {
        if (m->ml_doc == nullptr) {
            m->ml_doc = extract_docstring(class_name, m->ml_name);
        }
    }
}

// Pass 2: Stitch docstrings into PyGetSetDef getters
static void stitch_docs_getset(PyGetSetDef *getset, const char *class_name) {
    if (getset == nullptr) {
        return;
    }
    for (PyGetSetDef *g = getset; g->name != nullptr; g++) {
        if (g->doc == nullptr) {
            g->doc = extract_docstring(class_name, g->name);
        }
    }
}

// =============================================================================================

// --- Macros ---

#define CULV_CAST(m) (PyCFunction)(void (*)(void))(m)

#define CULV_FEAT(prefix, name, method_type)                                                       \
    {.ml_name  = #name,                                                                            \
     .ml_meth  = CULV_CAST(prefix##_##name),                                                       \
     .ml_flags = (method_type),                                                                    \
     .ml_doc   = nullptr} // Initialized to nullptr to be filled by stitcher

#define CULV_FEAT_INTERNAL(prefix, name, method_type)                                              \
    {.ml_name  = "_" #name,                                                                        \
     .ml_meth  = (PyCFunction)prefix##_##name,                                                     \
     .ml_flags = (method_type),                                                                    \
     .ml_doc   = nullptr}

// User-facing macros for context methods
#define PW_FASTCALL(name) CULV_FEAT(PhysicsWorld, name, METH_FASTCALL | METH_KEYWORDS)
#define PW_NOARGS(name) CULV_FEAT(PhysicsWorld, name, METH_NOARGS)
#define PW_O(name) CULV_FEAT(PhysicsWorld, name, METH_O)

#define CHAR_FASTCALL(name) CULV_FEAT(Character, name, METH_FASTCALL | METH_KEYWORDS)
#define CHAR_NOARGS(name) CULV_FEAT(Character, name, METH_NOARGS)
#define CHAR_O(name) CULV_FEAT(Character, name, METH_O)

#define VEH_FASTCALL(name) CULV_FEAT(Vehicle, name, METH_FASTCALL | METH_KEYWORDS)
#define VEH_NOARGS(name) CULV_FEAT(Vehicle, name, METH_NOARGS)

#define SKEL_FASTCALL(name) CULV_FEAT(Skeleton, name, METH_FASTCALL | METH_KEYWORDS)
#define SKEL_NOARGS(name) CULV_FEAT(Skeleton, name, METH_NOARGS)

#define RD_FASTCALL(name) CULV_FEAT(Ragdoll, name, METH_FASTCALL | METH_KEYWORDS)
#define RD_NOARGS(name) CULV_FEAT(Ragdoll, name, METH_NOARGS)

#define RDS_FASTCALL(name) CULV_FEAT(RagdollSettings, name, METH_FASTCALL | METH_KEYWORDS)
#define RDS_NOARGS(name) CULV_FEAT(RagdollSettings, name, METH_NOARGS)

#define SBSS_FASTCALL(name) CULV_FEAT(SoftBodySharedSettings, name, METH_FASTCALL | METH_KEYWORDS)
#define SBSS_NOARGS(name) CULV_FEAT(SoftBodySharedSettings, name, METH_NOARGS)
#define SBSS_O(name) CULV_FEAT(SoftBodySharedSettings, name, METH_O)

// Getter/Property macro - concise initialization
#define GETSET(name_str, getter_func)                                                              \
    {.name    = (name_str),                                                                        \
     .get     = (getter)(getter_func),                                                             \
     .set     = nullptr,                                                                           \
     .doc     = nullptr,                                                                           \
     .closure = nullptr}

// User-facing macros for module-level methods
#define MOD_FASTCALL(name) CULV_FEAT(culv, name, METH_FASTCALL | METH_KEYWORDS)
#define MOD_NOARGS(name) CULV_FEAT(culv, name, METH_NOARGS)

#define MOD_NOARGS_INTERNAL(name) CULV_FEAT_INTERNAL(culv, name, METH_NOARGS)

// --- Module Method Definitions ---
static PyMethodDef module_methods[] = {
    MOD_NOARGS_INTERNAL(dump_schema_json), // Stitches from ### dump_schema in ## class Module
    MOD_FASTCALL(mutate_tuple),            // Stitches from ### mutate_tuple in ## class Module
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef PhysicsWorld_getset[] = {
    GETSET("positions", get_positions),
    GETSET("rotations", get_rotations),
    GETSET("velocities", get_velocities),
    GETSET("angular_velocities", get_angular_velocities),
    GETSET("count", get_count),
    GETSET("time", get_time),
    GETSET("user_data", get_user_data_buffer),
    GETSET("shape_count", get_shape_count),
    GETSET("is_step_pending", get_is_step_pending),
    GETSET("max_bodies", PhysicsWorld_get_max_bodies),
    GETSET("remaining_capacity", PhysicsWorld_get_remaining_capacity),
    {.name = nullptr, .get = nullptr, .set = nullptr, .doc = nullptr, .closure = nullptr}};

static PyGetSetDef Character_getset[] = {GETSET("handle", Character_get_handle),
                                         {nullptr, nullptr, nullptr, nullptr, nullptr}};

static PyGetSetDef Vehicle_getset[] = {GETSET("wheel_count", Vehicle_get_wheel_count),
                                       {nullptr, nullptr, nullptr, nullptr, nullptr}};

// --- Method Definitions ---
// IMPORTANT: REMOVE 'const' so the memory is writable!
static PyMethodDef PhysicsWorld_methods[] = {
    // --- Lifecycle ---
    PW_FASTCALL(step),
    PW_FASTCALL(create_body),
    PW_FASTCALL(create_bodies_batch),
    PW_FASTCALL(destroy_body),
    PW_FASTCALL(destroy_bodies_batch),
    PW_FASTCALL(create_mesh_body),
    PW_FASTCALL(create_constraint),
    PW_FASTCALL(destroy_constraint),
    PW_FASTCALL(get_constraint_type),
    PW_FASTCALL(create_vehicle),
    PW_FASTCALL(create_tracked_vehicle),
    PW_FASTCALL(create_ragdoll_settings),
    PW_FASTCALL(create_ragdoll),
    PW_FASTCALL(create_heightfield),
    PW_FASTCALL(create_convex_hull),
    PW_FASTCALL(create_compound_body),
    PW_FASTCALL(create_soft_body),

    // --- Interaction ---
    PW_FASTCALL(apply_impulse),
    PW_FASTCALL(apply_angular_impulse),
    PW_FASTCALL(apply_impulse_at),
    PW_FASTCALL(apply_force),
    PW_FASTCALL(apply_torque),
    PW_FASTCALL(set_gravity),
    PW_NOARGS(get_gravity),
    PW_FASTCALL(apply_buoyancy),
    PW_FASTCALL(apply_buoyancy_batch),
    PW_FASTCALL(set_position),
    PW_FASTCALL(set_rotation),
    PW_FASTCALL(set_linear_velocity),
    PW_FASTCALL(set_angular_velocity),
    PW_FASTCALL(set_transform),
    PW_FASTCALL(set_collision_filter),
    PW_FASTCALL(register_material),
    PW_FASTCALL(set_constraint_target),

    // --- Motion Control ---
    PW_FASTCALL(get_motion_type),
    PW_FASTCALL(set_motion_type),
    PW_FASTCALL(activate),
    PW_FASTCALL(deactivate),
    PW_FASTCALL(set_ccd),

    // --- Queries ---
    PW_FASTCALL(get_soft_body_vertices),
    PW_FASTCALL(raycast),
    PW_FASTCALL(raycast_batch),
    PW_FASTCALL(shapecast),
    PW_FASTCALL(overlap_sphere),
    PW_FASTCALL(overlap_aabb),

    // --- Utilities ---
    PW_FASTCALL(get_index),
    PW_FASTCALL(is_alive),
    PW_FASTCALL(is_active),
    PW_NOARGS(get_active_indices),
    PW_FASTCALL(get_render_state),
    PW_FASTCALL(get_debug_data),
    PW_FASTCALL(get_body_stats),

    // --- User Data ---
    PW_FASTCALL(get_user_data),
    PW_FASTCALL(set_user_data),

    // -- Event Logic ---
    PW_NOARGS(get_contact_events),
    PW_NOARGS(get_contact_events_ex),
    PW_NOARGS(get_contact_events_raw),

    // --- State & Advanced ---
    PW_NOARGS(save_state),
    PW_FASTCALL(load_state),
    PW_FASTCALL(create_character),

    // --- Internal/Debug ---
    // Not for public use, therefore can't use macros
    {"_benchmark_parse", CULV_CAST(PhysicsWorld_benchmark_parse), METH_FASTCALL | METH_KEYWORDS,
     nullptr},

    {"_benchmark_build", CULV_CAST(PhysicsWorld_benchmark_build), METH_NOARGS, nullptr},

    {nullptr, nullptr, 0, nullptr}};

static PyMethodDef Character_methods[] = {
    CHAR_FASTCALL(move),          CHAR_NOARGS(get_position),     CHAR_FASTCALL(set_position),
    CHAR_FASTCALL(set_rotation),  CHAR_NOARGS(is_grounded),      CHAR_FASTCALL(set_strength),
    CHAR_O(get_render_transform), {nullptr, nullptr, 0, nullptr}};

static PyMethodDef Vehicle_methods[] = {VEH_FASTCALL(set_input),
                                        VEH_FASTCALL(set_tank_input),
                                        VEH_FASTCALL(get_wheel_transform),
                                        VEH_FASTCALL(get_wheel_local_transform),
                                        VEH_NOARGS(destroy),
                                        VEH_NOARGS(get_debug_state),
                                        {nullptr, nullptr, 0, nullptr}};

static PyMethodDef Skeleton_methods[] = {SKEL_FASTCALL(add_joint),
                                         SKEL_FASTCALL(get_joint_index),
                                         SKEL_NOARGS(finalize),
                                         {nullptr, nullptr, 0, nullptr}};

static PyMethodDef Ragdoll_methods[] = {RD_FASTCALL(drive_to_pose),
                                        RD_NOARGS(get_body_handles),
                                        RD_NOARGS(get_debug_info),
                                        {nullptr, nullptr, 0, nullptr}};

static PyMethodDef RagdollSettings_methods[] = {
    RDS_FASTCALL(add_part), RDS_NOARGS(stabilize), {nullptr, nullptr, 0, nullptr}};

static PyMethodDef SoftBodySharedSettings_methods[] = {
    SBSS_FASTCALL(add_vertex),         SBSS_FASTCALL(add_vertices), SBSS_O(add_pinned_vertex),
    SBSS_O(get_vertex_position),       SBSS_FASTCALL(add_face),     SBSS_FASTCALL(add_faces),
    SBSS_FASTCALL(create_constraints), SBSS_NOARGS(optimize),       {nullptr, nullptr, 0, nullptr}};

static const PyMemberDef PhysicsWorld_members[] = {
    {.name   = "__weaklistoffset__",
     .type   = Py_T_PYSSIZET,
     .offset = offsetof(PhysicsWorldObject, weakreflist),
     .flags  = Py_READONLY,
     .doc    = nullptr},
    {.name = nullptr, .type = 0, .offset = 0, .flags = 0, .doc = nullptr}};

static const PyType_Slot PhysicsWorld_slots[] = {
    {.slot = Py_tp_new, .pfunc = PyType_GenericNew},
    {.slot = Py_tp_init, .pfunc = PhysicsWorld_init},
    {.slot = Py_tp_dealloc, .pfunc = PhysicsWorld_dealloc},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)PhysicsWorld_methods},
    {.slot = Py_tp_members, .pfunc = (PyMemberDef *)PhysicsWorld_members},
    {.slot = Py_tp_getset, .pfunc = (PyGetSetDef *)PhysicsWorld_getset},
    {.slot = Py_bf_getbuffer, .pfunc = PhysicsWorld_getbuffer},
    {.slot = Py_bf_releasebuffer, .pfunc = PhysicsWorld_releasebuffer},
    {.slot = Py_tp_traverse, .pfunc = PhysicsWorld_traverse},
    {.slot = Py_tp_clear, .pfunc = PhysicsWorld_clear},
    {.slot = 0, .pfunc = nullptr},
};

static const PyType_Slot Character_slots[] = {
    {.slot = Py_tp_dealloc, .pfunc = Character_dealloc},
    {.slot = Py_tp_traverse, .pfunc = Character_traverse},
    {.slot = Py_tp_clear, .pfunc = Character_clear},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)Character_methods},
    {.slot = Py_tp_getset, .pfunc = (PyGetSetDef *)Character_getset},
    {.slot = 0, .pfunc = nullptr},
};

static const PyType_Slot Vehicle_slots[] = {
    {.slot = Py_tp_dealloc, .pfunc = Vehicle_dealloc},
    {.slot = Py_tp_traverse, .pfunc = Vehicle_traverse},
    {.slot = Py_tp_clear, .pfunc = Vehicle_clear},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)Vehicle_methods},
    {.slot = Py_tp_getset, .pfunc = (PyGetSetDef *)Vehicle_getset},
    {.slot = 0, .pfunc = nullptr},
};

static const PyType_Slot Skeleton_slots[] = {
    {.slot = Py_tp_new, .pfunc = Skeleton_new},
    {.slot = Py_tp_dealloc, .pfunc = Skeleton_dealloc},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)Skeleton_methods},
    {.slot = 0, .pfunc = nullptr},
};

static const PyType_Spec Skeleton_spec = {
    .name      = "culverin._culverin_c.Skeleton",
    .basicsize = sizeof(SkeletonObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)Skeleton_slots,
};

static const PyType_Slot RagdollSettings_slots[] = {
    {.slot = Py_tp_dealloc, .pfunc = RagdollSettings_dealloc},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)RagdollSettings_methods},
    {.slot = 0, nullptr},
};

static const PyType_Spec PhysicsWorld_spec = {
    .name      = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(PhysicsWorldObject),
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_MANAGED_DICT,
    .slots = (PyType_Slot *)PhysicsWorld_slots,
};

static const PyType_Spec Character_spec = {
    .name      = "culverin._culverin_c.Character",
    .basicsize = sizeof(CharacterObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = (PyType_Slot *)Character_slots,
};

static const PyType_Spec Vehicle_spec = {
    .name      = "culverin._culverin_c.Vehicle",
    .basicsize = sizeof(VehicleObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = (PyType_Slot *)Vehicle_slots,
};

static const PyType_Spec RagdollSettings_spec = {
    .name      = "culverin._culverin_c.RagdollSettings",
    .basicsize = sizeof(RagdollSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)RagdollSettings_slots,
};

// Update the SoftBodySharedSettings_slots array
static const PyType_Slot SoftBodySharedSettings_slots[] = {
    {.slot = Py_tp_new, .pfunc = PyType_GenericNew},
    {.slot = Py_tp_init, .pfunc = SoftBodySharedSettings_init},
    {.slot = Py_tp_dealloc, .pfunc = SoftBodySharedSettings_dealloc},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)SoftBodySharedSettings_methods},
    {.slot = 0, nullptr},
};

static const PyType_Spec SoftBodySharedSettings_spec = {
    .name      = "culverin._culverin_c.SoftBodySharedSettingsObject",
    .basicsize = sizeof(SoftBodySharedSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)SoftBodySharedSettings_slots};

static const PyType_Slot Ragdoll_slots[] = {
    {.slot = Py_tp_dealloc, .pfunc = Ragdoll_dealloc},
    {.slot = Py_tp_methods, .pfunc = (PyMethodDef *)Ragdoll_methods},
    {.slot = 0, .pfunc = nullptr},
};

static const PyType_Spec Ragdoll_spec = {
    .name      = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(RagdollObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)Ragdoll_slots,
};

// --- Module Initialization ---

// Embed the entire TOML file as a static string
static constexpr char PYPROJECT_TOML[] = {
// NOLINTNEXTLINE(readability-magic-numbers)
#embed "../../pyproject.toml" suffix(, 0)
};

// Helper function to extract the version string at runtime
static const char *extract_version_from_toml(void) {
    // Look for the specific pattern 'version = "'
    const char *key = "version = \"";
    auto start      = strstr((char *)PYPROJECT_TOML, key);

    if (!start) {
        return "0.0.0-unknown";
    }

    start += strlen(key); // Move pointer to the start of the actual version number

    // Find the closing quote
    const char *end = strchr(start, '\"');
    if (!end) {
        return "0.0.0-malformed";
    }

    // Create a static buffer to hold the version (e.g., "0.4.0")
    static constexpr size_t VERSION_BUFFER = 32;
    static char version_buffer[VERSION_BUFFER];
    size_t len = end - start;
    if (len >= sizeof(version_buffer)) {
        len = sizeof(version_buffer) - 1;
    }

    strncpy(version_buffer, start, len);
    version_buffer[len] = '\0';

    return version_buffer;
}

static int init_types(PyObject *m, CulverinState *st) {
    struct {
        const PyType_Spec *spec;
        PyObject **slot;
        const char *name;
    } types[] = {{(&PhysicsWorld_spec), &st->PhysicsWorldType, "PhysicsWorld"},
                 {(&Character_spec), &st->CharacterType, "Character"},
                 {(&Vehicle_spec), &st->VehicleType, "Vehicle"},
                 {(&RagdollSettings_spec), &st->RagdollSettingsType, "RagdollSettings"},
                 {(&Ragdoll_spec), &st->RagdollType, "Ragdoll"},
                 {(&Skeleton_spec), &st->SkeletonType, "Skeleton"},
                 {(&BufferProxy_spec), &st->BufferProxyType, "BufferProxyObject"},
                 {(&SoftBodySharedSettings_spec), &st->SoftBodySharedSettingsType,
                  "SoftBodySharedSettings"}};

    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        auto type = PyType_FromModuleAndSpec(m, (PyType_Spec *)types[i].spec, nullptr);
        if (!type) {
            return -1;
        }
        auto mod_name = PyUnicode_FromString("culverin");
        if (!mod_name) {
            Py_DECREF(type);
            return -1;
        }
        PyObject_SetAttrString(type, "__module__", mod_name);
        if (PyModule_AddObject(m, types[i].name, type) < 0) {
            Py_DECREF(type);
            return -1;
        }
        Py_DECREF(mod_name);
        *types[i].slot = type;
    }
    return 0;
}

static int init_constants(PyObject *m) {
    static const struct {
        const char *name;
        int value;
    } consts[] = {{.name = "SHAPE_BOX", .value = CULV_SHAPE_BOX},
                  {.name = "SHAPE_SPHERE", .value = CULV_SHAPE_SPHERE},
                  {.name = "SHAPE_CAPSULE", .value = CULV_SHAPE_CAPSULE},
                  {.name = "SHAPE_CYLINDER", .value = CULV_SHAPE_CYLINDER},
                  {.name = "SHAPE_PLANE", .value = CULV_SHAPE_PLANE},
                  {.name = "SHAPE_MESH", .value = CULV_SHAPE_MESH},
                  {.name = "SHAPE_HEIGHTFIELD", .value = CULV_SHAPE_HEIGHTFIELD},
                  {.name = "SHAPE_CONVEX_HULL", .value = CULV_SHAPE_CONVEX_HULL},
                  {.name = "MOTION_STATIC", .value = MOTION_STATIC},
                  {.name = "MOTION_KINEMATIC", .value = MOTION_KINEMATIC},
                  {.name = "MOTION_DYNAMIC", .value = MOTION_DYNAMIC},
                  {.name = "CONSTRAINT_FIXED", .value = CONSTRAINT_FIXED},
                  {.name = "CONSTRAINT_POINT", .value = CONSTRAINT_POINT},
                  {.name = "CONSTRAINT_HINGE", .value = CONSTRAINT_HINGE},
                  {.name = "CONSTRAINT_SLIDER", .value = CONSTRAINT_SLIDER},
                  {.name = "CONSTRAINT_DISTANCE", .value = CONSTRAINT_DISTANCE},
                  {.name = "CONSTRAINT_CONE", .value = CONSTRAINT_CONE},
                  {.name = "EVENT_ADDED", .value = EVENT_ADDED},
                  {.name = "EVENT_PERSISTED", .value = EVENT_PERSISTED},
                  {.name = "EVENT_REMOVED", .value = EVENT_REMOVED},
// Build Metadata
#if defined(JPH_DOUBLE_PRECISION)
                  {.name = "USE_DOUBLE_PRECISION", .value = 1},
#else
                  {.name = "USE_DOUBLE_PRECISION", .value = 0},
#endif
                  {.name = "FREE_THREADED",
                   .value =
#if defined(Py_GIL_DISABLED) && Py_GIL_DISABLED
                       1
#else
                       0
#endif
                  },
                  {.name = "DEBUG_BUILD",
                   .value =
#if defined(CULVERIN_DEBUG)
                       1
#else
                       0
#endif
                  },
                  {.name = "BEND_DIHEDRAL", .value = JPH_SoftBodyBendType_Dihedral},
                  {.name = "BEND_DISTANCE", .value = JPH_SoftBodyBendType_Distance},
                  {.name = "BEND_NONE", .value = JPH_SoftBodyBendType_None}};

    for (size_t i = 0; i < sizeof(consts) / sizeof(consts[0]); i++) {
        if (PyModule_AddIntConstant(m, consts[i].name, consts[i].value) < 0) {
            return -1;
        }
    }
    return 0;
}

static constexpr auto MAGIC_BUFFER = 128;
static char shared_version[MAGIC_BUFFER];

PyType_DeclareSlot_Status culverin_exec(PyObject *m) {
    CulverinState *st = get_culverin_state(m);

    // 1. THE MASTER GATE: Protects all static global memory in the process
    int expected = 0;
    if (atomic_compare_exchange_strong(&docs_status, &expected, 1)) {

        // --- 1A. VERSION & BUILD METADATA ---
        const char *ver_temp = extract_version_from_toml();

        const char *gil_status =
#if defined(Py_GIL_DISABLED) && Py_GIL_DISABLED
            "free-threaded";
#else
            "gil-enabled";
#endif

        const char *precision =
#if JPH_DOUBLE_PRECISION
            "double-precision";
#else
            "single-precision";
#endif

        const char *build_type =
#if defined(ENABLE_SANITIZER) || defined(__SANITIZE_THREAD__)
            "sanitized";
#elif defined(CULVERIN_DEBUG)
            "debug";
#else
                    "release";
#endif
        // Determine Compiler ID
        const char *compiler_id =
#if defined(__clang__)
            "Clang " __clang_version__;
#elif defined(__GNUC__)
            "GCC " __VERSION__;
#elif defined(_MSC_VER)
            "MSVC " _CRT_STRINGIZE(_MSC_VER);
#else
                "Unknown Compiler";
#endif

        // Result: e.g. "0.6.0 (free-threaded, double-precision, release, Clang 22.1.0)"
        snprintf(shared_version, MAGIC_BUFFER, "%s (%s, %s, %s, %s)", ver_temp, gil_status,
                 precision, build_type, compiler_id);

        // --- THE WINNER: Run exactly once per process life ---
        stitch_docs(module_methods, "Module");
        stitch_docs(PhysicsWorld_methods, "PhysicsWorld");
        stitch_docs(Character_methods, "Character");
        stitch_docs(Vehicle_methods, "Vehicle");
        stitch_docs(Skeleton_methods, "Skeleton");
        stitch_docs(Ragdoll_methods, "Ragdoll");
        stitch_docs(RagdollSettings_methods, "RagdollSettings");
        stitch_docs(SoftBodySharedSettings_methods, "SoftBodySharedSettings");
        stitch_docs_getset(PhysicsWorld_getset, "PhysicsWorld");
        stitch_docs_getset(Character_getset, "Character");
        stitch_docs_getset(Vehicle_getset, "Vehicle");

        // Gated Handler Registration
        JPH_SetTraceHandler(culv_jph_trace);
        JPH_SetAssertFailureHandler(culv_jph_assert);

        if (!JPH_Init()) {
            PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
            atomic_store_explicit(&docs_status, 0, memory_order_relaxed);
            return -1;
        }

        JPH_BroadPhaseLayerFilter_SetProcs(&global_bp_procs);
        JPH_ObjectLayerFilter_SetProcs(&global_obj_procs);
        JPH_BodyFilter_SetProcs(&global_bf_procs);
        JPH_ShapeFilter_SetProcs(&global_sf_procs);
        JPH_DebugRenderer_SetProcs(&debug_procs);
        JPH_ContactListener_SetProcs(&contact_procs);
        JPH_CharacterContactListener_SetProcs(&char_listener_procs);

        // Gated Mutex Initialization
        // (Fixes the race on g_jph_trampoline_lock write)
        if (INIT_NATIVE_MUTEX(g_jph_trampoline_lock) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to initialize global lock");
            return -1;
        }

        atomic_store_explicit(&docs_status, 2, memory_order_release);

    } else {
        // --- THE LOSERS: Wait for the Winner to finish ---
        while (atomic_load_explicit(&docs_status, memory_order_acquire) != 2) {
            culverin_yield();
        }
    }
    // --- 2. PER-INTERPRETER SETUP (Runs for every import) ---
    if (PyModule_AddStringConstant(m, "__version__", shared_version) < 0) {
        return -1;
    }

    culverin_init_all_parsers(&st->parsers);
    CULV_INIT_PROFILER();

    st->helper = PyImport_ImportModule("culverin._culverin");
    if (!st->helper) {
        return -1;
    }

    if (init_types(m, st) < 0) {
        return -1;
    }
    if (init_constants(m) < 0) {
        return -1;
    }

    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_Status culverin_traverse(PyObject *m, visitproc visit, void *arg) {
    CulverinState *st = get_culverin_state(m);
    Py_VISIT(st->helper);
    Py_VISIT(st->PhysicsWorldType);
    Py_VISIT(st->CharacterType);
    Py_VISIT(st->VehicleType);
    Py_VISIT(st->RagdollSettingsType);
    Py_VISIT(st->RagdollType);
    Py_VISIT(st->SkeletonType);
    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_Status culverin_clear(PyObject *m) {
    CulverinState *st = get_culverin_state(m);
    Py_CLEAR(st->helper);
    Py_CLEAR(st->PhysicsWorldType);
    Py_CLEAR(st->CharacterType);
    Py_CLEAR(st->VehicleType);
    Py_CLEAR(st->RagdollSettingsType);
    Py_CLEAR(st->RagdollType);
    Py_CLEAR(st->SkeletonType);
    // Clean up the parsers for this interpreter
    culverin_free_all_parsers(&st->parsers);
    return 0;
}

static const PyModuleDef_Slot culverin_slots[] = {
    {.slot = Py_mod_exec, .value = culverin_exec},

// 1. Handle the Free-threaded (No GIL) declaration (3.13+)
#if defined(Py_MOD_GIL_NOT_USED)
    {.slot = Py_mod_gil, .value = Py_MOD_GIL_NOT_USED},
#endif

    // 2. Handle Subinterpreter support
    {.slot = Py_mod_multiple_interpreters,
#if PY_VERSION_HEX >= 0x030D0000
     .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED
#else
     .value = Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
#endif
    },

    {.slot = 0, .value = nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyModuleDef culverin_module = {
    .m_base     = PyModuleDef_HEAD_INIT,
    .m_name     = "_culverin_c",
    .m_doc      = "Culverin Physics Engine Core",
    .m_size     = sizeof(CulverinState),
    .m_methods  = module_methods,
    .m_slots    = (PyModuleDef_Slot *)culverin_slots,
    .m_traverse = culverin_traverse,
    .m_clear    = culverin_clear,
};

extern PyMODINIT_FUNC PyInit__culverin_c(void) { return PyModuleDef_Init(&culverin_module); }
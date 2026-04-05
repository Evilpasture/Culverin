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
#include "culverin_filters.h"
#include "culverin_getters.h"
#include "culverin_parsers.h"
#include "culverin_physics_world_internal.h"
#include "culverin_query_methods.h"
#include "culverin_ragdoll.h"
#include "culverin_shadow_sync.h"
#include "culverin_vehicle.h"
#include "joltc.h"
#include <stdatomic.h>

// ============================================================================
// Semantic Constants - Magic Number Replacements
// ============================================================================

// Memory and Alignment
static constexpr size_t INITIAL_BODY_CAPACITY = 1024;
static constexpr size_t BODY_ID_SIZE_BYTES    = 8;

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
    self->view_export_count = 0;
    atomic_init(&self->waiting_threads, 0);
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
    JPH_DebugRenderer_SetProcs(&debug_procs);
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
    JPH_ContactListener_SetProcs(&contact_procs);
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &h_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ImpulseParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Impulse");

    // 2. CONCURRENCY & RESOLUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Wait for structural updates or buffer swaps
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to BodyHandle for atomic parameter unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures visibility of the creation data)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // CASE 1: Body is already in Jolt (ALIVE or CHARACTER)
    if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
        uint32_t dense_idx = self->slot_to_dense[slot];
        JPH_BodyID bid     = self->body_ids[dense_idx];

        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
        JPH_BodyInterface_AddImpulse(self->body_interface, bid, &imp);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body is queued but not yet flushed (PENDING_CREATE)
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        // Command queue is non-atomic; protected by SHADOW_LOCK and BLOCK_UNTIL_NOT_STEPPING
        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_IMPULSE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return nullptr;
    }

    Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_apply_impulse_at(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float ix;
    float iy;
    float iz;
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;

    void *targets[ImpAt_COUNT];
    targets[IDX_IMPAT_H]  = (void *)&h_raw;
    targets[IDX_IMPAT_IX] = (void *)&ix;
    targets[IDX_IMPAT_IY] = (void *)&iy;
    targets[IDX_IMPAT_IZ] = (void *)&iz;
    targets[IDX_IMPAT_PX] = (void *)&px;
    targets[IDX_IMPAT_PY] = (void *)&py;
    targets[IDX_IMPAT_PZ] = (void *)&pz;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ImpulseAtParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(ix, iy, iz, "Impulse");
    VALIDATE_FINITE_VEC3(px, py, pz, "Impulse position");

    // 2. CONCURRENCY & RESOLUTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic type for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // CASE 1: Immediate execution for active bodies or characters
    if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
        JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {ix, iy, iz};
        JPH_RVec3 v_pos                     = {px, py, pz};
        JPH_BodyInterface_AddImpulse2(self->body_interface, bid, &imp, &v_pos);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Deferred execution for pending bodies
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_IMPULSE_AT, slot);
        cmd->impulse_at.ix  = ix;
        cmd->impulse_at.iy  = iy;
        cmd->impulse_at.iz  = iz;
        cmd->impulse_at.px  = px;
        cmd->impulse_at.py  = py;
        cmd->impulse_at.pz  = pz;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return nullptr;
    }

    Py_RETURN_NONE;
}
PyCFunction_DeclareMethod PhysicsWorld_apply_angular_impulse(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = (void *)&h_raw;
    targets[IDX_V3_X] = (void *)&x;
    targets[IDX_V3_Y] = (void *)&y;
    targets[IDX_V3_Z] = (void *)&z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.AngImpulseParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Angular impulse");

    // 2. CONCURRENCY & RESOLUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Block only if a simulation step is currently swapping buffers
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic type for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // CASE 1: Body is already active in Jolt
    if (state == SLOT_ALIVE) {
        JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
        JPH_BodyInterface_AddAngularImpulse(self->body_interface, bid, &imp);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body is queued for creation (Causal Consistency)
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        // Standard write to non-atomic command queue
        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_ANG_IMPULSE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return nullptr;
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_apply_force(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &h_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ForceParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Force");

    // 2. CONCURRENCY & RESOLUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Ensure we aren't mutating while buffers are being swapped
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures visibility of the creation/sync data)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // CASE 1: Body is active in simulation
    if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
        uint32_t dense_idx = self->slot_to_dense[slot];
        JPH_BodyID bid     = self->body_ids[dense_idx];

        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 force_vec = {x, y, z};
        // Jolt forces are thread-safe accumulators
        JPH_BodyInterface_AddForce(self->body_interface, bid, &force_vec);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body is queued but not yet in Jolt (Order-preserving)
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_FORCE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return nullptr;
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_apply_torque(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &h_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.TorqueParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Torque");

    // 2. CONCURRENCY & RESOLUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Block if world is currently swapping buffers
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic type for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures visibility of the creation data)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // CASE 1: Body is already in Jolt
    if (state == SLOT_ALIVE) {
        uint32_t dense_idx = self->slot_to_dense[slot];
        JPH_BodyID bid     = self->body_ids[dense_idx];

        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 torque_vec = {x, y, z};
        JPH_BodyInterface_AddTorque(self->body_interface, bid, &torque_vec);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body was just created and not yet flushed
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        // Standard write to non-atomic command queue
        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_TORQUE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return nullptr;
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_gravity(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Unchanged)
    float x;
    float y;
    float z;

    void *targets[XYZ_COUNT];
    targets[IDX_XYZ_X] = &x;
    targets[IDX_XYZ_Y] = &y;
    targets[IDX_XYZ_Z] = &z;

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

PyCFunction_DeclareMethod PhysicsWorld_get_body_stats(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.HOnlyParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Don't read while buffers are being swapped/cleared
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures visibility of creator writes)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // We now allow CHARACTER stats retrieval as they occupy the same shadow buffer layout
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    uint32_t i = self->slot_to_dense[slot];

    // Snapshot values while holding the lock
    // Shadow buffers are non-atomic; protected by SHADOW_LOCK + BLOCK_UNTIL_NOT_STEPPING
    PosStride p = ((PosStride *)self->positions)[i];
    AuxStride r = ((AuxStride *)self->rotations)[i];
    AuxStride v = ((AuxStride *)self->linear_velocities)[i];

    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. RESULT CONSTRUCTION
    // Nested tuples: ((px, py, pz), (rx, ry, rz, rw), (vx, vy, vz))
    return FastBuild_Tuple(FastBuild_Tuple(p.x, p.y, p.z), FastBuild_Tuple(r.x, r.y, r.z, r.w),
                           FastBuild_Tuple(v.x, v.y, v.z));
}
PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    double surface_y;
    float buoyancy  = 1.0f;
    float lin_drag  = DEFAULT_LINEAR_DRAG;
    float ang_drag  = DEFAULT_ANGULAR_DRAG;
    float dt        = DEFAULT_FRAME_TIME;
    PyObject *o_vel = nullptr;

    // 2. FAST PARSE (Unchanged)
    void *targets[Buoy_COUNT];
    targets[IDX_BUOY_HANDLE]    = (void *)&h_raw;
    targets[IDX_BUOY_SURFACE_Y] = (void *)&surface_y;
    targets[IDX_BUOY_BUOYANCY]  = (void *)&buoyancy;
    targets[IDX_BUOY_LIN_DRAG]  = (void *)&lin_drag;
    targets[IDX_BUOY_ANG_DRAG]  = (void *)&ang_drag;
    targets[IDX_BUOY_DT]        = (void *)&dt;
    targets[IDX_BUOY_VEL]       = (void *)&o_vel;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.BuoyParser, targets)) {
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

    // 3. RESOLUTION PHASE (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to BodyHandle for atomic parameter unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_FALSE;
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with Stepper thread)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
    if (state != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_FALSE;
    }

    JPH_BodyID bid            = self->body_ids[self->slot_to_dense[slot]];
    JPH_BodyInterface *bi     = self->body_interface;
    JPH_PhysicsSystem *system = self->system;

    // TSan Fix: Register this as an active query so the Stepper thread
    // waits for us to finish before executing destruction commands.
    atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);

    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. EXECUTION PHASE (Unlocked & GIL-Friendly)
    bool submerged = false;
    Py_BEGIN_ALLOW_THREADS JPH_BodyInterface_ActivateBody(bi, bid);

    JPH_Vec3 gravity;
    JPH_PhysicsSystem_GetGravity(system, &gravity);

    JPH_STACK_ALLOC(JPH_RVec3, surf_pos);
    *surf_pos = (JPH_RVec3){0, (JPH_Real)surface_y, 0};
    JPH_STACK_ALLOC(JPH_Vec3, surf_norm);
    *surf_norm = (JPH_Vec3){0, 1.0f, 0};
    JPH_STACK_ALLOC(JPH_Vec3, fluid_vel);
    *fluid_vel = (JPH_Vec3){vx, vy, vz};

    submerged = JPH_BodyInterface_ApplyBuoyancyImpulse(bi, bid, surf_pos, surf_norm, buoyancy,
                                                       lin_drag, ang_drag, fluid_vel, &gravity, dt);

    // TSan Fix: Signal completion to the Stepper thread
    end_query_scope(self);
    Py_END_ALLOW_THREADS

        return PyBool_FromLong((int)submerged);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES (Unchanged)
    PyObject *o_handles = nullptr;
    JPH_Real surface_y  = 0.0;
    float buoyancy      = 1.0f;
    float lin_drag      = DEFAULT_LINEAR_DRAG;
    float ang_drag      = DEFAULT_ANGULAR_DRAG;
    float dt            = DEFAULT_FRAME_TIME;
    PyObject *o_vel     = nullptr;

    // 2. FAST PARSE (Unchanged)
    void *targets[BatchBuoy_COUNT];
    targets[IDX_BBUOY_HANDLES]   = (void *)&o_handles;
    targets[IDX_BBUOY_SURFACE_Y] = &surface_y;
    targets[IDX_BBUOY_BUOYANCY]  = &buoyancy;
    targets[IDX_BBUOY_LIN_DRAG]  = &lin_drag;
    targets[IDX_BBUOY_ANG_DRAG]  = &ang_drag;
    targets[IDX_BBUOY_DT]        = &dt;
    targets[IDX_BBUOY_VEL]       = (void *)&o_vel;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.BatchBuoyParser, targets)) {
        return nullptr;
    }

    // 3. BUFFER & VELOCITY EXTRACTION (Unchanged)
    Py_buffer h_view;
    if (PyObject_GetBuffer(o_handles, &h_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    if (UNLIKELY(h_view.itemsize != 8 && h_view.len % 8 != 0)) {
        PyBuffer_Release(&h_view);
        PyErr_SetString(PyExc_ValueError, "Handle buffer must be uint64");
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

    size_t count = (size_t)h_view.len / BODY_ID_SIZE_BYTES;
    if (count == 0) {
        PyBuffer_Release(&h_view);
        Py_RETURN_NONE;
    }

    // 4. TEMP ID RESOLUTION (ATOMIC REFACTOR)
    JPH_BodyID *ids = (JPH_BodyID *)CULV_RAW_MALLOC(count * sizeof(JPH_BodyID));
    if (!ids) {
        PyBuffer_Release(&h_view);
        return PyErr_NoMemory();
    }

    uint64_t *handles_raw = (uint64_t *)h_view.buf;
    size_t valid_count    = 0;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    for (size_t i = 0; i < count; i++) {
        uint32_t slot = 0;
        // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
        if (unpack_handle(self, (BodyHandle)handles_raw[i], &slot)) {
            // TSan Fix: Atomic acquire load to synchronize with Stepper thread
            uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);
            if (state == SLOT_ALIVE) {
                ids[valid_count++] = self->body_ids[self->slot_to_dense[slot]];
            }
        }
    }

    if (valid_count > 0) {
        // TSan Fix: Register active query so Stepper thread waits for batch completion
        atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_acquire);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&h_view);

    // 5. BATCH EXECUTION (Lockless)
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

        // TSan Fix: Notify Stepper thread
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
    CulverinState *st   = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    PyObject *state_obj = nullptr;
    void *targets[LoadState_COUNT];
    targets[IDX_LS_STATE] = (void *)&state_obj;

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

PyCFunction_DeclareMethod PhysicsWorld_step(PhysicsWorldObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    float dt          = DEFAULT_FRAME_TIME;
    void *targets[Step_COUNT];
    targets[IDX_STEP_DT] = (void *)&dt;

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

    // ANTI-STARVATION: Yield to waiting Python threads (Getters/Mutators)
    while (atomic_load_explicit(&self->waiting_threads, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS culverin_yield();
        Py_END_ALLOW_THREADS SHADOW_LOCK(&self->shadow_lock);
    }

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

    if (UNLIKELY(self->command_capacity > self->spare_capacity)) {
        void *new_spare = CULV_RAW_REALLOC(self->command_queue_spare,
                                           self->command_capacity * sizeof(PhysicsCommand));
        if (UNLIKELY(!new_spare)) {
            // Rollback flags on OOM
            atomic_store_explicit(&self->is_stepping, false, memory_order_relaxed);
            atomic_store_explicit(&self->step_requested, false, memory_order_relaxed);
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }
        self->command_queue_spare = (PhysicsCommand *)new_spare;
        self->spare_capacity      = self->command_capacity;
    }
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

    // 3. Jolt-to-Shadow Buffer Sync
    culverin_sync_shadow_buffers(self);

    CULV_PROFILE_END(jolt_step, "Jolt Physics Crunch", (unsigned int)captured_count);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        // --- PHASE 3: FINALIZATION ---
        SHADOW_LOCK(&self->shadow_lock);

    // 1. Cleanup retired buffers (Atomic gens/stats handled in free_new_buffers)
    if (self->trash_count > 0) {
        for (size_t i = 0; i < self->trash_count; i++) {
            free_new_buffers(&self->trash_buffers[i]);
        }
        self->trash_count = 0;
    }

    // 2. Metadata Updates
    size_t c_idx        = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    self->contact_count = (c_idx > self->contact_max_capacity) ? self->contact_max_capacity : c_idx;

    // TSan Fix: Snapshot final atomic body count to update Python memoryview layout
    size_t final_count  = atomic_load_explicit(&self->count, memory_order_acquire);
    self->view_shape[0] = (Py_ssize_t)final_count;

    self->time += (double)dt;

    // 3. Fence Release
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
    // 1. DEFAULT VALUES
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

    // 2. FAST PARSE (Using Unified HC Group)
    void *targets[HC_COUNT];
    targets[IDX_HC_POS]       = (void *)&o_pos;
    targets[IDX_HC_ROT]       = (void *)&o_rot;
    targets[IDX_HC_DATA]      = (void *)&o_points; // Schema Overlay interns this as "points"
    targets[IDX_HC_MOTION]    = (void *)&motion_type;
    targets[IDX_HC_MASS]      = (void *)&mass;
    targets[IDX_HC_USER_DATA] = (void *)&user_data;
    targets[IDX_HC_SENSOR]    = (void *)&is_sensor;
    targets[IDX_HC_CAT]       = (void *)&category;
    targets[IDX_HC_MASK]      = (void *)&mask;
    targets[IDX_HC_MAT_ID]    = (void *)&material_id;
    targets[IDX_HC_FRIC]      = (void *)&friction;
    targets[IDX_HC_REST]      = (void *)&restitution;
    targets[IDX_HC_CCD]       = (void *)&use_ccd;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ConvexHullParser, targets)) {
        return nullptr;
    }

    // 3. COMPLEX TYPE EXTRACTION
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

    /* Validate position and rotation components */
    VALIDATE_FINITE_VEC3(px, py, pz, "SetTransform position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "SetTransform rotation");

    Py_buffer points_view;
    if (PyObject_GetBuffer(o_points, &points_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    if (UNLIKELY(points_view.len % VERTEX_STRIDE_BYTES != 0)) {
        PyBuffer_Release(&points_view);
        return PyErr_Format(PyExc_ValueError, "Points buffer must be 3 * float32");
    }
    size_t num_points = points_view.len / VERTEX_STRIDE_BYTES;
    if (UNLIKELY(num_points < 3)) {
        PyBuffer_Release(&points_view);
        return PyErr_Format(PyExc_ValueError, "Convex Hull requires at least 3 points");
    }

    // 4. SHAPE CREATION (No GIL, No Shadow Lock)
    JPH_Shape *shape = nullptr;
    Py_BEGIN_ALLOW_THREADS auto *jolt_points =
        (JPH_Vec3 *)CULV_RAW_MALLOC(num_points * sizeof(JPH_Vec3));
    float *raw = (float *)points_view.buf;
    for (size_t i = 0; i < num_points; i++) {
        jolt_points[i] = (JPH_Vec3){raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]};
    }

    JPH_ConvexHullShapeSettings *hull_settings = JPH_ConvexHullShapeSettings_Create(
        jolt_points, (uint32_t)num_points, CONVEX_HULL_TOLERANCE);
    CULV_RAW_FREE(jolt_points);

    if (hull_settings) {
        shape = (JPH_Shape *)JPH_ConvexHullShapeSettings_CreateShape(hull_settings);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hull_settings);
    }
    Py_END_ALLOW_THREADS PyBuffer_Release(&points_view);

    if (!shape) {
        return PyErr_Format(PyExc_RuntimeError, "Jolt Convex Hull build failed");
    }

    // 5. BODY SETTINGS PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, (JPH_MotionType)motion_type,
        (motion_type == 0) ? 0 : 1);

    // Apply is_sensor
    if (is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }

    // Apply Mass/Friction/Restitution logic
    if (mass > 0.0f) {
        JPH_MassProperties mp;
        JPH_Shape_GetMassProperties(shape, &mp);
        float scale = mass / fmaxf(mp.mass, EPSILON_FLOAT);
        mp.mass     = mass;
        for (int i = 0; i < 3; i++) {
            mp.inertia.column[i].x *= scale;
            mp.inertia.column[i].y *= scale;
            mp.inertia.column[i].z *= scale;
        }
        JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
        JPH_BodyCreationSettings_SetOverrideMassProperties(
            settings, JPH_OverrideMassProperties_CalculateInertia);
    }

    JPH_BodyCreationSettings_SetFriction(settings, friction);
    JPH_BodyCreationSettings_SetRestitution(settings, restitution);

    if (use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }

    // 6. COMMIT PHASE (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Read counts atomically
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        if (PhysicsWorld_resize(self, (self->capacity == 0) ? INITIAL_BODY_CAPACITY
                                                            : self->capacity * 2) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            JPH_Shape_Destroy(shape);
            return nullptr;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Pop from free stack atomically
    uint32_t slot = self->free_slots[available - 1];
    atomic_store_explicit(&self->free_count, available - 1, memory_order_release);

    // Increment total count atomically
    uint32_t dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

    // Prepare Handle
    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);
    JPH_BodyCreationSettings_SetUserData(settings, raw_h);

    // Update Shadow Buffers (Non-atomic)
    ((PosStride *)self->positions)[dense]          = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense]          = (AuxStride){rx, ry, rz, rw};
    ((AuxStride *)self->linear_velocities)[dense]  = (AuxStride){};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){};

    self->categories[dense]    = category;
    self->masks[dense]         = mask;
    self->material_ids[dense]  = material_id;
    self->user_data[dense]     = user_data;
    self->body_ids[dense]      = JPH_INVALID_BODY_ID;
    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;

    // TSan Fix: Publish state atomically (Release creates synchronization boundary)
    atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Atomic Rollback
        atomic_fetch_sub_explicit(&self->count, 1, memory_order_relaxed);
        size_t r_idx = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
        self->free_slots[r_idx] = slot;
        atomic_store_explicit(&self->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);

        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

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

    JPH_BodyCreationSettings_SetFriction(settings, props.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, props.restitution);
}

// Orchestrator
PyCFunction_DeclareMethod PhysicsWorld_create_compound_body(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. DEFAULT VALUES
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

    // 2. TARGET MAPPING (Using the shared HullComp Index Group)
    void *targets[HC_COUNT];
    targets[IDX_HC_POS]       = (void *)&o_pos;
    targets[IDX_HC_ROT]       = (void *)&o_rot;
    targets[IDX_HC_DATA]      = (void *)&o_parts;
    targets[IDX_HC_MOTION]    = (void *)&motion_type;
    targets[IDX_HC_MASS]      = (void *)&mass;
    targets[IDX_HC_USER_DATA] = (void *)&user_data;
    targets[IDX_HC_SENSOR]    = (void *)&is_sensor;
    targets[IDX_HC_CAT]       = (void *)&category;
    targets[IDX_HC_MASK]      = (void *)&mask;
    targets[IDX_HC_MAT_ID]    = (void *)&material_id;
    targets[IDX_HC_FRIC]      = (void *)&friction;
    targets[IDX_HC_REST]      = (void *)&restitution;
    targets[IDX_HC_CCD]       = (void *)&use_ccd;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the HullCompParser initialized via X-Macro
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CompoundParser, targets)) {
        return nullptr;
    }

    // 3. COMPLEX TYPE EXTRACTION (Outside Shadow Lock)
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

    // SCHEMA defines IDX_HC_DATA as required, but we verify it's a list here
    if (UNLIKELY(!PyList_Check(o_parts))) {
        PyErr_SetString(PyExc_TypeError, "'parts' must be a list of tuples");
        return nullptr;
    }

    // 4. SHAPE BUILD (Heavy lifting - released GIL internally)
    JPH_Shape *final_shape = init_compound_shape(self, o_parts);
    if (!final_shape) {
        return nullptr;
    }

    // 5. JOLT PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        final_shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw},
        (JPH_MotionType)motion_type, (motion_type == 0) ? 0 : 1);

    BodyCreationProps props = {.mass        = mass,
                               .friction    = friction,
                               .restitution = restitution,
                               .is_sensor   = (int)is_sensor,
                               .use_ccd     = (int)use_ccd};
    apply_body_creation_props(settings, final_shape, props);

    // 6. COMMIT PHASE (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Load counters atomically
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        size_t needed = (self->capacity == 0) ? INITIAL_BODY_CAPACITY : self->capacity * 2;
        if (PhysicsWorld_resize(self, needed) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            JPH_Shape_Destroy(final_shape);
            return nullptr;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Atomic Pop from free stack
    uint32_t slot = self->free_slots[available - 1];
    atomic_store_explicit(&self->free_count, available - 1, memory_order_release);

    // Atomic dense index increment
    uint32_t dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

    // Prepare handle using explicit relaxed load to avoid implicit seq_cst
    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);
    JPH_BodyCreationSettings_SetUserData(settings, raw_h);

    // Update non-atomic Shadow Buffers
    PosStride p_val                            = {.x = px, .y = py, .z = pz, .w = 0.0};
    ((PosStride *)self->positions)[dense]      = p_val;
    ((PosStride *)self->prev_positions)[dense] = p_val;

    AuxStride q_val                            = {.x = rx, .y = ry, .z = rz, .w = rw};
    ((AuxStride *)self->rotations)[dense]      = q_val;
    ((AuxStride *)self->prev_rotations)[dense] = q_val;

    ((AuxStride *)self->linear_velocities)[dense]  = (AuxStride){};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){};

    self->categories[dense]    = category;
    self->masks[dense]         = mask;
    self->material_ids[dense]  = material_id;
    self->user_data[dense]     = user_data;
    self->body_ids[dense]      = JPH_INVALID_BODY_ID;
    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;

    // TSan Fix: Publish state and update view atomically
    atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    // 7. QUEUE COMMAND
    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Atomic Rollback
        atomic_fetch_sub_explicit(&self->count, 1, memory_order_relaxed);
        size_t r_idx = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
        self->free_slots[r_idx] = slot;
        atomic_store_explicit(&self->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);

        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(final_shape);
        return PyErr_NoMemory();
    }

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

// Helper 3: Apply mass, sensor, CCD, and sleeping settings to the creation
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
    void *targets[Body_COUNT];
    targets[IDX_POS]       = (void *)&o_pos;
    targets[IDX_ROT]       = (void *)&o_rot;
    targets[IDX_SIZE]      = (void *)&o_size;
    targets[IDX_SHAPE]     = (void *)&shape_type;
    targets[IDX_MOTION]    = (void *)&motion_type;
    targets[IDX_USER_DATA] = (void *)&user_data;
    targets[IDX_SENSOR]    = (void *)&is_sensor;
    targets[IDX_MASS]      = (void *)&mass;
    targets[IDX_CAT]       = (void *)&category;
    targets[IDX_MASK]      = (void *)&mask;
    targets[IDX_FRIC]      = (void *)&friction;
    targets[IDX_REST]      = (void *)&restitution;
    targets[IDX_MAT]       = (void *)&material_id;
    targets[IDX_CCD]       = (void *)&use_ccd;

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

    JPH_Shape *shape                   = nullptr;
    JPH_BodyCreationSettings *settings = nullptr;

    // --- CRITICAL SECTION: JOLT PREP ---
    Py_BEGIN_ALLOW_THREADS;
    SHADOW_LOCK(&self->shadow_lock);

    shape = find_or_create_shape_locked(self, shape_type, s);

    SHADOW_UNLOCK(&self->shadow_lock);

    if (LIKELY(shape)) {
        // Force local variables instead of compound literals in the function call
        JPH_RVec3 j_pos = {.x = px, .y = py, .z = pz};
        JPH_Quat j_rot  = {.x = rx, .y = ry, .z = rz, .w = rw};

        settings = JPH_BodyCreationSettings_Create3(
            shape, &j_pos, &j_rot, (JPH_MotionType)motion_type, (motion_type == 0) ? 0 : 1);

        if (settings) {
            BodyConfig config = {.mass        = mass,
                                 .friction    = mat.friction,
                                 .restitution = mat.restitution,
                                 .is_sensor   = (int)is_sensor,
                                 .use_ccd     = (int)use_ccd,
                                 .motion_type = motion_type};
            configure_body_settings(settings, shape, config);
        }
    }

    Py_END_ALLOW_THREADS;

    if (!shape) {
        return PyErr_Format(PyExc_RuntimeError, "Failed to create/find Shape");
    }
    if (!settings) {
        return PyErr_Format(PyExc_RuntimeError, "Failed to create BodySettings");
    }

    // --- COMMIT PHASE: SHADOW BUFFER UPDATE (ATOMIC REFACTOR) ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Load counts atomically to check Jolt and shadow limits
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);

    if (UNLIKELY(current_count >= self->max_jolt_bodies)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        return PyErr_Format(PyExc_RuntimeError, "PhysicsWorld limit reached: %zu bodies",
                            self->max_jolt_bodies);
    }

    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        size_t next_cap = (self->capacity == 0) ? INITIAL_BODY_CAPACITY : self->capacity * 2;
        if (next_cap > self->max_jolt_bodies) {
            next_cap = self->max_jolt_bodies;
        }
        if (PhysicsWorld_resize(self, next_cap) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            return nullptr;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Atomic Pop: Update free stack head
    uint32_t slot = self->free_slots[available - 1];
    atomic_store_explicit(&self->free_count, available - 1, memory_order_release);

    // Atomic Increment: Assign dense index
    uint32_t dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

    // Resolve handle metadata atomically
    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);
    JPH_BodyCreationSettings_SetUserData(settings, raw_h);

    // Update non-atomic shadow buffers (Safe under shadow_lock + block_stepping)
    PosStride p_val                            = {px, py, pz, 0.0};
    ((PosStride *)self->positions)[dense]      = p_val;
    ((PosStride *)self->prev_positions)[dense] = p_val;

    AuxStride q_val                            = {rx, ry, rz, rw};
    ((AuxStride *)self->rotations)[dense]      = q_val;
    ((AuxStride *)self->prev_rotations)[dense] = q_val;

    ((AuxStride *)self->linear_velocities)[dense]  = (AuxStride){};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){};

    self->categories[dense]    = category;
    self->masks[dense]         = mask;
    self->material_ids[dense]  = material_id;
    self->user_data[dense]     = user_data;
    self->body_ids[dense]      = JPH_INVALID_BODY_ID;
    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;

    // TSan Fix: Publish slot state and sync metadata atomically
    atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Atomic Rollback
        atomic_fetch_sub_explicit(&self->count, 1, memory_order_relaxed);
        size_t r_idx = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
        self->free_slots[r_idx] = slot;
        atomic_store_explicit(&self->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);

        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        return PyErr_NoMemory();
    }

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
    PyObject *py_positions = nullptr;
    PyObject *py_sizes     = nullptr;
    int shape_type         = 0;
    int motion_type        = 2;

    // DECLARE AND INITIALIZE AT THE TOP
    PosStride *pos_buf                      = nullptr;
    ShapeParams *size_buf                   = nullptr;
    JPH_BodyCreationSettings **settings_buf = nullptr;

    // Use BatchCreate Group count and schema IDs
    void *targets[BatchCreate_COUNT];
    targets[IDX_BC_POSITIONS] = (void *)&py_positions;
    targets[IDX_BC_SIZES]     = (void *)&py_sizes;
    targets[IDX_BC_SHAPE]     = (void *)&shape_type;
    targets[IDX_BC_MOTION]    = (void *)&motion_type;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.BatchCreateParser, targets)) {
        return nullptr;
    }
    // Initial Validation
    if (!PyList_Check(py_positions) || !PyList_Check(py_sizes)) {
        return PyErr_Format(PyExc_TypeError, "positions and sizes must be lists");
    }

    Py_ssize_t batch_count = PyList_GET_SIZE(py_positions);
    if (PyList_GET_SIZE(py_sizes) != batch_count) {
        return PyErr_Format(PyExc_ValueError, "List length mismatch");
    }

    // TSan Fix: Atomic load of current count for limit check
    if (UNLIKELY(atomic_load_explicit(&self->count, memory_order_acquire) + batch_count >
                 self->max_jolt_bodies)) {
        return PyErr_Format(PyExc_RuntimeError, "Batch would exceed Jolt body limit (%u)",
                            self->max_jolt_bodies);
    }

    // 2. TEMP ALLOCATION
    pos_buf      = (PosStride *)CULV_RAW_MALLOC(batch_count * sizeof(PosStride));
    size_buf     = (ShapeParams *)CULV_RAW_MALLOC(batch_count * sizeof(ShapeParams));
    settings_buf = (JPH_BodyCreationSettings **)CULV_RAW_CALLOC(batch_count,
                                                                sizeof(JPH_BodyCreationSettings *));

    if (!pos_buf || !size_buf || !settings_buf) {
        goto fail_oom;
    }

    // 3. PARSE INTO C BUFFERS (GIL HELD)
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        // Using GET_ITEM is safe here because we verified types/lengths above
        if (!parse_py_vec3(PyList_GET_ITEM(py_positions, i), &pos_buf[i])) {
            pos_buf[i] = (PosStride){.x = 0, .y = 0, .z = 0};
        }
        VALIDATE_FINITE_VEC3(pos_buf[i].x, pos_buf[i].y, pos_buf[i].z, "Batch Position");

        parse_body_size(PyList_GET_ITEM(py_sizes, i), size_buf[i].p);
        VALIDATE_FINITE_VEC4(size_buf[i].p[0], size_buf[i].p[1], size_buf[i].p[2], size_buf[i].p[3],
                             "Batch Size");
    }

    // 4. JOLT PREP (NO GIL)
    Py_BEGIN_ALLOW_THREADS SHADOW_LOCK(&self->shadow_lock);

    JPH_STACK_ALLOC(JPH_RVec3, j_pos);
    JPH_STACK_ALLOC(JPH_Quat, j_rot);
    *j_rot = (JPH_Quat){0.0f, 0.0f, 0.0f, 1.0f};

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        JPH_Shape *shape = find_or_create_shape_locked(self, shape_type, size_buf[i].p);
        if (shape) {
            *j_pos          = (JPH_RVec3){pos_buf[i].x, pos_buf[i].y, pos_buf[i].z};
            settings_buf[i] = JPH_BodyCreationSettings_Create3(
                shape, j_pos, j_rot, (JPH_MotionType)motion_type, (motion_type == 0 ? 0 : 1));
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_END_ALLOW_THREADS

        // 5. COMMIT PHASE (ATOMIC REFACTOR)
        SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Check atomic counts for structural availability
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (available < (size_t)batch_count || (current_count + batch_count) > self->capacity) {
        size_t needed = current_count + batch_count + INITIAL_BODY_CAPACITY;
        if (PhysicsWorld_resize(self, (needed > self->max_jolt_bodies) ? self->max_jolt_bodies
                                                                       : needed) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            goto fail;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Check command capacity in one block
    size_t needed_cmds = self->command_count + batch_count;
    if (self->command_capacity < needed_cmds) {
        void *new_q = CULV_RAW_REALLOC(self->command_queue, needed_cmds * sizeof(PhysicsCommand));
        if (!new_q) {
            SHADOW_UNLOCK(&self->shadow_lock);
            goto fail_oom;
        }
        self->command_queue    = (PhysicsCommand *)new_q;
        self->command_capacity = (uint32_t)needed_cmds;
    }

    PyObject *result_list = PyList_New(batch_count);
    if (!result_list) {
        SHADOW_UNLOCK(&self->shadow_lock);
        goto fail;
    }

    auto *shadow_pos  = (PosStride *)self->positions;
    auto *shadow_ppos = (PosStride *)self->prev_positions;
    auto *shadow_rot  = (AuxStride *)self->rotations;
    auto *shadow_prot = (AuxStride *)self->prev_rotations;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (!settings_buf[i]) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(result_list, i, Py_None);
            continue;
        }

        // TSan Fix: Atomic Pop and Increment
        size_t f_idx  = atomic_fetch_sub_explicit(&self->free_count, 1, memory_order_relaxed) - 1;
        uint32_t slot = self->free_slots[f_idx];
        auto dense    = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

        uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
        BodyHandle handle = make_handle(slot, gen);
        uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);
        JPH_BodyCreationSettings_SetUserData(settings_buf[i], raw_h);

        // Update non-atomic shadow buffers
        PosStride p          = {.x = pos_buf[i].x, .y = pos_buf[i].y, .z = pos_buf[i].z, .w = 0.0};
        shadow_pos[dense]    = p;
        shadow_ppos[dense]   = p;
        AuxStride identity_q = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};
        shadow_rot[dense]    = identity_q;
        shadow_prot[dense]   = identity_q;

        self->body_ids[dense]      = JPH_INVALID_BODY_ID;
        self->slot_to_dense[slot]  = dense;
        self->dense_to_slot[dense] = slot;

        // TSan Fix: Publish state atomically
        atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);

        // Queue Command
        PhysicsCommand *cmd  = &self->command_queue[self->command_count++];
        cmd->header          = CMD_HEADER(CMD_CREATE_BODY, slot);
        cmd->create.settings = settings_buf[i];

        PyList_SET_ITEM(result_list, i, PyLong_FromUnsignedLongLong(raw_h));
    }

    // Synchronize metadata for Python
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    CULV_RAW_FREE(pos_buf);
    CULV_RAW_FREE(size_buf);
    CULV_RAW_FREE((void *)settings_buf);
    return result_list;

fail_oom:
    PyErr_NoMemory();
fail:
    if (settings_buf) {
        for (Py_ssize_t i = 0; i < batch_count; i++) {
            if (settings_buf[i]) {
                JPH_BodyCreationSettings_Destroy(settings_buf[i]);
            }
        }
        CULV_RAW_FREE((void *)settings_buf);
    }
    if (pos_buf) {
        CULV_RAW_FREE(pos_buf);
    }
    if (size_buf) {
        CULV_RAW_FREE(size_buf);
    }
    return nullptr;
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
    // 1. DEFAULT VALUES
    PyObject *o_pos     = nullptr;
    PyObject *o_rot     = nullptr;
    PyObject *o_verts   = nullptr;
    PyObject *o_indices = nullptr;
    uint64_t user_data  = 0;
    uint32_t cat        = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask       = COLLISION_FILTER_ALL_MASKS;

    // 2. TARGET MAPPING (Using Mesh Group)
    void *targets[Mesh_COUNT]; // Mesh_COUNT generated by DEFINE_INDEX_GROUP
    targets[IDX_MSH_POS]       = (void *)&o_pos;
    targets[IDX_MSH_ROT]       = (void *)&o_rot;
    targets[IDX_MSH_VERTS]     = (void *)&o_verts;
    targets[IDX_MSH_INDICES]   = (void *)&o_indices;
    targets[IDX_MSH_USER_DATA] = (void *)&user_data;
    targets[IDX_MSH_CAT]       = (void *)&cat;
    targets[IDX_MSH_MASK]      = (void *)&mask;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use MeshParser initialized via SCHEMA_MESH
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.MeshParser, targets)) {
        return nullptr;
    }

    // 3. VECTOR/QUAT EXTRACTION (Precision Safe)
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

    // 2. Buffer Acquisition
    Py_buffer v_view = {0};
    Py_buffer i_view = {0};
    if (PyObject_GetBuffer(o_verts, &v_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (PyObject_GetBuffer(o_indices, &i_view, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&v_view);
        return nullptr;
    }

    if (UNLIKELY(v_view.len % VERTEX_STRIDE_BYTES != 0 || i_view.len % VERTEX_STRIDE_BYTES != 0)) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        goto buffer_fail;
    }

    MeshBounds bounds = {(uint32_t)(i_view.len / VERTEX_STRIDE_BYTES),
                         (uint32_t)(v_view.len / VERTEX_STRIDE_BYTES)};

    // 3. Jolt Shape Build (No GIL)
    JPH_Shape *shape = nullptr;
    Py_BEGIN_ALLOW_THREADS JPH_IndexedTriangle *tris =
        build_mesh_triangles((uint32_t *)i_view.buf, bounds);
    if (tris) {
        shape = build_mesh_shape(v_view.buf, bounds, tris);
        CULV_RAW_FREE(tris);
    }
    Py_END_ALLOW_THREADS

        // Release Python buffers IMMEDIATELY after copying data to Jolt.
        // This minimizes time holding Python references.
        PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);

    if (!shape) {
        return nullptr; // build_mesh_shape set the error
    }

    // 4. Creation Settings
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static, 0);

    if (!settings) {
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

    // 5. COMMIT PHASE (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Read atomic counts
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        if (PhysicsWorld_resize(self, (self->capacity == 0) ? INITIAL_BODY_CAPACITY
                                                            : self->capacity * 2) < 0) {
            goto commit_fail;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Atomic Pop and Increment
    uint32_t slot = self->free_slots[available - 1];
    atomic_store_explicit(&self->free_count, available - 1, memory_order_release);
    auto dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);
    JPH_BodyCreationSettings_SetUserData(settings, raw_h);

    // Update non-atomic shadow buffers
    ((PosStride *)self->positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->slot_to_dense[slot]             = dense;
    self->dense_to_slot[dense]            = slot;
    self->user_data[dense]                = user_data;
    self->categories[dense]               = cat;
    self->masks[dense]                    = mask;
    self->body_ids[dense]                 = JPH_INVALID_BODY_ID;

    // TSan Fix: Publish state and view length atomically
    atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Atomic Rollback
        atomic_fetch_sub_explicit(&self->count, 1, memory_order_relaxed);
        size_t r_idx = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
        self->free_slots[r_idx] = slot;
        atomic_store_explicit(&self->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);
        goto commit_fail;
    }

    PhysicsCommand *cmd   = &self->command_queue[self->command_count++];
    cmd->header           = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings  = settings;
    cmd->create.user_data = user_data;
    cmd->create.category  = cat;
    cmd->create.mask      = mask;

    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_Shape_Destroy(shape);
    return PyLong_FromUnsignedLongLong(raw_h);

commit_fail:
    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_BodyCreationSettings_Destroy(settings);
    JPH_Shape_Destroy(shape);
    return (PyErr_Occurred()) ? nullptr : PyErr_NoMemory();

buffer_fail:
    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    return nullptr;
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_body(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;

    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = &h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.DestroyParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes must wait for structural consistency
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // 3. MARK FOR DEFERRED DELETION (ATOMIC REFACTOR)

    // TSan Fix: Atomic load of state (Acquire ensures sync with creator thread)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE || state == SLOT_CHARACTER) {

        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        // Non-atomic command queue update (Exclusive ownership held via SHADOW_LOCK)
        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_DESTROY_BODY, slot);

        // TSan Fix: Atomic store to transition state.
        // Release creates a boundary: any thread that sees SLOT_PENDING_DESTROY
        // is guaranteed not to attempt any further Jolt or shadow buffer reads.
        atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_DESTROY, memory_order_release);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_bodies_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Unchanged)
    PyObject *py_handles_in = nullptr;
    void *targets[BatchDestroy_COUNT];
    targets[IDX_BD_HANDLES] = (void *)&py_handles_in;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.BatchDestroyParser, targets)) {
        return nullptr;
    }

    PyObject *py_handles = PySequence_Fast(py_handles_in, "handles must be a sequence");
    if (UNLIKELY(!py_handles)) {
        return nullptr;
    }

    Py_ssize_t batch_count = PySequence_Fast_GET_SIZE(py_handles);
    PyObject **items       = PySequence_Fast_ITEMS(py_handles);

    if (batch_count <= 0) {
        Py_DECREF(py_handles);
        Py_RETURN_NONE;
    }

    // 2. CRITICAL SECTION (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        PyObject *item = items[i];

        // TSan Fix: Extract value to standard register to avoid implicit atomic load overhead
        uint64_t h_val = PyLong_AsUnsignedLongLong(item);
        if (UNLIKELY(PyErr_Occurred())) {
            PyErr_Clear();
            continue;
        }

        uint32_t slot = 0;
        // TSan Fix: Cast to BodyHandle for atomic parameter validation
        if (unpack_handle(self, (BodyHandle)h_val, &slot)) {

            // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
            uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

            if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE || state == SLOT_CHARACTER) {
                if (UNLIKELY(!ensure_command_capacity(self))) {
                    SHADOW_UNLOCK(&self->shadow_lock);
                    Py_DECREF(py_handles);
                    return PyErr_NoMemory();
                }

                PhysicsCommand *cmd = &self->command_queue[self->command_count++];
                cmd->header         = CMD_HEADER(CMD_DESTROY_BODY, slot);

                // TSan Fix: Atomic store to transition state (Release ensures immediate visibility)
                atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_DESTROY,
                                      memory_order_release);
            }
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    Py_DECREF(py_handles);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_position(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;

    void *targets[SetPos_COUNT];
    targets[IDX_SETPOS_HANDLE] = &h_raw;
    targets[IDX_SETPOS_X]      = &x;
    targets[IDX_SETPOS_Y]      = &y;
    targets[IDX_SETPOS_Z]      = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetPosParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "SetPosition");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Wait for buffer consistency
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic type for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with creator/sync thread)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and virtual characters
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // 3. SHADOW BUFFER MIRROR (Zero-Streak Reset)
    uint32_t dense  = self->slot_to_dense[slot];
    PosStride p_val = {x, y, z, 0.0};

    // Update both CURRENT and PREVIOUS position buffers.
    // This forces LERP(prev, curr, alpha) to return exactly 'p_val' regardless of alpha.
    ((PosStride *)self->positions)[dense]      = p_val;
    ((PosStride *)self->prev_positions)[dense] = p_val;

    // 4. COMMAND COMMIT (For Jolt)
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_POS, slot);
    cmd->pos.x          = x;
    cmd->pos.y          = y;
    cmd->pos.z          = z;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_rotation(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;
    float w;

    void *targets[SetRot_COUNT];
    targets[IDX_SETROT_H] = (void *)&h_raw;
    targets[IDX_SETROT_X] = (void *)&x;
    targets[IDX_SETROT_Y] = (void *)&y;
    targets[IDX_SETROT_Z] = (void *)&z;
    targets[IDX_SETROT_W] = (void *)&w;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetRotParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_QUAT(x, y, z, w, "SetRotation");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Wait for buffer consistency
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to BodyHandle for atomic parameter validation
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support both rigid bodies and characters
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // 3. SHADOW BUFFER MIRROR (Zero-Streak Reset)
    uint32_t dense    = self->slot_to_dense[slot];
    AuxStride rot_val = {x, y, z, w};

    // Update CURRENT rotation
    ((AuxStride *)self->rotations)[dense] = rot_val;

    // Update PREVIOUS rotation (The Fix)
    // This forces the interpolation formula: NLERP(prev, curr, alpha)
    // to return exactly 'rot_val' for any alpha.
    ((AuxStride *)self->prev_rotations)[dense] = rot_val;

    // 4. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_ROT, slot);
    cmd->quat.x         = x;
    cmd->quat.y         = y;
    cmd->quat.z         = z;
    cmd->quat.w         = w;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &h_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetLinVelParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "LinearVelocity");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Wait for buffer consistency
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic BodyHandle for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with creator/sync thread)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and virtual characters
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for velocity update");
        return nullptr;
    }

    // 3. COMMAND COMMIT (For Jolt)
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_LINVEL, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    // 4. CAUSAL CONSISTENCY MIRROR
    // We update the shadow buffer immediately. This allows Python code
    // to read back the velocity in the same frame before Jolt has stepped.
    uint32_t dense    = self->slot_to_dense[slot];
    auto *shadow_lvel = (AuxStride *)self->linear_velocities;

    // Non-atomic stride update (Safe under shadow_lock + block_stepping)
    shadow_lvel[dense] = (AuxStride){x, y, z, 0.0f};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &h_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetAngVelParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "AngularVelocity");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Wait for buffer consistency
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic type for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and newly created ones.
    // Characters are usually upright-constrained and ignore angular velocity.
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError,
                        "Body is not in a valid state for angular velocity update");
        return nullptr;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_ANGVEL, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    // 4. CAUSAL CONSISTENCY MIRROR
    // Update the shadow buffer immediately so getters see the new value in the same frame.
    uint32_t dense     = self->slot_to_dense[slot];
    auto *shadow_avel  = (AuxStride *)self->angular_velocities;
    shadow_avel[dense] = (AuxStride){x, y, z, 0.0f};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = &h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.GetMotionParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Ensure indices are stable
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Characters are supported as they wrap an inner body with a motion type
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    JPH_BodyID bid        = self->body_ids[self->slot_to_dense[slot]];
    JPH_BodyInterface *bi = self->body_interface;

    // 3. JOLT INTERACTION
    // Native Jolt interaction remains safe inside shadow_lock once stepping is blocked
    JPH_MotionType mt = JPH_BodyInterface_GetMotionType(bi, bid);

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromLong((long)mt);
}

PyCFunction_DeclareMethod PhysicsWorld_set_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    int motion_type;

    void *targets[SetMotion_COUNT];
    targets[IDX_SM_H] = (void *)&h_raw;
    targets[IDX_SM_M] = (void *)&motion_type;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetMotionParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Ensure we aren't mutating while buffers are being rearranged
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with creators)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Motion types are only valid for rigid bodies (ALIVE or PENDING)
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Handle is stale or body is being destroyed");
        return nullptr;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // Command queue is non-atomic; protected by shadow_lock + block_stepping
    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_SET_MOTION, slot);
    cmd->motion.motion_type = motion_type;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    uint64_t data_raw;

    void *targets[SetUserData_COUNT];
    targets[IDX_SUD_H] = (void *)&h_raw;
    targets[IDX_SUD_D] = (void *)&data_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetUserDataParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies (ALIVE/PENDING) and virtual characters
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for UserData update");
        return nullptr;
    }

    // 3. COMMAND & MIRROR
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // MIRROR: Update the shadow buffer immediately so getters see the new value.
    // Non-atomic store is safe here as we hold the shadow_lock and verified no stepping.
    uint32_t dense         = self->slot_to_dense[slot];
    self->user_data[dense] = data_raw;

    // QUEUE: Command for Jolt (used for collision callbacks)
    PhysicsCommand *cmd          = &self->command_queue[self->command_count++];
    cmd->header                  = CMD_HEADER(CMD_SET_USER_DATA, slot);
    cmd->user_data.user_data_val = data_raw;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.GetUserDataParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Wait for buffer consistency (Stepper thread performs swaps outside this lock)
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Check liveness: ALIVE, CHARACTER, or PENDING_CREATE
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_CHARACTER && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // Shadow buffers are non-atomic; safe to access while holding shadow_lock + block_stepping
    uint64_t val = self->user_data[self->slot_to_dense[slot]];

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(val);
}

PyCFunction_DeclareMethod PhysicsWorld_activate(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;

    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = &h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ActivateParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes (like waking bodies) must wait for physics to be idle
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and virtual characters
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for activation");
        return nullptr;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // Command queue is non-atomic; protected by shadow_lock + block_stepping
    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_ACTIVATE, slot);

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_deactivate(PhysicsWorldObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;

    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ActivateParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Ensure we aren't mutating while buffers are being swapped
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic BodyHandle for helper unpacking
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with creator/sync thread)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and virtual characters
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE || state == SLOT_CHARACTER) {
        if (ensure_command_capacity(self)) {
            // Command queue is non-atomic; protected by shadow_lock + block_stepping
            PhysicsCommand *cmd = &self->command_queue[self->command_count++];
            cmd->header         = CMD_HEADER(CMD_DEACTIVATE, slot);
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_transform(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    PyObject *o_pos = nullptr;
    PyObject *o_rot = nullptr;

    void *targets[SetTrns_COUNT];
    targets[IDX_ST_HANDLE] = (void *)&h_raw;
    targets[IDX_ST_POS]    = (void *)&o_pos;
    targets[IDX_ST_ROT]    = (void *)&o_rot;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetTrnsParser, targets)) {
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

    // 3. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Support rigid bodies and virtual characters
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for transform update");
        return nullptr;
    }

    // 4. SHADOW BUFFER MIRROR (Zero-Streak Reset)
    uint32_t dense  = self->slot_to_dense[slot];
    PosStride p_val = {px, py, pz, 0.0};
    AuxStride r_val = {rx, ry, rz, rw};

    // Update CURRENT and PREVIOUS position/rotation buffers.
    // This forces LERP/NLERP to return the exact 'p_val' and 'r_val' for any alpha.
    ((PosStride *)self->positions)[dense]      = p_val;
    ((PosStride *)self->prev_positions)[dense] = p_val;
    ((AuxStride *)self->rotations)[dense]      = r_val;
    ((AuxStride *)self->prev_rotations)[dense] = r_val;

    // 5. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_TRNS, slot);
    cmd->transform.px   = px;
    cmd->transform.py   = py;
    cmd->transform.pz   = pz;
    cmd->transform.rx   = rx;
    cmd->transform.ry   = ry;
    cmd->transform.rz   = rz;
    cmd->transform.rw   = rw;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_ccd(PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Zero-Allocation)
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    bool enabled;

    void *targets[CCD_COUNT];
    targets[IDX_CCD_H] = (void *)&h_raw;
    targets[IDX_CCD_E] = (void *)&enabled;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CCDParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Modification of body properties requires idle physics
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures creator writes are visible)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Check if the body exists (Support rigid bodies and virtual characters)
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for CCD update");
        return nullptr;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_CCD, slot);

    // We cast the bool to int for storage in the union.
    // Jolt: 1 = LinearCast (CCD On), 0 = Discrete (CCD Off)
    cmd->motion.motion_type = (int)enabled;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_index(PhysicsWorldObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ActivateParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with the world state)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Verify handle is still valid
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // Shadow buffers are stable here (protected by SHADOW_LOCK + BLOCK_UNTIL_NOT_STEPPING)
    uint32_t idx = self->slot_to_dense[slot];

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLong(idx);
}

PyCFunction_DeclareMethod PhysicsWorld_is_alive(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;

    // Group Name: HOnly, Index ID: IDX_H_H
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ActivateParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    bool alive    = false;

    // TSan Fix: Cast to atomic BodyHandle for verification
    if (unpack_handle(self, (BodyHandle)h_raw, &slot)) {
        // TSan Fix: Atomic load of state (Acquire ensures sync with the world state)
        uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

        // Now recognizing SLOT_CHARACTER as a valid alive state
        if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE || state == SLOT_CHARACTER) {
            alive = true;
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    if (alive) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod PhysicsWorld_is_active(PhysicsWorldObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    uint64_t h_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Reuse ActivateParser or similar that only expects a handle
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ActivateParser, targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    // Only bodies actually in the Jolt system can be "active" or "sleeping"
    if (state == SLOT_ALIVE || state == SLOT_CHARACTER) {
        JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);

        // JPH_BodyInterface_IsActive returns true if the body is simulating
        bool active = JPH_BodyInterface_IsActive(self->body_interface, bid);

        if (active) {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    }

    // Pending bodies aren't technically "active" in the simulation yet
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
    void *targets[Render_COUNT];
    targets[IDX_RND_ALPHA] = (void *)&alpha;

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
    // 1. FAST PARSE (Zero-Allocation)
    // TSan Fix: Use standard uint64_t for parsing to avoid implicit seq_cst overhead
    uint64_t h_raw;
    uint32_t category;
    uint32_t mask;

    void *targets[ColFilter_COUNT];
    targets[IDX_CF_H] = (void *)&h_raw;
    targets[IDX_CF_C] = (void *)&category;
    targets[IDX_CF_M] = (void *)&mask;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ColFilterParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes (like collision filters) must block for both sim and queries
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic BodyHandle for verification
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)h_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // TSan Fix: Atomic load of state (Acquire ensures sync with the world state)
    uint8_t state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);

    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_CHARACTER)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // 3. IMMEDIATE WRITE
    // Since we hold the SHADOW_LOCK and the simulation is idle (via BLOCK),
    // these arrays are stable and non-atomic writes are safe here.
    uint32_t dense          = self->slot_to_dense[slot];
    self->categories[dense] = category;
    self->masks[dense]      = mask;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
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
    void *targets[RegMat_COUNT];
    targets[IDX_RM_ID]   = (void *)&id;
    targets[IDX_RM_FRIC] = (void *)&friction;
    targets[IDX_RM_REST] = (void *)&restitution;

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
    // 1. DEFAULT VALUES
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

    // 2. FAST PARSE (Zero-Allocation)
    void *targets[Heightfield_COUNT];
    targets[IDX_HF_POS]       = (void *)&o_pos;
    targets[IDX_HF_ROT]       = (void *)&o_rot;
    targets[IDX_HF_SCALE]     = (void *)&o_scale;
    targets[IDX_HF_HEIGHTS]   = (void *)&o_heights;
    targets[IDX_HF_GRID_SIZE] = (void *)&grid_size;
    targets[IDX_HF_USER_DATA] = (void *)&user_data;
    targets[IDX_HF_CAT]       = (void *)&category;
    targets[IDX_HF_MASK]      = (void *)&mask;
    targets[IDX_HF_MAT_ID]    = (void *)&material_id;
    targets[IDX_HF_FRIC]      = (void *)&friction;
    targets[IDX_HF_REST]      = (void *)&restitution;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.HeightfieldParser, targets)) {
        return nullptr;
    }

    // 3. EXTRACTION (Outside Lock)
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
    if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
        return nullptr;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }
    if (!parse_vec3_direct(o_scale, &sx, &sy, &sz)) {
        return nullptr;
    }

    Py_buffer h_view;
    if (PyObject_GetBuffer(o_heights, &h_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    // Validation
    if (UNLIKELY(h_view.len != (Py_ssize_t)((Py_ssize_t)grid_size * grid_size * sizeof(float)))) {
        PyBuffer_Release(&h_view);
        return PyErr_Format(PyExc_ValueError, "Height buffer size mismatch. Expected %d floats.",
                            grid_size * grid_size);
    }

    // 4. SHAPE CREATION (No GIL)
    JPH_Shape *shape                       = nullptr;
    Py_BEGIN_ALLOW_THREADS JPH_Vec3 offset = {0, 0, 0};
    JPH_Vec3 scale                         = {sx, sy, sz};

    JPH_HeightFieldShapeSettings *hf_settings = JPH_HeightFieldShapeSettings_Create(
        (float *)h_view.buf, &offset, &scale, (uint32_t)grid_size, nullptr);

    if (hf_settings) {
        shape = (JPH_Shape *)JPH_HeightFieldShapeSettings_CreateShape(hf_settings);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hf_settings);
    }
    Py_END_ALLOW_THREADS PyBuffer_Release(&h_view);

    if (!shape) {
        return PyErr_Format(PyExc_RuntimeError, "Failed to create HeightField shape");
    }

    // 4. COMMIT PHASE (ATOMIC REFACTOR)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // TSan Fix: Load counts atomically
    size_t available     = atomic_load_explicit(&self->free_count, memory_order_acquire);
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    if (UNLIKELY(available == 0 || current_count + 1 > self->capacity)) {
        if (PhysicsWorld_resize(self, self->capacity + INITIAL_BODY_CAPACITY) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_Shape_Destroy(shape);
            return nullptr;
        }
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Atomic Pop and Increment
    uint32_t slot = self->free_slots[available - 1];
    atomic_store_explicit(&self->free_count, available - 1, memory_order_release);
    auto dense = (uint32_t)atomic_fetch_add_explicit(&self->count, 1, memory_order_relaxed);

    uint32_t gen      = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle handle = make_handle(slot, gen);
    uint64_t raw_h    = atomic_load_explicit(&handle, memory_order_relaxed);

    // Shadow Write (Non-atomic)
    ((PosStride *)self->positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->slot_to_dense[slot]             = dense;
    self->dense_to_slot[dense]            = slot;
    self->user_data[dense]                = user_data;
    self->categories[dense]               = category;
    self->masks[dense]                    = mask;
    self->material_ids[dense]             = material_id;
    self->body_ids[dense]                 = JPH_INVALID_BODY_ID;

    // TSan Fix: Publish state and view length atomically
    atomic_store_explicit(&self->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);
    self->view_shape[0] = (Py_ssize_t)atomic_load_explicit(&self->count, memory_order_relaxed);

    // 5. COMMAND PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static, 0);

    JPH_BodyCreationSettings_SetFriction(settings, friction);
    JPH_BodyCreationSettings_SetRestitution(settings, restitution);
    JPH_BodyCreationSettings_SetUserData(settings, raw_h);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Atomic Rollback
        atomic_fetch_sub_explicit(&self->count, 1, memory_order_relaxed);
        size_t r_idx = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
        self->free_slots[r_idx] = slot;
        atomic_store_explicit(&self->slot_states[slot], SLOT_EMPTY, memory_order_relaxed);

        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

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
    void *targets[DebugData_COUNT];
    targets[IDX_DD_SHAPES]      = (void *)&draw_shapes;
    targets[IDX_DD_CONSTRAINTS] = (void *)&draw_constraints;
    targets[IDX_DD_BBOX]        = (void *)&draw_bounding_box;
    targets[IDX_DD_CENTERS]     = (void *)&draw_centers;
    targets[IDX_DD_WIREFRAME]   = (void *)&wireframe;

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

PyCFunction_DeclareMethod culv_dump_schema(PyObject *self, PyObject *Py_UNUSED(args)) {
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

// --- The Documentation System ---

#ifndef CULVERIN_DOCS_PATH
#    define CULVERIN_DOCS_PATH "docs/DOCS.md"
#endif

// Embed the markdown documentation. Note: Not 'const' because we will tokenize it.
static char ALL_DOCS[] = {
// NOLINTNEXTLINE(readability-magic-numbers)
#embed CULVERIN_DOCS_PATH suffix(, 0)
};

static_assert(_Generic((ALL_DOCS), char *: true, default: false));

// Global flag to ensure we only stitch once (important for subinterpreters)
static atomic_bool docs_stitched = false;

static void stitch_docs(PyMethodDef *methods, const char *prefix) {
    if (methods == nullptr) {
        return;
    }

    for (PyMethodDef *m = methods; m->ml_name != nullptr; m++) {
        // Skip if it already has a docstring (like internal benchmarks)
        if (m->ml_doc != nullptr) {
            continue;
        }

        // Search for markdown header: "## Prefix_method_name"
        static constexpr size_t KEY_BUFFER_SIZE = 128;
        char key[KEY_BUFFER_SIZE];
        snprintf(key, sizeof(key), "## %s_%s", prefix, m->ml_name);

        // Search for a complete match, skipping any substring matches
        auto search_start = ALL_DOCS;
        char *found       = nullptr;
        while ((found = strstr(search_start, key)) != nullptr) {
            // Verify we have a complete match (not substring of longer name)
            // The character after the key should be whitespace or newline, not alphanumeric or
            // underscore
            char *after_key = found + strlen(key);
            if (*after_key == '\0' || *after_key == '\r' || *after_key == '\n' ||
                *after_key == ' ' || *after_key == '\t') {
                // Complete match found!
                // Move past the header and any newlines
                char *doc_start = after_key;

                // Skip the newline(s) after the header
                while (*doc_start == '\r' || *doc_start == '\n') {
                    doc_start++;
                }

                m->ml_doc = doc_start;
                break; // Found the docstring for this method, move to next method
            }

            // This is a substring match, keep searching
            search_start = found + 1;
        }
    }
}

// Pass 1b: Stitch docstrings into PyGetSetDef getters
static void stitch_docs_getset(PyGetSetDef *getset, const char *prefix) {
    if (getset == nullptr) {
        return;
    }

    for (PyGetSetDef *g = getset; g->name != nullptr; g++) {
        // Skip if it already has a docstring
        if (g->doc != nullptr) {
            continue;
        }

        // Search for markdown header: "## Prefix_name"
        static constexpr size_t KEY_BUFFER_SIZE = 128;
        char key[KEY_BUFFER_SIZE];
        snprintf(key, sizeof(key), "## %s_%s", prefix, g->name);

        // Search for a complete match, skipping any substring matches
        auto search_start = ALL_DOCS;
        char *found       = nullptr;
        while ((found = strstr(search_start, key)) != nullptr) {
            // Verify we have a complete match (not substring of longer name)
            char *after_key = found + strlen(key);
            if (*after_key == '\0' || *after_key == '\r' || *after_key == '\n' ||
                *after_key == ' ' || *after_key == '\t') {
                // Complete match found!
                // Move past the header and any newlines
                char *doc_start = after_key;

                // Skip the newline(s) after the header
                while (*doc_start == '\r' || *doc_start == '\n') {
                    doc_start++;
                }

                g->doc = doc_start;
                break; // Found the docstring for this getter, move to next
            }

            // This is a substring match, keep searching
            search_start = found + 1;
        }
    }
}

// Pass 2: Null-terminate docstrings at markdown headers (## )
static void finalize_docs() {
    if (ALL_DOCS[0] == '\0') {
        return; // Empty docs, nothing to do
    }

    auto p = ALL_DOCS;
    while ((p = strstr(p, "## ")) != nullptr) {
        // Back up to find the newline before this header
        if (p > ALL_DOCS) {
            auto term_point = p - 1;

            // Skip back over spaces but stop at the first newline
            while (term_point > ALL_DOCS && *term_point != '\n' && *term_point != '\0') {
                if (*term_point != ' ' && *term_point != '\r') {
                    break;
                }
                term_point--;
            }

            // If we found a newline, null-terminate right after it
            if (term_point > ALL_DOCS && *term_point == '\n') {
                *term_point = '\0';
            }
        }

        // Move forward to next header search
        p += 3; // Skip "## "
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

// Getter/Property macro - concise initialization
#define GETSET(name_str, getter_func)                                                              \
    {.name    = (name_str),                                                                        \
     .get     = (getter)(getter_func),                                                             \
     .set     = nullptr,                                                                           \
     .doc     = nullptr,                                                                           \
     .closure = nullptr}

static PyMethodDef module_methods[] = {
    {.ml_name  = "_dump_schema_json",
     .ml_meth  = CULV_CAST(culv_dump_schema),
     .ml_flags = METH_NOARGS,
     .ml_doc   = "Internal: Dumps schema to culverin_schema.json"},
    {.ml_name = nullptr, .ml_meth = nullptr, .ml_flags = 0, .ml_doc = nullptr}};

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

    // --- Interaction ---
    PW_FASTCALL(apply_impulse),
    PW_FASTCALL(apply_angular_impulse),
    PW_FASTCALL(apply_impulse_at),
    PW_FASTCALL(apply_force),
    PW_FASTCALL(apply_torque),
    PW_FASTCALL(set_gravity),
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

static const PyMemberDef PhysicsWorld_members[] = {
    {.name   = "__weaklistoffset__",
     .type   = Py_T_PYSSIZET,
     .offset = offsetof(PhysicsWorldObject, weakreflist),
     .flags  = Py_READONLY,
     .doc    = nullptr},
    {.name = nullptr, .type = 0, .offset = 0, .flags = 0, .doc = nullptr}};

static const PyType_Slot PhysicsWorld_slots[] = {
    {Py_tp_new, PyType_GenericNew},
    {Py_tp_init, PhysicsWorld_init},
    {Py_tp_dealloc, PhysicsWorld_dealloc},
    {Py_tp_methods, (PyMethodDef *)PhysicsWorld_methods},
    {Py_tp_members, (PyMemberDef *)PhysicsWorld_members},
    {Py_tp_getset, (PyGetSetDef *)PhysicsWorld_getset},
    {Py_bf_getbuffer, PhysicsWorld_getbuffer},
    {Py_bf_releasebuffer, PhysicsWorld_releasebuffer},
    {Py_tp_traverse, PhysicsWorld_traverse},
    {Py_tp_clear, PhysicsWorld_clear},
    {0, nullptr},
};

static const PyType_Slot Character_slots[] = {
    {Py_tp_dealloc, Character_dealloc},
    {Py_tp_traverse, Character_traverse},
    {Py_tp_clear, Character_clear},
    {Py_tp_methods, (PyMethodDef *)Character_methods},
    {Py_tp_getset, (PyGetSetDef *)Character_getset},
    {0, nullptr},
};

static const PyType_Slot Vehicle_slots[] = {
    {Py_tp_dealloc, Vehicle_dealloc},
    {Py_tp_traverse, Vehicle_traverse},
    {Py_tp_clear, Vehicle_clear},
    {Py_tp_methods, (PyMethodDef *)Vehicle_methods},
    {Py_tp_getset, (PyGetSetDef *)Vehicle_getset},
    {0, nullptr},
};

static const PyType_Slot Skeleton_slots[] = {
    {Py_tp_new, Skeleton_new},
    {Py_tp_dealloc, Skeleton_dealloc},
    {Py_tp_methods, (PyMethodDef *)Skeleton_methods},
    {0, nullptr},
};

static const PyType_Spec Skeleton_spec = {
    .name      = "culverin._culverin_c.Skeleton",
    .basicsize = sizeof(SkeletonObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)Skeleton_slots,
};

static const PyType_Slot RagdollSettings_slots[] = {
    {Py_tp_dealloc, RagdollSettings_dealloc},
    {Py_tp_methods, (PyMethodDef *)RagdollSettings_methods},
    {0, nullptr},
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

static const PyType_Slot Ragdoll_slots[] = {
    {Py_tp_dealloc, Ragdoll_dealloc},
    {Py_tp_methods, (PyMethodDef *)Ragdoll_methods},
    {0, nullptr},
};

static const PyType_Spec Ragdoll_spec = {
    .name      = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(RagdollObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)Ragdoll_slots,
};

// --- Module Initialization ---

static int init_types(PyObject *m, CulverinState *st) {
    struct {
        PyType_Spec *spec;
        PyObject **slot;
        const char *name;
    } types[] = {
        {(PyType_Spec *)&PhysicsWorld_spec, &st->PhysicsWorldType, "PhysicsWorld"},
        {(PyType_Spec *)&Character_spec, &st->CharacterType, "Character"},
        {(PyType_Spec *)&Vehicle_spec, &st->VehicleType, "Vehicle"},
        {(PyType_Spec *)&RagdollSettings_spec, &st->RagdollSettingsType, "RagdollSettings"},
        {(PyType_Spec *)&Ragdoll_spec, &st->RagdollType, "Ragdoll"},
        {(PyType_Spec *)&Skeleton_spec, &st->SkeletonType, "Skeleton"}};

    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        PyObject *type = PyType_FromModuleAndSpec(m, types[i].spec, nullptr);
        if (!type) {
            return -1;
        }
        if (PyModule_AddObject(m, types[i].name, type) < 0) {
            Py_DECREF(type);
            return -1;
        }
        Py_INCREF(type);
        *types[i].slot = type;
    }
    return 0;
}

static int init_constants(PyObject *m) {
    static const struct {
        const char *name;
        int value;
    } consts[] = {{"SHAPE_BOX", CULV_SHAPE_BOX},
                  {"SHAPE_SPHERE", CULV_SHAPE_SPHERE},
                  {"SHAPE_CAPSULE", CULV_SHAPE_CAPSULE},
                  {"SHAPE_CYLINDER", CULV_SHAPE_CYLINDER},
                  {"SHAPE_PLANE", CULV_SHAPE_PLANE},
                  {"SHAPE_MESH", CULV_SHAPE_MESH},
                  {"SHAPE_HEIGHTFIELD", CULV_SHAPE_HEIGHTFIELD},
                  {"SHAPE_CONVEX_HULL", CULV_SHAPE_CONVEX_HULL},
                  {"MOTION_STATIC", 0},
                  {"MOTION_KINEMATIC", 1},
                  {"MOTION_DYNAMIC", 2},
                  {"CONSTRAINT_FIXED", 0},
                  {"CONSTRAINT_POINT", 1},
                  {"CONSTRAINT_HINGE", 2},
                  {"CONSTRAINT_SLIDER", 3},
                  {"CONSTRAINT_DISTANCE", 4},
                  {"CONSTRAINT_CONE", 5},
                  {"EVENT_ADDED", 0},
                  {"EVENT_PERSISTED", 1},
                  {"EVENT_REMOVED", 2}};

    for (size_t i = 0; i < sizeof(consts) / sizeof(consts[0]); i++) {
        if (PyModule_AddIntConstant(m, consts[i].name, consts[i].value) < 0) {
            return -1;
        }
    }
    return 0;
}

PyType_DeclareSlot_Status culverin_exec(PyObject *m) {
    CulverinState *st = get_culverin_state(m);

    // Atomic "test and set" to ensure only one thread ever runs the stitcher
    bool expected = false;
    if (atomic_compare_exchange_strong(&docs_stitched, &expected, true)) {
        stitch_docs(PhysicsWorld_methods, "PhysicsWorld");
        stitch_docs(Character_methods, "Character");
        stitch_docs(Vehicle_methods, "Vehicle");
        stitch_docs(Skeleton_methods, "Skeleton");
        stitch_docs(Ragdoll_methods, "Ragdoll");
        stitch_docs(RagdollSettings_methods, "RagdollSettings");

        // Stitch docstrings into getsets
        stitch_docs_getset(PhysicsWorld_getset, "PhysicsWorld");
        stitch_docs_getset(Character_getset, "Character");
        stitch_docs_getset(Vehicle_getset, "Vehicle");

        finalize_docs(); // Now safe to terminate strings
    }

    if (!JPH_Init()) {
        PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
        return -1;
    }

    culverin_init_all_parsers(&st->parsers);

    CULV_INIT_PROFILER();

    JPH_BroadPhaseLayerFilter_SetProcs(&global_bp_procs);
    JPH_ObjectLayerFilter_SetProcs(&global_obj_procs);
    JPH_BodyFilter_SetProcs(&global_bf_procs);
    JPH_ShapeFilter_SetProcs(&global_sf_procs);

    if (INIT_NATIVE_MUTEX(g_jph_trampoline_lock) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize global JPH trampoline lock");
        return -1;
    }

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

static const PyModuleDef_Slot culverin_slots[] = {{Py_mod_exec, culverin_exec},

// 1. Handle the Free-threaded (No GIL) declaration (3.13+)
#if defined(Py_MOD_GIL_NOT_USED)
                                                  {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif

                                                  // 2. Handle Subinterpreter support
                                                  {Py_mod_multiple_interpreters,
#if PY_VERSION_HEX >= 0x030D0000
                                                   Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED
#else
                                                   Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
#endif
                                                  },

                                                  {0, nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyModuleDef culverin_module = {
    PyModuleDef_HEAD_INIT,
    .m_name     = "_culverin_c",
    .m_doc      = "Culverin Physics Engine Core",
    .m_size     = sizeof(CulverinState),
    .m_methods  = module_methods,
    .m_slots    = (PyModuleDef_Slot *)culverin_slots,
    .m_traverse = culverin_traverse,
    .m_clear    = culverin_clear,
};

extern PyMODINIT_FUNC PyInit__culverin_c(void) { return PyModuleDef_Init(&culverin_module); }
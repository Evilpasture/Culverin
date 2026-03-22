#define _CRT_SECURE_NO_WARNINGS
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_character.h"
#include "culverin_compiler_specifics.h"
#include "culverin_constraint.h"
#include "culverin_contact_listener.h"
#include "culverin_fast_parse.h"
#include "culverin_filters.h"
#include "culverin_getters.h"
#include "culverin_parsers.h"
#include "culverin_physics_world_internal.h"
#include "culverin_query_methods.h"
#include "culverin_ragdoll.h"
#include "culverin_shadow_sync.h"
#include "culverin_vehicle.h"

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

    // 1. The GC "Safety Shield"
    // Use the check-then-untrack to avoid the 'already untracked' abort
    if (PyObject_GC_IsTracked((PyObject *)self)) {
        PyObject_GC_UnTrack(self);
    }

    // 2. Weakref cleanup
    if (self->weakreflist != NULL) {
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
    if (self->system != NULL) {
        // Please don't call __init__() again.
        PyErr_SetString(PyExc_RuntimeError,
                        "PhysicsWorld instance has already been initialized and "
                        "cannot be re-initialized.");
        return -1;
    }
    PyObject *settings_dict = NULL;
    PyObject *bodies_list   = NULL;
    PyObject *baked         = NULL;
    float gx;
    float gy;
    float gz;
    int max_bodies;
    int max_pairs;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", (char *[]){"settings", "bodies", NULL},
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

    self->material_count        = 0;
    self->material_capacity     = 0;
    self->free_count            = 0;
    self->slot_capacity         = 0;
    self->command_count         = 0;
    self->command_capacity      = 0;
    self->spare_capacity        = 0;
    self->shape_cache_count     = 0;
    self->shape_cache_capacity  = 0;
    self->count                 = 0;
    self->capacity              = 0;
    self->constraint_count      = 0;
    self->constraint_capacity   = 0;
    self->free_constraint_count = 0;
    self->time                  = 0.0;

    // 1.5. Query & Sync State
    self->max_jolt_bodies = 0;
    atomic_init(&self->active_queries, 0);
    self->view_export_count = 0;
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
        baked               = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
        Py_XDECREF(bake_func);
        if (!baked) {
            goto fail;
        }
        self->count = PyLong_AsSize_t(PyTuple_GetItem(baked, 0));
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

    for (auto i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity; i++) {
        self->generations[i]                 = 1;
        self->free_slots[self->free_count++] = i;
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
    BodyHandle handle_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &handle_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &ImpulseParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Impulse");

    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Don't mutate state while Jolt is updating buffers
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];

    // CASE 1: Body is already in Jolt (ALIVE)
    if (state == SLOT_ALIVE) {
        uint32_t dense_idx = self->slot_to_dense[slot];
        JPH_BodyID bid     = self->body_ids[dense_idx];

        // Release shadow lock, release GIL, and talk to Jolt immediately
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
        JPH_BodyInterface_AddImpulse(self->body_interface, bid, &imp);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body was just created but not yet stepped (PENDING_CREATE)
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        // QUEUE IT: Since the command queue preserves order, this impulse will
        // execute immediately after the creation command during the next step.
        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_IMPULSE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    }
    // CASE 3: Body is dead or being destroyed
    else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return NULL;
    }

    Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_apply_impulse_at(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    BodyHandle handle_raw;
    float ix;
    float iy;
    float iz;
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;

    void *targets[ImpAt_COUNT];
    targets[IDX_IMPAT_H]  = (void *)&handle_raw;
    targets[IDX_IMPAT_IX] = (void *)&ix;
    targets[IDX_IMPAT_IY] = (void *)&iy;
    targets[IDX_IMPAT_IZ] = (void *)&iz;
    targets[IDX_IMPAT_PX] = (void *)&px;
    targets[IDX_IMPAT_PY] = (void *)&py;
    targets[IDX_IMPAT_PZ] = (void *)&pz;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &ImpulseAtParser, targets)) {
        return NULL;
    }

    /* Validate parsed vectors before entering critical sections */
    VALIDATE_FINITE_VEC3(ix, iy, iz, "Impulse");
    VALIDATE_FINITE_VEC3(px, py, pz, "Impulse position");

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];

    // CASE 1: Immediate execution for active bodies
    if (state == SLOT_ALIVE) {
        JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {ix, iy, iz};
        JPH_RVec3 v_pos                     = {px, py, pz};
        JPH_BodyInterface_AddImpulse2(self->body_interface, bid, &imp, &v_pos);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Deferred execution for pending bodies (Causal Consistency)
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
        return NULL;
    }

    Py_RETURN_NONE;
}
PyCFunction_DeclareMethod PhysicsWorld_apply_angular_impulse(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = (void *)&handle_raw;
    targets[IDX_V3_X] = (void *)&x;
    targets[IDX_V3_Y] = (void *)&y;
    targets[IDX_V3_Z] = (void *)&z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &AngImpulseParser, targets)) {
        return NULL;
    }

    // Finite validation (Outside lock)
    VALIDATE_FINITE_VEC3(x, y, z, "Angular impulse");

    // 2. CONCURRENCY & EXECUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Block only if a simulation step is currently swapping buffers
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];

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

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_ANG_IMPULSE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return NULL;
    }

    Py_RETURN_NONE;
}
PyCFunction_DeclareMethod PhysicsWorld_apply_force(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = (void *)&handle_raw;
    targets[IDX_V3_X] = (void *)&x;
    targets[IDX_V3_Y] = (void *)&y;
    targets[IDX_V3_Z] = (void *)&z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &ForceParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Force");

    // 2. CONCURRENCY & EXECUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Only block if the world is currently updating its internal buffers
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];

    // CASE 1: Body is already in Jolt
    if (state == SLOT_ALIVE) {
        JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 force_vec = {x, y, z};
        // Jolt Force is an accumulator, safe to add outside the main step
        JPH_BodyInterface_AddForce(self->body_interface, bid, &force_vec);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body is queued for creation (Order-preserving)
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
        return NULL;
    }

    Py_RETURN_NONE;
}
PyCFunction_DeclareMethod PhysicsWorld_apply_torque(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero Allocation)
    BodyHandle handle_raw;
    float x;
    float y;
    float z;

    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = (void *)&handle_raw;
    targets[IDX_V3_X] = (void *)&x;
    targets[IDX_V3_Y] = (void *)&y;
    targets[IDX_V3_Z] = (void *)&z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &TorqueParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "Torque");

    // 2. CONCURRENCY & EXECUTION
    SHADOW_LOCK(&self->shadow_lock);

    // Block if world is updating buffers, but allow parallel execution with Queries
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];

    // CASE 1: Body is active in Jolt
    if (state == SLOT_ALIVE) {
        JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
        SHADOW_UNLOCK(&self->shadow_lock);

        Py_BEGIN_ALLOW_THREADS JPH_Vec3 torque_vec = {x, y, z};
        JPH_BodyInterface_AddTorque(self->body_interface, bid, &torque_vec);
        JPH_BodyInterface_ActivateBody(self->body_interface, bid);
        Py_END_ALLOW_THREADS
    }
    // CASE 2: Body is queued for creation
    else if (state == SLOT_PENDING_CREATE) {
        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_APPLY_TORQUE, slot);
        cmd->vec3f.x        = x;
        cmd->vec3f.y        = y;
        cmd->vec3f.z        = z;

        SHADOW_UNLOCK(&self->shadow_lock);
    } else {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is dead or being destroyed");
        return NULL;
    }

    Py_RETURN_NONE;
}
PyCFunction_DeclareMethod PhysicsWorld_set_gravity(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    float x;
    float y;
    float z;

    // Uses the shared XYZ group count and indices
    void *targets[XYZ_COUNT];
    targets[IDX_XYZ_X] = &x;
    targets[IDX_XYZ_Y] = &y;
    targets[IDX_XYZ_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &GravityParser, targets)) {
        return NULL;
    }

    // Validate gravity vector before touching global state
    VALIDATE_FINITE_VEC3(x, y, z, "Gravity");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Modification of global world properties requires the engine to be idle
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Jolt Interaction (JPH_Vec3 uses floats)
    JPH_Vec3 g = {x, y, z};
    JPH_PhysicsSystem_SetGravity(self->system, &g);

    // Safety check for body count overflow (Jolt limit)
    if (UNLIKELY(self->count > UINT32_MAX)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_OverflowError, "Body count exceeds Jolt limit");
        return NULL;
    }

    // Immediate reaction: Wake up all bodies so they fall in the new direction
    if (self->count > 0) {
        JPH_BodyInterface_ActivateBodies(self->body_interface, self->body_ids,
                                         (uint32_t)self->count);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}
PyCFunction_DeclareMethod PhysicsWorld_get_body_stats(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &HOnlyParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Don't read while buffers are being swapped/cleared
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot) ||
                 self->slot_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    uint32_t i = self->slot_to_dense[slot];

    // Snapshot values while holding the lock
    auto p = ((PosStride *)self->positions)[i];
    auto r = ((AuxStride *)self->rotations)[i];
    auto v = ((AuxStride *)self->linear_velocities)[i];

    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. RESULT CONSTRUCTION
    // Use Py_BuildValue to create the nested structure ((x,y,z), (x,y,z,w),
    // (vx,vy,vz)) JPH_REAL_STRING is "d" or "f" depending on double-precision
    // builds
    return Py_BuildValue("( " JPH_REAL_STRING JPH_REAL_STRING JPH_REAL_STRING ") " // Position
                         "(dddd) "                                                 // Rotation
                         "(ddd)",                                                  // Velocity
                         p.x, p.y, p.z, (double)r.x, (double)r.y, (double)r.z, (double)r.w,
                         (double)v.x, (double)v.y, (double)v.z);
}
PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    // 1. DEFAULT VALUES
    BodyHandle handle_raw;
    double surface_y;
    float buoyancy  = 1.0f;
    float lin_drag  = DEFAULT_LINEAR_DRAG;
    float ang_drag  = DEFAULT_ANGULAR_DRAG;
    float dt        = DEFAULT_FRAME_TIME;
    PyObject *o_vel = NULL;

    // 2. TARGET MAPPING (Using Buoy Group count and schema indices)
    void *targets[Buoy_COUNT];
    targets[IDX_BUOY_HANDLE]    = (void *)&handle_raw;
    targets[IDX_BUOY_SURFACE_Y] = (void *)&surface_y;
    targets[IDX_BUOY_BUOYANCY]  = (void *)&buoyancy;
    targets[IDX_BUOY_LIN_DRAG]  = (void *)&lin_drag;
    targets[IDX_BUOY_ANG_DRAG]  = (void *)&ang_drag;
    targets[IDX_BUOY_DT]        = (void *)&dt;
    targets[IDX_BUOY_VEL]       = (void *)&o_vel;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the BuoyParser generated via the X-Macro
    if (!FastParse_Unified(args, nargs, kwnames, &BuoyParser, targets)) {
        return NULL;
    }

    /* Validate numeric inputs */
    VALIDATE_FINITE_FLOAT(buoyancy, "buoyancy");
    VALIDATE_FINITE_FLOAT(lin_drag, "linear drag");
    VALIDATE_FINITE_FLOAT(ang_drag, "angular drag");
    VALIDATE_FINITE_FLOAT(dt, "dt");

    // Parse fluid velocity tuple if provided (Outside Lock)
    float vx = 0;
    float vy = 0;
    float vz = 0;
    if (o_vel && o_vel != Py_None) {
        // Generic dispatcher handles JPH_Real vs float automatically
        if (!parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
            return NULL;
        }
        VALIDATE_FINITE_VEC3(vx, vy, vz, "fluid velocity");
    }

    // 3. RESOLUTION PHASE (Locked)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot) ||
                 self->slot_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_FALSE;
    }

    JPH_BodyID bid            = self->body_ids[self->slot_to_dense[slot]];
    JPH_BodyInterface *bi     = self->body_interface;
    JPH_PhysicsSystem *system = self->system;

    // Release lock early to keep Shadow Buffers accessible during math
    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. EXECUTION PHASE (Unlocked & GIL-Friendly)
    bool submerged = false;
    Py_BEGIN_ALLOW_THREADS JPH_BodyInterface_ActivateBody(bi, bid);

    JPH_Vec3 gravity;
    JPH_PhysicsSystem_GetGravity(system, &gravity);

    // Stack allocate Jolt structures with alignment
    JPH_STACK_ALLOC(JPH_RVec3, surf_pos);
    *surf_pos = (JPH_RVec3){0, (JPH_Real)surface_y, 0};

    JPH_STACK_ALLOC(JPH_Vec3, surf_norm);
    *surf_norm = (JPH_Vec3){0, 1.0f, 0};

    JPH_STACK_ALLOC(JPH_Vec3, fluid_vel);
    *fluid_vel = (JPH_Vec3){vx, vy, vz};

    submerged = JPH_BodyInterface_ApplyBuoyancyImpulse(bi, bid, surf_pos, surf_norm, buoyancy,
                                                       lin_drag, ang_drag, fluid_vel, &gravity, dt);
    Py_END_ALLOW_THREADS

        return PyBool_FromLong((int)submerged);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    // 1. DEFAULT VALUES
    PyObject *o_handles = NULL;
    JPH_Real surface_y  = 0.0;
    float buoyancy      = 1.0f;
    float lin_drag      = DEFAULT_LINEAR_DRAG;
    float ang_drag      = DEFAULT_ANGULAR_DRAG;
    float dt            = DEFAULT_FRAME_TIME;
    PyObject *o_vel     = NULL;

    // 2. FAST PARSE
    void *targets[BatchBuoy_COUNT];
    targets[IDX_BBUOY_HANDLES]   = (void *)&o_handles;
    targets[IDX_BBUOY_SURFACE_Y] = &surface_y;
    targets[IDX_BBUOY_BUOYANCY]  = &buoyancy;
    targets[IDX_BBUOY_LIN_DRAG]  = &lin_drag;
    targets[IDX_BBUOY_ANG_DRAG]  = &ang_drag;
    targets[IDX_BBUOY_DT]        = &dt;
    targets[IDX_BBUOY_VEL]       = (void *)&o_vel;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &BatchBuoyParser, targets)) {
        return NULL;
    }

    // 3. BUFFER & VELOCITY EXTRACTION
    Py_buffer h_view;
    if (PyObject_GetBuffer(o_handles, &h_view, PyBUF_SIMPLE) != 0) {
        return NULL; // PyObject_GetBuffer sets the error
    }

    if (UNLIKELY(h_view.itemsize != 8 && h_view.len % 8 != 0)) {
        PyBuffer_Release(&h_view);
        PyErr_SetString(PyExc_ValueError, "Handle buffer must be uint64");
        return NULL;
    }

    float vx = 0;
    float vy = 0;
    float vz = 0;
    if (o_vel && o_vel != Py_None) {
        if (!parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
            PyBuffer_Release(&h_view);
            return NULL;
        }
    }

    size_t count = (size_t)h_view.len / BODY_ID_SIZE_BYTES;
    if (count == 0) {
        PyBuffer_Release(&h_view);
        Py_RETURN_NONE;
    }

    // 4. TEMP ID RESOLUTION (Locked)
    JPH_BodyID *ids = (JPH_BodyID *)CULV_RAW_MALLOC(count * sizeof(JPH_BodyID));
    if (!ids) {
        PyBuffer_Release(&h_view);
        return PyErr_NoMemory();
    }

    uint64_t *handles  = (uint64_t *)h_view.buf;
    size_t valid_count = 0;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    for (size_t i = 0; i < count; i++) {
        uint32_t slot = 0;
        if ((int)unpack_handle(self, handles[i], &slot) && self->slot_states[slot] == SLOT_ALIVE) {
            ids[valid_count++] = self->body_ids[self->slot_to_dense[slot]];
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&h_view);

    // 5. BATCH EXECUTION (No GIL, No Shadow Lock)
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
        Py_END_ALLOW_THREADS
    }

    CULV_RAW_FREE(ids);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_save_state(PhysicsWorldObject *self,
                                                  PyObject *Py_UNUSED(unused)) {
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // 1. Unambiguous Size Calculation using Stride Structs
    // Create a compile-time constant for the header size
    constexpr size_t HEADER_SIZE =
        sizeof(self->count) + sizeof(self->slot_capacity) + sizeof(self->time);

    // Stride 3 for Positions, Stride 4 for Rot/Vel/AngVel
    size_t pos_size_total = self->count * sizeof(PosStride);
    size_t aux_size_total = self->count * sizeof(AuxStride);

    size_t mapping_size =
        self->slot_capacity *
        (sizeof(typeof(*self->generations)) + sizeof(typeof(*self->slot_to_dense)) +
         sizeof(typeof(*self->dense_to_slot)) + sizeof(typeof(*self->slot_states)));

    // Total = Header + (1 * PosStride) + (3 * AuxStride) + Mapping
    size_t total_size = HEADER_SIZE + pos_size_total + (3 * aux_size_total) + mapping_size;

    PyObject *bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_size);
    if (!bytes) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return NULL;
    }

    char *ptr = PyBytes_AsString(bytes);

    // 2. Encode Header
    memcpy(ptr, &self->count, sizeof(typeof(self->count)));
    ptr += sizeof(typeof(self->count));
    memcpy(ptr, &self->time, sizeof(typeof(self->time)));
    ptr += sizeof(typeof(self->time));
    memcpy(ptr, &self->slot_capacity, sizeof(typeof(self->slot_capacity)));
    ptr += sizeof(typeof(self->slot_capacity));

    // 3. Encode Dense Buffers (Stride Sensitive)
    // Position (Stride 3 + 1 _pad)
    memcpy(ptr, self->positions, pos_size_total);
    ptr += pos_size_total;

    // Rotation (Stride 4)
    memcpy(ptr, self->rotations, aux_size_total);
    ptr += aux_size_total;

    // Linear Velocity (Stride 4)
    memcpy(ptr, self->linear_velocities, aux_size_total);
    ptr += aux_size_total;

    // Angular Velocity (Stride 4)
    memcpy(ptr, self->angular_velocities, aux_size_total);
    ptr += aux_size_total;

    // 4. Encode Mapping Tables (Source -> Destination)
    // generations
    size_t gen_sz = self->slot_capacity * sizeof(*self->generations);
    memcpy(ptr, self->generations, gen_sz);
    ptr += gen_sz;

    // slot_to_dense
    size_t s2d_sz = self->slot_capacity * sizeof(*self->slot_to_dense);
    memcpy(ptr, self->slot_to_dense, s2d_sz);
    ptr += s2d_sz;

    // dense_to_slot
    size_t d2s_sz = self->slot_capacity * sizeof(*self->dense_to_slot);
    memcpy(ptr, self->dense_to_slot, d2s_sz);
    ptr += d2s_sz;

    // slot_states
    size_t state_sz = self->slot_capacity * sizeof(*self->slot_states);
    memcpy(ptr, self->slot_states, state_sz);
    ptr += state_sz;

    SHADOW_UNLOCK(&self->shadow_lock);
    return bytes;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_load_state(PhysicsWorldObject *self, PyObject *const *args,
                                                  Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *state_obj = NULL;
    void *targets[LoadState_COUNT];
    targets[IDX_LS_STATE] = (void *)&state_obj;

    if (!FastParse_Unified(args, nargs, kwnames, &LoadStateParser, targets)) {
        return NULL;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(state_obj, &view, PyBUF_SIMPLE) != 0) {
        return NULL; // PyObject_GetBuffer sets the TypeError for us
    }
    // -------------------------

    // 1. IMMEDIATE SNAPSHOT (GIL held)
    void *local_state_copy = CULV_RAW_MALLOC(view.len);
    if (!local_state_copy) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    memcpy(local_state_copy, view.buf, view.len);
    auto total_len = (size_t)view.len;
    PyBuffer_Release(&view);

    SHADOW_LOCK(&self->shadow_lock);

    // 2. CONCURRENCY GUARD
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // 3. HEADER EXTRACTION
    auto *ptr = (char *)local_state_copy;
    constexpr size_t HEADER_SIZE =
        sizeof(self->count) + sizeof(self->slot_capacity) + sizeof(self->time);

    if (total_len < HEADER_SIZE) {
        goto size_fail;
    }

    auto saved_count    = (typeof(self->count))0;
    auto saved_slot_cap = (typeof(self->slot_capacity))0;
    auto saved_time     = (typeof(self->time))0.0;

    memcpy(&saved_count, ptr, sizeof(saved_count));
    ptr += sizeof(saved_count);
    memcpy(&saved_time, ptr, sizeof(saved_time));
    ptr += sizeof(saved_time);
    memcpy(&saved_slot_cap, ptr, sizeof(saved_slot_cap));
    ptr += sizeof(saved_slot_cap);

    if (saved_slot_cap != self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        CULV_RAW_FREE(local_state_copy);
        PyErr_Format(PyExc_ValueError, "Capacity mismatch: World is %zu, Snapshot is %zu",
                     self->slot_capacity, saved_slot_cap);
        return NULL;
    }

    // 4. FULL SIZE VALIDATION
    size_t pos_bytes = saved_count * sizeof(PosStride);
    size_t aux_bytes = saved_count * sizeof(AuxStride);
    size_t mapping_bytes =
        saved_slot_cap *
        (sizeof(typeof(*self->generations)) + sizeof(typeof(*self->slot_to_dense)) +
         sizeof(typeof(*self->dense_to_slot)) + sizeof(typeof(*self->slot_states)));

    size_t expected = HEADER_SIZE + pos_bytes + (aux_bytes * 3) + mapping_bytes;
    if (UNLIKELY(total_len != expected)) {
        goto size_fail;
    }

    // 5. RESTORE SHADOW STATE
    self->count         = saved_count;
    self->time          = saved_time;
    self->view_shape[0] = (Py_ssize_t)self->count;

    memcpy(self->positions, ptr, pos_bytes);
    ptr += pos_bytes;
    memcpy(self->rotations, ptr, aux_bytes);
    ptr += aux_bytes;
    memcpy(self->linear_velocities, ptr, aux_bytes);
    ptr += aux_bytes;
    memcpy(self->angular_velocities, ptr, aux_bytes);
    ptr += aux_bytes;

    size_t gen_sz = self->slot_capacity * sizeof(*self->generations);
    memcpy(self->generations, ptr, gen_sz);
    ptr += gen_sz;

    size_t s2d_sz = self->slot_capacity * sizeof(*self->slot_to_dense);
    memcpy(self->slot_to_dense, ptr, s2d_sz);
    ptr += s2d_sz;

    size_t d2s_sz = self->slot_capacity * sizeof(*self->dense_to_slot);
    memcpy(self->dense_to_slot, ptr, d2s_sz);
    ptr += d2s_sz;

    size_t state_sz = self->slot_capacity * sizeof(*self->slot_states);
    memcpy(self->slot_states, ptr, state_sz);
    ptr += state_sz;

    // 6. REBUILD FREE LIST
    self->free_count = 0;
    for (auto i = 0u; i < (uint32_t)self->slot_capacity; i++) {
        if (self->slot_states[i] == SLOT_EMPTY) {
            self->free_slots[self->free_count++] = i;
        }
    }

    auto *shadow_pos  = (PosStride *)self->positions;
    auto *shadow_rot  = (AuxStride *)self->rotations;
    auto *shadow_lvel = (AuxStride *)self->linear_velocities;
    auto *shadow_avel = (AuxStride *)self->angular_velocities;

    // 7. JOLT SYNC
    JPH_BodyID *bids      = self->body_ids;
    JPH_BodyInterface *bi = self->body_interface;
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

        uint32_t slot    = self->dense_to_slot[i];
        BodyHandle new_h = make_handle(slot, self->generations[slot]);
        JPH_BodyInterface_SetUserData(bi, bid, (uint64_t)new_h);

        uint32_t j_idx = JPH_ID_TO_INDEX(bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            self->id_to_handle_map[j_idx] = new_h;
        }
    }

    CULV_RAW_FREE(local_state_copy);
    Py_RETURN_NONE;

size_fail:
    SHADOW_UNLOCK(&self->shadow_lock);
    CULV_RAW_FREE(local_state_copy);
    PyErr_SetString(PyExc_ValueError, "Snapshot buffer truncated or stride mismatch");
    return NULL;
}

PyCFunction_DeclareMethod PhysicsWorld_step(PhysicsWorldObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
    float dt = DEFAULT_FRAME_TIME;
    void *targets[Step_COUNT];
    targets[IDX_STEP_DT] = (void *)&dt;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &StepParser, targets)) {
        return NULL;
    }

    /* Validate timestep */
    VALIDATE_FINITE_FLOAT(dt, "dt");

    // --- PHASE 0: RE-ENTRANCY GUARD ---
    if (atomic_load_explicit(&self->is_stepping, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "Concurrent step detected.");
        return NULL;
    }

    // --- PHASE 1: SHADOW STATE LOCK-DOWN ---
    SHADOW_LOCK(&self->shadow_lock);

    // Block ALL Python mutators and Raycasts
    atomic_store_explicit(&self->is_stepping, true, memory_order_relaxed);
    atomic_store_explicit(&self->step_requested, true, memory_order_relaxed);

    // Wait for in-flight queries to finish
    if (atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(self->step_sync.mutex);
        while (atomic_load_explicit(&self->active_queries, memory_order_relaxed) > 0) {
            NATIVE_COND_WAIT(self->step_sync.cond, self->step_sync.mutex);
        }
        NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
        Py_END_ALLOW_THREADS SHADOW_LOCK(&self->shadow_lock);
    }

    // Command Queue Swap
    PhysicsCommand *captured_queue = self->command_queue;
    size_t captured_count          = self->command_count;

    if (UNLIKELY(self->command_capacity > self->spare_capacity)) {
        self->command_queue_spare = (PhysicsCommand *)CULV_RAW_REALLOC(
            self->command_queue_spare, self->command_capacity * sizeof(PhysicsCommand));
        self->spare_capacity = self->command_capacity;
    }
    self->command_queue       = self->command_queue_spare;
    self->command_queue_spare = captured_queue;
    self->command_count       = 0;

    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- PHASE 2: JOLT CRUNCH (GIL Released) ---
    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    // 1. Process Batch Mutations (Shadow-to-Jolt)
    if (captured_count > 0) {
        flush_commands_internal(self, captured_queue, captured_count);
        self->needs_optimization = true;
    }

    // 2. Advance Simulation
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

    // 3. Sync Buffer Results (Jolt-to-Shadow)
    // is_stepping is STILL TRUE here, so Python mutators are still waiting.
    // This is the CRITICAL FIX for the stale handle race.
    culverin_sync_shadow_buffers(self);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        // --- PHASE 3: FINALIZATION & RELEASE ---
        SHADOW_LOCK(&self->shadow_lock);

    // Clear Trash
    if (self->trash_count > 0) {
        for (size_t i = 0; i < self->trash_count; i++) {
            free_new_buffers(&self->trash_buffers[i]);
        }
        self->trash_count = 0;
    }

    // Finalize Metadata
    size_t c_idx        = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    self->contact_count = (c_idx > self->contact_max_capacity) ? self->contact_max_capacity : c_idx;
    self->time += (double)dt;

    // RELEASE THE FENCE: Python threads can now enter safely
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
    // 1. DEFAULT VALUES
    PyObject *o_pos      = NULL;
    PyObject *o_rot      = NULL;
    PyObject *o_points   = NULL;
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
    if (!FastParse_Unified(args, nargs, kwnames, &ConvexHullParser, targets)) {
        return NULL;
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
        return NULL;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return NULL;
    }

    /* Validate position and rotation components */
    VALIDATE_FINITE_VEC3(px, py, pz, "SetTransform position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "SetTransform rotation");

    Py_buffer points_view;
    if (PyObject_GetBuffer(o_points, &points_view, PyBUF_SIMPLE) != 0) {
        return NULL;
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
    JPH_Shape *shape = NULL;
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

    // 6. COMMIT PHASE (Critical Section)
    SHADOW_LOCK(&self->shadow_lock);

    // Protect against concurrent simulation or shadow buffer queries
    BLOCK_UNTIL_NOT_STEPPING(self);

    if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
        if (PhysicsWorld_resize(self, (self->capacity == 0) ? INITIAL_BODY_CAPACITY
                                                            : self->capacity * 2) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            JPH_Shape_Destroy(shape);
            return NULL;
        }
    }

    uint32_t slot     = self->free_slots[--self->free_count];
    uint32_t dense    = (uint32_t)self->count++;
    BodyHandle handle = make_handle(slot, self->generations[slot]);
    JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

    // Update Shadow Buffers
    ((PosStride *)self->positions)[dense]         = (PosStride){.x = px, .y = py, .z = pz};
    ((AuxStride *)self->rotations)[dense]         = (AuxStride){.x = rx, .y = ry, .z = rz, .w = rw};
    ((AuxStride *)self->linear_velocities)[dense] = (AuxStride){};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){};

    self->categories[dense]   = category;
    self->masks[dense]        = mask;
    self->material_ids[dense] = material_id;
    self->user_data[dense]    = user_data;
    self->body_ids[dense]     = JPH_INVALID_BODY_ID;

    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;
    self->slot_states[slot]    = SLOT_PENDING_CREATE;
    self->view_shape[0]        = (Py_ssize_t)self->count;

    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Rollback structural changes
        self->count--;
        self->free_slots[self->free_count++] = slot;
        self->slot_states[slot]              = SLOT_EMPTY;
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

    // Queue Creation Command
    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings    = settings;
    cmd->create.user_data   = user_data;
    cmd->create.category    = category;
    cmd->create.mask        = mask;
    cmd->create.material_id = material_id;

    SHADOW_UNLOCK(&self->shadow_lock);

    // BodySettings/Jolt now owns the shape ref; destroy local handle
    JPH_Shape_Destroy(shape);

    return PyLong_FromUnsignedLongLong(handle);
}

// Helper 1: Build the Jolt Compound Shape from the Python parts list
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static JPH_Shape *init_compound_shape(PhysicsWorldObject *self, PyObject *parts) {
    if (!PyList_Check(parts)) {
        PyErr_SetString(PyExc_TypeError, "Compound parts must be a list");
        return NULL;
    }

    Py_ssize_t num_parts = PyList_Size(parts);
    if (num_parts == 0) {
        PyErr_SetString(PyExc_ValueError, "Compound shape must have at least one part");
        return NULL;
    }

    // --- 1. PARSE PHASE (GIL Held) ---
    // Allocate temp buffer to store parsed data so we can release GIL later
    CompoundPart *buffer = CULV_RAW_MALLOC(sizeof(CompoundPart) * num_parts);
    if (!buffer) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_parts; i++) {
        PyObject *item = PyList_GetItem(parts, i);
        // Expecting tuple: (pos, rot, type, size_params)
        if (!PyTuple_Check(item) || PyTuple_Size(item) < 4) {
            CULV_RAW_FREE(buffer);
            PyErr_Format(PyExc_ValueError, "Part %zd must be a tuple(pos, rot, type, size)", i);
            return NULL;
        }

        PyObject *p_pos  = PyTuple_GetItem(item, 0);
        PyObject *p_rot  = PyTuple_GetItem(item, 1);
        long type_l      = PyLong_AsLong(PyTuple_GetItem(item, 2));
        PyObject *p_size = PyTuple_GetItem(item, 3);

        if (PyErr_Occurred()) {
            CULV_RAW_FREE(buffer);
            return NULL;
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
        return NULL;
    }

    // --- 2. JOLT EXECUTION PHASE (Release GIL, Acquire Jolt Lock) ---
    JPH_Shape *final_shape = NULL;
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
        return NULL;
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
    // 1. DEFAULT VALUES
    PyObject *o_pos      = NULL;
    PyObject *o_rot      = NULL;
    PyObject *o_parts    = NULL;
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
    if (!FastParse_Unified(args, nargs, kwnames, &CompoundParser, targets)) {
        return NULL;
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
        return NULL;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return NULL;
    }

    // SCHEMA defines IDX_HC_DATA as required, but we verify it's a list here
    if (UNLIKELY(!PyList_Check(o_parts))) {
        PyErr_SetString(PyExc_TypeError, "'parts' must be a list of tuples");
        return NULL;
    }

    // 4. SHAPE BUILD (Heavy lifting - released GIL internally)
    JPH_Shape *final_shape = init_compound_shape(self, o_parts);
    if (!final_shape) {
        return NULL;
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

    // 6. COMMIT PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);

    // Critical: Block for both simulation and current queries (e.g. Raycasts)
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Check Capacity
    if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
        size_t needed = (self->capacity == 0) ? INITIAL_BODY_CAPACITY : self->capacity * 2;
        if (PhysicsWorld_resize(self, needed) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            JPH_Shape_Destroy(final_shape);
            return NULL;
        }
    }

    uint32_t slot     = self->free_slots[--self->free_count];
    uint32_t dense    = (uint32_t)self->count++;
    BodyHandle handle = make_handle(slot, self->generations[slot]);
    JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

    // --- IMMEDIATE SHADOW WRITE ---
    PosStride p                                = {.x = px, .y = py, .z = pz};
    ((PosStride *)self->positions)[dense]      = p;
    ((PosStride *)self->prev_positions)[dense] = p;

    AuxStride q                                = {.x = rx, .y = ry, .z = rz, .w = rw};
    ((AuxStride *)self->rotations)[dense]      = q;
    ((AuxStride *)self->prev_rotations)[dense] = q;

    ((AuxStride *)self->linear_velocities)[dense]  = (AuxStride){};
    ((AuxStride *)self->angular_velocities)[dense] = (AuxStride){};

    self->categories[dense]   = category;
    self->masks[dense]        = mask;
    self->material_ids[dense] = material_id;
    self->user_data[dense]    = user_data;
    self->body_ids[dense]     = JPH_INVALID_BODY_ID;

    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;
    self->slot_states[slot]    = SLOT_PENDING_CREATE;
    self->view_shape[0]        = (Py_ssize_t)self->count;

    // 7. QUEUE COMMAND
    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Rollback structural changes on failure
        self->count--;
        self->free_slots[self->free_count++] = slot;
        self->slot_states[slot]              = SLOT_EMPTY;
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

    // Local ref destroyed; Jolt Settings and the Body now own the shape ref.
    JPH_Shape_Destroy(final_shape);
    return PyLong_FromUnsignedLongLong(handle);
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
    auto nargs = PyVectorcall_NARGS(nargsf);

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
    if (!FastParse_Unified(args, nargs, kwnames, &BodyParser, targets)) {
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

    // --- COMMIT PHASE: SHADOW BUFFER UPDATE ---
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    // CRITICAL: Check Jolt limits before assigning a handle
    if (UNLIKELY(self->count >= self->max_jolt_bodies)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_Format(PyExc_RuntimeError,
                     "PhysicsWorld limit reached: %u/%u bodies. Increase 'max_bodies' in settings.",
                     (uint32_t)self->count, self->max_jolt_bodies);
        return NULL;
    }

    // Ensure Shadow Buffer Capacity (but cap it at max_jolt_bodies)
    if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
        size_t next_cap = (self->capacity == 0) ? INITIAL_BODY_CAPACITY : self->capacity * 2;
        if (next_cap > self->max_jolt_bodies) {
            next_cap = self->max_jolt_bodies;
        }

        if (PhysicsWorld_resize(self, next_cap) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            return NULL;
        }
    }

    uint32_t slot     = self->free_slots[--self->free_count];
    auto dense        = (uint32_t)self->count++;
    BodyHandle handle = make_handle(slot, self->generations[slot]);

    JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

    // Typed Pointers for clean struct assignment
    auto *shadow_pos  = (PosStride *)self->positions;
    auto *shadow_ppos = (PosStride *)self->prev_positions;
    auto *shadow_rot  = (AuxStride *)self->rotations;
    auto *shadow_prot = (AuxStride *)self->prev_rotations;
    auto *shadow_lvel = (AuxStride *)self->linear_velocities;
    auto *shadow_avel = (AuxStride *)self->angular_velocities;

    // 1. Position Commit (Stride 4)
    PosStride p        = {};
    p.x                = px;
    p.y                = py;
    p.z                = pz;
    shadow_pos[dense]  = p;
    shadow_ppos[dense] = p;

    // 2. Rotation Commit (Stride 4)
    AuxStride q        = {rx, ry, rz, rw};
    shadow_rot[dense]  = q;
    shadow_prot[dense] = q;

    // 3. Aux Data Commit (Stride 4 / Stride 1)
    AuxStride zero     = {};
    shadow_lvel[dense] = zero;
    shadow_avel[dense] = zero;

    self->categories[dense]   = category;
    self->masks[dense]        = mask;
    self->material_ids[dense] = material_id;
    self->user_data[dense]    = user_data;

    // 4. Indirection Commit
    self->slot_to_dense[slot]  = dense;
    self->dense_to_slot[dense] = slot;
    self->slot_states[slot]    = SLOT_PENDING_CREATE;

    self->view_shape[0] = (Py_ssize_t)self->count;

    // 5. Command Buffer Commit
    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Rollback
        self->count--;
        self->free_slots[self->free_count++] = slot;
        self->slot_states[slot]              = SLOT_EMPTY;
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

    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod PhysicsWorld_create_bodies_batch(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    // 1. FAST PARSE (Zero Lock Contention)
    PyObject *py_positions = NULL;
    PyObject *py_sizes     = NULL;
    int shape_type         = 0;
    int motion_type        = 2;

    // Use BatchCreate Group count and schema IDs
    void *targets[BatchCreate_COUNT];
    targets[IDX_BC_POSITIONS] = (void *)&py_positions;
    targets[IDX_BC_SIZES]     = (void *)&py_sizes;
    targets[IDX_BC_SHAPE]     = (void *)&shape_type;
    targets[IDX_BC_MOTION]    = (void *)&motion_type;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &BatchCreateParser, targets)) {
        return NULL;
    }
    // Initial Validation
    if (!PyList_Check(py_positions) || !PyList_Check(py_sizes)) {
        return PyErr_Format(PyExc_TypeError, "positions and sizes must be lists");
    }

    Py_ssize_t batch_count = PyList_GET_SIZE(py_positions);
    if (PyList_GET_SIZE(py_sizes) != batch_count) {
        return PyErr_Format(PyExc_ValueError, "List length mismatch");
    }

    if (UNLIKELY(self->count + batch_count > self->max_jolt_bodies)) {
        PyErr_Format(PyExc_RuntimeError, "Batch would exceed Jolt body limit (%u)",
                     self->max_jolt_bodies);
        goto fail;
    }

    // 2. TEMP ALLOCATION
    PosStride *pos_buf    = CULV_RAW_MALLOC(batch_count * sizeof(PosStride));
    ShapeParams *size_buf = CULV_RAW_MALLOC(batch_count * sizeof(ShapeParams));
    auto **settings_buf   = (JPH_BodyCreationSettings **)CULV_RAW_CALLOC(
        batch_count, sizeof(JPH_BodyCreationSettings *));

    if (!pos_buf || !size_buf || !settings_buf) {
        CULV_RAW_FREE(pos_buf);
        CULV_RAW_FREE(size_buf);
        CULV_RAW_FREE((void *)settings_buf);
        return PyErr_NoMemory();
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

        // 5. COMMIT PHASE (SHADOW LOCK)
        SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    // Bulk capacity check for slots and dense buffers
    if (self->free_count < (size_t)batch_count || (self->count + batch_count) > self->capacity) {
        size_t needed = self->count + batch_count + INITIAL_BODY_CAPACITY;
        if (PhysicsWorld_resize(self, needed) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            goto fail;
        }
    }

    // Bulk capacity check for command queue
    size_t needed_cmds = self->command_count + batch_count;
    if (self->command_capacity < needed_cmds) {
        void *new_q = CULV_RAW_REALLOC(self->command_queue, needed_cmds * sizeof(PhysicsCommand));
        if (!new_q) {
            SHADOW_UNLOCK(&self->shadow_lock);
            goto fail;
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
    auto *shadow_lvel = (AuxStride *)self->linear_velocities;
    auto *shadow_avel = (AuxStride *)self->angular_velocities;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (!settings_buf[i]) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(result_list, i, Py_None);
            continue;
        }

        uint32_t slot     = self->free_slots[--self->free_count];
        auto dense        = (uint32_t)self->count++;
        BodyHandle handle = make_handle(slot, self->generations[slot]);
        JPH_BodyCreationSettings_SetUserData(settings_buf[i], (uint64_t)handle);

        // Immediate Shadow Buffer Update
        PosStride p        = {.x = pos_buf[i].x, .y = pos_buf[i].y, .z = pos_buf[i].z};
        shadow_pos[dense]  = p;
        shadow_ppos[dense] = p;

        AuxStride identity_q = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};
        shadow_rot[dense]    = identity_q;
        shadow_prot[dense]   = identity_q;

        shadow_lvel[dense] = (AuxStride){};
        shadow_avel[dense] = (AuxStride){};

        // Indirection Mapping
        self->body_ids[dense]      = JPH_INVALID_BODY_ID;
        self->slot_to_dense[slot]  = dense;
        self->dense_to_slot[dense] = slot;
        self->slot_states[slot]    = SLOT_PENDING_CREATE;

        // Queue Creation Command
        PhysicsCommand *cmd  = &self->command_queue[self->command_count++];
        cmd->header          = CMD_HEADER(CMD_CREATE_BODY, slot);
        cmd->create.settings = settings_buf[i];

        // Store result handle
        PyList_SET_ITEM(result_list, i, PyLong_FromUnsignedLongLong(handle));
    }

    self->view_shape[0] = (Py_ssize_t)self->count;
    SHADOW_UNLOCK(&self->shadow_lock);

    CULV_RAW_FREE(pos_buf);
    CULV_RAW_FREE(size_buf);
    CULV_RAW_FREE((void *)settings_buf);
    return result_list;

fail:
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (settings_buf[i]) {
            JPH_BodyCreationSettings_Destroy(settings_buf[i]);
        }
    }
    CULV_RAW_FREE(pos_buf);
    CULV_RAW_FREE(size_buf);
    CULV_RAW_FREE((void *)settings_buf);
    return NULL;
}

/**
 * Helper 1: Build the Jolt triangle array while verifying index bounds.
 */
static JPH_IndexedTriangle *build_mesh_triangles(const uint32_t *raw, MeshBounds bounds) {
    auto *jolt_tris =
        (JPH_IndexedTriangle *)CULV_RAW_MALLOC(bounds.tri_count * sizeof(JPH_IndexedTriangle));
    if (!jolt_tris) {
        PyErr_NoMemory();
        return NULL;
    }

    for (uint32_t t = 0; t < bounds.tri_count; t++) {
        uint32_t i1 = raw[t * 3 + 0];
        uint32_t i2 = raw[t * 3 + 1];
        uint32_t i3 = raw[t * 3 + 2];

        if (i1 >= bounds.vertex_count || i2 >= bounds.vertex_count || i3 >= bounds.vertex_count) {
            CULV_RAW_FREE(jolt_tris);
            PyErr_Format(PyExc_ValueError, "Mesh index out of range: %u/%u/%u >= %u", i1, i2, i3,
                         bounds.vertex_count);
            return NULL;
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
        return NULL;
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
    // 1. DEFAULT VALUES
    PyObject *o_pos     = NULL;
    PyObject *o_rot     = NULL;
    PyObject *o_verts   = NULL;
    PyObject *o_indices = NULL;
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
    if (!FastParse_Unified(args, nargs, kwnames, &MeshParser, targets)) {
        return NULL;
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
        return NULL;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return NULL;
    }

    // 2. Buffer Acquisition
    Py_buffer v_view = {0};
    Py_buffer i_view = {0};
    if (PyObject_GetBuffer(o_verts, &v_view, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(o_indices, &i_view, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&v_view);
        return NULL;
    }

    if (UNLIKELY(v_view.len % VERTEX_STRIDE_BYTES != 0 || i_view.len % VERTEX_STRIDE_BYTES != 0)) {
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        goto buffer_fail;
    }

    MeshBounds bounds = {(uint32_t)(i_view.len / VERTEX_STRIDE_BYTES),
                         (uint32_t)(v_view.len / VERTEX_STRIDE_BYTES)};

    // 3. Jolt Shape Build (No GIL)
    JPH_Shape *shape = NULL;
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
        return NULL; // build_mesh_shape set the error
    }

    // 4. Creation Settings
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static, 0);

    if (!settings) {
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

    // 5. COMMIT PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
        if (PhysicsWorld_resize(self, (self->capacity == 0) ? INITIAL_BODY_CAPACITY
                                                            : self->capacity * 2) < 0) {
            goto commit_fail;
        }
    }

    uint32_t slot = self->free_slots[--self->free_count];
    auto dense    = (uint32_t)self->count++;
    auto handle   = make_handle(slot, self->generations[slot]);
    JPH_BodyCreationSettings_SetUserData(settings, handle);

    // Shadow Write
    ((PosStride *)self->positions)[dense] = (PosStride){.x = px, .y = py, .z = pz};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){.x = rx, .y = ry, .z = rz, .w = rw};
    self->slot_to_dense[slot]             = dense;
    self->dense_to_slot[dense]            = slot;
    self->slot_states[slot]               = SLOT_PENDING_CREATE;
    self->user_data[dense]                = user_data;
    self->categories[dense]               = cat;
    self->masks[dense]                    = mask;
    self->body_ids[dense]                 = JPH_INVALID_BODY_ID;
    self->view_shape[0]                   = (Py_ssize_t)self->count;

    if (UNLIKELY(!ensure_command_capacity(self))) {
        self->count--;
        self->free_slots[self->free_count++] = slot;
        self->slot_states[slot]              = SLOT_EMPTY;
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
    return PyLong_FromUnsignedLongLong(handle);

commit_fail:
    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_BodyCreationSettings_Destroy(settings);
    JPH_Shape_Destroy(shape);
    return (PyErr_Occurred()) ? NULL : PyErr_NoMemory();

buffer_fail:
    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    return NULL;
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_body(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;

    // Group name is HOnly, Index ID is IDX_H_H
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = &handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Uses the DestroyParser (which points to the HOnly group)
    if (!FastParse_Unified(args, nargs, kwnames, &DestroyParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // We only block for STEPPING.
    // Destroying a body is a "Deferred" command, so it doesn't
    // invalidate memory until the next flush_commands() call.
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // 3. MARK FOR DEFERRED DELETION
    uint8_t state = self->slot_states[slot];
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {

        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header         = CMD_HEADER(CMD_DESTROY_BODY, slot);

        // Transition the state so that other Python threads immediately
        // see the body as gone, even before Jolt processes the removal.
        self->slot_states[slot] = SLOT_PENDING_DESTROY;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_bodies_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    PyObject *py_handles_in = NULL;

    void *targets[BatchDestroy_COUNT];
    targets[IDX_BD_HANDLES] = (void *)&py_handles_in;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &BatchDestroyParser, targets)) {
        return NULL;
    }

    // Now it is safe to call PySequence_Fast
    PyObject *py_handles = PySequence_Fast(py_handles_in, "handles must be a sequence");
    if (UNLIKELY(!py_handles)) {
        return NULL;
    }

    Py_ssize_t batch_count = PySequence_Fast_GET_SIZE(py_handles);
    PyObject **items       = PySequence_Fast_ITEMS(py_handles);

    if (batch_count <= 0) {
        Py_DECREF(py_handles);
        Py_RETURN_NONE;
    }

    // --- LOCK SECTION ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    CULV_MAYBE_UNUSED int actual_destroyed = 0;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        PyObject *item = items[i];

        // This converts to int. In 3.14t, this is thread-safe on interned longs.
        BodyHandle h_raw = PyLong_AsUnsignedLongLong(item);

        if (UNLIKELY(PyErr_Occurred())) {
            PyErr_Clear();
            continue;
        }

        uint32_t slot = 0;
        if (unpack_handle(self, h_raw, &slot)) {
            uint8_t state = self->slot_states[slot];

            if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
                if (UNLIKELY(!ensure_command_capacity(self))) {
                    SHADOW_UNLOCK(&self->shadow_lock);
                    Py_DECREF(py_handles);
                    return PyErr_NoMemory();
                }

                PhysicsCommand *cmd = &self->command_queue[self->command_count++];
                cmd->header         = CMD_HEADER(CMD_DESTROY_BODY, slot);

                self->slot_states[slot] = SLOT_PENDING_DESTROY;
                actual_destroyed++;
            }
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    // --- UNLOCK SECTION ---

    Py_DECREF(py_handles);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_position(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;

    // Use auto-generated Group Count and Index IDs
    void *targets[SetPos_COUNT];
    targets[IDX_SETPOS_HANDLE] = &handle_raw;
    targets[IDX_SETPOS_X]      = &x;
    targets[IDX_SETPOS_Y]      = &y;
    targets[IDX_SETPOS_Z]      = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use SetPosParser initialized via SCHEMA_SET_POS
    if (!FastParse_Unified(args, nargs, kwnames, &SetPosParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "SetPosition");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Stale handle");
        return NULL;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_POS, slot);
    cmd->pos.x          = x;
    cmd->pos.y          = y;
    cmd->pos.z          = z;

    // Mirror to shadow buffer for immediate read-back
    auto *shadow_pos                      = (PosStride *)self->positions;
    shadow_pos[self->slot_to_dense[slot]] = (PosStride){x, y, z, 0};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_rotation(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    float x;
    float y;
    float z;
    float w;

    void *targets[SetRot_COUNT];
    targets[IDX_SETROT_H] = &handle_raw;
    targets[IDX_SETROT_X] = &x;
    targets[IDX_SETROT_Y] = &y;
    targets[IDX_SETROT_Z] = &z;
    targets[IDX_SETROT_W] = &w;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &SetRotParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_QUAT(x, y, z, w, "SetRotation");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Stale handle or body being destroyed");
        return NULL;
    }

    // 3. COMMAND COMMIT
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

    // --- CAUSAL CONSISTENCY ---
    // Mirror the new rotation to the shadow buffer immediately.
    // This ensures that Python-side get_rotation() calls return
    // the value just set, even before Jolt processes the command.
    uint32_t dense    = self->slot_to_dense[slot];
    auto *shadow_rot  = (AuxStride *)self->rotations;
    shadow_rot[dense] = (AuxStride){x, y, z, w};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    float x;
    float y;
    float z;

    // Use shared Vec3 Group count and IDs
    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = &handle_raw;
    targets[IDX_V3_X] = &x;
    targets[IDX_V3_Y] = &y;
    targets[IDX_V3_Z] = &z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &SetLinVelParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "LinearVelocity");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Stale handle or body being destroyed");
        return NULL;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_SET_LINVEL, slot);
    cmd->vec3f.x        = x;
    cmd->vec3f.y        = y;
    cmd->vec3f.z        = z;

    // --- CAUSAL CONSISTENCY ---
    // Update the linear_velocities shadow buffer immediately.
    // This allows immediate read-back of the value in Python.
    uint32_t dense     = self->slot_to_dense[slot];
    auto *shadow_lvel  = (AuxStride *)self->linear_velocities;
    shadow_lvel[dense] = (AuxStride){x, y, z, 0.0f};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    float x;
    float y;
    float z;

    // Use shared Vec3 Group count and IDs
    void *targets[Vec3_COUNT];
    targets[IDX_V3_H] = (void *)&handle_raw;
    targets[IDX_V3_X] = (void *)&x;
    targets[IDX_V3_Y] = (void *)&y;
    targets[IDX_V3_Z] = (void *)&z;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &SetAngVelParser, targets)) {
        return NULL;
    }

    VALIDATE_FINITE_VEC3(x, y, z, "AngularVelocity");

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    // Check state: allow if alive or newly created
    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Stale handle or body being destroyed");
        return NULL;
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

    // --- CAUSAL CONSISTENCY MIRROR ---
    // Update the angular_velocities shadow buffer immediately.
    uint32_t dense     = self->slot_to_dense[slot];
    auto *shadow_avel  = (AuxStride *)self->angular_velocities;
    shadow_avel[dense] = (AuxStride){x, y, z, 0.0f};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    // 1. FAST PARSE
    BodyHandle handle_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &GetMotionParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot) ||
                 self->slot_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    JPH_BodyID bid        = self->body_ids[self->slot_to_dense[slot]];
    JPH_BodyInterface *bi = self->body_interface;

    // 3. JOLT INTERACTION (Release GIL)
    int mt;
    Py_BEGIN_ALLOW_THREADS mt = (int)JPH_BodyInterface_GetMotionType(bi, bid);
    Py_END_ALLOW_THREADS

        SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromLong((long)mt);
}

PyCFunction_DeclareMethod PhysicsWorld_set_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    // 1. FAST PARSE
    BodyHandle handle_raw;
    int motion_type;

    void *targets[SetMotion_COUNT];
    targets[IDX_SM_H] = (void *)&handle_raw;
    targets[IDX_SM_M] = (void *)&motion_type;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &SetMotionParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes (like motion type) should wait for both simulation and
    // queries
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Handle is stale or body is being destroyed");
        return NULL;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd     = &self->command_queue[self->command_count++];
    cmd->header             = CMD_HEADER(CMD_SET_MOTION, slot);
    cmd->motion.motion_type = motion_type;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    uint64_t data_raw;

    void *targets[SetUserData_COUNT];
    targets[IDX_SUD_H] = (void *)&handle_raw;
    targets[IDX_SUD_D] = (void *)&data_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &SetUserDataParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state");
        return NULL;
    }

    // 3. COMMAND & MIRROR
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // MIRROR: Update the shadow buffer immediately so getters see the new value
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
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &GetUserDataParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Safety: Ensure indices aren't shifting while we read
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(
            !unpack_handle(self, handle_raw, &slot) ||
            (self->slot_states[slot] != SLOT_ALIVE && self->slot_states[slot] != SLOT_CHARACTER))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    uint64_t val = self->user_data[self->slot_to_dense[slot]];

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(val);
}

PyCFunction_DeclareMethod PhysicsWorld_activate(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;

    // Group Name: HOnly, Index ID: IDX_H_H
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the specific ActivateParser pointing to the HOnly layout
    if (!FastParse_Unified(args, nargs, kwnames, &ActivateParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Commands that modify Jolt state must wait for the simulation to be idle
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // Verify state: Only alive or pending bodies can be activated
    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for activation");
        return NULL;
    }

    // 3. COMMAND COMMIT
    if (UNLIKELY(!ensure_command_capacity(self))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header         = CMD_HEADER(CMD_ACTIVATE, slot);

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_deactivate(PhysicsWorldObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;

    // Group Name: HOnly, Index ID: IDX_H_H
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the specific ActivateParser pointing to the HOnly layout
    if (!FastParse_Unified(args, nargs, kwnames, &ActivateParser, targets)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    if (self->slot_states[slot] == SLOT_ALIVE || self->slot_states[slot] == SLOT_PENDING_CREATE) {
        if (ensure_command_capacity(self)) {
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
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    PyObject *o_pos = NULL;
    PyObject *o_rot = NULL;

    // Use auto-generated Group Count and Index IDs (IDX_ST_...)
    void *targets[SetTrns_COUNT];
    targets[IDX_ST_HANDLE] = (void *)&handle_raw;
    targets[IDX_ST_POS]    = (void *)&o_pos;
    targets[IDX_ST_ROT]    = (void *)&o_rot;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the SetTrnsParser defined via X-Macro
    if (!FastParse_Unified(args, nargs, kwnames, &SetTrnsParser, targets)) {
        return NULL;
    }

    // 2. VECTOR EXTRACTION (Outside of Lock)
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    // Uses your generic dispatchers for precision-safe parsing
    if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
        return NULL;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return NULL;
    }

    // 3. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    uint8_t state = self->slot_states[slot];
    if (UNLIKELY(state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for transform update");
        return NULL;
    }

    // 4. COMMAND COMMIT
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

    // --- CAUSAL CONSISTENCY MIRROR ---
    // Update shadow buffers immediately so immediate read-backs
    // from Python reflect the new state.
    uint32_t dense                        = self->slot_to_dense[slot];
    ((PosStride *)self->positions)[dense] = (PosStride){px, py, pz, 0};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){rx, ry, rz, rw};

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_set_ccd(PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    bool enabled;

    // Use auto-generated Group Count (CCD_COUNT) and IDs (IDX_CCD_H, IDX_CCD_E)
    void *targets[CCD_COUNT];
    targets[IDX_CCD_H] = (void *)&handle_raw;
    targets[IDX_CCD_E] = (void *)&enabled;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the CCDParser initialized via SCHEMA_CCD
    if (!FastParse_Unified(args, nargs, kwnames, &CCDParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Modification of body properties requires idle physics
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // Check if the body exists
    if (UNLIKELY(self->slot_states[slot] != SLOT_ALIVE &&
                 self->slot_states[slot] != SLOT_PENDING_CREATE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not in a valid state for CCD update");
        return NULL;
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
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;

    // Group Name: HOnly, Index ID: IDX_H_H
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the specific ActivateParser pointing to the HOnly layout
    if (!FastParse_Unified(args, nargs, kwnames, &ActivateParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // We don't necessarily need to block for STEPPING here because
    // we are reading the mapping table, which is stable during a step.
    // However, it's safer to block if you expect the index to change
    // mid-step due to a resize.

    uint32_t slot = 0;
    if (UNLIKELY(
            !unpack_handle(self, handle_raw, &slot) ||
            (self->slot_states[slot] != SLOT_ALIVE && self->slot_states[slot] != SLOT_CHARACTER))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    uint32_t idx = self->slot_to_dense[slot];

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLong(idx);
}

PyCFunction_DeclareMethod PhysicsWorld_is_alive(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;

    // Group Name: HOnly, Index ID: IDX_H_H
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    // Use the specific ActivateParser pointing to the HOnly layout
    if (!FastParse_Unified(args, nargs, kwnames, &ActivateParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot = 0;
    bool alive    = false;

    if (unpack_handle(self, handle_raw, &slot)) {
        uint8_t state = self->slot_states[slot];
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

PyCFunction_DeclareMethod PhysicsWorld_get_active_indices(PhysicsWorldObject *self,
                                                          PyObject *Py_UNUSED(args)) {
    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->count;
    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyBytes_FromStringAndSize(NULL, 0);
    }

    // 1. Snapshot the BodyIDs while locked (Fast)
    auto *id_scratch = (JPH_BodyID *)CULV_RAW_MALLOC(count * sizeof(JPH_BodyID));
    if (!id_scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }
    memcpy(id_scratch, self->body_ids, count * sizeof(JPH_BodyID));
    SHADOW_UNLOCK(&self->shadow_lock);

    // 2. Query activity state WHILE UNLOCKED (Deadlock safe)
    auto *results         = (uint32_t *)CULV_RAW_MALLOC(count * sizeof(uint32_t));
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
    // 1. FAST PARSE
    float alpha;
    void *targets[Render_COUNT];
    targets[IDX_RND_ALPHA] = (void *)&alpha;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &RenderParser, targets)) {
        return NULL;
    }

    // Clamp alpha to [0, 1]
    alpha        = fmaxf(0.0f, fminf(1.0f, alpha));
    auto d_alpha = (double)alpha; // Use double for position math

    SHADOW_LOCK(&self->shadow_lock);

    // Consistency: Ensure we aren't reading while a Step is finishing
    BLOCK_UNTIL_NOT_STEPPING(self);

    size_t count = self->count;
    if (UNLIKELY(count == 0)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyBytes_FromStringAndSize(NULL, 0);
    }
    size_t total_bytes = count * FLOATS_PER_INTERPOLATED_BODY * sizeof(float);

    PyObject *bytes_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_bytes);
    if (!bytes_obj) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    float *out = (float *)PyBytes_AsString(bytes_obj);

    // Map shadow buffers to Stride Structs
    auto *curr_p = (PosStride *)self->positions;
    auto *prev_p = (PosStride *)self->prev_positions;
    auto *curr_r = (AuxStride *)self->rotations;
    auto *prev_r = (AuxStride *)self->prev_rotations;

    for (size_t i = 0; i < count; i++) {
        size_t dst = i * FLOATS_PER_INTERPOLATED_BODY;

        // --- 1. Position Lerp (Performed in DOUBLE) ---
        // This prevents jittering when far from the world origin.
        JPH_Real px = prev_p[i].x + (curr_p[i].x - prev_p[i].x) * d_alpha;
        JPH_Real py = prev_p[i].y + (curr_p[i].y - prev_p[i].y) * d_alpha;
        JPH_Real pz = prev_p[i].z + (curr_p[i].z - prev_p[i].z) * d_alpha;

        out[dst + 0] = (float)px;
        out[dst + 1] = (float)py;
        out[dst + 2] = (float)pz;

        // --- 2. Rotation NLerp (Performed in FLOAT) ---
        // Rotations don't suffer from "large coordinate" precision loss,
        // so float is perfect here.
        float q1x = prev_r[i].x;
        float q1y = prev_r[i].y;
        float q1z = prev_r[i].z;
        float q1w = prev_r[i].w;

        float q2x = curr_r[i].x;
        float q2y = curr_r[i].y;
        float q2z = curr_r[i].z;
        float q2w = curr_r[i].w;

        // Shortest path correction
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

        // Re-normalize to ensure it's a valid quaternion
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
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    uint32_t category;
    uint32_t mask;

    void *targets[ColFilter_COUNT];
    targets[IDX_CF_H] = (void *)&handle_raw;
    targets[IDX_CF_C] = (void *)&category;
    targets[IDX_CF_M] = (void *)&mask;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &ColFilterParser, targets)) {
        return NULL;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Structural changes (like collision filters) must block for both sim and
    // queries
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, handle_raw, &slot) ||
                 self->slot_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // 3. IMMEDIATE WRITE
    // We update the dense buffers directly. Since we hold the shadow_lock
    // and verified we aren't stepping/querying, this is thread-safe.
    uint32_t dense          = self->slot_to_dense[slot];
    self->categories[dense] = category;
    self->masks[dense]      = mask;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_register_material(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
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
    if (!FastParse_Unified(args, nargs, kwnames, &RegMatParser, targets)) {
        return NULL;
    }

    // 3. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Update existing material if ID is already registered
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
    // 1. DEFAULT VALUES
    PyObject *o_pos      = NULL;
    PyObject *o_rot      = NULL;
    PyObject *o_scale    = NULL;
    PyObject *o_heights  = NULL;
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
    if (!FastParse_Unified(args, nargs, kwnames, &HeightfieldParser, targets)) {
        return NULL;
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
        return NULL;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return NULL;
    }
    if (!parse_vec3_direct(o_scale, &sx, &sy, &sz)) {
        return NULL;
    }

    Py_buffer h_view;
    if (PyObject_GetBuffer(o_heights, &h_view, PyBUF_SIMPLE) != 0) {
        return NULL;
    }

    // Validation
    if (UNLIKELY(h_view.len != (Py_ssize_t)((Py_ssize_t)grid_size * grid_size * sizeof(float)))) {
        PyBuffer_Release(&h_view);
        return PyErr_Format(PyExc_ValueError, "Height buffer size mismatch. Expected %d floats.",
                            grid_size * grid_size);
    }

    // 4. SHAPE CREATION (No GIL)
    JPH_Shape *shape                       = NULL;
    Py_BEGIN_ALLOW_THREADS JPH_Vec3 offset = {0, 0, 0};
    JPH_Vec3 scale                         = {sx, sy, sz};

    JPH_HeightFieldShapeSettings *hf_settings = JPH_HeightFieldShapeSettings_Create(
        (float *)h_view.buf, &offset, &scale, (uint32_t)grid_size, NULL);

    if (hf_settings) {
        shape = (JPH_Shape *)JPH_HeightFieldShapeSettings_CreateShape(hf_settings);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hf_settings);
    }
    Py_END_ALLOW_THREADS PyBuffer_Release(&h_view);

    if (!shape) {
        return PyErr_Format(PyExc_RuntimeError, "Failed to create HeightField shape");
    }

    // 5. COMMIT PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
        if (PhysicsWorld_resize(self, self->capacity + INITIAL_BODY_CAPACITY) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_Shape_Destroy(shape);
            return NULL;
        }
    }

    uint32_t slot     = self->free_slots[--self->free_count];
    auto dense        = (uint32_t)self->count++;
    BodyHandle handle = make_handle(slot, self->generations[slot]);

    // Update Shadow Buffers
    ((PosStride *)self->positions)[dense] = (PosStride){px, py, pz, 0.0};
    ((AuxStride *)self->rotations)[dense] = (AuxStride){rx, ry, rz, rw};
    self->slot_to_dense[slot]             = dense;
    self->dense_to_slot[dense]            = slot;
    self->slot_states[slot]               = SLOT_PENDING_CREATE;
    self->user_data[dense]                = user_data;
    self->categories[dense]               = category;
    self->masks[dense]                    = mask;
    self->material_ids[dense]             = material_id;
    self->body_ids[dense]                 = JPH_INVALID_BODY_ID;
    self->view_shape[0]                   = (Py_ssize_t)self->count;

    // 6. COMMAND PREP
    JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){px, py, pz}, &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static, 0);

    JPH_BodyCreationSettings_SetFriction(settings, friction);
    JPH_BodyCreationSettings_SetRestitution(settings, restitution);
    JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

    if (UNLIKELY(!ensure_command_capacity(self))) {
        self->count--;
        self->free_slots[self->free_count++] = slot;
        self->slot_states[slot]              = SLOT_EMPTY;
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
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod PhysicsWorld_get_debug_data(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
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
    if (!FastParse_Unified(args, nargs, kwnames, &DebugDataParser, targets)) {
        return NULL;
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
        JPH_PhysicsSystem_DrawBodies(self->system, &settings, self->debug_renderer, NULL);
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

PyCFunction_DeclareMethod culv_dump_schema(CULV_MAYBE_UNUSED PyObject *self,
                                       PyObject *Py_UNUSED(args)) {
    const char* filename = "culverin_schema.json";
    FILE *f              = fopen(filename, "w");
    if (!f) return PyErr_SetFromErrno(PyExc_IOError);

    fp_dump_schemas_json(f);
    fclose(f);

    Py_RETURN_NONE;
}

// --- Type Definition ---

static PyMethodDef module_methods[] = {
    {"_dump_schema_json", culv_dump_schema, METH_NOARGS, "Internal: Dumps schema to culverin_schema.json"},
    {NULL, NULL, 0, NULL}
};

static const PyGetSetDef PhysicsWorld_getset[] = {
    {"positions", (getter)get_positions, NULL, NULL, NULL},
    {"rotations", (getter)get_rotations, NULL, NULL, NULL},
    {"velocities", (getter)get_velocities, NULL, NULL, NULL},
    {"angular_velocities", (getter)get_angular_velocities, NULL, NULL, NULL},
    {"count", (getter)get_count, NULL, NULL, NULL},
    {"time", (getter)get_time, NULL, NULL, NULL},
    {"user_data", (getter)get_user_data_buffer, NULL, NULL, NULL},
    {"shape_count", (getter)get_shape_count, NULL, "Number of unique shapes in cache", NULL},
    {"is_step_pending", (getter)get_is_step_pending, NULL,
     "Whether a physics step is currently in progress. If True, structural changes are blocked.",
     NULL},
    {"max_bodies", (getter)PhysicsWorld_get_max_bodies, NULL,
     "The hard limit of bodies set at init.", NULL},
    {"remaining_capacity", (getter)PhysicsWorld_get_remaining_capacity, NULL,
     "Number of slots available before world.step() is required.", NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static const PyGetSetDef Character_getset[] = {{"handle", (getter)Character_get_handle, NULL,
                                                "The unique physics handle for this character.",
                                                NULL},
                                               {NULL, NULL, NULL, NULL, NULL}};

static const PyGetSetDef Vehicle_getset[] = {{"wheel_count", (getter)Vehicle_get_wheel_count, NULL,
                                              "Number of wheels attached to this vehicle.", NULL},
                                             {NULL, NULL, NULL, NULL, NULL}};

static const PyMethodDef PhysicsWorld_methods[] = {
    // --- Lifecycle ---
    {"step", (PyCFunction)(void (*)(void))PhysicsWorld_step, METH_FASTCALL | METH_KEYWORDS, NULL},
    {"create_body", (PyCFunction)(void (*)(void))PhysicsWorld_create_body,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"create_bodies_batch", (PyCFunction)(void (*)(void))PhysicsWorld_create_bodies_batch,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)(void (*)(void))PhysicsWorld_destroy_body,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"destroy_bodies_batch", (PyCFunction)(void (*)(void))PhysicsWorld_destroy_bodies_batch,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"create_mesh_body", (PyCFunction)(void (*)(void))PhysicsWorld_create_mesh_body,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"create_constraint", (PyCFunction)(void (*)(void))PhysicsWorld_create_constraint,
     METH_FASTCALL | METH_KEYWORDS,
     "Create a constraint between two bodies. Params depend on type."},
    {"destroy_constraint", (PyCFunction)(void (*)(void))PhysicsWorld_destroy_constraint,
     METH_FASTCALL | METH_KEYWORDS, "Remove and destroy a constraint by handle."},
    {"create_vehicle", (PyCFunction)(void (*)(void))PhysicsWorld_create_vehicle,
     METH_FASTCALL | METH_KEYWORDS, "Create a wheeled vehicle constraint"},
    {"create_tracked_vehicle", (PyCFunction)(void (*)(void))PhysicsWorld_create_tracked_vehicle,
     METH_FASTCALL | METH_KEYWORDS, "Create a tracked vehicle constraint (tanks, etc.)"},
    {"create_ragdoll_settings", (PyCFunction)(void (*)(void))PhysicsWorld_create_ragdoll_settings,
     METH_FASTCALL | METH_KEYWORDS, "Create settings for a ragdoll from a skeleton"},
    {"create_ragdoll", (PyCFunction)(void (*)(void))PhysicsWorld_create_ragdoll,
     METH_FASTCALL | METH_KEYWORDS, "Create a multi-body ragdoll from settings"},
    {"create_heightfield", (PyCFunction)(void (*)(void))PhysicsWorld_create_heightfield,
     METH_FASTCALL | METH_KEYWORDS, "Create a static terrain from a height grid."},
    {"create_convex_hull", (PyCFunction)(void (*)(void))PhysicsWorld_create_convex_hull,
     METH_FASTCALL | METH_KEYWORDS,
     "Create a body from a point cloud. Points are wrapped in a convex shell."},
    {"create_compound_body", (PyCFunction)(void (*)(void))PhysicsWorld_create_compound_body,
     METH_FASTCALL | METH_KEYWORDS,
     "Create a body made of multiple primitives. parts=[((x,y,z), "
     "(rx,ry,rz,rw), type, size), ...]"},

    // --- Interaction ---
    {"apply_impulse", (PyCFunction)(void (*)(void))PhysicsWorld_apply_impulse,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"apply_angular_impulse", (PyCFunction)(void (*)(void))PhysicsWorld_apply_angular_impulse,
     METH_FASTCALL | METH_KEYWORDS, "Apply rotational momentum."},
    {"apply_impulse_at", (PyCFunction)(void (*)(void))PhysicsWorld_apply_impulse_at,
     METH_FASTCALL | METH_KEYWORDS, "Apply impulse at world position."},
    {"apply_force", (PyCFunction)(void (*)(void))PhysicsWorld_apply_force,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"apply_torque", (PyCFunction)(void (*)(void))PhysicsWorld_apply_torque,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_gravity", (PyCFunction)(void (*)(void))PhysicsWorld_set_gravity,
     METH_FASTCALL | METH_KEYWORDS, "Set the world gravity vector (x, y, z)."},
    {"apply_buoyancy", (PyCFunction)(void (*)(void))PhysicsWorld_apply_buoyancy,
     METH_FASTCALL | METH_KEYWORDS, "Apply fluid forces to a body."},
    {"apply_buoyancy_batch", (PyCFunction)(void (*)(void))PhysicsWorld_apply_buoyancy_batch,
     METH_FASTCALL | METH_KEYWORDS,
     "Apply buoyancy to a list of bodies. handles must be a buffer of uint64."},
    {"set_position", (PyCFunction)(void (*)(void))PhysicsWorld_set_position,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_rotation", (PyCFunction)(void (*)(void))PhysicsWorld_set_rotation,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_linear_velocity", (PyCFunction)(void (*)(void))PhysicsWorld_set_linear_velocity,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_angular_velocity", (PyCFunction)(void (*)(void))PhysicsWorld_set_angular_velocity,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_transform", (PyCFunction)(void (*)(void))PhysicsWorld_set_transform,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_collision_filter", (PyCFunction)(void (*)(void))PhysicsWorld_set_collision_filter,
     METH_FASTCALL | METH_KEYWORDS, "Dynamically update collision bitmasks."},
    {"register_material", (PyCFunction)(void (*)(void))PhysicsWorld_register_material,
     METH_FASTCALL | METH_KEYWORDS, "Define properties for a material ID."},
    {"set_constraint_target", (PyCFunction)(void (*)(void))PhysicsWorld_set_constraint_target,
     METH_FASTCALL | METH_KEYWORDS, NULL},

    // --- Motion Control ---
    {"get_motion_type", (PyCFunction)(void (*)(void))PhysicsWorld_get_motion_type,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_motion_type", (PyCFunction)(void (*)(void))PhysicsWorld_set_motion_type,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"activate", (PyCFunction)(void (*)(void))PhysicsWorld_activate, METH_FASTCALL | METH_KEYWORDS,
     NULL},
    {"deactivate", (PyCFunction)(void (*)(void))PhysicsWorld_deactivate,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_ccd", (PyCFunction)(void (*)(void))PhysicsWorld_set_ccd, METH_FASTCALL | METH_KEYWORDS,
     "Enable/Disable Continuous Collision Detection."},

    // --- Queries ---
    {"raycast", (PyCFunction)(void (*)(void))PhysicsWorld_raycast, METH_FASTCALL | METH_KEYWORDS,
     NULL},
    {"raycast_batch", (PyCFunction)(void (*)(void))PhysicsWorld_raycast_batch,
     METH_FASTCALL | METH_KEYWORDS, "Execute multiple raycasts efficiently."},
    {"shapecast", (PyCFunction)(void (*)(void))PhysicsWorld_shapecast,
     METH_FASTCALL | METH_KEYWORDS,
     "Sweeps a shape along a direction vector. Returns (Handle, Fraction, "
     "ContactPoint, Normal) or None."},
    {"overlap_sphere", (PyCFunction)(void (*)(void))PhysicsWorld_overlap_sphere,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"overlap_aabb", (PyCFunction)(void (*)(void))PhysicsWorld_overlap_aabb,
     METH_FASTCALL | METH_KEYWORDS, NULL},

    // --- Utilities ---
    {"get_index", (PyCFunction)(void (*)(void))PhysicsWorld_get_index,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"is_alive", (PyCFunction)(void (*)(void))PhysicsWorld_is_alive, METH_FASTCALL | METH_KEYWORDS,
     NULL},
    {"get_active_indices", (PyCFunction)PhysicsWorld_get_active_indices, METH_NOARGS,
     "Returns a bytes object containing uint32 indices of all active bodies."},
    {"get_render_state", (PyCFunction)(void (*)(void))PhysicsWorld_get_render_state,
     METH_FASTCALL | METH_KEYWORDS,
     "Returns packed bytes of interpolated positions and rotations (float32)."},
    {"get_debug_data", (PyCFunction)(void (*)(void))PhysicsWorld_get_debug_data,
     METH_FASTCALL | METH_KEYWORDS,
     "Returns (lines_bytes, triangles_bytes). Each vertex is 16 bytes: [x, y, "
     "z, color_u32]."},
    {"get_body_stats", (PyCFunction)(void (*)(void))PhysicsWorld_get_body_stats,
     METH_FASTCALL | METH_KEYWORDS, NULL},

    // --- User Data ---
    {"get_user_data", (PyCFunction)(void (*)(void))PhysicsWorld_get_user_data,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_user_data", (PyCFunction)(void (*)(void))PhysicsWorld_set_user_data,
     METH_FASTCALL | METH_KEYWORDS, NULL},

    // -- Event Logic ---
    {"get_contact_events", (PyCFunction)PhysicsWorld_get_contact_events, METH_NOARGS, NULL},
    {"get_contact_events_ex", (PyCFunction)PhysicsWorld_get_contact_events_ex, METH_NOARGS,
     "Get rich collision data as dicts"},
    {"get_contact_events_raw", (PyCFunction)PhysicsWorld_get_contact_events_raw, METH_NOARGS,
     "Get raw collision buffer as memoryview"},

    // --- State & Advanced ---
    {"save_state", (PyCFunction)PhysicsWorld_save_state, METH_NOARGS, NULL},
    {"load_state", (PyCFunction)(void (*)(void))PhysicsWorld_load_state,
     METH_FASTCALL | METH_KEYWORDS, "Load world state snapshot"},
    {"create_character", (PyCFunction)(void (*)(void))PhysicsWorld_create_character,
     METH_FASTCALL | METH_KEYWORDS, "Create a virtual character"},

    {NULL, NULL, 0, NULL}};

static const PyMethodDef Character_methods[] = {
    {"move", (PyCFunction)(void (*)(void))Character_move, METH_FASTCALL | METH_KEYWORDS, NULL},
    {"get_position", (PyCFunction)(void (*)(void))Character_get_position, METH_NOARGS, NULL},
    {"set_position", (PyCFunction)(void (*)(void))Character_set_position,
     METH_FASTCALL | METH_KEYWORDS, "Teleport the character to a new position"},
    {"set_rotation", (PyCFunction)(void (*)(void))Character_set_rotation,
     METH_FASTCALL | METH_KEYWORDS, "Set the character's rotation quaternion (x, y, z, w)"},
    {"is_grounded", (PyCFunction)Character_is_grounded, METH_NOARGS, NULL},
    {"set_strength", (PyCFunction)(void (*)(void))Character_set_strength,
     METH_FASTCALL | METH_KEYWORDS, "Set the character's maximum pushing strength"},
    {"get_render_transform", (PyCFunction)Character_get_render_transform, METH_O,
     "Returns interpolated ((x,y,z), (rx,ry,rz,rw)) based on alpha [0-1]."},
    {NULL, NULL, 0, NULL}};

static const PyMethodDef Vehicle_methods[] = {
    {"set_input", (PyCFunction)(void (*)(void))Vehicle_set_input, METH_FASTCALL | METH_KEYWORDS,
     "Set vehicle driver inputs (forward, right, brake, handbrake)"},
    {"set_tank_input", (PyCFunction)(void (*)(void))Vehicle_set_tank_input,
     METH_FASTCALL | METH_KEYWORDS, "Set inputs for a tracked vehicle (left, right, brake)"},
    {"get_wheel_transform", (PyCFunction)(void (*)(void))Vehicle_get_wheel_transform,
     METH_FASTCALL | METH_KEYWORDS, "Get wheel transform in world space"},
    {"get_wheel_local_transform", (PyCFunction)(void (*)(void))Vehicle_get_wheel_local_transform,
     METH_FASTCALL | METH_KEYWORDS, "Get wheel transform in local chassis space"},
    {"destroy", (PyCFunction)Vehicle_destroy, METH_NOARGS,
     "Manually remove the vehicle from physics."},
    {"get_debug_state", (PyCFunction)Vehicle_get_debug_state, METH_NOARGS,
     "Print drivetrain and wheel status to stderr"},
    {NULL, NULL, 0, NULL}};

static const PyMethodDef Skeleton_methods[] = {
    {"add_joint", (PyCFunction)(void (*)(void))Skeleton_add_joint, METH_FASTCALL | METH_KEYWORDS,
     "Add a joint to the skeleton"},
    {"get_joint_index", (PyCFunction)(void (*)(void))Skeleton_get_joint_index,
     METH_FASTCALL | METH_KEYWORDS, "Find the index of a joint by name"},
    {"finalize", (PyCFunction)Skeleton_finalize, METH_NOARGS, "Bake skeleton hierarchy"},
    {NULL, NULL, 0, NULL}};

static const PyMethodDef Ragdoll_methods[] = {
    {"drive_to_pose", (PyCFunction)(void (*)(void))Ragdoll_drive_to_pose,
     METH_FASTCALL | METH_KEYWORDS, "Drive ragdoll motors to follow a specific pose"},
    {"get_body_handles", (PyCFunction)Ragdoll_get_body_ids, METH_NOARGS,
     "Get list of body handles"},
    {"get_debug_info", (PyCFunction)Ragdoll_get_debug_info, METH_NOARGS,
     "Returns list of dicts for each part"},
    {NULL, NULL, 0, NULL}};

static const PyMethodDef RagdollSettings_methods[] = {
    {"add_part", (PyCFunction)(void (*)(void))RagdollSettings_add_part,
     METH_FASTCALL | METH_KEYWORDS,
     "Add a body part and its parent constraint to the ragdoll settings"},
    {"stabilize", (PyCFunction)RagdollSettings_stabilize, METH_NOARGS, "Auto-detect collisions"},
    {NULL, NULL, 0, NULL}};

static PyMemberDef PhysicsWorld_members[] = {{"__weaklistoffset__", Py_T_PYSSIZET,
                                              offsetof(PhysicsWorldObject, weakreflist),
                                              Py_READONLY, NULL},
                                             {NULL, 0, 0, 0, NULL}};

static const PyType_Slot PhysicsWorld_slots[] = {
    {Py_tp_new, PyType_GenericNew},
    {Py_tp_init, PhysicsWorld_init},
    {Py_tp_dealloc, PhysicsWorld_dealloc},
    {Py_tp_methods, (PyMethodDef *)PhysicsWorld_methods},
    {Py_tp_members, (PyMemberDef *)PhysicsWorld_members},
    {Py_tp_getset, (PyGetSetDef *)PhysicsWorld_getset},
    {Py_bf_releasebuffer, PhysicsWorld_releasebuffer},
    {Py_tp_traverse, PhysicsWorld_traverse},
    {Py_tp_clear, PhysicsWorld_clear},
    {0, NULL},
};

static const PyType_Slot Character_slots[] = {
    {Py_tp_dealloc, Character_dealloc},
    {Py_tp_traverse, Character_traverse},
    {Py_tp_clear, Character_clear},
    {Py_tp_methods, (PyMethodDef *)Character_methods},
    {Py_tp_getset, (PyGetSetDef *)Character_getset},
    {0, NULL},
};

static const PyType_Slot Vehicle_slots[] = {
    {Py_tp_dealloc, Vehicle_dealloc},
    {Py_tp_traverse, Vehicle_traverse},
    {Py_tp_clear, Vehicle_clear},
    {Py_tp_methods, (PyMethodDef *)Vehicle_methods},
    {Py_tp_getset, (PyGetSetDef *)Vehicle_getset},
    {0, NULL},
};

static const PyType_Slot Skeleton_slots[] = {
    {Py_tp_new, Skeleton_new}, // We defined this
    {Py_tp_dealloc, Skeleton_dealloc},
    {Py_tp_methods, (PyMethodDef *)Skeleton_methods},
    {0, NULL},
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
    {0, NULL},
};

static const PyType_Spec PhysicsWorld_spec = {
    .name      = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(PhysicsWorldObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .slots     = (PyType_Slot *)PhysicsWorld_slots,
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
    {0, NULL},
};

static const PyType_Spec Ragdoll_spec = {
    .name      = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(RagdollObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = (PyType_Slot *)Ragdoll_slots,
};

// --- Module Initialization ---

// 1. Logic for registering types (from the previous refactor)
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
        PyObject *type = PyType_FromModuleAndSpec(m, types[i].spec, NULL);
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

// 2. Logic for constants
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

// 3. Main Entry (Complexity: ~5)
static int culverin_exec(PyObject *m) {
    CulverinState *st = get_culverin_state(m);

    if (!JPH_Init()) {
        PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
        return -1;
    }

    culverin_init_all_parsers();

    // REGISTER FILTERS ONCE HERE
    // This connects the logic (filter_allow_all_bp, UnifiedBodyFilter, etc.)
    // to the JoltC filter objects globally.
    JPH_BroadPhaseLayerFilter_SetProcs(&global_bp_procs);
    JPH_ObjectLayerFilter_SetProcs(&global_obj_procs);
    JPH_BodyFilter_SetProcs(&global_bf_procs);
    JPH_ShapeFilter_SetProcs(&global_sf_procs);

    // Initialize the GLOBAL lock for Jolt trampolines
    INIT_NATIVE_MUTEX(g_jph_trampoline_lock);
#if PY_VERSION_HEX < 0x030D0000
    if (!g_jph_trampoline_lock) {
        PyErr_NoMemory();
        return -1;
    }
#endif

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
static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
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
static int culverin_clear(PyObject *m) {
    CulverinState *st = get_culverin_state(m);
    Py_CLEAR(st->helper);
    Py_CLEAR(st->PhysicsWorldType);
    Py_CLEAR(st->CharacterType);
    Py_CLEAR(st->VehicleType);
    Py_CLEAR(st->RagdollSettingsType);
    Py_CLEAR(st->RagdollType);
    Py_CLEAR(st->SkeletonType);
    culverin_free_all_parsers();
    return 0;
}

static const PyModuleDef_Slot culverin_slots[] = {
    {Py_mod_exec, culverin_exec},
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
    {0, NULL}};

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

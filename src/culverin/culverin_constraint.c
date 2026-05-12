#include "culverin_constraint.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_constraint_factory.h"
#include "culverin_physics_sync.h"
#include "culverin_types.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                                   PyObject *const *args,
                                                                   size_t nargsf,
                                                                   PyObject *kwnames) {

    // 1. FAST PARSE
    int type           = 0;
    uint64_t h1_raw    = 0;
    uint64_t h2_raw    = 0;
    PyObject *o_params = nullptr;
    PyObject *o_motor  = nullptr;

    const void *const restrict targets[CreateConstr_COUNT] = {
        [IDX_CC_TYPE]   = (const void *const restrict)&type,
        [IDX_CC_BODY1]  = (const void *const restrict)&h1_raw,
        [IDX_CC_BODY2]  = (const void *const restrict)&h2_raw,
        [IDX_CC_PARAMS] = (const void *const restrict)&o_params,
        [IDX_CC_MOTOR]  = (const void *const restrict)&o_motor};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &self->parsers->CreateConstrParser, targets)) {
        return nullptr;
    }

    if (UNLIKELY(h1_raw == h2_raw)) {
        PyErr_SetString(PyExc_ValueError, "Cannot create a constraint between a body and itself");
        return nullptr;
    }

    // 2. PARAMS EXTRACTION
    ConstraintParams p;
    params_init(&p);
    int parse_ok = 1;

    switch (type) {
    case CONSTRAINT_FIXED:
        break;
    case CONSTRAINT_POINT:
        parse_ok = parse_point_params(o_params, &p);
        break;
    case CONSTRAINT_HINGE:
        parse_ok = parse_hinge_params(o_params, &p);
        break;
    case CONSTRAINT_SLIDER:
        parse_ok = parse_slider_params(o_params, &p);
        break;
    case CONSTRAINT_CONE:
        parse_ok = parse_cone_params(o_params, &p);
        break;
    case CONSTRAINT_DISTANCE:
        parse_ok = parse_distance_params(o_params, &p);
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown constraint type");
        return nullptr;
    }

    if (!parse_ok) {
        return nullptr; // PyErr set by helpers
    }
    if (o_motor) {
        parse_motor_config(o_motor, &p);
    }

    // 3. RESOURCE RESERVATION (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t s1;
    uint32_t s2;
    if (!unpack_handle(self, (BodyHandle)h1_raw, &s1) ||
        !unpack_handle(self, (BodyHandle)h2_raw, &s2)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    if (UNLIKELY(self->free_constraint_count == 0)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_MemoryError, "Max constraints reached");
    }

    // Allocate heap copy of params for the worker thread
    ConstraintParams *p_heap = (ConstraintParams *)CULV_RAW_MALLOC(sizeof(ConstraintParams));
    memcpy(p_heap, &p, sizeof(ConstraintParams));

    // Reserve constraint slot
    uint32_t c_slot                = self->free_constraint_slots[--self->free_constraint_count];
    self->constraint_types[c_slot] = type;
    atomic_store_explicit(&self->constraint_states[c_slot], SLOT_PENDING_CREATE,
                          memory_order_relaxed);
    uint32_t gen            = self->constraint_generations[c_slot];
    ConstraintHandle handle = ((uint64_t)gen << HANDLE_INDEX_BITS) | c_slot;

    // 4. QUEUE ASYNCHRONOUS COMMAND
    if (UNLIKELY(!ensure_command_capacity(self))) {
        CULV_RAW_FREE(p_heap); // DO NOT LEAK MEMORY
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }
    PhysicsCommand *cmd             = &self->command_queue[self->command_count++];
    cmd->header                     = CMD_HEADER(CMD_CREATE_CONSTRAINT, s1);
    cmd->constraint.body2_slot      = s2;
    cmd->constraint.constraint_slot = c_slot;
    cmd->constraint.type            = type;
    cmd->constraint.params          = p_heap;

    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLongLong(handle);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    // 1. FAST PARSE
    // TSan Fix: Use raw uint64 for the parser target to avoid atomic init overhead
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->DestroyConstrParser, targets)) {
        return nullptr;
    }

    JPH_Constraint *c_to_destroy = nullptr;

    // 2. RESOLUTION PHASE (Inside Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);

    // RESOLUTION PHASE
    uint32_t slot = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(h_raw >> HANDLE_INDEX_BITS);

    uint8_t state = atomic_load_explicit(&self->constraint_states[slot], memory_order_relaxed);

    // FIX: Allow destruction of ALIVE OR PENDING_CREATE
    if (slot >= self->constraint_capacity || self->constraint_generations[slot] != gen ||
        (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {

        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    c_to_destroy = self->constraints[slot];

    // Transition to EMPTY immediately.
    // If it was PENDING, the flush worker will see SLOT_EMPTY and skip creation.
    self->constraints[slot] = nullptr;
    atomic_store_explicit(&self->constraint_states[slot], SLOT_EMPTY, memory_order_release);
    self->constraint_generations[slot]++;
    self->free_constraint_slots[self->free_constraint_count++] = slot;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. JOLT DESTRUCTION PHASE (Outside Shadow Lock)
    if (c_to_destroy) {
        // Automatic Body Wake-up: Prevents bodies from floating if their joint is deleted.
        // ActivateBody is thread-safe in Jolt.
        if (JPH_Constraint_GetType(c_to_destroy) == JPH_ConstraintType_TwoBodyConstraint) {
            auto tbc     = (JPH_TwoBodyConstraint *)c_to_destroy;
            JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
            JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2(tbc);

            if (b1) {
                JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
            }
            if (b2) {
                JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
            }
        }

        JPH_PhysicsSystem_RemoveConstraint(self->system, c_to_destroy);
        JPH_Constraint_Destroy(c_to_destroy);
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                                                       PyObject *const *args,
                                                                       size_t nargsf,
                                                                       PyObject *kwnames) {
    // 1. FAST PARSE (Unchanged)
    uint64_t h_raw;
    float target;

    const void *const restrict targets[SetConstr_COUNT] = {
        [IDX_SCT_H] = (const void *const restrict)&h_raw,
        [IDX_SCT_T] = (const void *const restrict)&target,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &self->parsers->SetConstrTargetParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(h_raw >> HANDLE_INDEX_BITS);

    if (UNLIKELY(slot >= self->constraint_capacity || self->constraint_generations[slot] != gen ||
                 atomic_load_explicit(&self->constraint_states[slot], memory_order_relaxed) !=
                     SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    JPH_Constraint *c         = self->constraints[slot];
    JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

    // 3. JPH EXECUTION
    if (sub == JPH_ConstraintSubType_Hinge) {
        auto hc = (JPH_HingeConstraint *)c;
        JPH_HingeConstraint_SetMotorState(hc, JPH_MotorState_Position);
        JPH_HingeConstraint_SetTargetAngle(hc, target);
    } else if (sub == JPH_ConstraintSubType_Slider) {
        auto sc              = (JPH_SliderConstraint *)c;
        JPH_MotorState state = JPH_SliderConstraint_GetMotorState(sc);

        if (state == JPH_MotorState_Off) {
            state = JPH_MotorState_Position;
            JPH_SliderConstraint_SetMotorState(sc, state);
        }

        if (state == JPH_MotorState_Velocity) {
            JPH_SliderConstraint_SetTargetVelocity(sc, target);
        } else {
            JPH_SliderConstraint_SetTargetPosition(sc, target);
        }
    }

    // 4. WAKE UP BODIES (Locked sequence)
    JPH_ConstraintType type = JPH_Constraint_GetType(c);
    if (type == JPH_ConstraintType_TwoBodyConstraint) {
        JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1((JPH_TwoBodyConstraint *)c);
        JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2((JPH_TwoBodyConstraint *)c);

        if (b1) {
            JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
        }
        if (b2) {
            JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_get_constraint_type(PhysicsWorldObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames) {
    uint64_t handle_raw;
    const void *const restrict targets[HOnly_COUNT] = {[IDX_H_H] =
                                                           (const void *const restrict)&handle_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &self->parsers->HOnlyParser,
                           targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = (uint32_t)(handle_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(handle_raw >> HANDLE_INDEX_BITS);

    // Validate slot range and generation
    if (UNLIKELY(slot >= self->constraint_capacity || self->constraint_generations[slot] != gen)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // Load state atomically
    uint8_t state = atomic_load_explicit(&self->constraint_states[slot], memory_order_acquire);

    // NEW LOGIC: Accept both ALIVE and PENDING_CREATE
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
        // Return from shadow array directly.
        // This is safe even if Jolt hasn't built the object yet.
        int type = self->constraint_types[slot];
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyLong_FromLong((long)type);
    }

    // If state is SLOT_EMPTY or PENDING_DESTROY
    SHADOW_UNLOCK(&self->shadow_lock);
    RAISE_STALE_HANDLE();
}

/*
TODO: for constraints
def set_hinge_limits(self, handle: int, min_angle: float, max_angle: float) -> None: ...
def set_hinge_motor(self, handle: int, target_velocity: float, max_torque: float) -> None: ...
def set_slider_motor(self, handle: int, target_velocity: float, max_force: float) -> None: ...
def set_constraint_enabled(self, handle: int, enabled: bool) -> None: ...

*/
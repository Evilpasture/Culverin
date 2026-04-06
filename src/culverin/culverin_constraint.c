#include "culverin_constraint.h"
#include "culverin_arg_indices.h"
#include "culverin_constraint_factory.h"
#include "culverin_types.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                                   PyObject *const *args,
                                                                   size_t nargsf,
                                                                   PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE
    int type           = 0;
    uint64_t h1_raw    = 0;
    uint64_t h2_raw    = 0;
    PyObject *o_params = nullptr;
    PyObject *o_motor  = nullptr;

    void *targets[CreateConstr_COUNT] = {[IDX_CC_TYPE]   = (void *)&type,
                                         [IDX_CC_BODY1]  = (void *)&h1_raw,
                                         [IDX_CC_BODY2]  = (void *)&h2_raw,
                                         [IDX_CC_PARAMS] = (void *)&o_params,
                                         [IDX_CC_MOTOR]  = (void *)&o_motor};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
                           &st->parsers.CreateConstrParser, targets)) {
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

    // 3. RESOLUTION PHASE (Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    uint32_t s1 = 0;
    uint32_t s2 = 0;
    if (!unpack_handle(self, (BodyHandle)h1_raw, &s1) ||
        !unpack_handle(self, (BodyHandle)h2_raw, &s2)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    uint8_t state1 = atomic_load_explicit(&self->slot_states[s1], memory_order_acquire);
    uint8_t state2 = atomic_load_explicit(&self->slot_states[s2], memory_order_acquire);

    // Predicate: Are these bodies either Alive or about to be?
    const SlotPredicate pred1 = get_slot_predicate(state1, MASK_IMM_STANDARD);
    const SlotPredicate pred2 = get_slot_predicate(state2, MASK_IMM_STANDARD);

    if (!pred1.is_executable || !pred2.is_executable) {
        SHADOW_UNLOCK(&self->shadow_lock);
        RAISE_STALE_HANDLE();
    }

    // LAZY FLUSH: If either body is PENDING_CREATE, we must flush now.
    // Jolt requires valid JPH_BodyID/Pointers to instantiate a constraint.
    if (pred1.is_deferred || pred2.is_deferred) {
        sync_and_flush_internal(self);
        // Re-verify states after flush
        state1 = atomic_load_explicit(&self->slot_states[s1], memory_order_acquire);
        state2 = atomic_load_explicit(&self->slot_states[s2], memory_order_acquire);
    }

    // At this point, both must be SLOT_ALIVE or SLOT_CHARACTER
    JPH_BodyID bid1 = self->body_ids[self->slot_to_dense[s1]];
    JPH_BodyID bid2 = self->body_ids[self->slot_to_dense[s2]];

    if (UNLIKELY(self->free_constraint_count == 0)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_MemoryError, "Max constraints reached (%zu)",
                            self->constraint_capacity);
    }
    uint32_t c_slot = self->free_constraint_slots[--self->free_constraint_count];
    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. JOLT EXECUTION (Lock-safe body pointer retrieval)
    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock1;
    JPH_BodyLockWrite lock2;

    // Lock bodies in consistent ID order to prevent deadlocks
    if (bid1 < bid2) {
        JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
        JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
    } else {
        JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
        JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
    }

    JPH_Constraint *constraint = nullptr;
    if (lock1.body && lock2.body) {
        JPH_Body *b1 = (JPH_Body_GetID(lock1.body) == bid1) ? lock1.body : lock2.body;
        JPH_Body *b2 = (JPH_Body_GetID(lock1.body) == bid2) ? lock1.body : lock2.body;

        switch (type) {
        case CONSTRAINT_FIXED:
            constraint = create_fixed(&p, b1, b2);
            break;
        case CONSTRAINT_POINT:
            constraint = create_point(&p, b1, b2);
            break;
        case CONSTRAINT_HINGE:
            constraint = create_hinge(&p, b1, b2);
            break;
        case CONSTRAINT_SLIDER:
            constraint = create_slider(&p, b1, b2);
            break;
        case CONSTRAINT_CONE:
            constraint = create_cone(&p, b1, b2);
            break;
        case CONSTRAINT_DISTANCE:
            constraint = create_distance(&p, b1, b2);
            break;
        default:
            culv_unreachable();
        }
    }

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock1);
    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock2);

    // 5. COMMIT PHASE
    if (UNLIKELY(!constraint)) {
        SHADOW_LOCK(&self->shadow_lock);
        self->free_constraint_slots[self->free_constraint_count++] = c_slot;
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Jolt failed to instantiate constraint");
    }

    JPH_PhysicsSystem_AddConstraint(self->system, constraint);
    JPH_BodyInterface_ActivateBody(self->body_interface, bid1);
    JPH_BodyInterface_ActivateBody(self->body_interface, bid2);

    SHADOW_LOCK(&self->shadow_lock);
    self->constraints[c_slot]       = constraint;
    self->constraint_states[c_slot] = SLOT_ALIVE;
    uint32_t gen                    = self->constraint_generations[c_slot];
    ConstraintHandle handle         = ((uint64_t)gen << HANDLE_INDEX_BITS) | c_slot;
    SHADOW_UNLOCK(&self->shadow_lock);

    return PyLong_FromUnsignedLongLong(handle);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE
    // TSan Fix: Use raw uint64 for the parser target to avoid atomic init overhead
    uint64_t h_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.DestroyConstrParser, targets)) {
        return nullptr;
    }

    JPH_Constraint *c_to_destroy = nullptr;

    // 2. RESOLUTION PHASE (Inside Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // Manually unpack the constraint-specific handle bitmask
    uint32_t slot = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(h_raw >> HANDLE_INDEX_BITS);

    // Validate identity and state
    if (slot >= self->constraint_capacity || self->constraint_generations[slot] != gen ||
        self->constraint_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    // Snapshot pointer and invalidate the slot
    c_to_destroy = self->constraints[slot];

    self->constraints[slot]       = nullptr;
    self->constraint_states[slot] = SLOT_EMPTY;
    self->constraint_generations[slot]++; // Increment generation to kill existing handles
    self->free_constraint_slots[self->free_constraint_count++] = slot;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. JOLT DESTRUCTION PHASE (Outside Shadow Lock)
    if (c_to_destroy) {
        // Automatic Body Wake-up: Prevents bodies from floating if their joint is deleted.
        // ActivateBody is thread-safe in Jolt.
        if (JPH_Constraint_GetType(c_to_destroy) == JPH_ConstraintType_TwoBodyConstraint) {
            auto *tbc    = (JPH_TwoBodyConstraint *)c_to_destroy;
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Unchanged)
    uint64_t h_raw;
    float target;

    void *targets[SetConstr_COUNT];
    targets[IDX_SCT_H] = (void *)&h_raw;
    targets[IDX_SCT_T] = (void *)&target;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetConstrTargetParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(h_raw >> HANDLE_INDEX_BITS);

    if (UNLIKELY(slot >= self->constraint_capacity || self->constraint_generations[slot] != gen ||
                 self->constraint_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    JPH_Constraint *c         = self->constraints[slot];
    JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

    // 3. JPH EXECUTION
    if (sub == JPH_ConstraintSubType_Hinge) {
        auto *hc = (JPH_HingeConstraint *)c;
        JPH_HingeConstraint_SetMotorState(hc, JPH_MotorState_Position);
        JPH_HingeConstraint_SetTargetAngle(hc, target);
    } else if (sub == JPH_ConstraintSubType_Slider) {
        auto *sc             = (JPH_SliderConstraint *)c;
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
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // 1. FAST PARSE (Zero-Allocation)
    uint64_t handle_raw;
    void *targets[HOnly_COUNT] = {[IDX_H_H] = &handle_raw};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.HOnlyParser,
                           targets)) {
        return nullptr;
    }

    // 2. RESOLUTION PHASE
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Unpack constraint handle bits
    uint32_t slot = (uint32_t)(handle_raw & HANDLE_INDEX_MASK);
    uint32_t gen  = (uint32_t)(handle_raw >> HANDLE_INDEX_BITS);

    // Validate slot range, generation, and liveness
    if (UNLIKELY(slot >= self->constraint_capacity || self->constraint_generations[slot] != gen ||
                 self->constraint_states[slot] != SLOT_ALIVE)) {

        SHADOW_UNLOCK(&self->shadow_lock);

        // POLICY FIX: Use the shim macro to either return None (Silent) or raise ValueError
        // (Strict)
        RAISE_STALE_HANDLE();
    }

    // Extract subtype (Hinge, Slider, etc.) from Jolt
    JPH_Constraint *c         = self->constraints[slot];
    JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

    SHADOW_UNLOCK(&self->shadow_lock);

    return PyLong_FromLong((long)sub);
}
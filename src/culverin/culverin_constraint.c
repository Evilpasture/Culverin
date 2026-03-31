#include "culverin_constraint.h"
#include "culverin_arg_indices.h"
#include "culverin_constraint_factory.h"
#include "culverin_types.h"



// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                                   PyObject *const *args,
                                                                   size_t nargsf,
                                                                   PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    int type           = 0;
    uint64_t h1        = 0;
    uint64_t h2        = 0;
    PyObject *o_params = nullptr;
    PyObject *o_motor  = nullptr;

    void *targets[CreateConstr_COUNT];
    targets[IDX_CC_TYPE]   = (void *)&type;
    targets[IDX_CC_BODY1]  = (void *)&h1;
    targets[IDX_CC_BODY2]  = (void *)&h2;
    targets[IDX_CC_PARAMS] = (void *)&o_params;
    targets[IDX_CC_MOTOR]  = (void *)&o_motor;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &CreateConstrParser, targets)) {
        return nullptr;
    }

    if (UNLIKELY(h1 == h2)) {
        PyErr_SetString(PyExc_ValueError, "Cannot create a constraint between a body and itself");
        return nullptr;
    }

    // 2. PARAMS EXTRACTION (Outside Lock)
    ConstraintParams p;
    params_init(&p); // Assuming this zeros the struct
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
        return nullptr;
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
    if (!unpack_handle(self, h1, &s1) || self->slot_states[s1] != SLOT_ALIVE ||
        !unpack_handle(self, h2, &s2) || self->slot_states[s2] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale body handles");
        return nullptr;
    }

    JPH_BodyID bid1 = self->body_ids[self->slot_to_dense[s1]];
    JPH_BodyID bid2 = self->body_ids[self->slot_to_dense[s2]];

    if (UNLIKELY(self->free_constraint_count == 0)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_MemoryError, "Max constraints reached (%zu)",
                            self->constraint_capacity);
    }
    uint32_t c_slot = self->free_constraint_slots[--self->free_constraint_count];
    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. JOLT EXECUTION (Sorted Locking for Deadlock Prevention)
    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock1;
    JPH_BodyLockWrite lock2;

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
        return PyErr_Format(PyExc_RuntimeError, "Jolt failed to instantiate constraint type %d",
                            type);
    }

    JPH_PhysicsSystem_AddConstraint(self->system, constraint);

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
    // 1. FAST PARSE (Zero-Allocation)
    BodyHandle handle_raw;
    void *targets[HOnly_COUNT];
    targets[IDX_H_H] = (void *)&handle_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &DestroyConstrParser, targets)) {
        return nullptr;
    }

    JPH_Constraint *c_to_destroy = nullptr;

    // 2. RESOLUTION PHASE (Inside Shadow Lock)
    SHADOW_LOCK(&self->shadow_lock);

    // Guard against Physics Step AND active Queries (Raycasts/etc might touch constraints)
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    auto slot = (uint32_t)(handle_raw & HANDLE_INDEX_MASK);
    auto gen  = (uint32_t)(handle_raw >> HANDLE_INDEX_BITS);

    // Validate identity and state
    if (slot >= self->constraint_capacity || self->constraint_generations[slot] != gen ||
        self->constraint_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    // Capture pointer and IMMEDIATELY invalidate the slot
    c_to_destroy = self->constraints[slot];

    self->constraints[slot]       = nullptr;
    self->constraint_states[slot] = SLOT_EMPTY;
    self->constraint_generations[slot]++; // Invalidate future use of this handle
    self->free_constraint_slots[self->free_constraint_count++] = slot;

    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. JOLT DESTRUCTION PHASE (Outside Shadow Lock)
    if (c_to_destroy) {
        // Automatic Body Wake-up: Prevents bodies from floating if their joint is deleted
        if (JPH_Constraint_GetType(c_to_destroy) == JPH_ConstraintType_TwoBodyConstraint) {
            auto *tbc    = (JPH_TwoBodyConstraint *)c_to_destroy;
            JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
            JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2(tbc);

            // ActivateBody is thread-safe in Jolt
            if (b1) {
                JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
            }
            if (b2) {
                JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
            }
        }

        // Remove from Physics System and release C++ memory
        JPH_PhysicsSystem_RemoveConstraint(self->system, c_to_destroy);
        JPH_Constraint_Destroy(c_to_destroy);
    }

    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                                                       PyObject *const *args,
                                                                       size_t nargsf,
                                                                       PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    uint64_t handle_raw;
    float target;

    void *targets[SetConstr_COUNT];
    targets[IDX_SCT_H] = (void *)&handle_raw;
    targets[IDX_SCT_T] = (void *)&target;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &SetConstrTargetParser, targets)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    SHADOW_LOCK(&self->shadow_lock);

    // Modification of constraints requires simulation idle
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    // Note: We reuse unpack_handle logic for constraints as the bit-layout is identical
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)handle_raw, &slot) ||
                 self->constraint_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    JPH_Constraint *c         = self->constraints[slot];
    JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

    // 3. JPH EXECUTION (Release GIL potentially, but these setters are very fast)
    // HINGE
    if (sub == JPH_ConstraintSubType_Hinge) {
        auto *hc             = (JPH_HingeConstraint *)c;
        JPH_MotorState state = JPH_HingeConstraint_GetMotorState(hc);
        if (state == JPH_MotorState_Velocity) {
            JPH_HingeConstraint_SetTargetAngularVelocity(hc, target);
        } else if (state == JPH_MotorState_Position) {
            JPH_HingeConstraint_SetTargetAngle(hc, target);
        }
    }
    // SLIDER
    else if (sub == JPH_ConstraintSubType_Slider) {
        auto *sc             = (JPH_SliderConstraint *)c;
        JPH_MotorState state = JPH_SliderConstraint_GetMotorState(sc);
        if (state == JPH_MotorState_Velocity) {
            JPH_SliderConstraint_SetTargetVelocity(sc, target);
        } else if (state == JPH_MotorState_Position) {
            JPH_SliderConstraint_SetTargetPosition(sc, target);
        }
    }

    // 4. WAKE UP BODIES
    // If the constraint targets change, the bodies must wake up to react.
    JPH_ConstraintType type = JPH_Constraint_GetType(c);
    if (type == JPH_ConstraintType_TwoBodyConstraint) {
        JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1((JPH_TwoBodyConstraint *)c);
        JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2((JPH_TwoBodyConstraint *)c);

        JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
        JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

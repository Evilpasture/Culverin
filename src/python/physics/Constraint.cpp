#include "physics/Constraint.hpp"
#include "bindings/PythonCommon.hpp"
#include "core/Buffer.hpp"
#include "core/Types.hpp"
#include "physics/ConstraintFactory.hpp"
#include "physics/PhysicsSync.hpp"

namespace CPH::Python {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] PyObject * WorldCreateConstraint(PhysicsWorldObject *self,
                                                                 PyObject *const *args,
                                                                 size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE
    int type           = 0;
    uint64_t h1_raw    = 0;
    uint64_t h2_raw    = 0;
    PyObject *o_params = nullptr;
    PyObject *o_motor  = nullptr;

    if (!self->parsers->CreateConstr.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, type, h1_raw,
                                           h2_raw, o_params, o_motor)) {
        return nullptr;
    }

    if (CPH_UNLIKELY(h1_raw == h2_raw)) {
        PyErr_SetString(PyExc_ValueError, "Cannot create a constraint between a body and itself");
        return nullptr;
    }

    // 2. PARAMS EXTRACTION
    CPH_ConstraintParams p; // member-initialized defaults
    int parse_ok = 1;

    switch (type) {
    case CONSTRAINT_FIXED:
        break;
    case CONSTRAINT_POINT:
        parse_ok = ParsePointParams(o_params, &p);
        break;
    case CONSTRAINT_HINGE:
        parse_ok = ParseHingeParams(o_params, &p);
        break;
    case CONSTRAINT_SLIDER:
        parse_ok = ParseSliderParams(o_params, &p);
        break;
    case CONSTRAINT_CONE:
        parse_ok = ParseConeParams(o_params, &p);
        break;
    case CONSTRAINT_DISTANCE:
        parse_ok = ParseDistanceParams(o_params, &p);
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown constraint type");
        return nullptr;
    }

    if (!parse_ok) {
        return nullptr; // PyErr set by helpers
    }
    if (o_motor) {
        ParseMotorConfig(o_motor, &p);
    }

    // 3. RESOURCE RESERVATION (Shadow Lock)
    (self->physics.shadow_lock).lock();
    BlockUntilNotStepping(self);

    uint32_t s1;
    uint32_t s2;
    if (!unpack_handle(&self->physics, static_cast<BodyHandle>(h1_raw), &s1) ||
        !unpack_handle(&self->physics, static_cast<BodyHandle>(h2_raw), &s2)) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return nullptr;
    }

    if (CPH_UNLIKELY(self->physics.free_constraint_count == 0)) {
        (self->physics.shadow_lock).unlock();
        return PyErr_Format(PyExc_MemoryError, "Max constraints reached");
    }

    // 4. QUEUE ASYNCHRONOUS COMMAND (capacity first so failure paths
    // cannot leak the reserved slot; params travel by value).
    if (CPH_UNLIKELY(!CPH_EnsureCommandCapacity(&self->physics))) {
        (self->physics.shadow_lock).unlock();
        return PyErr_NoMemory();
    }

    // Reserve constraint slot
    uint32_t c_slot         = self->physics.free_constraint_slots[--self->physics.free_constraint_count];
    ConstraintSlot &entry   = self->physics.constraints[c_slot];
    entry.type              = type;
    entry.state.store(SLOT_PENDING_CREATE, std::memory_order::relaxed);
    ConstraintHandle handle = make_handle(c_slot, entry.generation);
    CPH_EnqueueCommand(&self->physics, s1,
                       CPH::cmd::CreateConstraint{s2, c_slot, type, p});

    (self->physics.shadow_lock).unlock();
    return PyLong_FromUnsignedLongLong(handle);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] PyObject * WorldDestroyConstraint(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {
    // 1. FAST PARSE
    // TSan Fix: Use raw uint64 for the parser target to avoid atomic init overhead
    uint64_t h_raw;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!self->parsers->DestroyConstr.Parse(args, nargs, kwnames, h_raw)) {
        return nullptr;
    }

    JPH_Constraint *c_to_destroy = nullptr;

    // 2. RESOLUTION PHASE (Inside Shadow Lock)
    (self->physics.shadow_lock).lock();

    // RESOLUTION PHASE
    uint32_t slot = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t gen  = static_cast<uint32_t>(h_raw >> kHandleIndexBits);

    // Range check before indexing (the legacy order read the state first).
    if (slot >= self->physics.constraints.size()) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }
    ConstraintSlot &entry = self->physics.constraints[slot];
    const uint8_t state   = entry.state.load(std::memory_order::relaxed);

    // Allow destruction of ALIVE OR PENDING_CREATE
    if (entry.generation != gen || (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE)) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    c_to_destroy = entry.constraint;

    // Transition to EMPTY immediately.
    // If it was PENDING, the flush worker will see SLOT_EMPTY and skip creation.
    entry.constraint = nullptr;
    entry.state.store(SLOT_EMPTY, std::memory_order::release);
    entry.generation++;
    self->physics.free_constraint_slots[self->physics.free_constraint_count++] = slot;

    (self->physics.shadow_lock).unlock();

    // 3. JOLT DESTRUCTION PHASE (Outside Shadow Lock)
    if (c_to_destroy) {
        // Automatic Body Wake-up: Prevents bodies from floating if their joint is deleted.
        // ActivateBody is thread-safe in Jolt.
        if (JPH_Constraint_GetType(c_to_destroy) == JPH_ConstraintType_TwoBodyConstraint) {
            auto tbc     = reinterpret_cast<JPH_TwoBodyConstraint *>(c_to_destroy);
            JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
            JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2(tbc);

            if (b1) {
                JPH_BodyInterface_ActivateBody(self->physics.body_interface, JPH_Body_GetID(b1));
            }
            if (b2) {
                JPH_BodyInterface_ActivateBody(self->physics.body_interface, JPH_Body_GetID(b2));
            }
        }

        JPH_PhysicsSystem_RemoveConstraint(self->physics.system, c_to_destroy);
        JPH_Constraint_Destroy(c_to_destroy);
    }

    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * WorldSetConstraintTarget(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    // 1. FAST PARSE (Unchanged)
    uint64_t h_raw;
    float target;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!self->parsers->SetConstrTarget.Parse(args, nargs, kwnames, h_raw, target)) {
        return nullptr;
    }

    // 2. CRITICAL SECTION
    (self->physics.shadow_lock).lock();
    BlockUntilNotStepping(self);

    uint32_t slot = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t gen  = static_cast<uint32_t>(h_raw >> kHandleIndexBits);

    if (CPH_UNLIKELY(slot >= self->physics.constraints.size() ||
                     self->physics.constraints[slot].generation != gen ||
                     self->physics.constraints[slot].state.load(std::memory_order::relaxed) !=
                         SLOT_ALIVE)) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    JPH_Constraint *c = self->physics.constraints[slot].constraint;
    JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

    // 3. JPH EXECUTION
    if (sub == JPH_ConstraintSubType_Hinge) {
        auto hc = reinterpret_cast<JPH_HingeConstraint *>(c);
        JPH_HingeConstraint_SetMotorState(hc, JPH_MotorState_Position);
        JPH_HingeConstraint_SetTargetAngle(hc, target);
    } else if (sub == JPH_ConstraintSubType_Slider) {
        auto sc              = reinterpret_cast<JPH_SliderConstraint *>(c);
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
        JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1(reinterpret_cast<JPH_TwoBodyConstraint *>(c));
        JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2(reinterpret_cast<JPH_TwoBodyConstraint *>(c));

        if (b1) {
            JPH_BodyInterface_ActivateBody(self->physics.body_interface, JPH_Body_GetID(b1));
        }
        if (b2) {
            JPH_BodyInterface_ActivateBody(self->physics.body_interface, JPH_Body_GetID(b2));
        }
    }

    (self->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * WorldGetConstraintType(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {
    uint64_t handle_raw;

    if (!self->parsers->HOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw)) {
        return nullptr;
    }

    (self->physics.shadow_lock).lock();
    BlockUntilNotStepping(self);

    uint32_t slot = static_cast<uint32_t>(handle_raw & kHandleIndexMask);
    uint32_t gen  = static_cast<uint32_t>(handle_raw >> kHandleIndexBits);

    // Validate slot range and generation
    if (CPH_UNLIKELY(slot >= self->physics.constraints.size() ||
                     self->physics.constraints[slot].generation != gen)) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return nullptr;
    }

    const uint8_t state =
        self->physics.constraints[slot].state.load(std::memory_order::acquire);

    // NEW LOGIC: Accept both ALIVE and PENDING_CREATE
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
        // Return from shadow array directly.
        // This is safe even if Jolt hasn't built the object yet.
        int type = self->physics.constraints[slot].type;
        (self->physics.shadow_lock).unlock();
        return PyLong_FromLong(static_cast<long>(type));
    }

    // If state is SLOT_EMPTY or PENDING_DESTROY
    (self->physics.shadow_lock).unlock();
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return nullptr;
}

/*
TODO: for constraints
def set_hinge_limits(self, handle: int, min_angle: float, max_angle: float) -> None: ...
def set_hinge_motor(self, handle: int, target_velocity: float, max_torque: float) -> None: ...
def set_slider_motor(self, handle: int, target_velocity: float, max_force: float) -> None: ...
def set_constraint_enabled(self, handle: int, enabled: bool) -> None: ...

*/

} // namespace CPH::Python

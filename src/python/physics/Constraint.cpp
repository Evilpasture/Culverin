#include "physics/Constraint.hpp"
#include "bindings/PythonCommon.hpp"
#include "core/Buffer.hpp"
#include "core/Types.hpp"
#include "physics/ConstraintFactory.hpp"
#include "physics/ConstraintOps.hpp"
#include "physics/PhysicsSync.hpp"
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>

namespace CPH::Python {
namespace {

/// Wakes a body whose constraint was just created, destroyed, or retargeted.
/// Prevents bodies from floating once their joint is gone. ActivateBody is
/// thread-safe in Jolt.
void ActivateBody(JPH::BodyInterface &bi, JPH::Body *body) {
    if (body) {
        bi.ActivateBody(body->GetID());
    }
}

} // namespace

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

    const ParseResult parsed =
        self->parsers->CreateConstr.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, type, h1_raw,
                                           h2_raw, o_params, o_motor);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (h1_raw == h2_raw) [[unlikely]] {
        PyErr_SetString(PyExc_ValueError, "Cannot create a constraint between a body and itself");
        return nullptr;
    }

    // 2. PARAMS EXTRACTION
    ConstraintParams p; // member-initialized defaults
    ParseResult parse_result{};

    switch (type) {
    case std::to_underlying(ConstraintType::Fixed):
        break;
    case std::to_underlying(ConstraintType::Point):
        parse_result = ParsePointParams(o_params, p);
        break;
    case std::to_underlying(ConstraintType::Hinge):
        parse_result = ParseHingeParams(o_params, p);
        break;
    case std::to_underlying(ConstraintType::Slider):
        parse_result = ParseSliderParams(o_params, p);
        break;
    case std::to_underlying(ConstraintType::Cone):
        parse_result = ParseConeParams(o_params, p);
        break;
    case std::to_underlying(ConstraintType::Distance):
        parse_result = ParseDistanceParams(o_params, p);
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown constraint type");
        return nullptr;
    }

    if (!parse_result) {
        RaiseParseError(parse_result.error());
        return nullptr;
    }
    if (o_motor) {
        ParseMotorConfig(o_motor, p);
    }

    // 3. RESOURCE RESERVATION (Shadow Lock)
    (self->physics.shadow_lock).lock();
    BlockUntilNotStepping(self);

    const std::optional<std::uint32_t> resolved1 =
        self->physics.Resolve(static_cast<BodyHandle>(h1_raw));
    const std::optional<std::uint32_t> resolved2 =
        self->physics.Resolve(static_cast<BodyHandle>(h2_raw));
    if (!resolved1 || !resolved2) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return nullptr;
    }
    const uint32_t s1 = *resolved1;
    const uint32_t s2 = *resolved2;

    if (self->physics.free_constraint_count == 0) [[unlikely]] {
        (self->physics.shadow_lock).unlock();
        return PyErr_Format(PyExc_MemoryError, "Max constraints reached");
    }

    // 4. QUEUE ASYNCHRONOUS COMMAND (capacity first so failure paths
    // cannot leak the reserved slot; params travel by value).
    if (!self->physics.EnsureCommandCapacity()) [[unlikely]] {
        (self->physics.shadow_lock).unlock();
        return PyErr_NoMemory();
    }

    // Reserve constraint slot
    uint32_t c_slot         = self->physics.free_constraint_slots[--self->physics.free_constraint_count];
    ConstraintSlot &entry   = self->physics.constraints[c_slot];
    entry.type              = type;
    entry.state.store(std::to_underlying(SlotState::PendingCreate), std::memory_order::relaxed);
    ConstraintHandle handle = ConstraintHandle::FromParts(c_slot, entry.generation);
    self->physics.EnqueueCommand(s1, CPH::cmd::CreateConstraint{s2, c_slot, type, p});

    (self->physics.shadow_lock).unlock();
    return PyLong_FromUnsignedLongLong(handle.Raw());
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
    const ParseResult parsed =
        self->parsers->DestroyConstr.Parse(args, nargs, kwnames, h_raw);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    JPH::Constraint *c_to_destroy = nullptr;

    // 2. RESOLUTION PHASE (Inside Shadow Lock)
    (self->physics.shadow_lock).lock();

    // RESOLUTION PHASE
    const ConstraintHandle handle(h_raw);
    uint32_t slot = handle.Index();
    uint32_t gen  = handle.Generation();

    // Range check before indexing (the legacy order read the state first).
    if (slot >= self->physics.constraints.size()) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }
    ConstraintSlot &entry = self->physics.constraints[slot];
    const uint8_t state   = entry.state.load(std::memory_order::relaxed);

    // Allow destruction of ALIVE OR PENDING_CREATE
    if (entry.generation != gen || (state != std::to_underlying(SlotState::Alive) && state != std::to_underlying(SlotState::PendingCreate))) {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }

    c_to_destroy = entry.constraint;

    // Transition to EMPTY immediately.
    // If it was PENDING, the flush worker will see SlotState::Empty and skip creation.
    entry.constraint = nullptr;
    entry.state.store(std::to_underlying(SlotState::Empty), std::memory_order::release);
    entry.generation++;
    self->physics.free_constraint_slots[self->physics.free_constraint_count++] = slot;

    (self->physics.shadow_lock).unlock();

    // 3. JOLT DESTRUCTION PHASE (Outside Shadow Lock)
    if (c_to_destroy) {
        // Automatic Body Wake-up: Prevents bodies from floating if their joint is deleted.
        // ActivateBody is thread-safe in Jolt.
        if (c_to_destroy->GetType() == JPH::EConstraintType::TwoBodyConstraint) {
            auto *tbc = static_cast<JPH::TwoBodyConstraint *>(c_to_destroy);
            JPH::BodyInterface &bi = self->physics.physics_system.GetBodyInterface();
            ActivateBody(bi, tbc->GetBody1());
            ActivateBody(bi, tbc->GetBody2());
        }

        self->physics.physics_system.RemoveConstraint(c_to_destroy);
        c_to_destroy->Release();
    }

    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * WorldSetConstraintTarget(PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    uint64_t h_raw;
    float target;

    const ParseResult parsed =
        self->parsers->SetConstrTarget.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw,
                                              target);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // The sub-type dispatch, the motor state machine and the body wake-ups are
    // all in CPH::SetConstraintTarget (src/physics/ConstraintOps.cpp); the GIL
    // is dropped for the call because it parks while a step is in flight.
    std::expected<void, CPH::Error> applied;
    Py_BEGIN_ALLOW_THREADS
    applied = CPH::SetConstraintTarget(self->physics, ConstraintHandle(h_raw), target);
    Py_END_ALLOW_THREADS

    if (!applied) {
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return nullptr;
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * WorldGetConstraintType(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {
    uint64_t handle_raw;

    const ParseResult parsed =
        self->parsers->HOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    (self->physics.shadow_lock).lock();
    BlockUntilNotStepping(self);

    const ConstraintHandle handle(handle_raw);
    uint32_t slot = handle.Index();
    uint32_t gen  = handle.Generation();

    // Validate slot range and generation
    if (slot >= self->physics.constraints.size() ||
                     self->physics.constraints[slot].generation != gen) [[unlikely]] {
        (self->physics.shadow_lock).unlock();
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return nullptr;
    }

    const uint8_t state =
        self->physics.constraints[slot].state.load(std::memory_order::acquire);

    // NEW LOGIC: Accept both ALIVE and PENDING_CREATE
    if (state == std::to_underlying(SlotState::Alive) || state == std::to_underlying(SlotState::PendingCreate)) {
        // Return from shadow array directly.
        // This is safe even if Jolt hasn't built the object yet.
        int type = self->physics.constraints[slot].type;
        (self->physics.shadow_lock).unlock();
        return PyLong_FromLong(static_cast<long>(type));
    }

    // If state is SlotState::Empty or PENDING_DESTROY
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

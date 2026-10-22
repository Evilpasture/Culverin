#include "physics/ConstraintOps.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>

namespace CPH {

namespace {

void ActivateBody(JPH::BodyInterface &body_interface, JPH::Body *body) {
    if (body != nullptr) {
        body_interface.ActivateBody(body->GetID());
    }
}

// Parks the caller while a step is in flight. Equivalent to the shadow-sync
// block the query paths use: shadow_lock is dropped while parked and
// re-acquired before returning. The binding wraps the whole call in
// Py_BEGIN_ALLOW_THREADS so the GIL is not held across this wait.
void WaitUntilNotStepping(std::unique_lock<std::mutex> &shadow, PhysicsWorld &world) {
    if (!world.isStepping.load(std::memory_order::relaxed)) {
        return;
    }
    world.waitingThreads.fetch_add(1, std::memory_order::relaxed);
    shadow.unlock();
    {
        std::unique_lock sync(world.stepSync.mutex);
        world.stepSync.cond.wait(
            sync, [&world] { return !world.isStepping.load(std::memory_order::relaxed); });
    }
    shadow.lock();
    world.waitingThreads.fetch_sub(1, std::memory_order::relaxed);
}

} // namespace

std::expected<void, Error> SetConstraintTarget(PhysicsWorld   &world,
                                                ConstraintHandle handle,
                                                float            target) {
    std::unique_lock shadow(world.shadowLock);
    WaitUntilNotStepping(shadow, world);

    const std::uint32_t slot = handle.Index();
    if (slot >= world.constraints.size() ||
        world.constraints[slot].generation != handle.Generation() ||
        world.constraints[slot].state.load(std::memory_order::relaxed) !=
            std::to_underlying(SlotState::Alive)) [[unlikely]] {
        return std::unexpected(ConstraintError::InvalidHandle);
    }

    JPH::Constraint *constraint = world.constraints[slot].constraint;
    const JPH::EConstraintSubType sub_type = constraint->GetSubType();

    if (sub_type == JPH::EConstraintSubType::Hinge) {
        auto *hinge = static_cast<JPH::HingeConstraint *>(constraint);
        hinge->SetMotorState(JPH::EMotorState::Position);
        hinge->SetTargetAngle(target);
    } else if (sub_type == JPH::EConstraintSubType::Slider) {
        auto *slider            = static_cast<JPH::SliderConstraint *>(constraint);
        JPH::EMotorState state  = slider->GetMotorState();

        // A slider that has never been driven starts in Position mode.
        if (state == JPH::EMotorState::Off) {
            state = JPH::EMotorState::Position;
            slider->SetMotorState(state);
        }

        // Follow the mode the constraint is already in: a velocity-driven
        // slider reads the target as a velocity, anything else as a position.
        if (state == JPH::EMotorState::Velocity) {
            slider->SetTargetVelocity(target);
        } else {
            slider->SetTargetPosition(target);
        }
    }

    // A constraint change is invisible until the bodies are awake.
    if (constraint->GetType() == JPH::EConstraintType::TwoBodyConstraint) {
        auto *two_body             = static_cast<JPH::TwoBodyConstraint *>(constraint);
        JPH::BodyInterface &bodies = world.physicsSystem.GetBodyInterface();
        ActivateBody(bodies, two_body->GetBody1());
        ActivateBody(bodies, two_body->GetBody2());
    }
    return {};
}

} // namespace CPH

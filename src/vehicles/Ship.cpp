#include "vehicles/Ship.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "jolt/Jolt.hpp"

#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/PhysicsStepListener.h>
#include <cmath>
#include <string_view>

namespace CPH {

// The flight model runs inside the step, while every body mutex is held, so it
// reaches the bodies through the no-lock interface and touches nothing that
// takes one.
class Ship::StepListener final : public JPH::PhysicsStepListener {
  public:
    explicit StepListener(Ship &owner) : owner_(owner) {}

    void OnStep([[maybe_unused]] const JPH::PhysicsStepListenerContext &context) override {
        Ship &ship = owner_;
        if (ship.sled_.IsInvalid()) {
            return;
        }

        const JPH::BodyID &sled = ship.sled_;
        JPH::BodyInterface &bodies =
            ship.world_.physics_system.GetBodyInterfaceNoLock();

        // 1. Current state.
        const JPH::Vec3 ang_vel = bodies.GetAngularVelocity(sled);
        const JPH::Quat rot     = bodies.GetRotation(sled);
        const JPH::Vec3 lin_vel = bodies.GetLinearVelocity(sled);

        const float fwd_input   = ship.input_forward_.load(std::memory_order::relaxed);
        const float steer_input = ship.input_right_.load(std::memory_order::relaxed);

        // 2. PD stabiliser and banking.
        const JPH::Vec3 current_up = rot * JPH::Vec3(0, 1.0f, 0);

        const float target_roll_z = -steer_input * ship.config_.banking_strength;

        // Torque X corrects pitch, torque Z corrects roll (plus the banking
        // target).
        const float tx =
            (-current_up.GetZ() * ship.config_.kp) - (ang_vel.GetX() * ship.config_.kd);
        const float tz = ((current_up.GetX() - target_roll_z) * ship.config_.kp) -
                         (ang_vel.GetZ() * ship.config_.kd);

        // 3. Yaw torque.
        const float current_yaw_vel = ang_vel.GetY();
        const float target_yaw_vel  = steer_input * ship.config_.steer_speed;
        const float ty = (target_yaw_vel - current_yaw_vel) * (ship.config_.kp * 0.5f);

        bodies.AddTorque(sled, JPH::Vec3(tx, ty, tz));

        // 4. Thrust.
        if (std::fabs(fwd_input) > 0.01f) {
            const JPH::Vec3 current_fwd = rot * JPH::Vec3(0, 0, 1.0f);
            const JPH::Vec3 force(current_fwd.GetX() * fwd_input * ship.config_.throttle_force,
                                  0.0f,
                                  current_fwd.GetZ() * fwd_input * ship.config_.throttle_force);
            bodies.AddForce(sled, force);
        }

        // 5. Lateral friction (anti-drift / keel effect).
        const JPH::Vec3 current_right = rot * JPH::Vec3(1.0f, 0, 0);

        // The dots and the drag division stay explicit scalar arithmetic
        // rather than Vec3::Dot / operator/: this runs every step, Jolt's Dot
        // reorders the additions and its scalar division multiplies by a
        // reciprocal. Either would drift the recorded-scene replays.
        const float lateral_speed = (lin_vel.GetX() * current_right.GetX()) +
                                    (lin_vel.GetY() * current_right.GetY()) +
                                    (lin_vel.GetZ() * current_right.GetZ());
        const float lateral_force_mag = -lateral_speed * ship.config_.lateral_grip;

        if (std::fabs(lateral_force_mag) > 0.01f) {
            const JPH::Vec3 side_force(current_right.GetX() * lateral_force_mag,
                                       current_right.GetY() * lateral_force_mag,
                                       current_right.GetZ() * lateral_force_mag);
            bodies.AddForce(sled, side_force);
        }

        // 6. Quadratic drag (speed limiter).
        const float speed_sq = (lin_vel.GetX() * lin_vel.GetX()) +
                               (lin_vel.GetY() * lin_vel.GetY()) +
                               (lin_vel.GetZ() * lin_vel.GetZ());
        if (speed_sq > 0.01f) {
            const float speed    = std::sqrt(speed_sq);
            const float drag_mag = speed_sq * ship.config_.linear_drag;
            const JPH::Vec3 drag_force(-(lin_vel.GetX() / speed) * drag_mag,
                                       -(lin_vel.GetY() / speed) * drag_mag,
                                       -(lin_vel.GetZ() / speed) * drag_mag);
            bodies.AddForce(sled, drag_force);
        }
    }

  private:
    Ship &owner_;
};

Ship::Ship(PhysicsWorld &world, JPH::BodyID sled, const ShipConfig &config)
    : world_(world), config_(config), sled_(sled) {}

std::unique_ptr<Ship> Ship::Create(PhysicsWorld &world, const ShipConfig &config) {
    // Resolve the sled while holding the shadow lock: the slot may be recycled
    // by another thread the moment it is released.
    std::uint32_t dense = 0;
    {
        const std::scoped_lock shadow(world.shadow_lock);
        const std::optional<std::uint32_t> slot = world.Resolve(config.sled_handle);
        if (!slot) {
            return nullptr;
        }
        dense = world.bodies.slot_to_dense[*slot];
    }

    std::unique_ptr<Ship> ship(new Ship(world, world.bodies.body_ids[dense], config));

    // Registration is a Jolt-system mutation, so it takes the trampoline lock
    // like every other such call. Acquisition order is
    // jph_trampoline_lock -> shadow_lock -> step_sync.
    {
        const std::scoped_lock trampoline(world.jph_trampoline_lock);
        ship->listener_ = std::make_unique<StepListener>(*ship);
        world.physics_system.AddStepListener(ship->listener_.get());
    }
    return ship;
}

Ship::~Ship() {
    if (listener_ != nullptr) {
        const std::scoped_lock trampoline(world_.jph_trampoline_lock);
        world_.physics_system.RemoveStepListener(listener_.get());
        listener_.reset();
    }
}

void Ship::SetInput(float forward, float right) noexcept {
    input_forward_.store(forward, std::memory_order::relaxed);
    input_right_.store(right, std::memory_order::relaxed);
}

} // namespace CPH

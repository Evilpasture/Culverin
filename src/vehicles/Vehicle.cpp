#include "vehicles/Vehicle.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Vehicle/TrackedVehicleController.h>
#include <Jolt/Physics/Vehicle/VehicleDifferential.h>
#include <Jolt/Physics/Vehicle/VehicleTrack.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <utility>

namespace CPH {

namespace {

// TODO(smell): PythonCommon.hpp has an identical DebugLog, but that header is
// binding glue and nothing under src/ may include it. A shared native logger
// belongs in src/core/; until then this is a local copy with the same format.
void DebugLog(const char *format, ...) noexcept {
#if defined(CULVERIN_DEBUG)
    std::fputs("[Culverin] ", stderr);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
#else
    (void)format;
#endif
}

// --- Wheeled wheel construction ---------------------------------------------

constexpr float kWheelInertiaDefault = 0.5f;
constexpr float kWheelSuspensionMin  = 0.05f;
constexpr float kWheelSteerThreshold = 0.1f;
constexpr float kWheelMaxSteerAngle  = 0.5f;

// TODO(smell): "does this wheel steer" is inferred from the sign of the wheel's
// local Z position, so a vehicle laid out along -Z gets no steering wheels at
// all. Preserved as-is; the wheel config should say so explicitly.
JPH::Ref<JPH::WheelSettingsWV> MakeWheeledWheel(const WheelConfig &config,
                                                const JPH::LinearCurve &friction_curve) {
    JPH::Ref<JPH::WheelSettingsWV> wheel = new JPH::WheelSettingsWV;

    wheel->mInertia             = kWheelInertiaDefault;
    wheel->mSuspensionMinLength = kWheelSuspensionMin;
    wheel->mSuspensionMaxLength = config.suspension;
    wheel->mSuspensionSpring    = JPH::SpringSettings(JPH::ESpringMode::FrequencyAndDamping,
                                                      config.springFreq, config.springDamp);

    wheel->mSteeringAxis        = JPH::Vec3(0, 1.0f, 0);
    wheel->mWheelUp             = JPH::Vec3(0, 1.0f, 0);
    wheel->mWheelForward        = JPH::Vec3(0, 0, 1.0f);
    wheel->mSuspensionDirection = JPH::Vec3(0, -1.0f, 0);
    wheel->mMaxBrakeTorque      = config.brakeTorque;

    if (config.z > kWheelSteerThreshold) {
        wheel->mMaxSteerAngle      = kWheelMaxSteerAngle;
        wheel->mMaxHandBrakeTorque = 0.0f;
    } else {
        wheel->mMaxSteerAngle      = 0.0f;
        wheel->mMaxHandBrakeTorque = config.handbrakeTorque;
    }

    wheel->mPosition = JPH::Vec3(config.x, config.y, config.z);
    wheel->mRadius   = config.radius;
    wheel->mWidth    = config.width;

    // LinearCurve is a value type, so each wheel gets its own copy.
    wheel->mLongitudinalFriction = friction_curve;
    wheel->mLateralFriction      = friction_curve;

    // TODO(smell): this rewrites mMaxSteerAngle with the value the branch above
    // already chose and does not re-touch mMaxHandBrakeTorque. Harmless today,
    // kept so the port stays reviewable.
    wheel->mMaxSteerAngle = (config.z > kWheelSteerThreshold) ? kWheelMaxSteerAngle : 0.0f;

    return wheel;
}

JPH::Ref<JPH::WheelSettingsTV> MakeTrackWheel(const TrackWheelConfig &config) {
    constexpr float kSuspensionMinLength = 0.05f;

    JPH::Ref<JPH::WheelSettingsTV> wheel = new JPH::WheelSettingsTV;
    wheel->mPosition             = JPH::Vec3(config.x, config.y, config.z);
    wheel->mRadius               = config.radius;
    wheel->mWidth                = config.width;
    wheel->mSuspensionMinLength  = kSuspensionMinLength;
    wheel->mSuspensionMaxLength  = config.suspension;
    wheel->mSuspensionSpring     = JPH::SpringSettings(JPH::ESpringMode::FrequencyAndDamping,
                                                       config.springFreq, config.springDamp);
    wheel->mLongitudinalFriction = config.friction;
    wheel->mLateralFriction      = config.friction;
    return wheel;
}

void SetupDifferentials(JPH::WheeledVehicleControllerSettings &controller, DriveMode drive,
                        std::uint32_t num_wheels) {
    switch (drive) {
    case DriveMode::Fwd:
        controller.mDifferentials.push_back(JPH::VehicleDifferentialSettings(0, 1));
        return;
    case DriveMode::Awd:
        if (num_wheels >= 4) {
            controller.mDifferentials.push_back(JPH::VehicleDifferentialSettings(0, 1));
            controller.mDifferentials.push_back(JPH::VehicleDifferentialSettings(2, 3));
            return;
        }
        break;
    case DriveMode::Rwd:
        break;
    }
    // RWD, and AWD on a vehicle too short for two axles.
    const int first  = (num_wheels >= 4) ? 2 : 0;
    const int second = (num_wheels >= 4) ? 3 : 1;
    controller.mDifferentials.push_back(JPH::VehicleDifferentialSettings(first, second));
}

constexpr float kTransmissionGearRatios[] = {2.66f, 1.78f, 1.30f, 1.00f, 0.74f, 0.50f};

} // namespace

// --- Chassis -----------------------------------------------------------------

std::expected<JPH::BodyID, Error> Vehicle::ResolveChassis(PhysicsWorld &world,
                                                          BodyHandle chassis) {
    const std::optional<std::uint32_t> resolved = world.Resolve(chassis);
    if (!resolved) {
        return std::unexpected(VehicleError::InvalidChassis);
    }
    const std::uint32_t slot = *resolved;
    const std::uint8_t  state = world.bodies.slotStates[slot].load(std::memory_order::acquire);
    if (state != std::to_underlying(SlotState::Alive)) {
        return std::unexpected(VehicleError::InvalidChassis);
    }
    return world.bodies.bodyIds[world.bodies.slotToDense[slot]];
}

std::expected<JPH::Ref<JPH::VehicleConstraint>, Error>
Vehicle::CommitConstraint(PhysicsWorld &world, JPH::BodyID chassis,
                          JPH::VehicleConstraintSettings &settings, JPH::ObjectLayer tester_layer,
                          float tester_scale) {
    JPH::Ref<JPH::VehicleConstraint> constraint;

    const std::scoped_lock trampoline(world.jphTrampolineLock);
    const JPH::BodyLockInterface &lock_interface = world.physicsSystem.GetBodyLockInterface();
    const JPH::BodyLockWrite      lock(lock_interface, chassis);
    if (!lock.Succeeded()) [[unlikely]] {
        return std::unexpected(VehicleError::CreationFailed);
    }

    constraint = new JPH::VehicleConstraint(lock.GetBody(), settings);

    // The constraint keeps its own RefConst to the tester, so the local
    // creation reference goes away with this scope.
    JPH::Ref<JPH::VehicleCollisionTester> tester =
        new JPH::VehicleCollisionTesterRay(tester_layer, JPH::Vec3(0, 1.0f, 0), tester_scale);

    constraint->SetVehicleCollisionTester(tester);
    world.physicsSystem.AddConstraint(constraint);
    // VehicleConstraint derives from PhysicsStepListener.
    world.physicsSystem.AddStepListener(constraint);

    return constraint;
}

// --- Construction ------------------------------------------------------------

Vehicle::Vehicle(PhysicsWorld &world, JPH::VehicleConstraint *constraint,
                 std::uint32_t num_wheels)
    : world_(&world), constraint_(constraint), numWheels_(num_wheels) {}

std::expected<std::unique_ptr<Vehicle>, Error>
Vehicle::CreateWheeled(PhysicsWorld &world, BodyHandle chassis,
                       const WheeledVehicleConfig &config) {
    const std::uint32_t num_wheels = static_cast<std::uint32_t>(config.wheels.size());
    if (num_wheels < 2) {
        return std::unexpected(VehicleError::TooFewWheels);
    }

    // The slot tables are only safe to read under the shadow lock, and the
    // lock has to be gone before the trampoline lock is taken.
    std::expected<JPH::BodyID, Error> chassis_id = std::unexpected(VehicleError::InvalidChassis);
    {
        const std::scoped_lock shadow(world.shadowLock);
        chassis_id = ResolveChassis(world, chassis);
    }
    if (!chassis_id) {
        return std::unexpected(chassis_id.error());
    }

    // Friction rises to 2.0 at 0.1 slip and settles at 1.2.
    JPH::LinearCurve friction_curve;
    friction_curve.AddPoint(0.0f, 1.0f);
    friction_curve.AddPoint(0.1f, 2.0f);
    friction_curve.AddPoint(1.0f, 1.2f);

    // Flat torque curve: full torque at every RPM.
    JPH::LinearCurve torque_curve;
    torque_curve.AddPoint(0.0f, 1.0f);
    torque_curve.AddPoint(1.0f, 1.0f);

    JPH::Ref<JPH::WheeledVehicleControllerSettings> controller =
        new JPH::WheeledVehicleControllerSettings;

    JPH::VehicleConstraintSettings settings;
    settings.mWheels.reserve(num_wheels);
    for (const WheelConfig &wheel : config.wheels) {
        settings.mWheels.push_back(MakeWheeledWheel(wheel, friction_curve).GetPtr());
    }

    // 1. Differentials first, so the transmission can find them for the ratio.
    SetupDifferentials(*controller, config.drive, num_wheels);

    // 2. Engine.
    controller->mEngine.mMaxTorque         = config.engine.maxTorque;
    controller->mEngine.mMaxRPM            = config.engine.maxRpm;
    controller->mEngine.mMinRPM            = config.engine.minRpm;
    controller->mEngine.mInertia           = config.engine.inertia;
    controller->mEngine.mNormalizedTorque  = torque_curve;

    // 3. Transmission.
    //
    // TODO(bug): the caller's transmission settings are assembled here and then
    // never copied into controller->mTransmission, exactly as before the port.
    // The controller therefore runs on Jolt's defaults (Auto mode, Jolt's own
    // gear ratios), so `transmission=` has never affected a wheeled vehicle.
    // Preserved deliberately so replays stay identical; wiring it up is
    // `controller->mTransmission = transmission;` and is a behaviour change.
    JPH::VehicleTransmissionSettings transmission;
    transmission.mMode           = static_cast<JPH::ETransmissionMode>(config.transmission.mode);
    transmission.mClutchStrength = config.transmission.clutchStrength;
    if (config.transmission.gearRatios.empty()) {
        transmission.mGearRatios.assign(std::begin(kTransmissionGearRatios),
                                        std::end(kTransmissionGearRatios));
    } else {
        transmission.mGearRatios.assign(config.transmission.gearRatios.begin(),
                                        config.transmission.gearRatios.end());
    }
    (void)transmission;

    // The differential ratio is the one transmission setting that does reach
    // the controller, because it lives on the differentials.
    for (JPH::VehicleDifferentialSettings &differential : controller->mDifferentials) {
        differential.mDifferentialRatio = config.transmission.differentialRatio;
    }

    settings.mController = controller;

    constexpr auto kLayerDynamic = static_cast<JPH::ObjectLayer>(2);
    // TODO(smell): the value is passed as the ray tester's maximum slope angle
    // in radians, i.e. ~114.6 degrees -- past vertical, so the tester accepts
    // essentially any surface normal. Preserved as-is.
    constexpr float kCollisionTesterScale = 2.0f;

    const std::expected<JPH::Ref<JPH::VehicleConstraint>, Error> constraint =
        CommitConstraint(world, *chassis_id, settings, kLayerDynamic, kCollisionTesterScale);
    if (!constraint) {
        return std::unexpected(constraint.error());
    }

    auto vehicle = std::unique_ptr<Vehicle>(new Vehicle(world, constraint->GetPtr(), num_wheels));
    vehicle->wheeled_ = true;
    return vehicle;
}

std::expected<std::unique_ptr<Vehicle>, Error>
Vehicle::CreateTracked(PhysicsWorld &world, BodyHandle chassis,
                       const TrackedVehicleConfig &config) {
    const std::uint32_t num_wheels = static_cast<std::uint32_t>(config.wheels.size());

    std::expected<JPH::BodyID, Error> chassis_id = std::unexpected(VehicleError::InvalidChassis);
    {
        const std::scoped_lock shadow(world.shadowLock);
        chassis_id = ResolveChassis(world, chassis);
    }
    if (!chassis_id) {
        return std::unexpected(chassis_id.error());
    }

    JPH::Ref<JPH::TrackedVehicleControllerSettings> controller =
        new JPH::TrackedVehicleControllerSettings;

    controller->mEngine.mMaxTorque = config.engine.maxTorque;
    controller->mEngine.mMaxRPM    = config.engine.maxRpm;
    controller->mEngine.mMinRPM    = config.engine.minRpm;
    controller->mEngine.mInertia   = config.engine.inertia;

    // Unlike the wheeled path, a tracked vehicle's transmission really is a
    // member of the controller settings, so this reaches the controller.
    controller->mTransmission.mMode = JPH::ETransmissionMode::Auto;
    constexpr float kForwardGears[] = {2.0f, 1.4f, 1.0f, 0.7f};
    constexpr float kReverseGears[] = {-1.5f};
    controller->mTransmission.mGearRatios.assign(std::begin(kForwardGears),
                                                 std::end(kForwardGears));
    controller->mTransmission.mReverseGearRatios.assign(std::begin(kReverseGears),
                                                        std::end(kReverseGears));

    // VehicleTrackSettings::mWheels is an Array, so the index lists are copied
    // into the controller and the config can go away with the call.
    const std::size_t track_count = std::min<std::size_t>(config.tracks.size(), 2);
    for (std::size_t t = 0; t < track_count; ++t) {
        JPH::VehicleTrackSettings &track = controller->mTracks[t];
        track.mWheels.assign(config.tracks[t].wheels.begin(), config.tracks[t].wheels.end());
        track.mDrivenWheel = static_cast<JPH::uint>(config.tracks[t].drivenWheel);
    }

    JPH::VehicleConstraintSettings settings;
    settings.mWheels.reserve(num_wheels);
    for (const TrackWheelConfig &wheel : config.wheels) {
        settings.mWheels.push_back(MakeTrackWheel(wheel).GetPtr());
    }
    settings.mController = controller;

    constexpr auto kTrackedLayerDrivable = static_cast<JPH::ObjectLayer>(2);
    constexpr float kTrackedCollisionTesterScale = 2.0f;

    const std::expected<JPH::Ref<JPH::VehicleConstraint>, Error> constraint =
        CommitConstraint(world, *chassis_id, settings, kTrackedLayerDrivable,
                         kTrackedCollisionTesterScale);
    if (!constraint) {
        return std::unexpected(VehicleError::TrackedCreationFailed);
    }

    auto vehicle = std::unique_ptr<Vehicle>(new Vehicle(world, constraint->GetPtr(), num_wheels));
    vehicle->wheeled_ = false;
    return vehicle;
}

// --- Teardown ----------------------------------------------------------------

void Vehicle::Detach() {
    if (world_ == nullptr || constraint_ == nullptr) {
        return;
    }

    JPH::VehicleConstraint *constraint = nullptr;
    {
        // Park until no step is in flight, then take the constraint out of the
        // handle table's view so no further call can reach it.
        std::unique_lock shadow(world_->LockShadowNotStepping());
        world_->WaitUntilNotQuerying(shadow);

        constraint  = constraint_;
        constraint_ = nullptr;
    }

    // Removing from the physics system is a Jolt mutation, so it takes the
    // trampoline lock on its own: holding the shadow lock here would invert the
    // documented order (jph_trampoline_lock outermost), and parking on
    // step_sync while holding the trampoline would deadlock against a step.
    {
        const std::scoped_lock trampoline(world_->jphTrampolineLock);
        world_->physicsSystem.RemoveStepListener(constraint);

        // The physics system owns the constraint's only reference: AddConstraint
        // took it, so RemoveConstraint releases it and the constraint is already
        // destroyed when this returns. `constraint` is a borrowed pointer from
        // here on -- releasing it (as this function used to, one statement
        // below, with a comment claiming the destructor needed to run outside
        // the locks) decrements the reference count of a freed block. That stray
        // 4-byte write landed in the allocator's free list and took out an
        // unrelated later allocation, which is why the crash surfaced as a
        // SIGSEGV inside realloc on a DebugRenderer rather than here.
        world_->physicsSystem.RemoveConstraint(constraint);
    }
}

Vehicle::~Vehicle() { Detach(); }

// --- Drive logic -------------------------------------------------------------

void Vehicle::SetInput(float forward, float right, float brake, float handbrake) {
    constexpr float kThrottleThreshold         = 0.01f;
    constexpr float kSpeedDirectionThreshold   = 0.5f;
    constexpr float kSpeedMinThreshold         = 0.1f;
    constexpr float kRollingResistanceCoasting = 0.05f;

    if (world_ == nullptr || constraint_ == nullptr) {
        return;
    }

    // Unlike the tank, a wheeled vehicle does not wait for a step to finish:
    // input arriving mid-step is dropped.
    std::unique_lock shadow(world_->shadowLock);
    if (world_->isStepping.load(std::memory_order::relaxed) || constraint_ == nullptr) {
        return;
    }

    auto *controller = static_cast<JPH::WheeledVehicleController *>(constraint_->GetController());
    const JPH::BodyID    chassis     = constraint_->GetVehicleBody()->GetID();
    JPH::BodyInterface  &bodies      = world_->physicsSystem.GetBodyInterface();

    // A sleeping vehicle ignores input.
    bodies.ActivateBody(chassis);

    const JPH::Vec3 linear_velocity  = bodies.GetLinearVelocity(chassis);
    const JPH::Quat chassis_rotation = bodies.GetRotation(chassis);

    const JPH::Vec3 world_forward = chassis_rotation * JPH::Vec3(0, 0, 1.0f);
    const float     speed         = linear_velocity.Dot(world_forward);

    float input_throttle = std::fabs(forward);
    float input_brake    = brake;

    JPH::VehicleTransmission &transmission = controller->GetTransmission();
    const int                 gear         = transmission.GetCurrentGear();

    if (forward > kThrottleThreshold) {
        transmission.mMode = JPH::ETransmissionMode::Auto;
        if (gear <= 0 && speed > -kSpeedDirectionThreshold) {
            transmission.Set(1, 1.0f);
        }
        if (speed < -kSpeedMinThreshold) {
            input_brake = 1.0f; // still rolling backwards: brake first
        }
    } else if (forward < -kThrottleThreshold) {
        transmission.mMode = JPH::ETransmissionMode::Manual;
        if (gear != -1 && speed < kSpeedDirectionThreshold) {
            transmission.Set(-1, 1.0f);
        }
        if (speed > kSpeedMinThreshold) {
            input_brake = 1.0f; // still rolling forwards: brake first
        }
    } else {
        input_throttle     = 0.0f;
        transmission.mMode = JPH::ETransmissionMode::Manual;
        if (gear != 0) {
            transmission.Set(0, 0.0f);
        }
        if (std::fabs(speed) > kSpeedMinThreshold) {
            input_brake = std::fmax(input_brake, kRollingResistanceCoasting);
        }
    }

    controller->SetDriverInput(input_throttle, right, input_brake, handbrake);
}

void Vehicle::SetTankInput(float left, float right, float brake) {
    constexpr float kKickstartThreshold = 0.01f;

    if (world_ == nullptr || constraint_ == nullptr) {
        return;
    }

    // The tank waits for the step to finish instead of dropping the input.
    std::unique_lock shadow(world_->LockShadowNotStepping());
    if (constraint_ == nullptr) {
        return;
    }

    auto *controller = static_cast<JPH::TrackedVehicleController *>(constraint_->GetController());
    const JPH::BodyID chassis = constraint_->GetVehicleBody()->GetID();

    // Wake up the tank to process inputs.
    world_->physicsSystem.GetBodyInterface().ActivateBody(chassis);

    JPH::VehicleTransmission &transmission = controller->GetTransmission();
    const int                 gear         = transmission.GetCurrentGear();

    // Throttle for a tracked vehicle is the hardest-driven side.
    const float throttle = std::fmax(std::fabs(left), std::fabs(right));

    // Kickstart: auto-shift from neutral into first when power is applied.
    if (throttle > kKickstartThreshold) {
        if (gear == 0) {
            transmission.Set(1, 1.0f);
        }
    } else if (gear != 0) {
        // Force neutral when idle so the tank does not crawl.
        transmission.Set(0, 0.0f);
    }

    // TODO(smell): TrackedVehicleController::SetDriverInput asserts
    // inLeftRatio != 0 && inRightRatio != 0, but set_tank_input(0, 0, 0) is a
    // perfectly reasonable call and reaches this line. Release builds compile
    // the assert out, so it has never fired; preserved as-is rather than
    // silently clamped.
    controller->SetDriverInput(throttle, left, right, brake);
}

// --- Queries -----------------------------------------------------------------

std::optional<WheelTransform> Vehicle::GetWheelTransform(std::uint32_t index) const {
    if (world_ == nullptr || constraint_ == nullptr || index >= numWheels_) {
        return std::nullopt;
    }

    std::unique_lock shadow(world_->LockShadowNotStepping());

    // RMat44::GetQuaternion() rotates the 3x3 part only.
    const JPH::RMat44 transform = constraint_->GetWheelWorldTransform(
        index, JPH::Vec3(1.0f, 0.0f, 0.0f), JPH::Vec3(0.0f, 1.0f, 0.0f));
    const JPH::DVec3 position = transform.GetTranslation();
    const JPH::Quat  rotation = transform.GetQuaternion();

    return WheelTransform{.px = position.GetX(),
                          .py = position.GetY(),
                          .pz = position.GetZ(),
                          .rx = rotation.GetX(),
                          .ry = rotation.GetY(),
                          .rz = rotation.GetZ(),
                          .rw = rotation.GetW()};
}

std::optional<WheelTransform> Vehicle::GetWheelLocalTransform(std::uint32_t index) const {
    if (world_ == nullptr || constraint_ == nullptr || index >= numWheels_) {
        return std::nullopt;
    }

    std::unique_lock shadow(world_->LockShadowNotStepping());

    // The wheel keeps a RefConst to its settings, so the attachment point is
    // readable at runtime.
    const JPH::Wheel         *wheel    = constraint_->GetWheel(index);
    const JPH::WheelSettings *settings = wheel->GetSettings();
    const JPH::Vec3 local_position     = settings->mPosition;

    const JPH::Vec3 right((local_position.GetX() >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
    const JPH::Vec3 up(0.0f, 1.0f, 0.0f);

    const JPH::Mat44 transform = constraint_->GetWheelLocalTransform(index, right, up);
    const JPH::Vec3  position  = transform.GetTranslation();
    const JPH::Quat  rotation  = transform.GetQuaternion();

    return WheelTransform{.px = position.GetX(),
                          .py = position.GetY(),
                          .pz = position.GetZ(),
                          .rx = rotation.GetX(),
                          .ry = rotation.GetY(),
                          .rz = rotation.GetZ(),
                          .rw = rotation.GetW()};
}

void Vehicle::LogDebugState() const {
    if (world_ == nullptr || constraint_ == nullptr) {
        DebugLog("Warning: Vehicle has been destroyed.");
        return;
    }

    std::unique_lock shadow(world_->shadowLock);
    if (world_->isStepping.load(std::memory_order::relaxed)) {
        DebugLog("Warning: Cannot get debug state while physics is stepping.");
        return;
    }
    if (constraint_ == nullptr) {
        return;
    }

    // Only a wheeled vehicle has a WheeledVehicleController; a tracked one
    // gets no drivetrain dump, exactly as before the port.
    if (!wheeled_) {
        return;
    }

    auto *controller = static_cast<JPH::WheeledVehicleController *>(constraint_->GetController());
    if (controller == nullptr) {
        return;
    }

    const JPH::VehicleEngine       &engine         = controller->GetEngine();
    const JPH::VehicleTransmission &transmission   = controller->GetTransmission();

    const float forward_input = controller->GetForwardInput();
    const float brake_input   = controller->GetBrakeInput();
    const float rpm           = engine.GetCurrentRPM();
    const float torque        = engine.GetTorque(forward_input);
    const int   gear          = transmission.GetCurrentGear();
    const float clutch        = transmission.GetClutchFriction();

    DebugLog("=== VEHICLE DEBUG STATE ===");
    DebugLog("  Inputs: Fwd=%.2f | Brk=%.2f", forward_input, brake_input);
    DebugLog("  Engine: %.2f RPM | Torque: %.2f Nm", rpm, torque);
    DebugLog("  Trans : Gear %d | Clutch Friction: %.2f", gear, clutch);

    for (std::uint32_t i = 0; i < numWheels_; i++) {
        const JPH::Wheel         *wheel    = constraint_->GetWheel(i);
        const JPH::WheelSettings *settings = wheel->GetSettings();

        const bool  contact          = wheel->HasContact();
        const float suspension       = wheel->GetSuspensionLength();
        const float angular_velocity = wheel->GetAngularVelocity();
        const float tire_speed       = angular_velocity * settings->mRadius;

        DebugLog("  Wheel %u: %s", i, contact ? "GROUND" : "AIR   ");
        DebugLog("    Susp: %.3fm | AngVel: %.2f rad/s | SurfSpd: %.2f m/s", suspension,
                 angular_velocity, tire_speed);
        DebugLog("    Trac: Long=%.2f | Lat=%.2f", wheel->GetLongitudinalLambda(),
                 wheel->GetLateralLambda());
    }
    DebugLog("===========================");
}

} // namespace CPH

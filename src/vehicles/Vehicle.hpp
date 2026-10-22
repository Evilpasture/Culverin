#pragma once

// Wheeled and tracked vehicles: a JPH::VehicleConstraint plus the arcade drive
// logic that decides what to feed its controller.
//
// This is engine code, not binding glue. It used to live in the Python
// binding, which meant the transmission state machine -- gear engagement,
// rolling resistance, the kickstart threshold -- could only run from Python.
//
// Configuration is plain C++ structs. Turning a Python wheel dictionary into
// one of these is the binding's job; nothing here knows what a dict is.

#include "core/Error.hpp"
#include "core/Handle.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace CPH {

struct PhysicsWorld;

// --- Wheeled configuration ---------------------------------------------------

struct WheelConfig {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    float radius           = 0.4f;
    float width            = 0.2f;
    float brakeTorque     = 1500.0f;
    float handbrakeTorque = 4000.0f;
    float suspension       = 0.3f;
    float springFreq      = 1.5f;
    float springDamp      = 0.5f;
};

enum class DriveMode : std::uint8_t { Fwd = 1, Rwd = 2, Awd = 3 };

struct EngineConfig {
    float maxTorque = 500.0f;
    float maxRpm    = 7000.0f;
    float minRpm    = 1000.0f;
    float inertia    = 0.5f;
};

struct TransmissionConfig {
    int                mode            = 1; // JPH::ETransmissionMode
    float              clutchStrength = 2000.0f;
    float              differentialRatio = 3.42f;
    std::vector<float> gearRatios;        // empty means "use the defaults"
};

struct WheeledVehicleConfig {
    std::vector<WheelConfig> wheels;
    DriveMode                drive = DriveMode::Rwd;
    EngineConfig             engine;
    TransmissionConfig       transmission;
};

// --- Tracked configuration ---------------------------------------------------

struct TrackWheelConfig {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    float radius     = 0.4f;
    float width      = 0.2f;
    float suspension = 0.5f;
    float friction   = 1.0f;
    float springFreq = 2.0f;
    float springDamp = 0.5f;
};

struct TrackConfig {
    std::vector<int> wheels;      // indices into TrackedVehicleConfig::wheels
    int              drivenWheel = 0;
};

struct TrackedVehicleConfig {
    std::vector<TrackWheelConfig> wheels;
    std::vector<TrackConfig>      tracks; // at most two
    EngineConfig                  engine;
};

// --- Results -----------------------------------------------------------------

struct WheelTransform {
    double px = 0.0, py = 0.0, pz = 0.0;
    float  rx = 0.0f, ry = 0.0f, rz = 0.0f, rw = 1.0f;
};

enum class VehicleError : std::uint8_t {
    InvalidChassis = 1,
    TooFewWheels,
    CreationFailed,
    TrackedCreationFailed,
};

// Observable strings: the binding raises these verbatim and they are what
// callers have always seen.
inline std::string_view ErrorMessage(VehicleError error) noexcept {
    switch (error) {
    case VehicleError::InvalidChassis:
        return "Invalid chassis handle (Body is dead or pending)";
    case VehicleError::TooFewWheels:
        return "Wheels must be a list of at least 2 dictionaries";
    case VehicleError::CreationFailed:
        return "Jolt vehicle creation failed";
    case VehicleError::TrackedCreationFailed:
        return "Tracked vehicle creation failed";
    }
    return "vehicle error";
}

class Vehicle {
  public:
    // Builds the constraint, attaches the collision tester and registers the
    // constraint and its step listener with the world.
    [[nodiscard]] static std::expected<std::unique_ptr<Vehicle>, Error>
    CreateWheeled(PhysicsWorld &world, BodyHandle chassis, const WheeledVehicleConfig &config);

    [[nodiscard]] static std::expected<std::unique_ptr<Vehicle>, Error>
    CreateTracked(PhysicsWorld &world, BodyHandle chassis, const TrackedVehicleConfig &config);

    ~Vehicle();

    Vehicle(const Vehicle &)            = delete;
    Vehicle &operator=(const Vehicle &) = delete;
    Vehicle(Vehicle &&)                 = delete;
    Vehicle &operator=(Vehicle &&)      = delete;

    // Detaches from the world without waiting for the destructor; `destroy()`
    // is idempotent because Python can call it before teardown.
    void Detach();

    // Wheeled controls: forward/right/brake/handbrake, with the arcade
    // transmission state machine deciding gear engagement and rolling
    // resistance. A no-op while the world is stepping.
    void SetInput(float forward, float right, float brake, float handbrake);

    // Tracked controls: independent left/right track ratios plus a brake.
    void SetTankInput(float left, float right, float brake);

    [[nodiscard]] std::optional<WheelTransform> GetWheelTransform(std::uint32_t index) const;
    [[nodiscard]] std::optional<WheelTransform> GetWheelLocalTransform(std::uint32_t index) const;

    [[nodiscard]] std::uint32_t wheel_count() const noexcept { return numWheels_; }

    // Writes the engine/transmission/wheel state to the debug log. Compiled
    // out unless CULVERIN_DEBUG.
    void LogDebugState() const;

  private:
    Vehicle(PhysicsWorld &world, JPH::VehicleConstraint *constraint,
            std::uint32_t num_wheels);

    // Shared tail of both factories: builds the constraint under the Jolt
    // trampoline lock and registers it with the world.
    [[nodiscard]] static std::expected<JPH::Ref<JPH::VehicleConstraint>, Error>
    CommitConstraint(PhysicsWorld &world, JPH::BodyID chassis, JPH::VehicleConstraintSettings &settings,
                     JPH::ObjectLayer tester_layer, float tester_scale);

    [[nodiscard]] static std::expected<JPH::BodyID, Error>
    ResolveChassis(PhysicsWorld &world, BodyHandle chassis);

    PhysicsWorld *world_;
    // Borrowed: the physics system owns the constraint's only reference and
    // RemoveConstraint destroys it. Never Release() this pointer -- see
    // Vehicle::Detach().
    JPH::VehicleConstraint *constraint_ = nullptr;
    std::uint32_t           numWheels_ = 0;
    bool                    wheeled_    = true;
};

} // namespace CPH

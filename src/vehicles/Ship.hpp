#pragma once

// A hover-ship flight controller: a per-ship JPH::PhysicsStepListener running
// a PD stabiliser, banking, yaw steering, thrust, keel friction and quadratic
// drag. This used to be embedded in the Python binding, which meant the flight
// model could only be run from Python; it is engine code and lives here now.
//
// One instance owns exactly one listener, so there is no shared state between
// ships and no C trampoline recovering an owner from a void*.

#include "core/Error.hpp"
#include "core/Handle.hpp"
#include "jolt/Jolt.hpp"
#include <atomic>
#include <cstdint>
#include <memory>

namespace CPH {

struct PhysicsWorld;

// Everything that configures a ship's flight model. Defaults are the values
// the Python API documents.
struct ShipConfig {
    BodyHandle sledHandle;
    float      kp               = 0.0f;
    float      kd               = 0.0f;
    float      throttleForce   = 0.0f;
    float      steerSpeed      = 0.0f;
    float      bankingStrength = 0.15f;
    float      lateralGrip     = 500.0f;
    float      linearDrag      = 10.0f;
};

enum class ShipError : std::uint8_t {
    // The sled handle was stale when the ship was created.
    InvalidSled = 1,
};

inline std::string_view ErrorMessage(ShipError) noexcept {
    return "Invalid sled handle.";
}

class Ship {
  public:
    // Resolves the sled handle and registers the step listener. Returns
    // nullptr when the sled handle is stale.
    [[nodiscard]] static std::unique_ptr<Ship> Create(PhysicsWorld &world,
                                                      const ShipConfig &config);

    ~Ship();

    Ship(const Ship &)            = delete;
    Ship &operator=(const Ship &) = delete;
    Ship(Ship &&)                 = delete;
    Ship &operator=(Ship &&)      = delete;

    // Called from Python (or any control thread); the listener reads these
    // from the physics step, so they are atomics and never locked.
    void SetInput(float forward, float right) noexcept;

    [[nodiscard]] BodyHandle SledHandle() const noexcept { return config_.sledHandle; }

  private:
    class StepListener;

    Ship(PhysicsWorld &world, JPH::BodyID sled, const ShipConfig &config);

    PhysicsWorld            &world_;
    ShipConfig               config_;
    JPH::BodyID              sled_;
    std::unique_ptr<StepListener> listener_;

    std::atomic<float>       inputForward_{0.0f};
    std::atomic<float>       inputRight_{0.0f};
};

} // namespace CPH

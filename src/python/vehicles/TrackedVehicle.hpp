#pragma once
#include "core/Compiler.hpp"
#include "vehicles/Vehicle.hpp"

namespace CPH::Python {

[[maybe_unused]] static constexpr uint32_t kTrackedLayerMoving   = 0;
[[maybe_unused]] static constexpr uint32_t kTrackedLayerStatic   = 1;
[[maybe_unused]] static constexpr uint32_t kTrackedLayerDrivable = 2;

struct PhysicsWorldObject;


struct VehicleObject;


struct TrackedEngineConfig {
    float torque;
    float max_rpm;
    float min_rpm;
};

} // namespace CPH::Python

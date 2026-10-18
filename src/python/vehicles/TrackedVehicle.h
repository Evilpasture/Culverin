#pragma once
#include "core/CompilerSpecifics.h"
#include "vehicles/Vehicle.h"

[[maybe_unused]] static constexpr uint32_t kTrackedLayerMoving   = 0;
[[maybe_unused]] static constexpr uint32_t kTrackedLayerStatic   = 1;
[[maybe_unused]] static constexpr uint32_t kTrackedLayerDrivable = 2;

struct CPH_PhysicsWorldObject;
typedef struct CPH_PhysicsWorldObject CPH_PhysicsWorldObject;

struct CPH_VehicleObject;
typedef struct CPH_VehicleObject CPH_VehicleObject;

typedef struct {
    float torque;
    float max_rpm;
    float min_rpm;
} CPH_TrackedEngineConfig;
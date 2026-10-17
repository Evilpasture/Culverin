#pragma once
#include "core/CompilerSpecifics.h"
#include "vehicles/Vehicle.h"

[[maybe_unused]] static constexpr uint32_t TRACKED_LAYER_MOVING   = 0;
[[maybe_unused]] static constexpr uint32_t TRACKED_LAYER_STATIC   = 1;
[[maybe_unused]] static constexpr uint32_t TRACKED_LAYER_DRIVABLE = 2;

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

struct VehicleObject;
typedef struct VehicleObject VehicleObject;

typedef struct {
    float torque;
    float max_rpm;
    float min_rpm;
} TrackedEngineConfig;
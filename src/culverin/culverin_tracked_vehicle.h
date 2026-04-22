#pragma once
#include "culverin_compiler_specifics.h"
#include "culverin_vehicle.h"

CULV_MAYBE_UNUSED static constexpr uint32_t TRACKED_LAYER_MOVING   = 0;
CULV_MAYBE_UNUSED static constexpr uint32_t TRACKED_LAYER_STATIC   = 1;
CULV_MAYBE_UNUSED static constexpr uint32_t TRACKED_LAYER_DRIVABLE = 2;

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

struct VehicleObject;
typedef struct VehicleObject VehicleObject;

typedef struct {
    float torque;
    float max_rpm;
    float min_rpm;
} TrackedEngineConfig;
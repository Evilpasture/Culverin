#pragma once

#include "core/Types.h"
#include <stddef.h>

// Plain native data structures shared by parser adapters and the physics core.
enum {
    CULV_BAKED_POSITION_STRIDE = 4,
    CULV_BAKED_ROTATION_STRIDE = 4,
    CULV_BAKED_SHAPE_STRIDE    = 5,
};

typedef struct {
    size_t count;
    const JPH_Real *positions;
    const float *rotations;
    const float *shapes;
    const uint8_t *motion_types;
    const uint8_t *object_layers;
    const uint64_t *user_data;
} CPH_BakedSceneData;

typedef struct {
    float p[4];
} CPH_ShapeParams;

typedef struct {
    uint32_t *indices;
    uint32_t count;
    uint32_t driven_idx;
} CPH_TrackData;

typedef struct {
    float px, py, pz, ax, ay, az;
    float limit_min, limit_max, half_cone_angle;
    bool has_motor;
    int motor_type;
    float motor_target, max_torque, frequency, damping, max_force;
} CPH_ConstraintParams;

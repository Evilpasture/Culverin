#pragma once

#include "core/Types.h"

// Plain native data structures shared by parser adapters and the physics core.
typedef struct {
    float p[4];
} ShapeParams;

typedef struct {
    uint32_t *indices;
    uint32_t count;
    uint32_t driven_idx;
} TrackData;

typedef struct {
    float px, py, pz, ax, ay, az;
    float limit_min, limit_max, half_cone_angle;
    bool has_motor;
    int motor_type;
    float motor_target, max_torque, frequency, damping, max_force;
} ConstraintParams;

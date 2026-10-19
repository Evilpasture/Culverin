#pragma once

// Plain native data shared by the parser adapters and the physics core.

#include "core/Types.hpp"
#include <joltc.h>

#include <cstddef>
#include <cstdint>
#include <limits>

enum {
    CPH_BAKED_POSITION_STRIDE = 4,
    CPH_BAKED_ROTATION_STRIDE = 4,
    CPH_BAKED_SHAPE_STRIDE    = 5,
};

struct CPH_BakedSceneData {
    std::size_t count;
    const JPH_Real *positions;
    const float *rotations;
    const float *shapes;
    const std::uint8_t *motion_types;
    const std::uint8_t *object_layers;
    const std::uint64_t *user_data;
};

struct CPH_ShapeParams {
    float p[4];
};

struct CPH_TrackData {
    std::uint32_t *indices;
    std::uint32_t count;
    std::uint32_t driven_idx;
};

// Constraint parameters. Default member initializers carry the sane
// defaults the old CPH_ParamsInit() applied (20 Hz / critically damped
// motors, unlimited limits), so a value-initialized struct is ready to use.
struct CPH_ConstraintParams {
    float px = 0.0F, py = 0.0F, pz = 0.0F;
    float ax = 0.0F, ay = 1.0F, az = 0.0F;
    float limit_min       = -std::numeric_limits<float>::max();
    float limit_max       = std::numeric_limits<float>::max();
    float half_cone_angle = 0.0F;
    bool has_motor        = false;
    int motor_type        = 0;
    float motor_target    = 0.0F;
    float max_torque      = 1e6F;
    float frequency       = 20.0F;
    float damping         = 1.0F;
    float max_force       = 1e6F;
};

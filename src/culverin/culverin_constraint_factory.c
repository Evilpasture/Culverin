#include "culverin_constraint_factory.h"
#include "culverin_math.h"
#include <float.h>

// Named constants to avoid magic numbers
static constexpr float CCF_DEFAULT_FREQUENCY = 20.0f;
static constexpr float CCF_DEFAULT_DAMPING = 1.0f;
static constexpr float CCF_MAX_TORQUE = 1e6f;
static constexpr float CCF_MAX_FORCE = 1e6f;
static constexpr float CCF_EPSILON = 1e-6f;
static constexpr float CCF_TINY_EPSILON = 1e-9f;

// Constraints

// Initialize defaults to avoid garbage data
void params_init(ConstraintParams *p) {
    memset(p, 0, sizeof(ConstraintParams));
    p->ay         = 1.0f;
    p->limit_min  = -FLT_MAX;
    p->limit_max  = FLT_MAX;
    p->frequency  = CCF_DEFAULT_FREQUENCY; // Sane default for motors (20Hz)
    p->damping    = CCF_DEFAULT_DAMPING;  // Critical damping
    p->max_torque = CCF_MAX_TORQUE;       // 1,000,000 Nm - enough to move heavy crates
    p->max_force  = CCF_MAX_FORCE;        // For sliders
}

// --- Jolt Creator Helpers ---

JPH_Constraint *create_fixed(const ConstraintParams *Py_UNUSED(p), JPH_Body *b1, JPH_Body *b2) {
    JPH_FixedConstraintSettings s;
    JPH_FixedConstraintSettings_Init(&s);
    s.base.enabled    = true;
    s.autoDetectPoint = true;
    return (JPH_Constraint *)JPH_FixedConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_point(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2) {
    JPH_PointConstraintSettings s;
    JPH_PointConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space        = JPH_ConstraintSpace_WorldSpace;
    s.point1.x     = p->px;
    s.point1.y     = p->py;
    s.point1.z     = p->pz;
    s.point2       = s.point1;
    return (JPH_Constraint *)JPH_PointConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_hinge(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2) {
    JPH_HingeConstraintSettings s;
    JPH_HingeConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space        = JPH_ConstraintSpace_WorldSpace;

    s.point1.x = p->px;
    s.point1.y = p->py;
    s.point1.z = p->pz;
    s.point2   = s.point1;

    JPH_Vec3 axis = {p->ax, p->ay, p->az};
    if (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z < CCF_EPSILON) {
        axis.y = 1.0f;
    }
    JPH_Vec3_Normalize(&axis, &axis);

    // Compute normal from pivot-to-b2 projected onto the hinge plane,
    // so that angle=0 means "b2 at its initial position" and SetTargetAngle
    // is meaningful without the caller needing to know the reference frame.
    JPH_Vec3 norm;
    JPH_RVec3 b2pos;
    JPH_Body_GetPosition(b2, &b2pos);
    JPH_Vec3 pivot_to_b2 = {
        (float)b2pos.x - p->px,
        (float)b2pos.y - p->py,
        (float)b2pos.z - p->pz,
    };
    float dot = pivot_to_b2.x * axis.x + pivot_to_b2.y * axis.y + pivot_to_b2.z * axis.z;
    pivot_to_b2.x -= dot * axis.x;
    pivot_to_b2.y -= dot * axis.y;
    pivot_to_b2.z -= dot * axis.z;
    float len = sqrtf(pivot_to_b2.x * pivot_to_b2.x + pivot_to_b2.y * pivot_to_b2.y +
                      pivot_to_b2.z * pivot_to_b2.z);
    if (len > CCF_EPSILON) {
        float inv = 1.0f / len;
        norm.x    = pivot_to_b2.x * inv;
        norm.y    = pivot_to_b2.y * inv;
        norm.z    = pivot_to_b2.z * inv;
    } else {
        // b2 is on the hinge axis itself — fall back to arbitrary perpendicular
        vec3_get_perpendicular(&axis, &norm);
    }

    s.hingeAxis1  = axis;
    s.hingeAxis2  = axis;
    s.normalAxis1 = norm;
    s.normalAxis2 = norm;
    s.limitsMin   = p->limit_min;
    s.limitsMax   = p->limit_max;

    s.motorSettings.springSettings.mode                 = JPH_SpringMode_FrequencyAndDamping;
    s.motorSettings.springSettings.frequencyOrStiffness = p->frequency > 0 ? p->frequency : CCF_DEFAULT_FREQUENCY;
    s.motorSettings.springSettings.damping              = p->damping > 0 ? p->damping : CCF_DEFAULT_DAMPING;
    s.motorSettings.maxTorqueLimit                      = CCF_MAX_TORQUE;
    s.motorSettings.minTorqueLimit                      = -CCF_MAX_TORQUE;

    JPH_HingeConstraint *c = JPH_HingeConstraint_Create(&s, b1, b2);

    if (c && p->has_motor && p->motor_type > 0) {
        JPH_HingeConstraint_SetMotorState(c, (JPH_MotorState)p->motor_type);
        if (p->motor_type == 1) {
            JPH_HingeConstraint_SetTargetAngularVelocity(c, p->motor_target);
        }
        if (p->motor_type == 2) {
            JPH_HingeConstraint_SetTargetAngle(c, p->motor_target);
        }
    }

    return (JPH_Constraint *)c;
}

JPH_Constraint *create_slider(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2) {
    JPH_SliderConstraintSettings s;
    JPH_SliderConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space        = JPH_ConstraintSpace_WorldSpace;

    s.point1.x = p->px;
    s.point1.y = p->py;
    s.point1.z = p->pz;
    s.point2   = s.point1;

    JPH_Vec3 axis = {p->ax, p->ay, p->az};
    if (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z < CCF_EPSILON) {
        axis.x = 1.0f;
    }
    JPH_Vec3_Normalize(&axis, &axis);

    JPH_Vec3 norm;
    vec3_get_perpendicular(&axis, &norm);

    s.sliderAxis1 = axis;
    s.sliderAxis2 = axis;
    s.normalAxis1 = norm;
    s.normalAxis2 = norm;
    s.limitsMin   = p->limit_min;
    s.limitsMax   = p->limit_max;

    // --- Motor Support ---
    s.motorSettings.springSettings.mode                 = JPH_SpringMode_FrequencyAndDamping;
    s.motorSettings.springSettings.frequencyOrStiffness = p->frequency;
    s.motorSettings.springSettings.damping              = p->damping;
    s.motorSettings.maxForceLimit                       = p->max_force;
    s.motorSettings.minForceLimit                       = -p->max_force;

    return (JPH_Constraint *)JPH_SliderConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_cone(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2) {
    JPH_ConeConstraintSettings s;
    JPH_ConeConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space        = JPH_ConstraintSpace_WorldSpace;

    s.point1.x = p->px;
    s.point1.y = p->py;
    s.point1.z = p->pz;
    s.point2   = s.point1;

    JPH_Vec3 axis = {p->ax, p->ay, p->az};
    float len_sq  = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;

    // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
    if (len_sq < CCF_TINY_EPSILON) {
        axis.x = 0.0f;
        axis.y = 1.0f;
        axis.z = 0.0f;
    } else {
        JPH_Vec3_Normalize(&axis, &axis);
    }

    s.twistAxis1    = axis;
    s.twistAxis2    = axis;
    s.halfConeAngle = p->half_cone_angle;

    return (JPH_Constraint *)JPH_ConeConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_distance(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2) {
    JPH_DistanceConstraintSettings s;
    JPH_DistanceConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space        = JPH_ConstraintSpace_WorldSpace;

    // Check if the user provided a specific pivot point
    if (fabsf(p->px) > CCF_EPSILON || fabsf(p->py) > CCF_EPSILON || fabsf(p->pz) > CCF_EPSILON) {
        s.point1.x = p->px;
        s.point1.y = p->py;
        s.point1.z = p->pz;
        s.point2   = s.point1;
    } else {
        // Fallback: Default to current body centers if no pivot was provided
        JPH_Body_GetPosition(b1, &s.point1);
        JPH_Body_GetPosition(b2, &s.point2);
    }

    s.minDistance = p->limit_min;
    s.maxDistance = p->limit_max;

    return (JPH_Constraint *)JPH_DistanceConstraint_Create(&s, b1, b2);
}

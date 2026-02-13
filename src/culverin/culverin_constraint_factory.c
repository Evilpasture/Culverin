#include "culverin_constraint_factory.h"
#include "culverin_math.h"
#include <float.h>

// Constraints

// Initialize defaults to avoid garbage data
void params_init(ConstraintParams *p) {
  memset(p, 0, sizeof(ConstraintParams)); // Zero everything first
  p->ay = 1;                              // Default Up
  p->limit_min = -FLT_MAX;
  p->limit_max = FLT_MAX;
  p->frequency = 20.0f; // Default decent stiffness
  p->damping = 1.0f;
}

// --- Jolt Creator Helpers ---

JPH_Constraint *create_fixed(const ConstraintParams *p, JPH_Body *b1,
                             JPH_Body *b2) {
  JPH_FixedConstraintSettings s;
  JPH_FixedConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.autoDetectPoint = true;
  return (JPH_Constraint *)JPH_FixedConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_point(const ConstraintParams *p, JPH_Body *b1,
                             JPH_Body *b2) {
  JPH_PointConstraintSettings s;
  JPH_PointConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;
  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;
  return (JPH_Constraint *)JPH_PointConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_hinge(const ConstraintParams *p, JPH_Body *b1,
                             JPH_Body *b2) {
  JPH_HingeConstraintSettings s;
  JPH_HingeConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  // ... (Keep existing normalization logic) ...
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
  if (len_sq > 1e-9f) {
    JPH_Vec3_Normalize(&axis, &axis);
  } else {
    axis.x = 0;
    axis.y = 1;
    axis.z = 0;
  }

  JPH_Vec3 norm;
  vec3_get_perpendicular(&axis, &norm);

  s.hingeAxis1 = axis;
  s.hingeAxis2 = axis;
  s.normalAxis1 = norm;
  s.normalAxis2 = norm;
  s.limitsMin = p->limit_min;
  s.limitsMax = p->limit_max;

  // --- MOTOR CONFIG ---
  if (p->has_motor) {
    s.motorSettings.springSettings.mode =
        (p->frequency > 0) ? JPH_SpringMode_FrequencyAndDamping
                           : JPH_SpringMode_StiffnessAndDamping;
    s.motorSettings.springSettings.frequencyOrStiffness = p->frequency;
    s.motorSettings.springSettings.damping = p->damping;
    s.motorSettings.maxTorqueLimit = p->max_torque;
    s.motorSettings.minTorqueLimit = -p->max_torque;
  }

  JPH_HingeConstraint *c = JPH_HingeConstraint_Create(&s, b1, b2);

  // Apply Runtime State immediately
  if (c && (int)p->has_motor && p->motor_type > 0) {
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

JPH_Constraint *create_slider(const ConstraintParams *p, JPH_Body *b1,
                              JPH_Body *b2) {
  JPH_SliderConstraintSettings s;
  JPH_SliderConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;
  s.autoDetectPoint = false;

  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;

  // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
  if (len_sq < 1e-9f) {
    axis.x = 0.0f;
    axis.y = 1.0f;
    axis.z = 0.0f;
  } else {
    JPH_Vec3_Normalize(&axis, &axis);
  }

  JPH_Vec3 norm;
  vec3_get_perpendicular(&axis, &norm);

  s.sliderAxis1 = axis;
  s.sliderAxis2 = axis;
  s.normalAxis1 = norm;
  s.normalAxis2 = norm;
  s.limitsMin = p->limit_min;
  s.limitsMax = p->limit_max;

  return (JPH_Constraint *)JPH_SliderConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_cone(const ConstraintParams *p, JPH_Body *b1,
                            JPH_Body *b2) {
  JPH_ConeConstraintSettings s;
  JPH_ConeConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;

  // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
  if (len_sq < 1e-9f) {
    axis.x = 0.0f;
    axis.y = 1.0f;
    axis.z = 0.0f;
  } else {
    JPH_Vec3_Normalize(&axis, &axis);
  }

  s.twistAxis1 = axis;
  s.twistAxis2 = axis;
  s.halfConeAngle = p->half_cone_angle;

  return (JPH_Constraint *)JPH_ConeConstraint_Create(&s, b1, b2);
}

JPH_Constraint *create_distance(const ConstraintParams *p, JPH_Body *b1,
                                JPH_Body *b2) {
  JPH_DistanceConstraintSettings s;
  JPH_DistanceConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  // Check if the user provided a specific pivot point
  if (fabsf(p->px) > 1e-6f || fabsf(p->py) > 1e-6f || fabsf(p->pz) > 1e-6f) {
    s.point1.x = p->px;
    s.point1.y = p->py;
    s.point1.z = p->pz;
    s.point2 = s.point1;
  } else {
    // Fallback: Default to current body centers if no pivot was provided
    JPH_Body_GetPosition(b1, &s.point1);
    JPH_Body_GetPosition(b2, &s.point2);
  }

  s.minDistance = p->limit_min;
  s.maxDistance = p->limit_max;

  return (JPH_Constraint *)JPH_DistanceConstraint_Create(&s, b1, b2);
}
#pragma once
#include "joltc.h"
#include <math.h>

// Helper to find an arbitrary vector perpendicular to 'in'
static inline void vec3_get_perpendicular(const JPH_Vec3 *in, JPH_Vec3 *out) {
  if (fabsf(in->x) > fabsf(in->z)) {
    out->x = -in->y;
    out->y = in->x;
    out->z = 0.0f; // Cross(in, Z)
  } else {
    out->x = 0.0f;
    out->y = -in->z;
    out->z = in->y; // Cross(in, X)
  }
  // Normalize
  float len = sqrtf(out->x * out->x + out->y * out->y + out->z * out->z);
  if (len > 1e-6f) {
    float inv = 1.0f / len;
    out->x *= inv;
    out->y *= inv;
    out->z *= inv;
  } else {
    // Fallback if 'in' is zero
    out->x = 1.0f;
    out->y = 0.0f;
    out->z = 0.0f;
  }
}

// Helper to rotate a vector by a quaternion manually (v' = q * v * q^-1)
static inline void manual_vec3_rotate_by_quat(const JPH_Vec3 *v,
                                              const JPH_Quat *q,
                                              JPH_Vec3 *out) {
  float tx = 2.0f * (q->y * v->z - q->z * v->y);
  float ty = 2.0f * (q->z * v->x - q->x * v->z);
  float tz = 2.0f * (q->x * v->y - q->y * v->x);

  float cx = q->y * tz - q->z * ty;
  float cy = q->z * tx - q->x * tz;
  float cz = q->x * ty - q->y * tx;

  out->x = v->x + q->w * tx + cx;
  out->y = v->y + q->w * ty + cy;
  out->z = v->z + q->w * tz + cz;
}

// Helper for quaternion multiplication (q_out = q_a * q_b)
static inline void manual_quat_multiply(const JPH_Quat *a, const JPH_Quat *b,
                                        JPH_Quat *__restrict out) {
  out->x = a->w * b->x + a->x * b->w + a->y * b->z - a->z * b->y;
  out->y = a->w * b->y - a->x * b->z + a->y * b->w + a->z * b->x;
  out->z = a->w * b->z + a->x * b->y - a->y * b->x + a->z * b->w;
  out->w = a->w * b->w - a->x * b->x - a->y * b->y - a->z * b->z;
}
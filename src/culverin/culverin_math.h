#pragma once
#include "joltc.h"
#include <math.h>

#ifdef _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif


// Helper to find an arbitrary vector perpendicular to 'in'
static inline void vec3_get_perpendicular(const JPH_Vec3 *RESTRICT in, JPH_Vec3 *RESTRICT out) {
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
static inline void manual_vec3_rotate_by_quat(const JPH_Vec3 *RESTRICT v,
                                              const JPH_Quat *RESTRICT q,
                                              JPH_Vec3 *RESTRICT out) {
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
static inline void manual_quat_multiply(const JPH_Quat *RESTRICT a, const JPH_Quat *RESTRICT b,
                                        JPH_Quat *RESTRICT out) {
  out->x = a->w * b->x + a->x * b->w + a->y * b->z - a->z * b->y;
  out->y = a->w * b->y - a->x * b->z + a->y * b->w + a->z * b->x;
  out->z = a->w * b->z + a->x * b->y - a->y * b->x + a->z * b->w;
  out->w = a->w * b->w - a->x * b->x - a->y * b->y - a->z * b->z;
}


// Helper: Convert Quaternion to Mat4 (Rotation Matrix)
static inline void manual_mat4_from_quat(const JPH_Quat *RESTRICT q, JPH_Mat4 *RESTRICT out) {
    float x2 = q->x + q->x;
    float y2 = q->y + q->y;
    float z2 = q->z + q->z;

    float xx = q->x * x2;
    float xy = q->x * y2;
    float xz = q->x * z2;
    float yy = q->y * y2;
    float yz = q->y * z2;
    float zz = q->z * z2;
    float wx = q->w * x2;
    float wy = q->w * y2;
    float wz = q->w * z2;

    // Column 0
    out->column[0].x = 1.0f - (yy + zz);
    out->column[0].y = xy + wz;
    out->column[0].z = xz - wy;
    out->column[0].w = 0.0f;

    // Column 1
    out->column[1].x = xy - wz;
    out->column[1].y = 1.0f - (xx + zz);
    out->column[1].z = yz + wx;
    out->column[1].w = 0.0f;

    // Column 2
    out->column[2].x = xz + wy;
    out->column[2].y = yz - wx;
    out->column[2].z = 1.0f - (xx + yy);
    out->column[2].w = 0.0f;

    // Column 3
    out->column[3].x = 0.0f;
    out->column[3].y = 0.0f;
    out->column[3].z = 0.0f;
    out->column[3].w = 1.0f;
}

// Helper: Matrix Multiplication (C = A * B)
static inline void manual_mat4_multiply(const JPH_Mat4 *RESTRICT A, const JPH_Mat4 *RESTRICT B, JPH_Mat4 *RESTRICT out) {
    // A is left, B is right.
    // Iterating columns of B
    for (int i = 0; i < 4; ++i) {
        float x = B->column[i].x;
        float y = B->column[i].y;
        float z = B->column[i].z;
        float w = B->column[i].w;

        // out_col_i = A * b_col_i
        out->column[i].x = A->column[0].x * x + A->column[1].x * y + A->column[2].x * z + A->column[3].x * w;
        out->column[i].y = A->column[0].y * x + A->column[1].y * y + A->column[2].y * z + A->column[3].y * w;
        out->column[i].z = A->column[0].z * x + A->column[1].z * y + A->column[2].z * z + A->column[3].z * w;
        out->column[i].w = A->column[0].w * x + A->column[1].w * y + A->column[2].w * z + A->column[3].w * w;
    }
}

#undef RESTRICT
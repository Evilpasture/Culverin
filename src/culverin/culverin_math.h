#pragma once
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"
#include "joltc.h"
#include <Python.h>

// --- Double Precision Refinement ---
CULV_MAYBE_UNUSED
static inline double newton_raphson_iterate_d(double number) {
    constexpr double threehalfs = 1.5;
    const double half_number    = number * 0.5;
    return number * (threehalfs - (half_number * number * number));
}

static inline double culverin_fast_rsqrt_d(double number) {
    // Clang will likely emit RSQRTD (if available) or SQRTD + DIVD
    return 1.0 / sqrt(number);
}

// --- Float Precision Refinement ---
CULV_MAYBE_UNUSED
static inline float newton_raphson_iterate_f(float number) {
    constexpr float threehalfs = 1.5f;
    const float half_number    = number * 0.5f;
    return number * (threehalfs - (half_number * number * number));
}

static inline float culverin_fast_rsqrt_f(float number) { return 1.0f / sqrtf(number); }

// --- The Type-Generic Interface ---
// NOLINTNEXTLINE(readability-identifier-naming)
#define culverin_fast_rsqrt(x)                                                                     \
    _Generic((x), float: culverin_fast_rsqrt_f, double: culverin_fast_rsqrt_d)(x)

// Helper to find an arbitrary vector perpendicular to 'in'
CULV_MAYBE_UNUSED
static inline void vec3_get_perpendicular(const JPH_Vec3 *CULV_RESTRICT in,
                                          JPH_Vec3 *CULV_RESTRICT out) {
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

#if defined(__cplusplus)
extern "C" {
#endif
void culverin_compute_interpolation_loop(const PosStride *__restrict curr_p,
                                         const PosStride *__restrict prev_p,
                                         const AuxStride *__restrict curr_r,
                                         const AuxStride *__restrict prev_r, float alpha,
                                         float *__restrict out, size_t count);

void culverin_math_get_perspective(float fovy_rad, float aspect, float near_p, float far_p,
                                   float *__restrict out);
void culverin_math_get_ortho(float left, float right, float bottom, float top, float near_p,
                             float far_p, float *__restrict out);
void culverin_math_get_look_at(const float *__restrict eye, const float *__restrict target,
                               const float *__restrict up, float *__restrict out);
void culverin_math_get_trs(const float *__restrict pos, const float *__restrict rot_q,
                           const float *__restrict scale, float *__restrict out);
void culverin_math_get_trs_batch(size_t count, const float *__restrict pos,
                                 const float *__restrict rot_q, const float *__restrict scale,
                                 float *__restrict out);
void culverin_math_mat44_inverse(const float *__restrict in, float *__restrict out);
void culverin_math_mat44_mul(const float *__restrict a, const float *__restrict b,
                             float *__restrict out);
void culverin_math_mat44_mul_batch(const float *__restrict single_mat,
                                   const float *__restrict batch_mats, size_t count,
                                   float *__restrict out);
void culverin_math_transform_vec3(const float *__restrict mat, const float *__restrict vec,
                                  float *__restrict out);
void culverin_math_transform_vec3_batch(const float *__restrict mat, const float *__restrict vecs,
                                        size_t count, float *__restrict out);
// Returns 1 if visible, 0 if culled
int culverin_math_cull_aabb(const float *__restrict vp_mat, const float *__restrict aabb_min,
                            const float *__restrict aabb_max);
void culverin_math_cull_aabb_batch(const float *__restrict vp_mat,
                                   const float *__restrict aabb_data, size_t count,
                                   uint8_t *__restrict out_visibility);
#if defined(__cplusplus)
}
#endif

extern PyType_Spec MathService_spec;
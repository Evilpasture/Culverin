#pragma once
#include <math.h>
#include "core/CompilerSpecifics.h"
#include "core/Types.h"
#include "joltc.h"

// --- Double Precision Refinement ---
[[maybe_unused]]
static inline double newton_raphson_iterate_d(double number) {
    constexpr double threehalfs = 1.5;
    const double half_number    = number * 0.5;
    return number * (threehalfs - (half_number * number * number));
}

static inline double CPH_FastRsqrtD(double number) {
    // Clang will likely emit RSQRTD (if available) or SQRTD + DIVD
    return 1.0 / sqrt(number);
}

// --- Float Precision Refinement ---
[[maybe_unused]]
static inline float newton_raphson_iterate_f(float number) {
    constexpr float threehalfs = 1.5f;
    const float half_number    = number * 0.5f;
    return number * (threehalfs - (half_number * number * number));
}

static inline float CPH_FastRsqrtF(float number) { return 1.0f / sqrtf(number); }

// --- The Type-Generic Interface ---
// NOLINTNEXTLINE(readability-identifier-naming)
#define CPH_FastRsqrt(x)                                                                     \
    _Generic((x), float: CPH_FastRsqrtF, double: CPH_FastRsqrtD)(x)

// Helper to find an arbitrary vector perpendicular to 'in'
[[maybe_unused]]
static inline void vec3_get_perpendicular(const JPH_Vec3 *CPH_RESTRICT in,
                                          JPH_Vec3 *CPH_RESTRICT out) {
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
void CPH_MathComputeInterpolationLoop(const CPH_PosStride *__restrict curr_p,
                                         const CPH_PosStride *__restrict prev_p,
                                         const CPH_AuxStride *__restrict curr_r,
                                         const CPH_AuxStride *__restrict prev_r, float alpha,
                                         float *__restrict out, size_t count);

void CPH_MathInterpolateCharacterTransform(const CPH_PosStride *__restrict start_p,
                                                   const CPH_AuxStride *__restrict start_r,
                                                   const JPH_RVec3 *__restrict end_p,
                                                   const JPH_Quat *__restrict end_r, float alpha,
                                                   float *__restrict out_p,
                                                   float *__restrict out_r);
void CPH_MathGetPerspective(float fovy_rad, float aspect, float near_p, float far_p,
                                   float *__restrict out);
void CPH_MathGetOrtho(float left, float right, float bottom, float top, float near_p,
                             float far_p, float *__restrict out);
void CPH_MathGetLookAt(const float *__restrict eye, const float *__restrict target,
                               const float *__restrict up, float *__restrict out);
void CPH_MathGetTrs(const float *__restrict pos, const float *__restrict rot_q,
                           const float *__restrict scale, float *__restrict out);
void CPH_MathGetTrsBatch(size_t count, const float *__restrict pos,
                                 const float *__restrict rot_q, const float *__restrict scale,
                                 float *__restrict out);
void CPH_MathMat44Inverse(const float *__restrict in, float *__restrict out);
void CPH_MathMat44Mul(const float *__restrict a, const float *__restrict b,
                             float *__restrict out);
void CPH_MathMat44MulBatch(const float *__restrict single_mat,
                                   const float *__restrict batch_mats, size_t count,
                                   float *__restrict out);
void CPH_MathTransformVec3(const float *__restrict mat, const float *__restrict vec,
                                  float *__restrict out);
void CPH_MathTransformVec3Batch(const float *__restrict mat, const float *__restrict vecs,
                                        size_t count, float *__restrict out);
// Returns 1 if visible, 0 if culled
int CPH_MathCullAabb(const float *__restrict vp_mat, const float *__restrict aabb_min,
                            const float *__restrict aabb_max);
void CPH_MathCullAabbBatch(const float *__restrict vp_mat,
                                   const float *__restrict aabb_data, size_t count,
                                   uint8_t *__restrict out_visibility);
void CPH_MathVec3NormalizeBatch(const float *__restrict in, size_t count,
                                        float *__restrict out);
void CPH_MathQuatFromEuler(float x, float y, float z, float *__restrict out);
void CPH_MathQuatToEuler(const float *__restrict in_q, float *__restrict out_euler);
void CPH_MathQuatSlerp(const float *__restrict q1, const float *__restrict q2, float t,
                              float *__restrict out);
void CPH_MathQuatMul(const float *__restrict a, const float *__restrict b,
                            float *__restrict out);
void CPH_MathVec3LerpBatch(const float *__restrict a, const float *__restrict b,
                                   float alpha, size_t count, float *__restrict out);
void CPH_MathQuatRotateVec3(const float *__restrict q, const float *__restrict v,
                                    float *__restrict out);
void CPH_MathQuatRotateVec3Batch(const float *__restrict q, const float *__restrict vecs,
                                          size_t count, float *__restrict out);
void CPH_MathQuatInverse(const float *__restrict q, float *__restrict out);
void CPH_MathProject(const float *__restrict v, const float *__restrict mvp,
                           const int *__restrict viewport, float *__restrict out);
void CPH_MathUnproject(const float *__restrict v, const float *__restrict mvp,
                             const int *__restrict viewport, float *__restrict out);
void CPH_MathQuatFromTo(const float *__restrict v1, const float *__restrict v2,
                                float *__restrict out);
float CPH_MathVec3Dot(const float *__restrict v1, const float *__restrict v2);
void CPH_MathVec3Cross(const float *__restrict v1, const float *__restrict v2,
                              float *__restrict out);
int CPH_MathIntersectRayPlane(const float *__restrict ro, const float *__restrict rd,
                                      const float *__restrict po, const float *__restrict pn,
                                      float *__restrict out_t, float *__restrict out_p);
void CPH_MathQuatGetAxisAngle(const float *__restrict in_q, float *__restrict out_axis,
                                       float *__restrict out_angle);
void CPH_MathQuatFromAxisAngle(const float *__restrict axis, float angle,
                                        float *__restrict out);
void CPH_MathVec3DistanceBatch(const float *__restrict a, const float *__restrict b,
                                       size_t count, float *__restrict out);
void CPH_MathVec3Normalize(const float *__restrict v, float *__restrict out);
void CPH_MathMat44GetTranslation(const float *__restrict in_mat, float *__restrict out_vec);
void CPH_MathMat44GetRotation(const float *__restrict in_mat, float *__restrict out_quat);
void CPH_MathMat44Identity(float *__restrict out);
void CPH_MathVec3Reflect(const float *__restrict v, const float *__restrict n,
                                float *__restrict out);
float CPH_MathVec3Distance(const float *__restrict v1, const float *__restrict v2);
void CPH_MathQuatRotateVec3Inverse(const float *__restrict q, const float *__restrict v,
                                            float *__restrict out);
void CPH_MathEulerToQuat(const float *__restrict euler, float *__restrict out_q);
void CPH_MathEulerToQuatBatch(const float *__restrict eulers, size_t count,
                                       float *__restrict out_qs);
#if defined(__cplusplus)
}
#endif

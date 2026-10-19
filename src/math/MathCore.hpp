#pragma once
#include <cmath>
#include "core/Compiler.hpp"
#include "core/Types.hpp"
#include "joltc.h"


namespace CPH {

inline double FastRsqrtD(double number) {
    // Clang will likely emit RSQRTD (if available) or SQRTD + DIVD
    return 1.0 / std::sqrt(number);
}

inline float FastRsqrtF(float number) { return 1.0f / std::sqrt(number); }

// Overloaded reciprocal square root (replaces the C _Generic dispatch).
[[nodiscard]] inline float FastRsqrt(float number) noexcept { return FastRsqrtF(number); }
[[nodiscard]] inline double FastRsqrt(double number) noexcept { return FastRsqrtD(number); }

// Helper to find an arbitrary vector perpendicular to 'in'
inline void vec3_get_perpendicular(const JPH_Vec3 *CPH_RESTRICT in,
                                          JPH_Vec3 *CPH_RESTRICT out) {
    if (std::fabs(in->x) > std::fabs(in->z)) {
        out->x = -in->y;
        out->y = in->x;
        out->z = 0.0f; // Cross(in, Z)
    } else {
        out->x = 0.0f;
        out->y = -in->z;
        out->z = in->y; // Cross(in, X)
    }
    // Normalize
    float len = std::sqrt(out->x * out->x + out->y * out->y + out->z * out->z);
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

void MathComputeInterpolationLoop(const PosStride *__restrict curr_p,
                                         const PosStride *__restrict prev_p,
                                         const AuxStride *__restrict curr_r,
                                         const AuxStride *__restrict prev_r, float alpha,
                                         float *__restrict out, size_t count);

void MathInterpolateCharacterTransform(const PosStride *__restrict start_p,
                                                   const AuxStride *__restrict start_r,
                                                   const JPH_RVec3 *__restrict end_p,
                                                   const JPH_Quat *__restrict end_r, float alpha,
                                                   float *__restrict out_p,
                                                   float *__restrict out_r);
void MathGetPerspective(float fovy_rad, float aspect, float near_p, float far_p,
                                   float *__restrict out);
void MathGetOrtho(float left, float right, float bottom, float top, float near_p,
                             float far_p, float *__restrict out);
void MathGetLookAt(const float *__restrict eye, const float *__restrict target,
                               const float *__restrict up, float *__restrict out);
void MathGetTrs(const float *__restrict pos, const float *__restrict rot_q,
                           const float *__restrict scale, float *__restrict out);
void MathGetTrsBatch(size_t count, const float *__restrict pos,
                                 const float *__restrict rot_q, const float *__restrict scale,
                                 float *__restrict out);
void MathMat44Inverse(const float *__restrict in, float *__restrict out);
void MathMat44Mul(const float *__restrict a, const float *__restrict b,
                             float *__restrict out);
void MathMat44MulBatch(const float *__restrict single_mat,
                                   const float *__restrict batch_mats, size_t count,
                                   float *__restrict out);
void MathTransformVec3(const float *__restrict mat, const float *__restrict vec,
                                  float *__restrict out);
void MathTransformVec3Batch(const float *__restrict mat, const float *__restrict vecs,
                                        size_t count, float *__restrict out);
// Returns 1 if visible, 0 if culled
int MathCullAabb(const float *__restrict vp_mat, const float *__restrict aabb_min,
                            const float *__restrict aabb_max);
void MathCullAabbBatch(const float *__restrict vp_mat,
                                   const float *__restrict aabb_data, size_t count,
                                   uint8_t *__restrict out_visibility);
void MathVec3NormalizeBatch(const float *__restrict in, size_t count,
                                        float *__restrict out);
void MathQuatFromEuler(float x, float y, float z, float *__restrict out);
void MathQuatToEuler(const float *__restrict in_q, float *__restrict out_euler);
void MathQuatSlerp(const float *__restrict q1, const float *__restrict q2, float t,
                              float *__restrict out);
void MathQuatMul(const float *__restrict a, const float *__restrict b,
                            float *__restrict out);
void MathVec3LerpBatch(const float *__restrict a, const float *__restrict b,
                                   float alpha, size_t count, float *__restrict out);
void MathQuatRotateVec3(const float *__restrict q, const float *__restrict v,
                                    float *__restrict out);
void MathQuatRotateVec3Batch(const float *__restrict q, const float *__restrict vecs,
                                          size_t count, float *__restrict out);
void MathQuatInverse(const float *__restrict q, float *__restrict out);
void MathProject(const float *__restrict v, const float *__restrict mvp,
                           const int *__restrict viewport, float *__restrict out);
void MathUnproject(const float *__restrict v, const float *__restrict mvp,
                             const int *__restrict viewport, float *__restrict out);
void MathQuatFromTo(const float *__restrict v1, const float *__restrict v2,
                                float *__restrict out);
float MathVec3Dot(const float *__restrict v1, const float *__restrict v2);
void MathVec3Cross(const float *__restrict v1, const float *__restrict v2,
                              float *__restrict out);
int MathIntersectRayPlane(const float *__restrict ro, const float *__restrict rd,
                                      const float *__restrict po, const float *__restrict pn,
                                      float *__restrict out_t, float *__restrict out_p);
void MathQuatGetAxisAngle(const float *__restrict in_q, float *__restrict out_axis,
                                       float *__restrict out_angle);
void MathQuatFromAxisAngle(const float *__restrict axis, float angle,
                                        float *__restrict out);
void MathVec3DistanceBatch(const float *__restrict a, const float *__restrict b,
                                       size_t count, float *__restrict out);
void MathVec3Normalize(const float *__restrict v, float *__restrict out);
void MathMat44GetTranslation(const float *__restrict in_mat, float *__restrict out_vec);
void MathMat44GetRotation(const float *__restrict in_mat, float *__restrict out_quat);
void MathMat44Identity(float *__restrict out);
void MathVec3Reflect(const float *__restrict v, const float *__restrict n,
                                float *__restrict out);
float MathVec3Distance(const float *__restrict v1, const float *__restrict v2);
void MathQuatRotateVec3Inverse(const float *__restrict q, const float *__restrict v,
                                            float *__restrict out);
void MathEulerToQuat(const float *__restrict euler, float *__restrict out_q);
void MathEulerToQuatBatch(const float *__restrict eulers, size_t count,
                                       float *__restrict out_qs);

} // namespace CPH

#include "culverin_types.h" // Needed for PosStride/AuxStride
// clang-format off
#include <Jolt/Jolt.h>
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Geometry/Plane.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Math/Vec4.h>
#include <Python.h>
#include <cmath>
// clang-format on

extern "C" {

// Internal math helper for C, Python doesn't need this
void culverin_compute_interpolation_loop(const PosStride *__restrict curr_p,
                                         const PosStride *__restrict prev_p,
                                         const AuxStride *__restrict curr_r,
                                         const AuxStride *__restrict prev_r, float alpha,
                                         float *__restrict out, size_t count) {
    const double d_alpha = static_cast<double>(alpha);
    const float f_alpha  = alpha;

    CULV_UNROLL_LOOP(4)
    for (size_t i = 0; i < count; ++i) {
        const uint32_t base_idx = i * 7;

        // 1. Position Interpolation
        // Using component construction is the safest way to handle RVec3 (Float or Double)
        JPH::RVec3 p1(prev_p[i].x, prev_p[i].y, prev_p[i].z);
        JPH::RVec3 p2(curr_p[i].x, curr_p[i].y, curr_p[i].z);
        JPH::RVec3 p_interp = p1 + (p2 - p1) * d_alpha;

        // 2. Rotation Interpolation (Shortest-path NLerp)
        // Reinterpret the AuxStride pointer as a JPH::Float4 pointer for sLoadFloat4
        JPH::Vec4 v1 = JPH::Vec4::sLoadFloat4(reinterpret_cast<const JPH::Float4 *>(&prev_r[i]));
        JPH::Vec4 v2 = JPH::Vec4::sLoadFloat4(reinterpret_cast<const JPH::Float4 *>(&curr_r[i]));

        JPH::Quat q1(v1);
        JPH::Quat q2(v2);

        // Vectorized dot product of the Quat components
        float dot = v1.Dot(v2);

        // Shortest path hemisphere check
        JPH::Quat q2_short = (dot < 0.0f) ? -q2 : q2;

        // NLerp: q1 + (q2 - q1) * alpha, followed by normalization
        JPH::Quat q_interp = (q1 + (q2_short - q1) * f_alpha).Normalized();

        // 3. Store to Output Buffer
        // Store Position (Double -> Float conversion happens here)
        out[base_idx + 0] = static_cast<float>(p_interp.GetX());
        out[base_idx + 1] = static_cast<float>(p_interp.GetY());
        out[base_idx + 2] = static_cast<float>(p_interp.GetZ());

        // Store Rotation (4 floats starting at index 3)
        // Cast the output pointer to Float4* to satisfy Jolt's Store method
        q_interp.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(&out[base_idx + 3]));
    }
}

// Internal helpers for Python

// -----------------------------------------------------------------------------
// Projection Matrices (Standard Column-Major)
// -----------------------------------------------------------------------------
void culverin_math_get_perspective(float fovy_rad, float aspect, float near_p, float far_p,
                                   float *__restrict out) {
    float f         = 1.0f / std::tan(fovy_rad * 0.5f);
    float range_inv = 1.0f / (near_p - far_p);

    JPH::Mat44 m;
    // Column 0
    m.SetColumn4(0, JPH::Vec4(f / aspect, 0.0f, 0.0f, 0.0f));
    // Column 1
    m.SetColumn4(1, JPH::Vec4(0.0f, f, 0.0f, 0.0f));
    // Column 2: Contains the Z-range mapping and the W-divider (-1)
    m.SetColumn4(2, JPH::Vec4(0.0f, 0.0f, (far_p + near_p) * range_inv, -1.0f));
    // Column 3: Contains the Z-precision offset
    m.SetColumn4(3, JPH::Vec4(0.0f, 0.0f, (2.0f * far_p * near_p) * range_inv, 0.0f));

    // Jolt's StoreFloat4x4 writes Col0, Col1, Col2, Col3 (Standard Column-Major)
    m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_get_ortho(float left, float right, float bottom, float top, float near_p,
                             float far_p, float *__restrict out) {
    float r_l = 1.0f / (right - left);
    float t_b = 1.0f / (top - bottom);
    float f_n = 1.0f / (far_p - near_p);

    JPH::Mat44 m;
    m.SetColumn4(0, JPH::Vec4(2.0f * r_l, 0.0f, 0.0f, 0.0f));
    m.SetColumn4(1, JPH::Vec4(0.0f, 2.0f * t_b, 0.0f, 0.0f));
    m.SetColumn4(2, JPH::Vec4(0.0f, 0.0f, -2.0f * f_n, 0.0f));
    m.SetColumn4(
        3, JPH::Vec4(-(right + left) * r_l, -(top + bottom) * t_b, -(far_p + near_p) * f_n, 1.0f));

    m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

// -----------------------------------------------------------------------------
// View Matrix
// -----------------------------------------------------------------------------
void culverin_math_get_look_at(const float *__restrict eye, const float *__restrict target,
                               const float *__restrict up, float *__restrict out) {
    JPH::Vec3 e(eye[0], eye[1], eye[2]);
    JPH::Vec3 t(target[0], target[1], target[2]);
    JPH::Vec3 u(up[0], up[1], up[2]);

    // Use Jolt's built-in LookAt (it's optimized and handles edge cases)
    // Note: Jolt's sLookAt creates a View Matrix (World -> Camera)
    JPH::Mat44 m = JPH::Mat44::sLookAt(e, t, u);

    m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

// -----------------------------------------------------------------------------
// Model Matrices (TRS)
// -----------------------------------------------------------------------------
void culverin_math_get_trs(const float *__restrict pos, const float *__restrict rot_q,
                           const float *__restrict scale, float *__restrict out) {
    JPH::Quat q(rot_q[0], rot_q[1], rot_q[2], rot_q[3]);

    // 1. Create rotation matrix (Column-Major)
    JPH::Mat44 m = JPH::Mat44::sRotation(q);

    // 2. Scale the basis vectors (Columns 0, 1, 2)
    m.SetColumn4(0, m.GetColumn4(0) * scale[0]);
    m.SetColumn4(1, m.GetColumn4(1) * scale[1]);
    m.SetColumn4(2, m.GetColumn4(2) * scale[2]);

    // 3. Set translation (Column 3)
    m.SetTranslation(JPH::Vec3(pos[0], pos[1], pos[2]));

    // 4. FIX: Remove .Transposed(). Native Jolt layout matches the test
    // expectation where translation starts at index 12.
    m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_get_trs_batch(size_t count, const float *__restrict pos,
                                 const float *__restrict rot_q, const float *__restrict scale,
                                 float *__restrict out) {
    for (size_t i = 0; i < count; ++i) {
        JPH::Quat q(rot_q[i * 4 + 0], rot_q[i * 4 + 1], rot_q[i * 4 + 2], rot_q[i * 4 + 3]);

        JPH::Mat44 m = JPH::Mat44::sRotation(q);
        m.SetColumn4(0, m.GetColumn4(0) * scale[i * 3 + 0]);
        m.SetColumn4(1, m.GetColumn4(1) * scale[i * 3 + 1]);
        m.SetColumn4(2, m.GetColumn4(2) * scale[i * 3 + 2]);
        m.SetTranslation(JPH::Vec3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]));

        // FIX: Remove .Transposed()
        m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(&out[i * 16]));
    }
}

void culverin_math_mat44_inverse(const float *__restrict in, float *__restrict out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(in));
    // Jolt's Inversed() is highly optimized SIMD
    m.Inversed().StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_mat44_mul(const float *__restrict a, const float *__restrict b,
                             float *__restrict out) {
    JPH::Mat44 ma = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(a));
    JPH::Mat44 mb = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(b));
    (ma * mb).StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

// Computes: Out[i] = Single * Batch[i]
// Use case: MVP_Batch = ViewProj * Model_Batch
void culverin_math_mat44_mul_batch(const float *__restrict single_mat,
                                   const float *__restrict batch_mats, size_t count,
                                   float *__restrict out) {
    JPH::Mat44 s = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(single_mat));
    for (size_t i = 0; i < count; ++i) {
        JPH::Mat44 b =
            JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(&batch_mats[i * 16]));
        (s * b).StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(&out[i * 16]));
    }
}

void culverin_math_transform_vec3(const float *__restrict mat, const float *__restrict vec,
                                  float *__restrict out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mat));
    JPH::Vec3 v(vec[0], vec[1], vec[2]);
    JPH::Vec3 res = m * v;

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

// Transforms N Vec3s by 1 Matrix
void culverin_math_transform_vec3_batch(const float *__restrict mat, const float *__restrict vecs,
                                        size_t count, float *__restrict out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mat));
    for (size_t i = 0; i < count; ++i) {
        JPH::Vec3 v(vecs[i * 3], vecs[i * 3 + 1], vecs[i * 3 + 2]);
        JPH::Vec3 res  = m * v;
        out[i * 3]     = res.GetX();
        out[i * 3 + 1] = res.GetY();
        out[i * 3 + 2] = res.GetZ();
    }
}

int culverin_math_cull_aabb(const float *__restrict vp_mat, const float *__restrict aabb_min,
                            const float *__restrict aabb_max) {
    // 1. Load Matrix (Column-Major)
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(vp_mat));

    // 2. Get the Transposed matrix to access original rows as columns
    // The compiler confirmed 'Transposed()' is the correct member name
    JPH::Mat44 mt = m.Transposed();

    // In the transposed matrix, columns are the rows of the original matrix
    JPH::Vec4 r0 = mt.GetColumn4(0);
    JPH::Vec4 r1 = mt.GetColumn4(1);
    JPH::Vec4 r2 = mt.GetColumn4(2);
    JPH::Vec4 r3 = mt.GetColumn4(3);

    // 3. Extract Frustum Planes using Gribb-Hartmann
    // Planes are stored as Vec4(A, B, C, D) where Ax + By + Cz + D = 0
    JPH::Vec4 planes[6] = {
        r3 + r0, // Left
        r3 - r0, // Right
        r3 + r1, // Bottom
        r3 - r1, // Top
        r3 + r2, // Near
        r3 - r2  // Far
    };

    // 4. Setup AABox
    JPH::AABox box(JPH::Vec3(aabb_min[0], aabb_min[1], aabb_min[2]),
                   JPH::Vec3(aabb_max[0], aabb_max[1], aabb_max[2]));

    // 5. Test Box against Planes
    for (int i = 0; i < 6; ++i) {
        // Extract the normal (xyz) from the Vec4 plane.
        // In Jolt, the Vec3 constructor from a Vec4 is the standard way to drop W.
        JPH::Vec3 normal(planes[i]);

        // Find the box corner furthest in the normal's direction
        JPH::Vec3 support = box.GetSupport(normal);

        // Signed Distance test: (Normal . Support) + Plane_D
        // We can do this efficiently by dotting the Plane Vec4 with (Support.xyz, 1.0)
        // This calculates Ax + By + Cz + D in one SIMD operation.
        float dist = planes[i].Dot(JPH::Vec4(support, 1.0f));

        if (dist < 0.0f) {
            return 0; // Fully outside this plane -> Culled
        }
    }

    return 1; // Visible or Intersecting
}

void culverin_math_cull_aabb_batch(const float *__restrict vp_mat,
                                   const float *__restrict aabb_data, size_t count,
                                   uint8_t *__restrict out_visibility) {
    JPH::Mat44 m  = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(vp_mat));
    JPH::Mat44 mt = m.Transposed();
    JPH::Vec4 r0  = mt.GetColumn4(0);
    JPH::Vec4 r1  = mt.GetColumn4(1);
    JPH::Vec4 r2  = mt.GetColumn4(2);
    JPH::Vec4 r3  = mt.GetColumn4(3);

    JPH::Vec4 planes[6] = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2};

    for (size_t i = 0; i < count; ++i) {
        const float *b = &aabb_data[i * 6];
        JPH::AABox box(JPH::Vec3(b[0], b[1], b[2]), JPH::Vec3(b[3], b[4], b[5]));

        bool visible = true;
        for (int p = 0; p < 6; ++p) {
            if (planes[p].Dot(JPH::Vec4(box.GetSupport(JPH::Vec3(planes[p])), 1.0f)) < 0.0f) {
                visible = false;
                break;
            }
        }
        out_visibility[i] = visible ? 1 : 0;
    }
}

void culverin_math_vec3_normalize_batch(const float *__restrict in, size_t count,
                                        float *__restrict out) {
    for (size_t i = 0; i < count; ++i) {
        // Load from float[3] into SIMD register
        JPH::Vec3 v(in[i * 3 + 0], in[i * 3 + 1], in[i * 3 + 2]);

        float len_sq = v.LengthSq();
        if (len_sq > 1e-12f) {
            // Jolt uses Reciprocal Square Root (RSQRT) for normalization
            v = v / std::sqrt(len_sq);
        } else {
            v = JPH::Vec3::sZero();
        }

        // Store back to float[3]
        out[i * 3 + 0] = v.GetX();
        out[i * 3 + 1] = v.GetY();
        out[i * 3 + 2] = v.GetZ();
    }
}

void culverin_math_quat_from_euler(float x, float y, float z, float *__restrict out) {
    // Jolt uses sEulerAngles for (x, y, z) radian inputs
    // Order: Rotation around Y, then X, then Z
    JPH::Quat q = JPH::Quat::sEulerAngles(JPH::Vec3(x, y, z));

    // Store back to out[4] as (x, y, z, w)
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_quat_to_euler(const float *__restrict in_q, float *__restrict out_euler) {
    // Load components into Jolt Quat
    JPH::Quat q(in_q[0], in_q[1], in_q[2], in_q[3]);

    // GetEulerAngles returns Vec3(x, y, z) in radians
    JPH::Vec3 euler = q.GetEulerAngles();

    out_euler[0] = euler.GetX();
    out_euler[1] = euler.GetY();
    out_euler[2] = euler.GetZ();
}

void culverin_math_quat_slerp(const float *__restrict q1, const float *__restrict q2, float t,
                              float *__restrict out) {
    // Jolt Quat stores as [x, y, z, w]
    JPH::Quat a(q1[0], q1[1], q1[2], q1[3]);
    JPH::Quat b(q2[0], q2[1], q2[2], q2[3]);

    // Corrected: Jolt uses uppercase SLERP
    JPH::Quat res = a.SLERP(b, t);

    res.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_quat_mul(const float *__restrict a, const float *__restrict b,
                            float *__restrict out) {
    JPH::Quat qa(a[0], a[1], a[2], a[3]);
    JPH::Quat qb(b[0], b[1], b[2], b[3]);

    // Combine rotations: Apply b then a
    JPH::Quat res = qa * qb;

    res.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_vec3_lerp_batch(const float *__restrict a, const float *__restrict b,
                                   float alpha, size_t count, float *__restrict out) {
    // Optimization: If alpha is 0 or 1, we can just memcpy,
    // but usually, the caller handles that.
    // We proceed with the standard SIMD-accelerated loop.
    for (size_t i = 0; i < count; ++i) {
        size_t idx = i * 3;

        // Load the two 3D vectors into Jolt registers
        JPH::Vec3 vA(a[idx], a[idx + 1], a[idx + 2]);
        JPH::Vec3 vB(b[idx], b[idx + 1], b[idx + 2]);

        // Lerp Formula: Result = A + (B - A) * alpha
        JPH::Vec3 res = vA + (vB - vA) * alpha;

        // Store the result back into the output buffer
        out[idx]     = res.GetX();
        out[idx + 1] = res.GetY();
        out[idx + 2] = res.GetZ();
    }
}

void culverin_math_quat_rotate_vec3(const float *__restrict q, const float *__restrict v,
                                    float *__restrict out) {
    JPH::Quat rotation(q[0], q[1], q[2], q[3]);
    JPH::Vec3 point(v[0], v[1], v[2]);

    // Rotate the vector
    JPH::Vec3 res = rotation * point;

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

void culverin_math_quat_rotate_vec3_batch(const float *__restrict q, const float *__restrict vecs,
                                          size_t count, float *__restrict out) {
    // Load the rotation once
    JPH::Quat rotation(q[0], q[1], q[2], q[3]);

    for (size_t i = 0; i < count; ++i) {
        size_t idx = i * 3;
        JPH::Vec3 v(vecs[idx], vecs[idx + 1], vecs[idx + 2]);

        // Rotate vector
        JPH::Vec3 res = rotation * v;

        out[idx]     = res.GetX();
        out[idx + 1] = res.GetY();
        out[idx + 2] = res.GetZ();
    }
}

void culverin_math_quat_inverse(const float *__restrict q, float *__restrict out) {
    JPH::Quat in_q(q[0], q[1], q[2], q[3]);

    // Calculate the inverse
    JPH::Quat inv = in_q.Inversed();

    inv.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_project(const float *__restrict v, const float *__restrict mvp,
                           const int *__restrict viewport, float *__restrict out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mvp));
    JPH::Vec3 world_pos(v[0], v[1], v[2]);

    // Transform to Clip Space
    JPH::Vec4 clip = m * JPH::Vec4(world_pos, 1.0f);

    float w = clip.GetW();
    if (std::abs(w) > 1e-6f) {
        // NDC Space (-1 to 1)
        float inv_w = 1.0f / w;
        float ndc_x = clip.GetX() * inv_w;
        float ndc_y = clip.GetY() * inv_w;
        float ndc_z = clip.GetZ() * inv_w;

        // Screen Space (Pixels)
        out[0] = viewport[0] + (viewport[2] * (ndc_x + 1.0f) * 0.5f);
        out[1] =
            viewport[1] + (viewport[3] * (1.0f - (ndc_y + 1.0f) * 0.5f)); // Flip Y for Screen space
        out[2] = ndc_z; // Z is usually preserved for depth sorting
    } else {
        out[0] = out[1] = out[2] = 0.0f;
    }
}

void culverin_math_unproject(const float *__restrict v, const float *__restrict mvp, const int *__restrict viewport, float *__restrict out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mvp));
    JPH::Mat44 inv_m = m.Inversed();

    float ndc_x = (v[0] - (float)viewport[0]) / (float)viewport[2] * 2.0f - 1.0f;
    float ndc_y = 1.0f - (v[1] - (float)viewport[1]) / (float)viewport[3] * 2.0f;
    
    // FIX: Maintain symmetry with project()
    float ndc_z = v[2]; 

    JPH::Vec4 world_pos = inv_m * JPH::Vec4(ndc_x, ndc_y, ndc_z, 1.0f);

    float w = world_pos.GetW();
    if (std::abs(w) > 1e-6f) {
        float inv_w = 1.0f / w;
        out[0] = world_pos.GetX() * inv_w;
        out[1] = world_pos.GetY() * inv_w;
        out[2] = world_pos.GetZ() * inv_w;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
    }
}

void culverin_math_quat_from_to(const float *__restrict v1, const float *__restrict v2,
                                float *__restrict out) {
    JPH::Vec3 from(v1[0], v1[1], v1[2]);
    JPH::Vec3 to(v2[0], v2[1], v2[2]);

    // Create quaternion that rotates from 'from' to 'to' along the shortest path
    JPH::Quat q = JPH::Quat::sFromTo(from, to);

    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

float culverin_math_vec3_dot(const float *__restrict v1, const float *__restrict v2) {
    JPH::Vec3 a(v1[0], v1[1], v1[2]);
    JPH::Vec3 b(v2[0], v2[1], v2[2]);

    return a.Dot(b);
}

void culverin_math_vec3_cross(const float *__restrict v1, const float *__restrict v2,
                              float *__restrict out) {
    JPH::Vec3 a(v1[0], v1[1], v1[2]);
    JPH::Vec3 b(v2[0], v2[1], v2[2]);

    // Calculate perpendicular vector
    JPH::Vec3 res = a.Cross(b);

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

int culverin_math_intersect_ray_plane(const float *__restrict ro, const float *__restrict rd,
                                      const float *__restrict po, const float *__restrict pn,
                                      float *__restrict out_t, float *__restrict out_p) {
    JPH::Vec3 ray_o(ro[0], ro[1], ro[2]);
    JPH::Vec3 ray_d(rd[0], rd[1], rd[2]);
    JPH::Vec3 plane_p(po[0], po[1], po[2]);
    JPH::Vec3 plane_n(pn[0], pn[1], pn[2]);

    float denom = ray_d.Dot(plane_n);

    // If denominator is near 0, the ray is parallel to the plane
    if (std::abs(denom) > 1e-6f) {
        float t = (plane_p - ray_o).Dot(plane_n) / denom;

        // We only care about intersections in front of the ray (t >= 0)
        if (t >= 0.0f) {
            JPH::Vec3 hit_point = ray_o + ray_d * t;
            *out_t              = t;
            out_p[0]            = hit_point.GetX();
            out_p[1]            = hit_point.GetY();
            out_p[2]            = hit_point.GetZ();
            return 1;
        }
    }

    return 0;
}

void culverin_math_quat_get_axis_angle(const float *__restrict in_q, float *__restrict out_axis,
                                       float *__restrict out_angle) {
    JPH::Quat q(in_q[0], in_q[1], in_q[2], in_q[3]);

    JPH::Vec3 axis;
    float angle;
    q.GetAxisAngle(axis, angle);

    out_axis[0] = axis.GetX();
    out_axis[1] = axis.GetY();
    out_axis[2] = axis.GetZ();
    *out_angle  = angle;
}

void culverin_math_quat_from_axis_angle(const float *__restrict axis, float angle,
                                        float *__restrict out) {
    JPH::Vec3 v(axis[0], axis[1], axis[2]);

    // Ensure axis is normalized as JPH::Quat::sRotation asserts this
    float len_sq = v.LengthSq();
    if (len_sq > 1e-12f) {
        v = v / std::sqrt(len_sq);
    } else {
        // Default to Y-axis if provided vector is zero
        v = JPH::Vec3::sAxisY();
    }

    JPH::Quat q = JPH::Quat::sRotation(v, angle);
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_vec3_distance_batch(const float *__restrict a, const float *__restrict b,
                                       size_t count, float *__restrict out) {
    for (size_t i = 0; i < count; ++i) {
        size_t idx = i * 3;
        JPH::Vec3 vA(a[idx], a[idx + 1], a[idx + 2]);
        JPH::Vec3 vB(b[idx], b[idx + 1], b[idx + 2]);

        // Euclidean distance
        out[i] = (vA - vB).Length();
    }
}

void culverin_math_vec3_normalize(const float *__restrict v, float *__restrict out) {
    JPH::Vec3 vec(v[0], v[1], v[2]);
    
    float len_sq = vec.LengthSq();
    if (len_sq > 1e-12f) {
        // Jolt's Normalized() uses optimized reciprocal square root
        JPH::Vec3 res = vec.Normalized();
        out[0] = res.GetX();
        out[1] = res.GetY();
        out[2] = res.GetZ();
    } else {
        // Return zero vector for degenerate inputs
        out[0] = out[1] = out[2] = 0.0f;
    }
}

void culverin_math_mat44_get_translation(const float *__restrict in_mat, float *__restrict out_vec) {
    // Load 4x4 matrix from float buffer
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(in_mat));
    
    // Extract translation component (Column 3)
    JPH::Vec3 t = m.GetTranslation();
    
    out_vec[0] = t.GetX();
    out_vec[1] = t.GetY();
    out_vec[2] = t.GetZ();
}

void culverin_math_mat44_get_rotation(const float *__restrict in_mat, float *__restrict out_quat) {
    // Load 4x4 matrix
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(in_mat));
    
    // Extract rotation as a Quaternion
    JPH::Quat q = m.GetQuaternion();
    
    // Store as [x, y, z, w]
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out_quat));
}

void culverin_math_mat44_identity(float *__restrict out) {
    // Jolt's static identity matrix
    JPH::Mat44::sIdentity().StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_vec3_reflect(const float *__restrict v, const float *__restrict n, float *__restrict out) {
    JPH::Vec3 vec(v[0], v[1], v[2]);
    JPH::Vec3 norm(n[0], n[1], n[2]);
    
    // Formula: v - 2 * dot(v, n) * n
    JPH::Vec3 res = vec - 2.0f * vec.Dot(norm) * norm;
    
    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

float culverin_math_vec3_distance(const float *__restrict v1, const float *__restrict v2) {
    JPH::Vec3 a(v1[0], v1[1], v1[2]);
    JPH::Vec3 b(v2[0], v2[1], v2[2]);
    
    // Euclidean Distance: Length of the difference vector
    return (a - b).Length();
}

void culverin_math_quat_rotate_vec3_inverse(const float *__restrict q, const float *__restrict v, float *__restrict out) {
    JPH::Quat rotation(q[0], q[1], q[2], q[3]);
    JPH::Vec3 point(v[0], v[1], v[2]);
    
    // Rotate the vector by the inverse (conjugate) of the quaternion
    JPH::Vec3 res = rotation.InverseRotate(point);
    
    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

} // extern "C"
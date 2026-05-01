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
#include "CulverinCPP"
// clang-format on

extern "C" {

// Internal math helper for C, Python doesn't need this
void culverin_compute_interpolation_loop(const PosStride *CPH_RESTRICT curr_p,
                                         const PosStride *CPH_RESTRICT prev_p,
                                         const AuxStride *CPH_RESTRICT curr_r,
                                         const AuxStride *CPH_RESTRICT prev_r, CPH::Float32 alpha,
                                         CPH::Float32 *CPH_RESTRICT out, CPH::SizeType count) {
    const auto d_alpha         = static_cast<CPH::Float64>(alpha);
    const CPH::Float32 f_alpha = alpha;

    CULV_UNROLL_LOOP(4)
    for (CPH::SizeType i = 0; i < count; ++i) {
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
        CPH::Float32 dot = v1.Dot(v2);

        // Shortest path hemisphere check
        JPH::Quat q2_short = (dot < 0.0F) ? -q2 : q2;

        // NLerp: q1 + (q2 - q1) * alpha, followed by normalization
        JPH::Quat q_interp = (q1 + (q2_short - q1) * f_alpha).Normalized();

        // 3. Store to Output Buffer
        // Store Position (Double -> Float conversion happens here)
        out[base_idx + 0] = static_cast<CPH::Float32>(p_interp.GetX());
        out[base_idx + 1] = static_cast<CPH::Float32>(p_interp.GetY());
        out[base_idx + 2] = static_cast<CPH::Float32>(p_interp.GetZ());

        // Store Rotation (4 floats starting at index 3)
        // Cast the output pointer to Float4* to satisfy Jolt's Store method
        q_interp.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(&out[base_idx + 3]));
    }
}

/**
 * @brief SIMD Interpolation for Character Transforms.
 * start_p is JPH_Real (potentially CPH::Float64), start_r is CPH::Float32.
 */
[[gnu::hot, gnu::nonnull]]
void culverin_math_interpolate_character_transform(
    const PosStride *CPH_RESTRICT start_p, const AuxStride *CPH_RESTRICT start_r,
    const JPH_RVec3 *CPH_RESTRICT end_p, const JPH_Quat *CPH_RESTRICT end_r,
    const CPH::Float32 alpha, CPH::Float32 *CPH_RESTRICT out_p, CPH::Float32 *CPH_RESTRICT out_r) {
    using namespace JPH;

    // 1. POSITION LERP
    // SAFETY: Explicitly construct from components.
    // This prevents the Jolt SIMD engine from trying to
    // perform a 32-byte (4-CPH::Float64) load on a 24-byte (3-CPH::Float64) C-struct.
    const RVec3 p1(static_cast<Real>(start_p->x), static_cast<Real>(start_p->y),
                   static_cast<Real>(start_p->z));

    const RVec3 p2(static_cast<Real>(end_p->x), static_cast<Real>(end_p->y),
                   static_cast<Real>(end_p->z));

    const auto p_res = p1 + (p2 - p1) * static_cast<Real>(alpha);

    // 2. ROTATION NLERP
    // start_r is AuxStride (CPH::Float32[4], 16 bytes). sLoadFloat4 is safe here.
    const auto v1 = Vec4::sLoadFloat4(reinterpret_cast<const Float4 *>(start_r));

    // end_r is JPH_Quat (CPH::Float32[4], 16 bytes).
    // Construct explicitly to avoid any potential AVX-512 over-reads.
    const Quat q1(v1);
    const Quat q2(end_r->x, end_r->y, end_r->z, end_r->w);

    const auto v2          = q2.mValue;
    const CPH::Float32 dot = v1.Dot(v2);

    // Shortest path hemisphere check
    const Quat q2_shortest = (dot < 0.0F) ? -q2 : q2;

    // NLerp: (q1 + (q2 - q1) * alpha).Normalized()
    const Quat q_res = (q1 + (q2_shortest - q1) * alpha).Normalized();

    // 3. STORE RESULTS
    // Cast back to CPH::Float32 for the renderer
    out_p[0] = static_cast<CPH::Float32>(p_res.GetX());
    out_p[1] = static_cast<CPH::Float32>(p_res.GetY());
    out_p[2] = static_cast<CPH::Float32>(p_res.GetZ());

    // out_r is CPH::Float32[4] (16 bytes). StoreFloat4 is safe.
    q_res.mValue.StoreFloat4(reinterpret_cast<Float4 *>(out_r));
}

// Internal helpers for Python

// -----------------------------------------------------------------------------
// Projection Matrices (Standard Column-Major)
// -----------------------------------------------------------------------------
void culverin_math_get_perspective(CPH::Float32 fovy_rad, CPH::Float32 aspect, CPH::Float32 near_p,
                                   CPH::Float32 far_p, CPH::Float32 *CPH_RESTRICT out) {
    CPH::Float32 f         = 1.0F / std::tan(fovy_rad * 0.5F);
    CPH::Float32 range_inv = 1.0F / (near_p - far_p);

    JPH::Mat44 m;
    // Column 0
    m.SetColumn4(0, JPH::Vec4(f / aspect, 0.0F, 0.0F, 0.0F));
    // Column 1
    m.SetColumn4(1, JPH::Vec4(0.0F, f, 0.0F, 0.0F));
    // Column 2: Contains the Z-range mapping and the W-divider (-1)
    m.SetColumn4(2, JPH::Vec4(0.0F, 0.0F, (far_p + near_p) * range_inv, -1.0F));
    // Column 3: Contains the Z-precision offset
    m.SetColumn4(3, JPH::Vec4(0.0F, 0.0F, (2.0F * far_p * near_p) * range_inv, 0.0F));

    // Jolt's StoreFloat4x4 writes Col0, Col1, Col2, Col3 (Standard Column-Major)
    m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_get_ortho(CPH::Float32 left, CPH::Float32 right, CPH::Float32 bottom,
                             CPH::Float32 top, CPH::Float32 near_p, CPH::Float32 far_p,
                             CPH::Float32 *CPH_RESTRICT out) {
    CPH::Float32 r_l = 1.0F / (right - left);
    CPH::Float32 t_b = 1.0F / (top - bottom);
    CPH::Float32 f_n = 1.0F / (far_p - near_p);

    JPH::Mat44 m;
    m.SetColumn4(0, JPH::Vec4(2.0F * r_l, 0.0F, 0.0F, 0.0F));
    m.SetColumn4(1, JPH::Vec4(0.0F, 2.0F * t_b, 0.0F, 0.0F));
    m.SetColumn4(2, JPH::Vec4(0.0F, 0.0F, -2.0F * f_n, 0.0F));
    m.SetColumn4(
        3, JPH::Vec4(-(right + left) * r_l, -(top + bottom) * t_b, -(far_p + near_p) * f_n, 1.0F));

    m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

// -----------------------------------------------------------------------------
// View Matrix
// -----------------------------------------------------------------------------
void culverin_math_get_look_at(const CPH::Float32 *CPH_RESTRICT eye,
                               const CPH::Float32 *CPH_RESTRICT target,
                               const CPH::Float32 *CPH_RESTRICT up,
                               CPH::Float32 *CPH_RESTRICT out) {
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
void culverin_math_get_trs(const CPH::Float32 *CPH_RESTRICT pos,
                           const CPH::Float32 *CPH_RESTRICT rot_q,
                           const CPH::Float32 *CPH_RESTRICT scale, CPH::Float32 *CPH_RESTRICT out) {
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

void culverin_math_get_trs_batch(CPH::SizeType count, const CPH::Float32 *CPH_RESTRICT pos,
                                 const CPH::Float32 *CPH_RESTRICT rot_q,
                                 const CPH::Float32 *CPH_RESTRICT scale,
                                 CPH::Float32 *CPH_RESTRICT out) {
    for (CPH::SizeType i = 0; i < count; ++i) {
        const float *r = &rot_q[i * 4];
        JPH::Quat q(r[0], r[1], r[2], r[3]);

        JPH::Mat44 m = JPH::Mat44::sRotation(q);
        m.SetColumn4(0, m.GetColumn4(0) * scale[i * 3 + 0]);
        m.SetColumn4(1, m.GetColumn4(1) * scale[i * 3 + 1]);
        m.SetColumn4(2, m.GetColumn4(2) * scale[i * 3 + 2]);
        m.SetTranslation(JPH::Vec3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]));

        // FIX: Remove .Transposed()
        m.StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(&out[i * 16]));
    }
}

void culverin_math_mat44_inverse(const CPH::Float32 *CPH_RESTRICT in,
                                 CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(in));
    // Jolt's Inversed() is highly optimized SIMD
    m.Inversed().StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_mat44_mul(const CPH::Float32 *CPH_RESTRICT a, const CPH::Float32 *CPH_RESTRICT b,
                             CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 ma = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(a));
    JPH::Mat44 mb = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(b));
    (ma * mb).StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

// Computes: Out[i] = Single * Batch[i]
// Use case: MVP_Batch = ViewProj * Model_Batch
void culverin_math_mat44_mul_batch(const CPH::Float32 *CPH_RESTRICT single_mat,
                                   const CPH::Float32 *CPH_RESTRICT batch_mats, CPH::SizeType count,
                                   CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 s = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(single_mat));
    for (CPH::SizeType i = 0; i < count; ++i) {
        JPH::Mat44 b =
            JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(&batch_mats[i * 16]));
        (s * b).StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(&out[i * 16]));
    }
}

void culverin_math_transform_vec3(const CPH::Float32 *CPH_RESTRICT mat,
                                  const CPH::Float32 *CPH_RESTRICT vec,
                                  CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mat));
    JPH::Vec3 v(vec[0], vec[1], vec[2]);
    JPH::Vec3 res = m * v;

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

// Transforms N Vec3s by 1 Matrix
void culverin_math_transform_vec3_batch(const CPH::Float32 *CPH_RESTRICT mat,
                                        const CPH::Float32 *CPH_RESTRICT vecs, CPH::SizeType count,
                                        CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mat));
    for (CPH::SizeType i = 0; i < count; ++i) {
        JPH::Vec3 v(vecs[i * 3], vecs[i * 3 + 1], vecs[i * 3 + 2]);
        JPH::Vec3 res  = m * v;
        out[i * 3]     = res.GetX();
        out[i * 3 + 1] = res.GetY();
        out[i * 3 + 2] = res.GetZ();
    }
}

int culverin_math_cull_aabb(const CPH::Float32 *CPH_RESTRICT vp_mat,
                            const CPH::Float32 *CPH_RESTRICT aabb_min,
                            const CPH::Float32 *CPH_RESTRICT aabb_max) {
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
    for (auto plane : planes) {
        // Extract the normal (xyz) from the Vec4 plane.
        // In Jolt, the Vec3 constructor from a Vec4 is the standard way to drop W.
        JPH::Vec3 normal(plane);

        // Find the box corner furthest in the normal's direction
        JPH::Vec3 support = box.GetSupport(normal);

        // Signed Distance test: (Normal . Support) + Plane_D
        // We can do this efficiently by dotting the Plane Vec4 with (Support.xyz, 1.0)
        // This calculates Ax + By + Cz + D in one SIMD operation.
        CPH::Float32 dist = plane.Dot(JPH::Vec4(support, 1.0F));

        if (dist < 0.0F) {
            return 0; // Fully outside this plane -> Culled
        }
    }

    return 1; // Visible or Intersecting
}

void culverin_math_cull_aabb_batch(const CPH::Float32 *CPH_RESTRICT vp_mat,
                                   const CPH::Float32 *CPH_RESTRICT aabb_data, CPH::SizeType count,
                                   uint8_t *CPH_RESTRICT out_visibility) {
    JPH::Mat44 m  = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(vp_mat));
    JPH::Mat44 mt = m.Transposed();
    JPH::Vec4 r0  = mt.GetColumn4(0);
    JPH::Vec4 r1  = mt.GetColumn4(1);
    JPH::Vec4 r2  = mt.GetColumn4(2);
    JPH::Vec4 r3  = mt.GetColumn4(3);

    JPH::Vec4 planes[6] = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2};

    for (CPH::SizeType i = 0; i < count; ++i) {
        const CPH::Float32 *b = &aabb_data[i * 6];
        JPH::AABox box(JPH::Vec3(b[0], b[1], b[2]), JPH::Vec3(b[3], b[4], b[5]));

        bool visible = true;
        for (auto plane : planes) {
            if (plane.Dot(JPH::Vec4(box.GetSupport(JPH::Vec3(plane)), 1.0F)) < 0.0F) {
                visible = false;
                break;
            }
        }
        out_visibility[i] = visible ? 1 : 0;
    }
}

void culverin_math_vec3_normalize_batch(const CPH::Float32 *CPH_RESTRICT in, CPH::SizeType count,
                                        CPH::Float32 *CPH_RESTRICT out) {
    for (CPH::SizeType i = 0; i < count; ++i) {
        // Load from CPH::Float32[3] into SIMD register
        JPH::Vec3 v(in[i * 3 + 0], in[i * 3 + 1], in[i * 3 + 2]);

        CPH::Float32 len_sq = v.LengthSq();
        if (len_sq > 1e-12f) {
            // Jolt uses Reciprocal Square Root (RSQRT) for normalization
            v = v / JPH::Sqrt(len_sq);
        } else {
            v = JPH::Vec3::sZero();
        }

        // Store back to CPH::Float32[3]
        out[i * 3 + 0] = v.GetX();
        out[i * 3 + 1] = v.GetY();
        out[i * 3 + 2] = v.GetZ();
    }
}

void culverin_math_quat_from_euler(CPH::Float32 x, CPH::Float32 y, CPH::Float32 z,
                                   CPH::Float32 *CPH_RESTRICT out) {
    // Jolt uses sEulerAngles for (x, y, z) radian inputs
    // Order: Rotation around Y, then X, then Z
    JPH::Quat q = JPH::Quat::sEulerAngles(JPH::Vec3(x, y, z));

    // Store back to out[4] as (x, y, z, w)
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_quat_to_euler(const CPH::Float32 *CPH_RESTRICT in_q,
                                 CPH::Float32 *CPH_RESTRICT out_euler) {
    // Load components into Jolt Quat
    JPH::Quat q(in_q[0], in_q[1], in_q[2], in_q[3]);

    // GetEulerAngles returns Vec3(x, y, z) in radians
    JPH::Vec3 euler = q.GetEulerAngles();

    out_euler[0] = euler.GetX();
    out_euler[1] = euler.GetY();
    out_euler[2] = euler.GetZ();
}

void culverin_math_quat_slerp(const CPH::Float32 *CPH_RESTRICT q1,
                              const CPH::Float32 *CPH_RESTRICT q2, CPH::Float32 t,
                              CPH::Float32 *CPH_RESTRICT out) {
    // Jolt Quat stores as [x, y, z, w]
    JPH::Quat a(q1[0], q1[1], q1[2], q1[3]);
    JPH::Quat b(q2[0], q2[1], q2[2], q2[3]);

    // Corrected: Jolt uses uppercase SLERP
    JPH::Quat res = a.SLERP(b, t);

    res.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_quat_mul(const CPH::Float32 *CPH_RESTRICT a, const CPH::Float32 *CPH_RESTRICT b,
                            CPH::Float32 *CPH_RESTRICT out) {
    JPH::Quat qa(a[0], a[1], a[2], a[3]);
    JPH::Quat qb(b[0], b[1], b[2], b[3]);

    // Combine rotations: Apply b then a
    JPH::Quat res = qa * qb;

    res.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_vec3_lerp_batch(const CPH::Float32 *CPH_RESTRICT a,
                                   const CPH::Float32 *CPH_RESTRICT b, CPH::Float32 alpha,
                                   CPH::SizeType count, CPH::Float32 *CPH_RESTRICT out) {
    // Optimization: If alpha is 0 or 1, we can just memcpy,
    // but usually, the caller handles that.
    // We proceed with the standard SIMD-accelerated loop.
    for (CPH::SizeType i = 0; i < count; ++i) {
        CPH::SizeType idx = i * 3;

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

void culverin_math_quat_rotate_vec3(const CPH::Float32 *CPH_RESTRICT q,
                                    const CPH::Float32 *CPH_RESTRICT v,
                                    CPH::Float32 *CPH_RESTRICT out) {
    JPH::Quat rotation(q[0], q[1], q[2], q[3]);
    JPH::Vec3 point(v[0], v[1], v[2]);

    // Rotate the vector
    JPH::Vec3 res = rotation * point;

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

void culverin_math_quat_rotate_vec3_batch(const CPH::Float32 *CPH_RESTRICT q,
                                          const CPH::Float32 *CPH_RESTRICT vecs,
                                          CPH::SizeType count, CPH::Float32 *CPH_RESTRICT out) {
    // Load the rotation once
    JPH::Quat rotation(q[0], q[1], q[2], q[3]);

    for (CPH::SizeType i = 0; i < count; ++i) {
        CPH::SizeType idx = i * 3;
        JPH::Vec3 v(vecs[idx], vecs[idx + 1], vecs[idx + 2]);

        // Rotate vector
        JPH::Vec3 res = rotation * v;

        out[idx]     = res.GetX();
        out[idx + 1] = res.GetY();
        out[idx + 2] = res.GetZ();
    }
}

void culverin_math_quat_inverse(const CPH::Float32 *CPH_RESTRICT q,
                                CPH::Float32 *CPH_RESTRICT out) {
    JPH::Quat in_q(q[0], q[1], q[2], q[3]);

    // Calculate the inverse
    JPH::Quat inv = in_q.Inversed();

    inv.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_project(const CPH::Float32 *CPH_RESTRICT v, const CPH::Float32 *CPH_RESTRICT mvp,
                           const int *CPH_RESTRICT viewport, CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mvp));
    JPH::Vec3 world_pos(v[0], v[1], v[2]);

    // Transform to Clip Space
    JPH::Vec4 clip = m * JPH::Vec4(world_pos, 1.0F);

    CPH::Float32 w = clip.GetW();
    if (std::abs(w) > 1e-6f) {
        // NDC Space (-1 to 1)
        CPH::Float32 inv_w = 1.0F / w;
        CPH::Float32 ndc_x = clip.GetX() * inv_w;
        CPH::Float32 ndc_y = clip.GetY() * inv_w;
        CPH::Float32 ndc_z = clip.GetZ() * inv_w;

        // Screen Space (Pixels)
        out[0] = viewport[0] + (viewport[2] * (ndc_x + 1.0F) * 0.5F);
        out[1] =
            viewport[1] + (viewport[3] * (1.0F - (ndc_y + 1.0F) * 0.5F)); // Flip Y for Screen space
        out[2] = ndc_z; // Z is usually preserved for depth sorting
    } else {
        out[0] = out[1] = out[2] = 0.0F;
    }
}

void culverin_math_unproject(const CPH::Float32 *CPH_RESTRICT v,
                             const CPH::Float32 *CPH_RESTRICT mvp, const int *CPH_RESTRICT viewport,
                             CPH::Float32 *CPH_RESTRICT out) {
    JPH::Mat44 m     = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(mvp));
    JPH::Mat44 inv_m = m.Inversed();

    CPH::Float32 ndc_x =
        (v[0] - (CPH::Float32)viewport[0]) / (CPH::Float32)viewport[2] * 2.0F - 1.0F;
    CPH::Float32 ndc_y =
        1.0F - (v[1] - (CPH::Float32)viewport[1]) / (CPH::Float32)viewport[3] * 2.0F;

    // FIX: Maintain symmetry with project()
    CPH::Float32 ndc_z = v[2];

    JPH::Vec4 world_pos = inv_m * JPH::Vec4(ndc_x, ndc_y, ndc_z, 1.0F);

    CPH::Float32 w = world_pos.GetW();
    if (std::abs(w) > 1e-6f) {
        CPH::Float32 inv_w = 1.0F / w;
        out[0]             = world_pos.GetX() * inv_w;
        out[1]             = world_pos.GetY() * inv_w;
        out[2]             = world_pos.GetZ() * inv_w;
    } else {
        out[0] = out[1] = out[2] = 0.0F;
    }
}

void culverin_math_quat_from_to(const CPH::Float32 *CPH_RESTRICT v1,
                                const CPH::Float32 *CPH_RESTRICT v2,
                                CPH::Float32 *CPH_RESTRICT out) {
    JPH::Vec3 from(v1[0], v1[1], v1[2]);
    JPH::Vec3 to(v2[0], v2[1], v2[2]);

    // Create quaternion that rotates from 'from' to 'to' along the shortest path
    JPH::Quat q = JPH::Quat::sFromTo(from, to);

    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

CPH::Float32 culverin_math_vec3_dot(const CPH::Float32 *CPH_RESTRICT v1,
                                    const CPH::Float32 *CPH_RESTRICT v2) {
    JPH::Vec3 a(v1[0], v1[1], v1[2]);
    JPH::Vec3 b(v2[0], v2[1], v2[2]);

    return a.Dot(b);
}

void culverin_math_vec3_cross(const CPH::Float32 *CPH_RESTRICT v1,
                              const CPH::Float32 *CPH_RESTRICT v2, CPH::Float32 *CPH_RESTRICT out) {
    JPH::Vec3 a(v1[0], v1[1], v1[2]);
    JPH::Vec3 b(v2[0], v2[1], v2[2]);

    // Calculate perpendicular vector
    JPH::Vec3 res = a.Cross(b);

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

int culverin_math_intersect_ray_plane(const CPH::Float32 *CPH_RESTRICT ro,
                                      const CPH::Float32 *CPH_RESTRICT rd,
                                      const CPH::Float32 *CPH_RESTRICT po,
                                      const CPH::Float32 *CPH_RESTRICT pn,
                                      CPH::Float32 *CPH_RESTRICT out_t,
                                      CPH::Float32 *CPH_RESTRICT out_p) {
    JPH::Vec3 ray_o(ro[0], ro[1], ro[2]);
    JPH::Vec3 ray_d(rd[0], rd[1], rd[2]);
    JPH::Vec3 plane_p(po[0], po[1], po[2]);
    JPH::Vec3 plane_n(pn[0], pn[1], pn[2]);

    CPH::Float32 denom = ray_d.Dot(plane_n);

    // If denominator is near 0, the ray is parallel to the plane
    if (std::abs(denom) > 1e-6f) {
        CPH::Float32 t = (plane_p - ray_o).Dot(plane_n) / denom;

        // We only care about intersections in front of the ray (t >= 0)
        if (t >= 0.0F) {
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

void culverin_math_quat_get_axis_angle(const CPH::Float32 *CPH_RESTRICT in_q,
                                       CPH::Float32 *CPH_RESTRICT out_axis,
                                       CPH::Float32 *CPH_RESTRICT out_angle) {
    JPH::Quat q(in_q[0], in_q[1], in_q[2], in_q[3]);

    JPH::Vec3 axis;
    CPH::Float32 angle;
    q.GetAxisAngle(axis, angle);

    out_axis[0] = axis.GetX();
    out_axis[1] = axis.GetY();
    out_axis[2] = axis.GetZ();
    *out_angle  = angle;
}

void culverin_math_quat_from_axis_angle(const CPH::Float32 *CPH_RESTRICT axis, CPH::Float32 angle,
                                        CPH::Float32 *CPH_RESTRICT out) {
    JPH::Vec3 v(axis[0], axis[1], axis[2]);

    // Ensure axis is normalized as JPH::Quat::sRotation asserts this
    CPH::Float32 len_sq = v.LengthSq();
    if (len_sq > 1e-12f) {
        v = v / JPH::Sqrt(len_sq);
    } else {
        // Default to Y-axis if provided vector is zero
        v = JPH::Vec3::sAxisY();
    }

    JPH::Quat q = JPH::Quat::sRotation(v, angle);
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_vec3_distance_batch(const CPH::Float32 *CPH_RESTRICT a,
                                       const CPH::Float32 *CPH_RESTRICT b, CPH::SizeType count,
                                       CPH::Float32 *CPH_RESTRICT out) {
    for (CPH::SizeType i = 0; i < count; ++i) {
        CPH::SizeType idx = i * 3;
        JPH::Vec3 vA(a[idx], a[idx + 1], a[idx + 2]);
        JPH::Vec3 vB(b[idx], b[idx + 1], b[idx + 2]);

        // Euclidean distance
        out[i] = (vA - vB).Length();
    }
}

void culverin_math_vec3_normalize(const CPH::Float32 *CPH_RESTRICT v,
                                  CPH::Float32 *CPH_RESTRICT out) {
    JPH::Vec3 vec(v[0], v[1], v[2]);

    CPH::Float32 len_sq = vec.LengthSq();
    if (len_sq > 1e-12f) {
        // Jolt's Normalized() uses optimized reciprocal square root
        JPH::Vec3 res = vec.Normalized();
        out[0]        = res.GetX();
        out[1]        = res.GetY();
        out[2]        = res.GetZ();
    } else {
        // Return zero vector for degenerate inputs
        out[0] = out[1] = out[2] = 0.0F;
    }
}

void culverin_math_mat44_get_translation(const CPH::Float32 *CPH_RESTRICT in_mat,
                                         CPH::Float32 *CPH_RESTRICT out_vec) {
    // Load 4x4 matrix from CPH::Float32 buffer
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(in_mat));

    // Extract translation component (Column 3)
    JPH::Vec3 t = m.GetTranslation();

    out_vec[0] = t.GetX();
    out_vec[1] = t.GetY();
    out_vec[2] = t.GetZ();
}

void culverin_math_mat44_get_rotation(const CPH::Float32 *CPH_RESTRICT in_mat,
                                      CPH::Float32 *CPH_RESTRICT out_quat) {
    // Load 4x4 matrix
    JPH::Mat44 m = JPH::Mat44::sLoadFloat4x4(reinterpret_cast<const JPH::Float4 *>(in_mat));

    // Extract rotation as a Quaternion
    JPH::Quat q = m.GetQuaternion();

    // Store as [x, y, z, w]
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out_quat));
}

void culverin_math_mat44_identity(CPH::Float32 *CPH_RESTRICT out) {
    // Jolt's static identity matrix
    JPH::Mat44::sIdentity().StoreFloat4x4(reinterpret_cast<JPH::Float4 *>(out));
}

void culverin_math_vec3_reflect(const CPH::Float32 *CPH_RESTRICT v,
                                const CPH::Float32 *CPH_RESTRICT n,
                                CPH::Float32 *CPH_RESTRICT out) {
    JPH::Vec3 vec(v[0], v[1], v[2]);
    JPH::Vec3 norm(n[0], n[1], n[2]);

    // Formula: v - 2 * dot(v, n) * n
    JPH::Vec3 res = vec - 2.0F * vec.Dot(norm) * norm;

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

CPH::Float32 culverin_math_vec3_distance(const CPH::Float32 *CPH_RESTRICT v1,
                                         const CPH::Float32 *CPH_RESTRICT v2) {
    JPH::Vec3 a(v1[0], v1[1], v1[2]);
    JPH::Vec3 b(v2[0], v2[1], v2[2]);

    // Euclidean Distance: Length of the difference vector
    return (a - b).Length();
}

void culverin_math_quat_rotate_vec3_inverse(const CPH::Float32 *CPH_RESTRICT q,
                                            const CPH::Float32 *CPH_RESTRICT v,
                                            CPH::Float32 *CPH_RESTRICT out) {
    JPH::Quat rotation(q[0], q[1], q[2], q[3]);
    JPH::Vec3 point(v[0], v[1], v[2]);

    // Rotate the vector by the inverse (conjugate) of the quaternion
    JPH::Vec3 res = rotation.InverseRotate(point);

    out[0] = res.GetX();
    out[1] = res.GetY();
    out[2] = res.GetZ();
}

void culverin_math_euler_to_quat(const CPH::Float32 *CPH_RESTRICT euler,
                                 CPH::Float32 *CPH_RESTRICT out_q) {
    // Jolt implementation: Euler XYZ -> Quat
    JPH::Quat q = JPH::Quat::sEulerAngles(JPH::Vec3(euler[0], euler[1], euler[2]));
    q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(out_q));
}

void culverin_math_euler_to_quat_batch(const CPH::Float32 *CPH_RESTRICT eulers, CPH::SizeType count,
                                       CPH::Float32 *CPH_RESTRICT out_qs) {
    for (CPH::SizeType i = 0; i < count; ++i) {
        const CPH::Float32 *e = &eulers[i * 3];
        JPH::Quat q           = JPH::Quat::sEulerAngles(JPH::Vec3(e[0], e[1], e[2]));

        // Store into the output buffer (16 bytes per quaternion)
        q.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(&out_qs[i * 4]));
    }
}

} // extern "C"
#include <Jolt/Jolt.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Math/Vec4.h>
#include <Jolt/Math/Quat.h>
#include <cmath>
#include <Python.h>
#include "culverin_types.h" // Needed for PosStride/AuxStride

extern "C" {

// Internal math helper for C, Python doesn't need this
void culverin_compute_interpolation_loop(
    const PosStride* __restrict curr_p,
    const PosStride* __restrict prev_p,
    const AuxStride* __restrict curr_r,
    const AuxStride* __restrict prev_r,
    float alpha,
    float* __restrict out,
    size_t count) 
{
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
        JPH::Vec4 v1 = JPH::Vec4::sLoadFloat4(reinterpret_cast<const JPH::Float4*>(&prev_r[i]));
        JPH::Vec4 v2 = JPH::Vec4::sLoadFloat4(reinterpret_cast<const JPH::Float4*>(&curr_r[i]));

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
        q_interp.GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4*>(&out[base_idx + 3]));
    }
}

// Internal helpers for Python

// -----------------------------------------------------------------------------
// Projection Matrices
// -----------------------------------------------------------------------------
void culverin_math_get_perspective(float fovy_rad, float aspect, float near_p, float far_p, float* __restrict out) {
    float f = 1.0f / std::tan(fovy_rad * 0.5f);
    JPH::Mat44 m;
    
    // TRICK: Jolt is Column-Major, but Python/OpenGL expects Row-Major.
    // By packing our ROWS into Jolt's COLUMNS, StoreFloat4x4 outputs a 
    // perfect C-contiguous Row-Major array directly into memory.
    m.SetColumn4(0, JPH::Vec4(f / aspect, 0.0f, 0.0f, 0.0f));
    m.SetColumn4(1, JPH::Vec4(0.0f, f, 0.0f, 0.0f));
    m.SetColumn4(2, JPH::Vec4(0.0f, 0.0f, (far_p + near_p) / (near_p - far_p), (2.0f * far_p * near_p) / (near_p - far_p)));
    m.SetColumn4(3, JPH::Vec4(0.0f, 0.0f, -1.0f, 0.0f));
    
    m.StoreFloat4x4(reinterpret_cast<JPH::Float4*>(out));
}

void culverin_math_get_ortho(float left, float right, float bottom, float top, float near_p, float far_p, float* __restrict out) {
    JPH::Mat44 m;
    m.SetColumn4(0, JPH::Vec4(2.0f / (right - left), 0.0f, 0.0f, -(right + left) / (right - left)));
    m.SetColumn4(1, JPH::Vec4(0.0f, 2.0f / (top - bottom), 0.0f, -(top + bottom) / (top - bottom)));
    m.SetColumn4(2, JPH::Vec4(0.0f, 0.0f, -2.0f / (far_p - near_p), -(far_p + near_p) / (far_p - near_p)));
    m.SetColumn4(3, JPH::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    
    m.StoreFloat4x4(reinterpret_cast<JPH::Float4*>(out));
}

// -----------------------------------------------------------------------------
// View Matrix
// -----------------------------------------------------------------------------
void culverin_math_get_look_at(const float* __restrict eye, const float* __restrict target, const float* __restrict up, float* __restrict out) {
    JPH::Vec3 e(eye[0], eye[1], eye[2]);
    JPH::Vec3 t(target[0], target[1], target[2]);
    JPH::Vec3 u(up[0], up[1], up[2]);

    JPH::Vec3 f = (t - e).Normalized();
    JPH::Vec3 r = f.Cross(u).Normalized();
    JPH::Vec3 u_new = r.Cross(f);

    JPH::Mat44 m;
    m.SetColumn4(0, JPH::Vec4(r.GetX(), r.GetY(), r.GetZ(), -r.Dot(e)));
    m.SetColumn4(1, JPH::Vec4(u_new.GetX(), u_new.GetY(), u_new.GetZ(), -u_new.Dot(e)));
    m.SetColumn4(2, JPH::Vec4(-f.GetX(), -f.GetY(), -f.GetZ(), f.Dot(e)));
    m.SetColumn4(3, JPH::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    
    m.StoreFloat4x4(reinterpret_cast<JPH::Float4*>(out));
}

// -----------------------------------------------------------------------------
// Model Matrices
// -----------------------------------------------------------------------------
void culverin_math_get_trs(const float* __restrict pos, const float* __restrict rot_q, const float* __restrict scale, float* __restrict out) {
    JPH::Quat q(rot_q[0], rot_q[1], rot_q[2], rot_q[3]);
    
    // 1. Create SIMD rotation matrix
    JPH::Mat44 m = JPH::Mat44::sRotation(q);
    
    // 2. Apply scale to the axes directly
    m.SetColumn4(0, m.GetColumn4(0) * scale[0]);
    m.SetColumn4(1, m.GetColumn4(1) * scale[1]);
    m.SetColumn4(2, m.GetColumn4(2) * scale[2]);
    
    // 3. Set translation
    m.SetTranslation(JPH::Vec3(pos[0], pos[1], pos[2]));
    
    // 4. Jolt is Column-Major. GetTransposed() perfectly aligns the 
    // SIMD registers so StoreFloat4x4 writes it as Row-Major!
    m.Transposed().StoreFloat4x4(reinterpret_cast<JPH::Float4*>(out));
}

// -----------------------------------------------------------------------------
// MASSIVE ECS SCALE BATCHING
// -----------------------------------------------------------------------------
void culverin_math_get_trs_batch(
    size_t count, 
    const float* __restrict pos,     // Input: N * 3 floats
    const float* __restrict rot_q,   // Input: N * 4 floats
    const float* __restrict scale,   // Input: N * 3 floats
    float* __restrict out)           // Output: N * 16 floats
{
    // If you have 500 entities, doing this in a Python loop is slow.
    // Call this via the Python C-API to generate 500 model matrices in nanoseconds.
    for(size_t i = 0; i < count; ++i) {
        JPH::Quat q(rot_q[i*4+0], rot_q[i*4+1], rot_q[i*4+2], rot_q[i*4+3]);
        
        JPH::Mat44 m = JPH::Mat44::sRotation(q);
        m.SetColumn4(0, m.GetColumn4(0) * scale[i*3+0]);
        m.SetColumn4(1, m.GetColumn4(1) * scale[i*3+1]);
        m.SetColumn4(2, m.GetColumn4(2) * scale[i*3+2]);
        m.SetTranslation(JPH::Vec3(pos[i*3+0], pos[i*3+1], pos[i*3+2]));
        
        m.Transposed().StoreFloat4x4(reinterpret_cast<JPH::Float4*>(&out[i*16]));
    }
}

} // extern "C"
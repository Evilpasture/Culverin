#include <Jolt/Jolt.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Real.h>
#include "culverin_types.h"

extern "C" void culverin_compute_interpolation_loop(
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
    for (auto i = 0ll; i < count; ++i) {
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
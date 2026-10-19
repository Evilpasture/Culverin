#pragma once

#include "core/Types.hpp"
#include <joltc.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>


namespace CPH {

struct PhysicsWorld;

inline constexpr float kPhysicsQueryDefaultMaxDistance = 1000.0F;
inline constexpr std::size_t kPhysicsQueryInlineHitCapacity = 64;

enum PhysicsQueryStatus : int {
    PHYSICS_QUERY_STATUS_OK               = 0,
    PHYSICS_QUERY_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_QUERY_STATUS_INVALID_ARGUMENT = -2,
    PHYSICS_QUERY_STATUS_SHAPE_FAILURE    = -3,
};

// Hit list with small-buffer optimization: overlap queries construct one
// per call, and the common case fits in the 64 inline handles without a
// heap allocation.
struct PhysicsQueryHitBuffer {
    std::array<BodyHandle, kPhysicsQueryInlineHitCapacity> inline_hits{};
    BodyHandle *hits                    = inline_hits.data();
    std::size_t count                   = 0;
    std::size_t capacity                = kPhysicsQueryInlineHitCapacity;

    PhysicsQueryHitBuffer() = default;
    ~PhysicsQueryHitBuffer() {
        if (hits != inline_hits.data()) {
            delete[] hits;
        }
    }

    PhysicsQueryHitBuffer(const PhysicsQueryHitBuffer &)            = delete;
    PhysicsQueryHitBuffer &operator=(const PhysicsQueryHitBuffer &) = delete;

    [[nodiscard]] bool append(BodyHandle handle) noexcept {
        if (count >= capacity) {
            const std::size_t new_capacity = capacity * 2;
            if (new_capacity < capacity ||
                new_capacity > static_cast<std::size_t>(-1) / sizeof(BodyHandle)) {
                return false;
            }
            auto *new_hits = new (std::nothrow) BodyHandle[new_capacity];
            if (new_hits != nullptr) {
                for (std::size_t i = 0; i < count; ++i) {
                    new_hits[i] = hits[i];
                }
                if (hits != inline_hits.data()) {
                    delete[] hits;
                }
            }
            if (new_hits == nullptr) {
                return false;
            }
            hits     = new_hits;
            capacity = new_capacity;
        }
        hits[count++] = handle;
        return true;
    }
};

// Packed output for the Python raycast-batch byte buffer. Keep this wire
// layout stable: consumers decode each record as exactly 48 bytes.
struct
#ifndef _MSC_VER
    __attribute__((packed))
#endif
    RayCastBatchResult {
    std::uint64_t handle;
    float fraction;
    float nx, ny, nz;
    float px, py, pz;
    std::uint32_t subShapeID;
    std::uint32_t material_id;
    std::uint32_t _pad;
};

inline constexpr std::size_t kRaycastResultSize = 48;
static_assert(sizeof(RayCastBatchResult) == kRaycastResultSize);

struct PhysicsRaycastHit {
    BodyHandle handle;
    float fraction;
    JPH_Vec3 normal;
    bool has_hit;
};

struct PhysicsShapecastHit {
    BodyHandle handle;
    float fraction;
    JPH_Vec3 point;
    JPH_Vec3 normal;
    bool has_hit;
};

// Query lifetime: balances the active_queries registration performed by
// the query entry points (also used by native ops that register as queries).
void PhysicsWorldQueryEnd(PhysicsWorld *world);

PhysicsQueryStatus PhysicsWorldOverlapSphere(PhysicsWorld *world, JPH_RVec3 center,
                                                 float radius, PhysicsQueryHitBuffer *hits);
PhysicsQueryStatus PhysicsWorldOverlapAabb(PhysicsWorld *world, JPH_RVec3 min,
                                               JPH_RVec3 max, PhysicsQueryHitBuffer *hits);
PhysicsQueryStatus PhysicsWorldRaycast(PhysicsWorld *world, JPH_RVec3 origin,
                                           JPH_Vec3 direction, float max_distance,
                                           BodyHandle ignore_handle, PhysicsRaycastHit *hit);
PhysicsQueryStatus PhysicsWorldRaycastBatch(PhysicsWorld *world, const float *starts,
                                                const float *directions, std::size_t count,
                                                float max_distance, RayCastBatchResult *results);
PhysicsQueryStatus PhysicsWorldShapecast(PhysicsWorld *world, int shape_type,
                                              JPH_RVec3 position, JPH_Quat rotation,
                                              JPH_Vec3 sweep_direction, const float size[4],
                                              BodyHandle ignore_handle,
                                              PhysicsShapecastHit *hit);

} // namespace CPH

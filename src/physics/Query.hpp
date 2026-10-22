#pragma once

#include "core/Error.hpp"
#include "core/Types.hpp"
#include "jolt/Jolt.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <new>


namespace CPH {

inline constexpr float kPhysicsQueryDefaultMaxDistance = 1000.0F;
inline constexpr std::size_t kPhysicsQueryInlineHitCapacity = 64;

enum class QueryError : std::uint32_t {
    OutOfMemory     = 1,
    InvalidArgument = 2,
    ShapeFailure    = 3,
};

// Hit list with small-buffer optimization: overlap queries construct one
// per call, and the common case fits in the 64 inline handles without a
// heap allocation.
struct PhysicsQueryHitBuffer {
    std::array<BodyHandle, kPhysicsQueryInlineHitCapacity> inlineHits{};
    BodyHandle *hits                    = inlineHits.data();
    std::size_t count                   = 0;
    std::size_t capacity                = kPhysicsQueryInlineHitCapacity;

    PhysicsQueryHitBuffer() = default;
    ~PhysicsQueryHitBuffer() {
        if (hits != inlineHits.data()) {
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
                if (hits != inlineHits.data()) {
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
    std::uint32_t materialId;
    std::uint32_t pad;
};

inline constexpr std::size_t kRaycastResultSize = 48;
static_assert(sizeof(RayCastBatchResult) == kRaycastResultSize);

struct PhysicsRaycastHit {
    BodyHandle handle;
    float fraction;
    JPH::Vec3 normal;
    bool hasHit;
};

struct PhysicsShapecastHit {
    BodyHandle handle;
    float fraction;
    JPH::Vec3 point;
    JPH::Vec3 normal;
    bool hasHit;
};

} // namespace CPH

#pragma once

#include "core/Buffer.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Renderer/DebugRendererSimple.h>

#include <cstddef>
#include <cstdint>

// 16-byte strided vertex for easy GPU upload: [X, Y, Z, Color(u32)]

namespace CPH {

struct DebugVertex {
    float x, y, z;
    std::uint32_t color;
};

static_assert(sizeof(DebugVertex) == 16);

// Append-only vertex stream filled by Jolt debug-draw callbacks. Growth
// failure drops vertices silently: debug output must never fail a step.
struct DebugBuffer {
    Buffer<DebugVertex> data;
    std::size_t count = 0;

    void clear() noexcept { count = 0; }

    [[nodiscard]] bool ensure(std::size_t additional) noexcept {
        if (count + additional <= data.size()) {
            return true;
        }
        std::size_t new_capacity = data.size() == 0 ? 4096 : data.size() * 2;
        while (count + additional > new_capacity) {
            new_capacity *= 2;
        }
        return data.reallocate(new_capacity);
    }

    void push(float x, float y, float z, std::uint32_t color) noexcept {
        if (count < data.size()) {
            data[count++] = {x, y, z, color};
        }
    }
};

// One frame of debug geometry produced by PhysicsWorld::BuildDebugData.
struct DebugData {
    Buffer<DebugVertex> lines;
    Buffer<DebugVertex> triangles;
};

struct PhysicsWorld;

// Native debug renderer (Phase 2, step A3). Only the three draw calls Culverin
// batches are overridden; DebugRendererSimple supplies the triangle-batch /
// DrawGeometry machinery that BuildDebugData never reaches.
//
// Jolt treats DebugRenderer as a process-wide singleton by contract
// (JPH_ASSERT(sInstance == nullptr) in its constructor). Culverin keeps one
// renderer per world — exactly what the JoltC wrapper did before this change —
// so assert-enabled builds still see the assert when a second world is alive;
// release builds are unaffected. Resolving that (a shared renderer with a
// current-world pointer) is a separate design change, not part of this phase.
class WorldDebugRenderer final : public JPH::DebugRendererSimple {
public:
    explicit WorldDebugRenderer(PhysicsWorld *world) noexcept : world_(world) {}

    void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) override;
    void DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3,
                      JPH::ColorArg inColor, ECastShadow inCastShadow) override;
    void DrawText3D(JPH::RVec3Arg inPosition, const JPH::string_view &inString,
                    JPH::ColorArg inColor, float inHeight) override;

private:
    PhysicsWorld *world_;
};

} // namespace CPH

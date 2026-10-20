#pragma once

#include "core/Buffer.hpp"
#include "jolt/JoltConfig.hpp"
#include <joltc.h>

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

extern const JPH_DebugRenderer_Procs kDebugProcs;

} // namespace CPH

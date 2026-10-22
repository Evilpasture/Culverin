#pragma once

#include "core/Buffer.hpp"
#include "core/Types.hpp"

// Native shadow storage for a Jolt soft body's world-space vertices
// (PosStride per vertex). Owned storage: freed with the body slot.
// The legacy struct also carried normals/velocities pointers that were
// always null; vertex data is the only shadow soft bodies maintain.

namespace CPH {

struct SoftBodyShadow {
    Buffer<PosStride, kAvxAlignment> vertices;

    [[nodiscard]] std::uint32_t num_vertices() const noexcept {
        return static_cast<std::uint32_t>(vertices.size());
    }
};

} // namespace CPH

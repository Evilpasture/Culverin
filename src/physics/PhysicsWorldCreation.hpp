#pragma once

#include "core/Error.hpp"
#include "physics/PhysicsParams.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>


namespace CPH {

enum class WorldCreationError : std::uint32_t {
    BodySettingsFailed = 1,
    InvalidArgument    = 2,
    ShapeFailed        = 3,
    OutOfMemory        = 4,
    InvalidMeshIndex   = 5,
    RagdollFailed      = 6,
};

// First triangle (flat XYZ-index triples) that references a vertex at
// or past `vertex_count`, if any. The mesh binding reports the offending
// indices in its error message; PhysicsWorld::CreateMeshBody re-checks
// and reports WorldCreationError::InvalidMeshIndex.
[[nodiscard]] inline std::optional<std::array<std::uint32_t, 3>>
FindInvalidMeshTriangle(std::span<const std::uint32_t> indices, std::size_t vertex_count) {
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const std::uint32_t i1 = indices[i];
        const std::uint32_t i2 = indices[i + 1];
        const std::uint32_t i3 = indices[i + 2];
        if (i1 >= vertex_count || i2 >= vertex_count || i3 >= vertex_count) {
            return std::array<std::uint32_t, 3>{i1, i2, i3};
        }
    }
    return std::nullopt;
}

// Ragdoll settings factory (the skeleton/settings pair is the subject,
// not a world). Defined in PhysicsWorldCreation.cpp.
JPH_RagdollSettings *PhysicsWorldCreateRagdollSettings(JPH_Skeleton *skeleton);

} // namespace CPH

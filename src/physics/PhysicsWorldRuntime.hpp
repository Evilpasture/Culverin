#pragma once

// Error categories for the runtime operations declared as PhysicsWorld
// members in PhysicsWorldState.hpp, plus the standalone snapshot-header
// reader shared by the load path and the binding's error reporting.

#include "physics/PhysicsParams.hpp"

#include <cstdint>
#include <optional>
#include <span>


namespace CPH {

// Parses the fixed snapshot prefix; empty when the buffer is shorter
// than the header.
[[nodiscard]] std::optional<SnapshotHeader> ReadSnapshotHeader(std::span<const std::uint8_t> data);


enum class WorldRuntimeError : std::uint32_t {
    OutOfMemory     = 1,
    StaleHandle     = 2,
    PendingCreate   = 3,
    Overflow        = 4,
    InvalidArgument = 5,
    WrongBodyType   = 6,
    OutOfRange      = 7,
    MissingBuffer   = 8,
};

enum class WorldSnapshotError : std::uint32_t {
    OutOfMemory      = 1,
    InvalidArgument  = 2,
    Truncated        = 3,
    CapacityMismatch = 4,
    Overflow         = 5,
};

} // namespace CPH

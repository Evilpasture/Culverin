#pragma once

// Error category for the body-operation members declared in
// PhysicsWorldState.hpp.

#include <cstdint>


namespace CPH {

enum class BodyOperationError : std::uint32_t {
    OutOfMemory     = 1,
    StaleHandle     = 2,
    InvalidArgument = 3,
};

} // namespace CPH

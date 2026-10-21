#pragma once

// Constraint mutations that depend on the concrete constraint sub-type.
//
// Inspecting JPH::EConstraintSubType and downcasting to Hinge/Slider is physics,
// not binding glue: the Python layer used to do it because that is where the
// entry point happened to live. It lives here now, so a headless caller gets
// the same behaviour and the binding only parses arguments and forwards.

#include "core/Error.hpp"
#include "core/Handle.hpp"
#include <cstdint>
#include <expected>
#include <string_view>

namespace CPH {

struct PhysicsWorld;

enum class ConstraintError : std::uint8_t {
    // The handle's slot is out of range, its generation moved on, or the
    // constraint is not alive. The Python API reports this as
    // "Invalid or stale constraint handle" (ValueError).
    InvalidHandle = 1,
};

inline std::string_view ErrorMessage(ConstraintError) noexcept {
    return "Invalid or stale constraint handle";
}

// Drives a constraint's motor to `target`: a hinge angle, or a slider
// position/velocity depending on the motor state it is already in. Both bodies
// of a two-body constraint are woken so the change takes effect immediately.
// Takes shadow_lock itself and parks while a step is in flight.
[[nodiscard]] std::expected<void, Error> SetConstraintTarget(PhysicsWorld   &world,
                                                              ConstraintHandle handle,
                                                              float            target);

} // namespace CPH

#pragma once

#include "core/Types.hpp"
#include "jolt/Jolt.hpp"
#include "physics/PhysicsParams.hpp"
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Constraints/Constraint.h>

namespace CPH {

/// Creates a Jolt constraint of `type` between two bodies from Culverin's
/// flat parameter struct. The command flush is the only caller.
///
/// The returned constraint has a reference count of zero: the caller owns
/// nothing. `PhysicsSystem::AddConstraint` takes the system's reference and
/// `PhysicsSystem::RemoveConstraint` releases it, so the pointer is borrowed
/// (and dangling) from the moment it is added. Never `Release()` it.
JPH::Constraint *CreateConstraint(ConstraintType type, const ConstraintParams &params,
                                  JPH::Body &body1, JPH::Body &body2);

} // namespace CPH

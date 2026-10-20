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
/// The returned constraint carries one reference owned by the caller;
/// `PhysicsSystem::AddConstraint` takes its own, so release the returned
/// reference once the constraint is added.
JPH::Constraint *CreateConstraint(ConstraintType type, const ConstraintParams &params,
                                  JPH::Body &body1, JPH::Body &body2);

} // namespace CPH

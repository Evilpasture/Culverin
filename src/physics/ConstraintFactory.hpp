#pragma once

#include "core/Types.hpp"
#include "physics/PhysicsParams.hpp"
#include <joltc.h>

// Creates a Jolt constraint of `type` between two bodies from Culverin's
// flat parameter struct. Replaces the six per-type factory functions; the
// command flush is the only caller.
JPH_Constraint *CPH_CreateConstraint(ConstraintType type, const CPH_ConstraintParams &params,
                                     JPH_Body *body1, JPH_Body *body2);

#pragma once

// Free helpers whose subject is a settings value rather than a world,
// plus the module-init ABI self-check. World operations are declared
// as PhysicsWorld members in PhysicsWorldState.hpp.

#include "core/Types.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>


namespace CPH {

void ConfigureBodySettings(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                               BodyConfig config);
void ApplyBodyCreationProps(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                                BodyCreationProps props);

// Startup self-check, run once per world. Creates a static box at a known
// position, reads it back through the body interface and destroys it again.
// It used to straddle the JoltC boundary and so could catch a C/native ABI
// mismatch; with JoltC gone both sides are the same headers, so what remains
// is a runtime assertion that positions really round-trip at the configured
// precision (i.e. that DOUBLE_PRECISION took effect). Returns 0 or -1.
bool PrimeBodyIdFreeList(JPH::BodyInterface &body_interface);


} // namespace CPH
